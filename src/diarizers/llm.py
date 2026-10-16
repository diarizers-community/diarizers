"""DiarizationLM post-processor supporting GGUF, Transformers, and LG-TPST."""

from collections.abc import Callable, Sequence
import os
from typing import Any, Dict, List, Optional, Tuple

from diarizers import alignment
from diarizers import configs


class DiarizationLMRefiner:
  """Post-processes diarized transcripts with DiarizationLM + LG-TPST."""

  def __init__(
      self,
      model_path_or_repo: Optional[str] = None,
      gguf_filename: Optional[str] = None,
      backend: str = "auto",
      prompt_config: configs.PromptConfig = configs.DEFAULT_PROMPT_CONFIG,
      device: str = "cpu",
      n_ctx: int = 4096,
      n_threads: int = 8,
      completion_fn: Optional[Callable[[str], str]] = None,
      model_path: Optional[str] = None,
  ):
    self.model_path_or_repo = model_path_or_repo or model_path
    self.gguf_filename = gguf_filename
    self.backend = backend
    self.prompt_config = prompt_config
    self.device = device
    self.n_ctx = n_ctx
    self.n_threads = n_threads
    self.completion_fn = completion_fn
    self._llm_instance: Any = None
    self._hf_tokenizer: Any = None
    self._hf_model: Any = None

  def format_model_input(self, raw_prompt: str) -> str:
    """Formats chunk prompt with Gemma-4 turn template if enabled."""
    if self.prompt_config.use_gemma_turn_template:
      return f"<|turn>user\n{raw_prompt}<turn|>\n<|turn>model\n"
    return raw_prompt

  def _ensure_loaded(self) -> None:
    if (
        self.completion_fn is not None
        or self._llm_instance is not None
        or self._hf_model is not None
    ):
      return
    if not self.model_path_or_repo:
      return

    use_gguf = (
        self.backend == "gguf"
        or (self.gguf_filename is not None)
        or self.model_path_or_repo.endswith(".gguf")
    )
    if use_gguf:
      from llama_cpp import Llama

      if os.path.isfile(self.model_path_or_repo):
        gguf_path = self.model_path_or_repo
      else:
        from huggingface_hub import hf_hub_download

        fname = (
            self.gguf_filename or "DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf"
        )
        gguf_path = hf_hub_download(
            repo_id=self.model_path_or_repo, filename=fname
        )
      self._llm_instance = Llama(
          model_path=gguf_path,
          n_ctx=self.n_ctx,
          n_threads=self.n_threads,
          verbose=False,
      )
    else:
      import torch
      import transformers

      self._hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
          self.model_path_or_repo
      )
      self._hf_model = transformers.AutoModelForCausalLM.from_pretrained(
          self.model_path_or_repo,
          torch_dtype=(
              torch.bfloat16 if self.device != "cpu" else torch.float32
          ),
      )
      self._hf_model.to(self.device)
      self._hf_model.eval()

  def generate_chunk_completion(self, raw_prompt: str) -> str:
    """Generates raw text completion for a single chunk prompt."""
    if self.completion_fn is not None:
      return self.completion_fn(raw_prompt)

    self._ensure_loaded()
    formatted = self.format_model_input(raw_prompt)

    if self._llm_instance is not None:
      out = self._llm_instance(
          formatted,
          max_tokens=1024,
          temperature=0.0,
          stop=[self.prompt_config.completion_suffix.strip(), "<turn|>"],
      )
      return str(out["choices"][0]["text"]).strip()

    if self._hf_model is not None and self._hf_tokenizer is not None:
      import torch

      inputs = self._hf_tokenizer(formatted, return_tensors="pt").to(
          self.device
      )
      in_len = int(inputs.input_ids.shape[1])
      with torch.no_grad():
        out_ids = self._hf_model.generate(
            **inputs,
            max_new_tokens=max(128, int(in_len * 1.2)),
            do_sample=False,
            pad_token_id=self._hf_tokenizer.eos_token_id,
        )
      comp = self._hf_tokenizer.decode(
          out_ids[0][in_len:], skip_special_tokens=True
      )
      return str(comp).strip()

    # Identity pass-through when no LLM is configured
    body = raw_prompt
    if body.endswith(self.prompt_config.prompt_suffix):
      body = body[:-len(self.prompt_config.prompt_suffix)]
    return body + self.prompt_config.completion_suffix

  def refine_utterance(
      self,
      hyp_text: str,
      hyp_spk: str,
      utt_id: str = "utt",
      precomputed_completions: Optional[Sequence[str]] = None,
  ) -> Tuple[str, str, List[str]]:
    """Refines speaker sequence and returns (raw_tpst, lg_tpst, completions)."""
    words = alignment.normalize_text(hyp_text).split()
    spks = alignment.speakers_transform(hyp_spk.split())
    if not words or len(words) != len(spks):
      return hyp_spk, hyp_spk, []

    chunker = alignment.UtteranceChunker(
        max_input_length=self.prompt_config.emit_input_length,
        max_target_length=self.prompt_config.emit_target_length,
        prompt_prefix=self.prompt_config.prompt_prefix,
        prompt_suffix=self.prompt_config.prompt_suffix,
        completion_suffix=self.prompt_config.completion_suffix,
        speaker_prefix=self.prompt_config.speaker_prefix,
        speaker_suffix=self.prompt_config.speaker_suffix,
        overlap_words=self.prompt_config.overlap_words,
    )

    chunks = list(chunker.chunk_utterance_fields(words, spks, utt_id=utt_id))
    if precomputed_completions is not None:
      completions = list(precomputed_completions)
    else:
      completions = [
          self.generate_chunk_completion(ch["prompt"]) for ch in chunks
      ]

    raw_tpst, lg_tpst = alignment.reconcile_chunk_speakers_hungarian(
        utt_words=words,
        utt_hyp_spks=spks,
        chunk_completions=completions,
        chunker=chunker,
        use_locality_gate=self.prompt_config.use_locality_gate,
        max_local_span_words=self.prompt_config.max_local_span_words,
    )
    return raw_tpst, lg_tpst, completions

  def refine_transcript(
      self,
      diarized_text: str,
      utt_id: str = "utt",
  ) -> Tuple[str, str]:
    """Refines a speaker-tagged transcript and returns (text, refined_spk)."""
    hyp_text, hyp_spk = alignment.extract_text_and_spk(
        diarized_text,
        speaker_prefix=self.prompt_config.speaker_prefix,
        speaker_suffix=self.prompt_config.speaker_suffix,
    )
    _, lg_tpst, _ = self.refine_utterance(
        hyp_text=hyp_text,
        hyp_spk=hyp_spk,
        utt_id=utt_id,
    )
    return hyp_text, lg_tpst

  def refine_json_dict(
      self, data_dict: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Refines all utterances in a dataset dictionary."""
    out_utts: List[Dict[str, Any]] = []
    for idx, utt in enumerate(data_dict.get("utterances", [])):
      u_copy = dict(utt)
      u_id = str(u_copy.get("utterance_id", f"utt_{idx}"))
      h_text = str(u_copy.get("hyp_text", u_copy.get("ref_text", "")))
      h_spk = str(u_copy.get("hyp_spk", u_copy.get("ref_spk_degraded", "")))
      pre_comp = u_copy.get("completions")
      raw_tpst, lg_tpst, comps = self.refine_utterance(
          hyp_text=h_text,
          hyp_spk=h_spk,
          utt_id=u_id,
          precomputed_completions=pre_comp,
      )
      u_copy["completions"] = comps
      u_copy["hyp_spk_llm_raw"] = raw_tpst
      u_copy["hyp_spk_llm"] = lg_tpst
      out_utts.append(u_copy)
    return {"utterances": out_utts}

