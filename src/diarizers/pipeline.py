"""Unified 4-stage ASR + Diarization + Alignment + DiarizationLM Pipeline."""

from collections.abc import Callable, Sequence
from typing import List, Optional, Tuple, Union

import numpy as np

from diarizers import alignment
from diarizers import asr as asr_lib
from diarizers import configs
from diarizers import diarization as diar_lib
from diarizers import llm as llm_lib
from diarizers import types


class DiarizersPipeline:
  """End-to-end 4-stage pipeline for speech recognition and diarization."""

  def __init__(
      self,
      asr_backend: Optional[Callable[..., List[types.WordSegment]]] = None,
      diarizer_backend: Optional[Callable[..., List[types.SpeakerTurn]]] = None,
      llm_refiner: Optional[llm_lib.DiarizationLMRefiner] = None,
      prompt_config: Optional[configs.PromptConfig] = None,
      asr: Optional[Callable[..., List[types.WordSegment]]] = None,
      diarizer: Optional[Callable[..., List[types.SpeakerTurn]]] = None,
      use_locality_gate: Optional[bool] = None,
  ):
    self.asr_backend = asr_backend or asr or asr_lib.WhisperWordASR()
    self.diarizer_backend = (
        diarizer_backend or diarizer or diar_lib.ECAPASpectralDiarizer()
    )
    if prompt_config is None:
      if use_locality_gate is not None:
        prompt_config = configs.PromptConfig(
            use_locality_gate=use_locality_gate
        )
      else:
        prompt_config = configs.DEFAULT_PROMPT_CONFIG
    elif use_locality_gate is not None:
      prompt_config = configs.PromptConfig(
          speaker_prefix=prompt_config.speaker_prefix,
          speaker_suffix=prompt_config.speaker_suffix,
          prompt_prefix=prompt_config.prompt_prefix,
          prompt_suffix=prompt_config.prompt_suffix,
          completion_suffix=prompt_config.completion_suffix,
          emit_input_length=prompt_config.emit_input_length,
          emit_target_length=prompt_config.emit_target_length,
          use_gemma_turn_template=prompt_config.use_gemma_turn_template,
          use_locality_gate=use_locality_gate,
          max_local_span_words=prompt_config.max_local_span_words,
          max_two_spk_span_words=prompt_config.max_two_spk_span_words,
          max_ghost_cluster_words=prompt_config.max_ghost_cluster_words,
          overlap_words=prompt_config.overlap_words,
      )
    self.llm_refiner = llm_refiner
    self.prompt_config = prompt_config

  @classmethod
  def from_pretrained(
      cls,
      asr_model: str = "openai/whisper-large-v3-turbo",
      diarizer_preset: configs.DiarizationPreset = (
          configs.FISHER_TELEPHONY_PRESET
      ),
      segmentation_model: Optional[str] = "default",
      llm_model_or_path: Optional[str] = None,
      llm_gguf_filename: Optional[str] = None,
      llm_backend: str = "auto",
      device: str = "cpu",
      use_locality_gate: bool = True,
  ) -> "DiarizersPipeline":
    """Constructs a complete DiarizersPipeline from open-source checkpoints."""
    asr_engine = asr_lib.WhisperWordASR(
        model_name_or_path=asr_model, device=device
    )
    diar_engine = diar_lib.ECAPASpectralDiarizer(
        preset=diarizer_preset,
        device=device,
        segmentation_model=segmentation_model,
    )
    p_cfg = configs.PromptConfig(use_locality_gate=use_locality_gate)
    refiner = (
        llm_lib.DiarizationLMRefiner(
            model_path_or_repo=llm_model_or_path,
            gguf_filename=llm_gguf_filename,
            backend=llm_backend,
            prompt_config=p_cfg,
            device=device,
        )
        if llm_model_or_path
        else None
    )
    return cls(
        asr_backend=asr_engine,
        diarizer_backend=diar_engine,
        llm_refiner=refiner,
        prompt_config=p_cfg,
    )

  def __call__(
      self,
      audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]],
      utterance_id: str = "audio",
      num_speakers: Optional[int] = None,
      precomputed_words: Optional[Sequence[types.WordSegment]] = None,
      precomputed_turns: Optional[Sequence[types.SpeakerTurn]] = None,
  ) -> types.DiarizedTranscript:
    """Executes ASR, Diarization, Word Alignment, and DiarizationLM."""
    # Stage 1: Word-timestamped ASR
    if precomputed_words is not None:
      words = list(precomputed_words)
    else:
      words = self.asr_backend(audio_input)

    # Stage 2: Acoustic Speaker Diarization
    if precomputed_turns is not None:
      turns = list(precomputed_turns)
    else:
      turns = self.diarizer_backend(audio_input, num_speakers=num_speakers)

    # Stage 3: Word-Speaker Alignment
    acoustic_spks = alignment.assign_speakers(
        words, turns, normalize_speaker_ids=True
    )
    word_strs = [w.word for w in words]
    hyp_text = " ".join(word_strs)
    hyp_spk = " ".join(acoustic_spks)

    # Stage 4: DiarizationLM Post-Processing + LG-TPST
    if self.llm_refiner is not None and word_strs:
      raw_tpst, lg_tpst, _ = self.llm_refiner.refine_utterance(
          hyp_text=hyp_text,
          hyp_spk=hyp_spk,
          utt_id=utterance_id,
      )
    else:
      raw_tpst = hyp_spk
      lg_tpst = hyp_spk

    final_spks = lg_tpst.split()
    if len(final_spks) != len(words):
      final_spks = acoustic_spks

    diarized_words = [
        types.DiarizedWord(
            word=w.word,
            start=w.start,
            end=w.end,
            acoustic_speaker=a_spk,
            speaker=f_spk,
        )
        for w, a_spk, f_spk in zip(words, acoustic_spks, final_spks)
    ]

    before_llm_str = alignment.create_diarized_text(
        word_strs,
        acoustic_spks,
        speaker_prefix=self.prompt_config.speaker_prefix,
        speaker_suffix=self.prompt_config.speaker_suffix,
    )
    after_llm_str = alignment.create_diarized_text(
        word_strs,
        final_spks,
        speaker_prefix=self.prompt_config.speaker_prefix,
        speaker_suffix=self.prompt_config.speaker_suffix,
    )

    return types.DiarizedTranscript(
        utterance_id=utterance_id,
        words=diarized_words,
        hyp_text=hyp_text,
        hyp_spk=hyp_spk,
        hyp_spk_llm_raw=raw_tpst,
        hyp_spk_llm=lg_tpst,
        diarized_before_llm=before_llm_str,
        diarized_after_llm=after_llm_str,
    )


DiarizationPipeline = DiarizersPipeline
Pipeline = DiarizersPipeline
