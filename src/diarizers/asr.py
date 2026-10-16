"""Open-source Apache-2.0 ASR backends with word-level timestamps."""

from collections.abc import Sequence
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
  import soundfile as sf
except ImportError:  # pragma: no cover
  sf = None  # type: ignore

try:
  import torch
except ImportError:  # pragma: no cover
  torch = None  # type: ignore

from diarizers import alignment
from diarizers import diarization
from diarizers import types


def _deduplicate_repetitions(
    words: List[str],
    max_ngram: int = 6,
    max_repeats: int = 2,
) -> List[str]:
  """Collapses consecutive repeated n-grams from ASR hallucination loops."""
  if len(words) <= max_repeats:
    return words
  out = list(words)
  for n in range(1, max_ngram + 1):
    cleaned: List[str] = []
    i = 0
    while i < len(out):
      cleaned.append(out[i])
      if len(cleaned) >= n * (max_repeats + 1):
        tail = cleaned[-n:]
        is_rep = all(
            cleaned[-n * (r + 1):-n * r] == tail
            for r in range(1, max_repeats + 1)
        )
        if is_rep:
          del cleaned[-n:]
      i += 1
    out = cleaned
  return out


def turns_to_word_segments(
    turns: Sequence[Dict[str, Any]],
) -> List[types.WordSegment]:
  """Interpolates word-level timestamps from turn-level transcripts."""
  words_out: List[types.WordSegment] = []
  for turn in turns:
    txt = str(turn.get("transcript", turn.get("text", "")))
    tw = alignment.normalize_text(txt).split()
    if not tw:
      continue
    ts = float(turn.get("start", 0.0))
    te = float(turn.get("end", ts + 0.5))
    dur = max(0.05, te - ts)
    tot_chars = max(1, sum(len(w) for w in tw))
    cur = ts
    spk = turn.get("speaker")
    spk_str = str(spk) if spk is not None else None
    for w in tw:
      wd = dur * (len(w) / tot_chars)
      words_out.append(
          types.WordSegment(
              word=w,
              start=cur,
              end=cur + wd,
              speaker=spk_str,
          )
      )
      cur += wd
  return words_out


class WhisperWordASR:
  """MIT/Apache-2.0 Whisper ASR backend producing WordSegments on CPU or GPU."""

  def __init__(
      self,
      model_name_or_path: str = "openai/whisper-large-v3-turbo",
      device: str = "cpu",
      batch_size: int = 4,
      chunk_seconds: int = 30,
      num_threads: int = 8,
      language: Optional[str] = "en",
  ):
    self.model_name_or_path = model_name_or_path
    self.device = device
    self.batch_size = max(1, int(batch_size))
    self.chunk_seconds = max(5, int(chunk_seconds))
    self.num_threads = max(1, int(num_threads))
    self.language = language
    self.processor: Any = None
    self.model: Any = None

  def _ensure_loaded(self) -> None:
    if self.model is not None and self.processor is not None:
      return
    if torch is not None and self.device == "cpu":
      torch.set_num_threads(self.num_threads)
    from transformers import WhisperForConditionalGeneration
    from transformers import WhisperProcessor

    resolved_path = self.model_name_or_path
    if resolved_path == "openai/whisper-large-v3-turbo":
      model_root = os.environ.get(
          "DIARIZERS_MODEL_DIR", "/tmp/diarizers_models"
      )
      cand = os.path.join(model_root, "whisper_large_v3_turbo")
      if os.path.isdir(cand) and os.path.isfile(
          os.path.join(cand, "config.json")
      ):
        resolved_path = cand

    self.processor = WhisperProcessor.from_pretrained(resolved_path)
    use_fp16 = (
        torch is not None
        and str(self.device).startswith("cuda")
        and torch.cuda.is_available()
    )
    if use_fp16:
      self.model = WhisperForConditionalGeneration.from_pretrained(
          resolved_path, torch_dtype=torch.float16
      )
    else:
      self.model = WhisperForConditionalGeneration.from_pretrained(
          resolved_path
      )
    self.model.to(self.device)
    self.model.eval()

  def __call__(
      self,
      audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]],
  ) -> List[types.WordSegment]:
    """Transcribes audio and returns normalized WordSegments."""
    self._ensure_loaded()
    wav = diarization.load_audio_16k(audio_input, target_sr=16000)
    if wav.size == 0:
      return []

    sr = 16000
    chunk_samples = self.chunk_seconds * sr
    chunks: List[np.ndarray] = []
    offsets: List[float] = []
    for s in range(0, len(wav), chunk_samples):
      chunks.append(wav[s:s + chunk_samples])
      offsets.append(s / float(sr))

    ts_pattern = re.compile(r"<\|(\d+\.\d+)\|>")
    words_out: List[types.WordSegment] = []
    gen_cfg = getattr(self.model, "generation_config", None)
    is_multilingual = bool(getattr(gen_cfg, "is_multilingual", False))
    gen_kwargs: Dict[str, Any] = {"return_timestamps": True}
    if is_multilingual and self.language:
      gen_kwargs["language"] = self.language
      gen_kwargs["task"] = "transcribe"

    for b_start in range(0, len(chunks), self.batch_size):
      b_chunks = chunks[b_start:b_start + self.batch_size]
      b_offsets = offsets[b_start:b_start + self.batch_size]
      inputs = self.processor(
          b_chunks,
          sampling_rate=sr,
          return_tensors="pt",
          return_attention_mask=True,
      )
      feats = (
          inputs.input_features.to(self.device)
          if hasattr(inputs.input_features, "to")
          else inputs.input_features
      )
      if (
          torch is not None
          and hasattr(self.model, "dtype")
          and hasattr(feats, "to")
          and isinstance(self.model.dtype, torch.dtype)
      ):
        feats = feats.to(dtype=self.model.dtype)
      attn_mask = (
          inputs.attention_mask.to(self.device)
          if hasattr(inputs.attention_mask, "to")
          else inputs.attention_mask
      )
      if torch is not None:
        with torch.no_grad():
          out_ids = self.model.generate(
              feats,
              attention_mask=attn_mask,
              **gen_kwargs,
          )
      else:
        out_ids = self.model.generate(
            feats,
            attention_mask=attn_mask,
            **gen_kwargs,
        )
      seq_ids = (
          out_ids.tolist() if hasattr(out_ids, "tolist") else out_ids
      )
      decoded = self.processor.batch_decode(
          seq_ids,
          skip_special_tokens=False,
          decode_with_timestamps=True,
      )

      for raw_str, offset in zip(decoded, b_offsets):
        cleaned = re.sub(
            r"<\|(?!\d+\.\d+\|)[a-zA-Z0-9_]+\|>", "", raw_str
        )
        parts = ts_pattern.split(cleaned)
        if len(parts) < 3:
          dehyphen = re.sub(r"(?<=\w)-(?=\w)", " ", cleaned)
          seg_words = _deduplicate_repetitions(
              alignment.normalize_text(dehyphen).split()
          )
          if seg_words:
            dur = float(self.chunk_seconds)
            tot_c = max(1, sum(len(w) for w in seg_words))
            cur_t = offset
            for w in seg_words:
              wd = dur * (len(w) / tot_c)
              words_out.append(
                  types.WordSegment(word=w, start=cur_t, end=cur_t + wd)
              )
              cur_t += wd
          continue

        i = 1
        while i < len(parts):
          t_val = float(parts[i])
          seg_text = parts[i + 1] if (i + 1) < len(parts) else ""
          if seg_text.strip():
            t_end = (
                float(parts[i + 2])
                if (i + 2) < len(parts)
                else min(float(self.chunk_seconds), t_val + 2.5)
            )
            if t_end <= t_val:
              t_end = min(float(self.chunk_seconds), t_val + 1.0)
            dehyphen = re.sub(r"(?<=\w)-(?=\w)", " ", seg_text)
            seg_words = _deduplicate_repetitions(
                alignment.normalize_text(dehyphen).split()
            )
            if seg_words:
              tot_c = max(1, sum(len(w) for w in seg_words))
              dur = max(0.05, t_end - t_val)
              cur_t = offset + t_val
              for w in seg_words:
                wd = dur * (len(w) / tot_c)
                words_out.append(
                    types.WordSegment(word=w, start=cur_t, end=cur_t + wd)
                )
                cur_t += wd
          i += 2

    return words_out


class AudioCppASR:
  """Local CPU ASR backend using audio.cpp CLI and Qwen3-ASR GGUF."""

  def __init__(
      self,
      gguf_model_path: str,
      audiocpp_cli_path: str = "audiocpp_cli",
      family: str = "qwen3_asr",
      threads: int = 8,
      chunk_seconds: int = 25,
  ):
    self.gguf_model_path = gguf_model_path
    self.audiocpp_cli_path = audiocpp_cli_path
    self.family = family
    self.threads = max(1, int(threads))
    self.chunk_seconds = max(5, int(chunk_seconds))

  def __call__(
      self,
      audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]],
  ) -> List[types.WordSegment]:
    """Transcribes audio in chunks via audiocpp_cli and returns WordSegments."""
    wav = diarization.load_audio_16k(audio_input, target_sr=16000)
    if wav.size == 0:
      return []

    sr = 16000
    chunk_samples = self.chunk_seconds * sr
    words_out: List[types.WordSegment] = []

    with tempfile.TemporaryDirectory() as tmpdir:
      for s in range(0, len(wav), chunk_samples):
        chunk_wav = wav[s:s + chunk_samples]
        offset = s / float(sr)
        dur = len(chunk_wav) / float(sr)
        wav_path = os.path.join(tmpdir, "chunk.wav")
        txt_path = os.path.join(tmpdir, "chunk.txt")
        sf.write(wav_path, chunk_wav, sr)

        cmd = [
            self.audiocpp_cli_path,
            "--task",
            "asr",
            "--family",
            self.family,
            "--model",
            self.gguf_model_path,
            "--backend",
            "cpu",
            "--threads",
            str(self.threads),
            "--audio",
            wav_path,
            "--text-out",
            txt_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.isfile(txt_path):
          with open(txt_path, "r", encoding="utf-8") as f:
            txt = f.read()
          seg_words = alignment.normalize_text(txt).split()
          if seg_words:
            tot_c = max(1, sum(len(w) for w in seg_words))
            cur_t = offset
            for w in seg_words:
              wd = dur * (len(w) / tot_c)
              words_out.append(
                  types.WordSegment(word=w, start=cur_t, end=cur_t + wd)
              )
              cur_t += wd

    return words_out


class PrecomputedASR:
  """ASR adapter that returns precomputed WordSegments or turn transcripts."""

  def __init__(
      self,
      words_or_turns: Optional[Sequence[Any]] = None,
  ):
    self.words_or_turns = list(words_or_turns) if words_or_turns else []

  def __call__(
      self,
      audio_input: Any = None,
      words_or_turns: Optional[Sequence[Any]] = None,
  ) -> List[types.WordSegment]:
    del audio_input
    items = words_or_turns if words_or_turns is not None else (
        self.words_or_turns
    )
    if not items:
      return []
    first = items[0]
    if isinstance(first, types.WordSegment):
      return list(items)
    if isinstance(first, dict) and "word" in first:
      return [
          types.WordSegment(
              word=alignment.normalize_text(str(w["word"])),
              start=float(w.get("start", 0.0)),
              end=float(w.get("end", 0.0)),
              speaker=str(w["speaker"]) if w.get("speaker") else None,
          )
          for w in items
          if alignment.normalize_text(str(w["word"]))
      ]
    return turns_to_word_segments(items)
