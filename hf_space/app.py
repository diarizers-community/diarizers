"""Gradio 5.50.0 CPU end-to-end audio demo for the Diarizers pipeline."""

from __future__ import annotations

from collections.abc import Iterator
import os
import sys
import time
from typing import Tuple

import gradio as gr

# Allow running directly from repo checkout or HF Space root
REPO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
  sys.path.insert(0, REPO_SRC)

import diarizers  # noqa: E402

DEFAULT_GGUF_REPO = "diarizers-community/DiarizationLM-Gemma-4-E4B-v1"
DEFAULT_GGUF_FILE = "DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf"
EXAMPLE_AUDIO_PATH = os.path.join(
    os.path.dirname(__file__), "examples", "example_conversation.wav"
)

_ASR_CACHE: dict[str, diarizers.WhisperWordASR] = {}
_DIARIZER_CACHE: dict[str, diarizers.ECAPASpectralDiarizer] = {}
_REFINER_CACHE: dict[bool, diarizers.DiarizationLMRefiner] = {}


def _get_asr(model_name: str) -> diarizers.WhisperWordASR:
  if model_name not in _ASR_CACHE:
    _ASR_CACHE[model_name] = diarizers.WhisperWordASR(
        model_name_or_path=model_name,
        device="cpu",
    )
  return _ASR_CACHE[model_name]


def _get_diarizer(preset_name: str) -> diarizers.ECAPASpectralDiarizer:
  ami_preset = getattr(
      diarizers, "AMI_MEETING_PRESET", diarizers.MEETING_MULTISPK_PRESET
  )
  preset_map = {
      "Fisher Telephony (2 speakers)": diarizers.FISHER_TELEPHONY_PRESET,
      "Callhome Telephony (2-6 speakers)": diarizers.CALLHOME_TELEPHONY_PRESET,
      "AMI Meeting (3-5 speakers)": ami_preset,
      "Meeting Multi-Speaker (2-10 speakers)": (
          diarizers.MEETING_MULTISPK_PRESET
      ),
  }
  preset = preset_map.get(preset_name, diarizers.FISHER_TELEPHONY_PRESET)
  if preset.name not in _DIARIZER_CACHE:
    _DIARIZER_CACHE[preset.name] = diarizers.ECAPASpectralDiarizer(
        preset=preset,
        device="cpu",
    )
  return _DIARIZER_CACHE[preset.name]


def _get_refiner(use_lg_tpst: bool) -> diarizers.DiarizationLMRefiner:
  key = bool(use_lg_tpst)
  if key not in _REFINER_CACHE:
    p_cfg = diarizers.PromptConfig(use_locality_gate=key)
    _REFINER_CACHE[key] = diarizers.DiarizationLMRefiner(
        model_path_or_repo=DEFAULT_GGUF_REPO,
        gguf_filename=DEFAULT_GGUF_FILE,
        backend="gguf",
        prompt_config=p_cfg,
    )
  return _REFINER_CACHE[key]


def run_audio_pipeline(
    audio_path: str,
    asr_model_name: str,
    preset_name: str,
    num_speakers: int,
    use_gguf_model: bool = True,
    use_lg_tpst: bool = True,
) -> Iterator[Tuple[str, str, str, str, str]]:
  """Runs end-to-end Whisper ASR + Acoustic Diarization + DiarizationLM."""
  if not audio_path:
    yield (
        "Please upload, record, or click an example audio file.",
        "",
        "",
        "",
        "",
    )
    return

  t0 = time.time()
  yield (
      f"[Stage 1/3] Running Whisper ASR ({asr_model_name})...",
      "",
      "",
      "",
      "",
  )
  asr_backend = _get_asr(asr_model_name)
  word_segments = asr_backend(audio_path)
  t_asr = time.time() - t0
  if not word_segments:
    yield (
        "No speech words detected in the audio file.",
        "",
        "",
        "",
        "",
    )
    return

  t1 = time.time()
  yield (
      f"[Stage 2/3] ASR finished in {t_asr:.1f}s ({len(word_segments)} words). "
      f"Running Acoustic Diarization ({preset_name})...",
      "",
      "",
      "",
      "",
  )
  diarizer_backend = _get_diarizer(preset_name)
  n_spk = int(num_speakers) if int(num_speakers) > 0 else None
  turns = diarizer_backend(audio_path, num_speakers=n_spk)
  t_diar = time.time() - t1

  # Stage 3: Align words to speaker turns
  asr_dicts = [
      {"word": w.word, "timestamp": (w.start, w.end)} for w in word_segments
  ]
  turn_dicts = [
      {"segment": {"start": t.start, "end": t.end}, "speaker": t.speaker}
      for t in turns
  ]
  acoustic_spks = diarizers.assign_speakers(asr_dicts, turn_dicts)
  words_str = " ".join(w.word for w in word_segments)
  hyp_spk_str = " ".join(acoustic_spks)
  before_llm_text = diarizers.create_diarized_text(
      [w.word for w in word_segments], acoustic_spks
  )

  # Build initial transcript before LLM
  diar_words_pre = [
      diarizers.DiarizedWord(
          word=w.word,
          start=w.start,
          end=w.end,
          acoustic_speaker=s,
          speaker=s,
      )
      for w, s in zip(word_segments, acoustic_spks)
  ]
  dt_pre = diarizers.DiarizedTranscript(
      utterance_id="recording",
      words=diar_words_pre,
      hyp_text=words_str,
      hyp_spk=hyp_spk_str,
      hyp_spk_llm_raw=hyp_spk_str,
      hyp_spk_llm=hyp_spk_str,
      diarized_before_llm=before_llm_text,
      diarized_after_llm=before_llm_text,
  )

  if not use_gguf_model:
    status = (
        f"Done in {t_asr + t_diar:.1f}s (ASR: {t_asr:.1f}s, "
        f"Diarization: {t_diar:.1f}s) | Words: {len(word_segments)} | "
        f"Speakers: {len(set(acoustic_spks))} | DiarizationLM: disabled"
    )
    yield (
        status,
        before_llm_text,
        before_llm_text,
        dt_pre.to_rttm(),
        dt_pre.to_srt(),
    )
    return

  yield (
      f"[Stage 3/3] Acoustic frontend finished in {t_asr + t_diar:.1f}s. "
      f"Running {DEFAULT_GGUF_REPO} (Q4_K_M GGUF) + LG-TPST...",
      before_llm_text,
      "Running DiarizationLM-Gemma-4-E4B-v1 on CPU...",
      dt_pre.to_rttm(),
      dt_pre.to_srt(),
  )

  t2 = time.time()
  refiner = _get_refiner(use_lg_tpst)
  raw_llm_spk, final_llm_spk, _ = refiner.refine_utterance(
      words_str, hyp_spk_str
  )
  t_llm = time.time() - t2
  final_spks = final_llm_spk.split()
  after_llm_text = diarizers.create_diarized_text(
      [w.word for w in word_segments], final_spks
  )

  diar_words_post = [
      diarizers.DiarizedWord(
          word=w.word,
          start=w.start,
          end=w.end,
          acoustic_speaker=ac_s,
          speaker=fn_s,
      )
      for w, ac_s, fn_s in zip(word_segments, acoustic_spks, final_spks)
  ]
  dt_post = diarizers.DiarizedTranscript(
      utterance_id="recording",
      words=diar_words_post,
      hyp_text=words_str,
      hyp_spk=hyp_spk_str,
      hyp_spk_llm_raw=raw_llm_spk,
      hyp_spk_llm=final_llm_spk,
      diarized_before_llm=before_llm_text,
      diarized_after_llm=after_llm_text,
  )
  changed = sum(1 for a, b in zip(acoustic_spks, final_spks) if a != b)
  total_t = t_asr + t_diar + t_llm
  status = (
      f"Completed in {total_t:.1f}s (ASR: {t_asr:.1f}s, "
      f"Diarization: {t_diar:.1f}s, DiarizationLM: {t_llm:.1f}s) | "
      f"Words: {len(word_segments)} | Speakers: {len(set(final_spks))} | "
      f"Words relabeled by DiarizationLM: {changed}/{len(word_segments)}"
  )
  yield (
      status,
      before_llm_text,
      after_llm_text,
      dt_post.to_rttm(),
      dt_post.to_srt(),
  )


def build_demo() -> gr.Blocks:
  """Builds the Gradio Blocks interface for the end-to-end audio pipeline."""
  with gr.Blocks(
      title="Diarizers — End-to-End Audio ASR + Diarization + DiarizationLM"
  ) as demo:
    gr.Markdown(
        "# 🎙️ Diarizers — End-to-End Open-Source ASR, Speaker Diarization & "
        "DiarizationLM\n"
        "> End-to-end Apache-2.0/MIT audio pipeline combining **Whisper ASR** "
        "+ **`pyannote/segmentation-3.0`** + "
        "**`speechbrain/spkrec-ecapa-voxceleb`** "
        "+ **Symmetric Normalized Laplacian `SpectralClusterer`** + "
        "**[`DiarizationLM-Gemma-4-E4B-v1`]"
        "(https://huggingface.co/diarizers-community/"
        "DiarizationLM-Gemma-4-E4B-v1)** (`Q4_K_M` GGUF + `LG-TPST`).\n>\n"
        "> 📦 **GitHub**: [`diarizers-community/diarizers`]"
        "(https://github.com/diarizers-community/diarizers) "
        "| **PyPI**: `pip install diarizers` "
        "| 💬 **Text-only DiarizationLM GGUF Demo**: "
        "[`spaces/diarizers-community/DiarizationLM-GGUF`]"
        "(https://huggingface.co/spaces/diarizers-community/"
        "DiarizationLM-GGUF)"
    )

    with gr.Row():
      with gr.Column(scale=1):
        audio_in = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            value=(
                EXAMPLE_AUDIO_PATH
                if os.path.isfile(EXAMPLE_AUDIO_PATH)
                else None
            ),
            label="Input Conversation Audio (.wav / .mp3 / .flac)",
        )
        asr_dd = gr.Dropdown(
            choices=[
                "openai/whisper-base.en",
                "openai/whisper-small.en",
                "openai/whisper-large-v3-turbo",
            ],
            value="openai/whisper-base.en",
            label=(
                "Stage 1: Whisper ASR Model "
                "(`whisper-base.en` recommended on CPU)"
            ),
        )
        preset_dd = gr.Dropdown(
            choices=[
                "Fisher Telephony (2 speakers)",
                "Callhome Telephony (2-6 speakers)",
                "AMI Meeting (3-5 speakers)",
                "Meeting Multi-Speaker (2-10 speakers)",
            ],
            value="Fisher Telephony (2 speakers)",
            label="Stage 2: Acoustic Diarization Preset",
        )
        num_spk_slider = gr.Slider(
            minimum=0,
            maximum=8,
            step=1,
            value=2,
            label=(
                "Known Speaker Count "
                "(0 = Auto-detect via Laplacian Eigengap)"
            ),
        )
        audio_use_gguf = gr.Checkbox(
            value=True,
            label=(
                "Stage 4: Enable DiarizationLM-Gemma-4-E4B-v1 "
                "(Q4_K_M GGUF on CPU)"
            ),
        )
        use_lg = gr.Checkbox(
            value=True,
            label="Enable Locality-Gated Speaker Transfer (LG-TPST)",
        )
        run_audio_btn = gr.Button(
            "Run End-to-End Audio Pipeline", variant="primary"
        )

      with gr.Column(scale=1):
        out_status = gr.Textbox(
            label="Pipeline Status & Diagnostics", lines=2
        )
        out_before_llm = gr.Textbox(
            label="Before DiarizationLM (Acoustic Baseline)", lines=5
        )
        out_after_llm = gr.Textbox(
            label="After DiarizationLM-Gemma-4-E4B-v1 (Refined Transcript)",
            lines=5,
        )
        out_rttm = gr.Textbox(label="NIST RTTM Output", lines=5)
        out_srt = gr.Textbox(label="SRT Subtitles", lines=5)

    if os.path.isfile(EXAMPLE_AUDIO_PATH):
      gr.Examples(
          examples=[
              [
                  EXAMPLE_AUDIO_PATH,
                  "openai/whisper-base.en",
                  "Fisher Telephony (2 speakers)",
                  2,
                  True,
                  True,
              ]
          ],
          inputs=[
              audio_in,
              asr_dd,
              preset_dd,
              num_spk_slider,
              audio_use_gguf,
              use_lg,
          ],
      )

    run_audio_btn.click(
        fn=run_audio_pipeline,
        inputs=[
            audio_in,
            asr_dd,
            preset_dd,
            num_spk_slider,
            audio_use_gguf,
            use_lg,
        ],
        outputs=[
            out_status,
            out_before_llm,
            out_after_llm,
            out_rttm,
            out_srt,
        ],
    )

  return demo


if __name__ == "__main__":
  app = build_demo()
  app.queue(max_size=3).launch()
