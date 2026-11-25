"""Gradio 5.50.0 CPU end-to-end audio demo for Diarizers v1.2.0."""

from __future__ import annotations

from collections.abc import Iterator
import gc
import os
import sys
import threading
import time
from typing import Optional, Tuple

import gradio as gr

# Allow running directly from repo checkout or HF Space root
REPO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
  sys.path.insert(0, REPO_SRC)

import diarizers  # noqa: E402

PROFILE_LABELS = {
    (
        "fast — ECAPA + Gemma-4-E4B-v2 "
        "(~7.6 GiB RAM, 2x faster, 4.13% WDER)"
    ): "fast",
    (
        "best-quality — FlaxSpeaker-XL-Diar-Adapt + Gemma-4-12B-v1 "
        "(3.82% micro / 5.16% macro WDER, 0.177 MAE)"
    ): "best-quality",
    (
        "max — FlaxSpeaker-XL-Diar-Adapt + ECAPA Fusion + Gemma-4-12B-v1 "
        "(3.79% micro / 5.14% macro WDER, 7.92% AMI)"
    ): "max",
    (
        "balanced — ECAPA + Gemma-4-12B-v1 "
        "(15x faster encoder, ~14.3 GiB RAM, 3.88% WDER)"
    ): "balanced",
    (
        "acoustic-only — ECAPA powerset_ward "
        "(<2 GiB RAM, no LLM)"
    ): "acoustic-only",
}
DEFAULT_PROFILE_LABEL = next(iter(PROFILE_LABELS.keys()))

PRESET_LABELS = {
    "Profile Default (Recommended)": None,
    "FlaxSpeaker-XL-Diar-Adapt Auto (powerset_ward)": (
        diarizers.FLAXSPEAKER_XL_DIAR_PRESET
    ),
    "FlaxSpeaker-XL-Diar + ECAPA Fusion (powerset_ward)": (
        diarizers.FLAXSPEAKER_DIAR_ECAPA_FUSION_PRESET
    ),
    "FlaxSpeaker-V2-Diar Auto (compact powerset_ward)": (
        diarizers.FLAXSPEAKER_V2_DIAR_PRESET
    ),
    "Universal Auto (ECAPA powerset_ward)": diarizers.UNIVERSAL_AUTO_PRESET,
    "FlaxSpeaker-XL Auto (legacy VoxCeleb powerset_ward)": (
        diarizers.FLAXSPEAKER_XL_PRESET
    ),
    "FlaxSpeaker-Large Auto (powerset_ward)": (
        diarizers.FLAXSPEAKER_LARGE_PRESET
    ),
    "Fisher Telephony (2 speakers)": diarizers.FISHER_TELEPHONY_PRESET,
    "Callhome Telephony (2-6 speakers)": diarizers.CALLHOME_TELEPHONY_PRESET,
    "AMI Meeting (3-5 speakers)": diarizers.AMI_MEETING_PRESET,
    "Meeting Multi-Speaker (2-10 speakers)": (
        diarizers.MEETING_MULTISPK_PRESET
    ),
}

EXAMPLE_AUDIO_1 = os.path.join(
    os.path.dirname(__file__), "examples", "example_conversation.wav"
)
EXAMPLE_AUDIO_2 = os.path.join(
    os.path.dirname(__file__), "examples", "example_2_meeting_3spk.wav"
)

_LOCK = threading.Lock()
_ASR_CACHE: dict[str, diarizers.WhisperWordASR] = {}
_DIARIZER_CACHE: dict[str, diarizers.ECAPASpectralDiarizer] = {}
_ACTIVE_REFINER_REPO: Optional[str] = None
_ACTIVE_REFINER: Optional[diarizers.DiarizationLMRefiner] = None


def _get_asr(model_name: str) -> diarizers.WhisperWordASR:
  if model_name not in _ASR_CACHE:
    _ASR_CACHE[model_name] = diarizers.WhisperWordASR(
        model_name_or_path=model_name,
        device="cpu",
    )
  return _ASR_CACHE[model_name]


def _get_diarizer(
    profile_key: str,
    preset_label: str,
) -> diarizers.ECAPASpectralDiarizer:
  override = PRESET_LABELS.get(preset_label)
  prof = diarizers.resolve_pipeline_profile(profile_key)
  preset = override if override is not None else prof.diarization
  if preset.name not in _DIARIZER_CACHE:
    _DIARIZER_CACHE[preset.name] = diarizers.ECAPASpectralDiarizer(
        preset=preset,
        device="cpu",
    )
  return _DIARIZER_CACHE[preset.name]


def _get_refiner(
    repo_id: str,
    use_lg_tpst: bool,
) -> diarizers.DiarizationLMRefiner:
  """Loads one active GGUF model at a time so 16 GiB RAM machines stay safe."""
  global _ACTIVE_REFINER, _ACTIVE_REFINER_REPO
  p_cfg = diarizers.PromptConfig(use_locality_gate=bool(use_lg_tpst))
  if _ACTIVE_REFINER is not None and _ACTIVE_REFINER_REPO == repo_id:
    _ACTIVE_REFINER.prompt_config = p_cfg
    return _ACTIVE_REFINER
  if _ACTIVE_REFINER is not None:
    _ACTIVE_REFINER = None
    _ACTIVE_REFINER_REPO = None
    gc.collect()
  n_threads = max(2, min(8, os.cpu_count() or 4))
  _ACTIVE_REFINER = diarizers.DiarizationLMRefiner(
      model_path_or_repo=repo_id,
      backend="gguf",
      n_ctx=4096,
      n_threads=n_threads,
      prompt_config=p_cfg,
  )
  _ACTIVE_REFINER_REPO = repo_id
  return _ACTIVE_REFINER


def run_audio_pipeline(
    audio_path: str,
    profile_label: str,
    asr_model_name: str,
    preset_label: str,
    num_speakers: int,
    use_lg_tpst: bool = True,
) -> Iterator[Tuple[str, str, str, str, str]]:
  """Runs end-to-end Whisper ASR + powerset_ward Diarization + DiarizationLM."""
  if not audio_path:
    yield (
        "Please upload, record, or click an example audio file.",
        "",
        "",
        "",
        "",
    )
    return

  profile_key = PROFILE_LABELS.get(profile_label, "fast")
  prof = diarizers.resolve_pipeline_profile(profile_key)

  with _LOCK:
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

    diarizer_backend = _get_diarizer(profile_key, preset_label)
    t1 = time.time()
    yield (
        f"[Stage 2/3] ASR finished in {t_asr:.1f}s "
        f"({len(word_segments)} words). Running {diarizer_backend.preset.name}"
        f" ({diarizer_backend.preset.diarization_mode})...",
        "",
        "",
        "",
        "",
    )
    n_spk = int(num_speakers) if int(num_speakers) > 0 else None
    turns = diarizer_backend(audio_path, num_speakers=n_spk)
    t_diar = time.time() - t1

    if diarizer_backend.last_frame_probs is not None:
      acoustic_spks = diarizers.assign_speakers_from_probs(
          word_segments,
          diarizer_backend.last_frame_probs,
          diarizer_backend.last_frame_dt,
      )
    else:
      acoustic_spks = diarizers.assign_speakers(word_segments, turns)

    words_list = [w.word for w in word_segments]
    words_str = " ".join(words_list)
    hyp_spk_str = " ".join(acoustic_spks)
    before_llm_text = diarizers.create_diarized_text(words_list, acoustic_spks)

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

    if not prof.llm_model_or_path:
      status = (
          f"Done in {t_asr + t_diar:.1f}s (ASR: {t_asr:.1f}s, "
          f"Diarization: {t_diar:.1f}s) | Profile: {prof.name} | "
          f"Words: {len(word_segments)} | Speakers: {len(set(acoustic_spks))}"
      )
      yield (
          status,
          before_llm_text,
          before_llm_text,
          dt_pre.to_rttm(),
          dt_pre.to_srt(),
      )
      return

    mode_str = "LG-TPST" if use_lg_tpst else "Raw-TPST"
    yield (
        f"[Stage 3/3] Acoustic frontend finished in {t_asr + t_diar:.1f}s. "
        f"Running {prof.llm_model_or_path} (Q4_K_M GGUF) + {mode_str}...",
        before_llm_text,
        f"Running {prof.llm_model_or_path} on CPU...",
        dt_pre.to_rttm(),
        dt_pre.to_srt(),
    )

    t2 = time.time()
    refiner = _get_refiner(prof.llm_model_or_path, use_lg_tpst)
    raw_llm_spk, lg_llm_spk, _ = refiner.refine_utterance(
        words_str, hyp_spk_str
    )
    t_llm = time.time() - t2
    chosen_spk_str = lg_llm_spk if use_lg_tpst else raw_llm_spk
    final_spks = chosen_spk_str.split()
    after_llm_text = diarizers.create_diarized_text(words_list, final_spks)

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
        hyp_spk_llm=chosen_spk_str,
        diarized_before_llm=before_llm_text,
        diarized_after_llm=after_llm_text,
    )
    changed = sum(1 for a, b in zip(acoustic_spks, final_spks) if a != b)
    total_t = t_asr + t_diar + t_llm
    status = (
        f"Completed in {total_t:.1f}s (ASR: {t_asr:.1f}s, "
        f"Diarization: {t_diar:.1f}s, DiarizationLM: {t_llm:.1f}s) | "
        f"Profile: {prof.name} | Words: {len(word_segments)} | "
        f"Speakers: {len(set(final_spks))} | "
        f"Relabeled: {changed}/{len(word_segments)} ({mode_str})"
    )
    yield (
        status,
        before_llm_text,
        after_llm_text,
        dt_post.to_rttm(),
        dt_post.to_srt(),
    )


def build_demo() -> gr.Blocks:
  """Builds the Gradio Blocks interface for Diarizers v1.2.0."""
  with gr.Blocks(
      title="Diarizers v1.2.0 — End-to-End Audio ASR + Diarization + LLM"
  ) as demo:
    gr.Markdown(
        "# 🎙️ Diarizers v1.2.0 — End-to-End Open-Source ASR, Speaker "
        "Diarization & DiarizationLM\n"
        "> End-to-end Apache-2.0/MIT audio pipeline combining **Whisper ASR** "
        "+ **`powerset_ward` (`pyannote/segmentation-3.0` + "
        "[`FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) "
        "`univ_full_eres2conformer_xl_diar_adapt` / `ECAPA-TDNN`)** + "
        "**[`DiarizationLM-Gemma-4-12B-v1`]"
        "(https://huggingface.co/diarizers-community/"
        "DiarizationLM-Gemma-4-12B-v1)** / "
        "**[`E4B-v2`](https://huggingface.co/diarizers-community/"
        "DiarizationLM-Gemma-4-E4B-v2)** (`Q4_K_M` GGUF + `LG-TPST`).\n>\n"
        "> 📦 **GitHub**: [`diarizers-community/diarizers`]"
        "(https://github.com/diarizers-community/diarizers) "
        "| **PyPI**: `pip install -U diarizers` "
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
                EXAMPLE_AUDIO_1
                if os.path.isfile(EXAMPLE_AUDIO_1)
                else None
            ),
            label="1. Input Conversation Audio (.wav / .mp3 / .flac)",
        )
        profile_dd = gr.Dropdown(
            choices=list(PROFILE_LABELS.keys()),
            value=DEFAULT_PROFILE_LABEL,
            label="2. Pipeline Quality / Speed Profile",
        )
        asr_dd = gr.Dropdown(
            choices=[
                "openai/whisper-base.en",
                "openai/whisper-small.en",
                "openai/whisper-large-v3-turbo",
            ],
            value="openai/whisper-base.en",
            label=(
                "3. Stage 1: Whisper ASR Model "
                "(`whisper-base.en` fastest on CPU)"
            ),
        )
        preset_dd = gr.Dropdown(
            choices=list(PRESET_LABELS.keys()),
            value="Profile Default (Recommended)",
            label="4. Stage 2: Optional Acoustic Preset Override",
        )
        num_spk_slider = gr.Slider(
            minimum=0,
            maximum=10,
            step=1,
            value=2,
            label="5. Known Speaker Count (0 = Auto-detect 1..10 speakers)",
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
            label="Before DiarizationLM (Acoustic powerset_ward Baseline)",
            lines=5,
        )
        out_after_llm = gr.Textbox(
            label="After DiarizationLM (Refined Transcript)",
            lines=5,
        )
        out_rttm = gr.Textbox(label="NIST RTTM Output", lines=5)
        out_srt = gr.Textbox(label="SRT Subtitles", lines=5)

    example_rows = []
    labels_list = list(PROFILE_LABELS.keys())
    if os.path.isfile(EXAMPLE_AUDIO_1):
      example_rows.append([
          EXAMPLE_AUDIO_1,
          labels_list[1],  # best-quality
          "openai/whisper-base.en",
          "Profile Default (Recommended)",
          2,
          True,
      ])
    if os.path.isfile(EXAMPLE_AUDIO_2):
      example_rows.append([
          EXAMPLE_AUDIO_2,
          labels_list[2],  # max (FlaxSpeaker-XL)
          "openai/whisper-base.en",
          "Profile Default (Recommended)",
          3,
          True,
      ])
    if example_rows:
      gr.Examples(
          examples=example_rows,
          inputs=[
              audio_in,
              profile_dd,
              asr_dd,
              preset_dd,
              num_spk_slider,
              use_lg,
          ],
      )

    run_audio_btn.click(
        fn=run_audio_pipeline,
        inputs=[
            audio_in,
            profile_dd,
            asr_dd,
            preset_dd,
            num_spk_slider,
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
