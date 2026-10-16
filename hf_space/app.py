"""Gradio 5.50.0 CPU interactive demo for Diarizers v1.0.0."""

import os
import sys
from typing import Tuple

import gradio as gr

# Allow running directly from repo checkout or HF Space root
REPO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
  sys.path.insert(0, REPO_SRC)

import diarizers  # noqa: E402

EXAMPLE_DIARIZED_INPUT = (
    "<speaker:1> hello my name is angela how are you doing today colin "
    "<speaker:2> oh very nice to meet you i am calling from tennessee where "
    "are you located <speaker:1> i am out in california near san francisco"
)

DEFAULT_GGUF_REPO = (
    "diarizers-community/DiarizationLM-Gemma-4-E4B-v1"
)
DEFAULT_GGUF_FILE = "DiarizationLM-Gemma-4-E4B-v1-q4_k_m.gguf"

_REFINER_CACHE: dict[tuple[bool, int], diarizers.DiarizationLMRefiner] = {}
_PIPELINE_CACHE: dict[tuple[str, bool], diarizers.DiarizersPipeline] = {}


def _get_cached_refiner(
    use_lg_tpst: bool,
    max_span_words: int,
) -> diarizers.DiarizationLMRefiner:
  key = (bool(use_lg_tpst), int(max_span_words))
  if key not in _REFINER_CACHE:
    p_cfg = diarizers.PromptConfig(
        use_locality_gate=bool(use_lg_tpst),
        max_local_span_words=int(max_span_words),
    )
    _REFINER_CACHE[key] = diarizers.DiarizationLMRefiner(
        model_path_or_repo=DEFAULT_GGUF_REPO,
        gguf_filename=DEFAULT_GGUF_FILE,
        backend="gguf",
        prompt_config=p_cfg,
    )
  return _REFINER_CACHE[key]


def run_text_refinement(
    diarized_input: str,
    use_lg_tpst: bool,
    max_span_words: int,
    use_gguf_model: bool,
) -> Tuple[str, str, str]:
  """Runs DiarizationLM + LG-TPST on a diarized transcript string."""
  if not diarized_input or not diarized_input.strip():
    return "", "", "Please provide a non-empty diarized transcript."

  words_str, spks_str = diarizers.extract_text_and_spk(diarized_input)
  words = words_str.split()
  hyp_spks = spks_str.split()
  if not words:
    return "", "", "No valid words found in input."

  if use_gguf_model:
    refiner = _get_cached_refiner(use_lg_tpst, int(max_span_words))
    _, lg_spks, _ = refiner.refine_utterance(words_str, spks_str)
    final_spks = lg_spks.split()
    mode_str = "GGUF DiarizationLM + LG-TPST"
  else:
    final_spks = diarizers.apply_locality_gated_speaker_transfer(
        hyp_spks,
        hyp_spks,
        max_local_span_words=int(max_span_words),
    )
    mode_str = "Ghost-speaker filter only (GGUF LLM disabled)"

  refined_text = diarizers.create_diarized_text(words, final_spks)
  changed = sum(1 for a, b in zip(hyp_spks, final_spks) if a != b)
  summary = (
      f"Mode: {mode_str} | Words: {len(words)} (100% preserved via TPST) | "
      f"Speaker labels adjusted: {changed}/{len(words)} | "
      f"LG-TPST enabled: {use_lg_tpst} (max_span={max_span_words})"
  )
  return refined_text, " ".join(final_spks), summary


def run_audio_pipeline(
    audio_path: str,
    preset_name: str,
    num_speakers: int,
    use_gguf_model: bool = False,
) -> Tuple[str, str, str]:
  """Runs WhisperWordASR + ECAPASpectralDiarizer on an uploaded audio file."""
  if not audio_path:
    return "", "", "Please upload or select an audio file."

  preset_map = {
      "Fisher Telephony (2 speakers)": diarizers.FISHER_TELEPHONY_PRESET,
      "Callhome Telephony (2-6 speakers)": diarizers.CALLHOME_TELEPHONY_PRESET,
      "AMI Meeting (3-5 speakers)": diarizers.AMI_MEETING_PRESET,
      "Meeting Multi-Speaker (2-10 speakers)": (
          diarizers.MEETING_MULTISPK_PRESET
      ),
  }
  preset = preset_map.get(preset_name, diarizers.FISHER_TELEPHONY_PRESET)
  cache_key = (preset.name, bool(use_gguf_model))
  if cache_key not in _PIPELINE_CACHE:
    _PIPELINE_CACHE[cache_key] = diarizers.DiarizersPipeline.from_pretrained(
        asr_model="openai/whisper-large-v3-turbo",
        diarizer_preset=preset,
        llm_model_or_path=DEFAULT_GGUF_REPO if use_gguf_model else None,
        llm_gguf_filename=DEFAULT_GGUF_FILE if use_gguf_model else None,
        llm_backend="gguf" if use_gguf_model else "auto",
        device="cpu",
    )
  pipe = _PIPELINE_CACHE[cache_key]
  n_spk = int(num_speakers) if int(num_speakers) > 0 else None
  transcript = pipe(audio_path, num_speakers=n_spk)
  return (
      transcript.to_text(),
      transcript.to_rttm(),
      transcript.to_srt(),
  )


def build_demo() -> gr.Blocks:
  """Builds the Gradio Blocks interface."""
  with gr.Blocks(title="Diarizers v1.0 — Open-Source DiarizationLM") as demo:
    gr.Markdown(
        "# 🎙️ Diarizers v1.0 — Unified Open-Source ASR, Diarization & "
        "DiarizationLM\n"
        "> Apache-2.0/MIT open-source pipeline combining "
        "`openai/whisper-large-v3-turbo`, `pyannote/segmentation-3.0`, "
        "`speechbrain/spkrec-ecapa-voxceleb`, `SpectralClusterer`, and "
        "`DiarizationLM-Gemma-4-E4B-v1` (`LG-TPST`)."
    )

    with gr.Tabs():
      with gr.TabItem("1. End-to-End Audio Pipeline (CPU)"):
        with gr.Row():
          with gr.Column():
            audio_in = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Input Conversation Audio (.wav / .mp3)",
            )
            preset_dd = gr.Dropdown(
                choices=[
                    "Fisher Telephony (2 speakers)",
                    "Callhome Telephony (2-6 speakers)",
                    "AMI Meeting (3-5 speakers)",
                    "Meeting Multi-Speaker (2-10 speakers)",
                ],
                value="Fisher Telephony (2 speakers)",
                label="Acoustic Diarization Preset",
            )
            num_spk_slider = gr.Slider(
                minimum=0,
                maximum=8,
                step=1,
                value=2,
                label="Known Speaker Count (0 = Auto-detect via Eigengap)",
            )
            audio_use_gguf = gr.Checkbox(
                value=False,
                label=(
                    "Enable GGUF DiarizationLM + LG-TPST Refinement "
                    f"({DEFAULT_GGUF_REPO})"
                ),
            )
            run_audio_btn = gr.Button(
                "Run ASR + Speaker Diarization", variant="primary"
            )
          with gr.Column():
            out_diar_text = gr.Textbox(
                label="Diarized Transcript (<speaker:k> format)", lines=8
            )
            out_rttm = gr.Textbox(label="NIST RTTM Output", lines=6)
            out_srt = gr.Textbox(label="SRT Subtitles", lines=6)

        run_audio_btn.click(
            fn=run_audio_pipeline,
            inputs=[audio_in, preset_dd, num_spk_slider, audio_use_gguf],
            outputs=[out_diar_text, out_rttm, out_srt],
        )

      with gr.TabItem("2. DiarizationLM + LG-TPST Text Refinement"):
        with gr.Row():
          with gr.Column():
            text_in = gr.Textbox(
                value=EXAMPLE_DIARIZED_INPUT,
                label="Input Diarized Transcript",
                lines=6,
            )
            use_lg = gr.Checkbox(
                value=True,
                label="Enable Locality-Gated TPST (LG-TPST)",
            )
            max_span = gr.Slider(
                minimum=1,
                maximum=25,
                step=1,
                value=5,
                label="Max Contiguous Modification Span (words)",
            )
            use_gguf = gr.Checkbox(
                value=False,
                label=(
                    "Load GGUF LLM from Hugging Face Hub "
                    f"({DEFAULT_GGUF_REPO})"
                ),
            )
            run_txt_btn = gr.Button(
                "Refine Speaker Boundaries", variant="primary"
            )
          with gr.Column():
            out_refined = gr.Textbox(
                label="Refined Diarized Transcript", lines=6
            )
            out_spks = gr.Textbox(
                label="Word-Level Speaker Sequence", lines=3
            )
            out_stats = gr.Textbox(label="TPST / LG-TPST Diagnostics", lines=2)

        run_txt_btn.click(
            fn=run_text_refinement,
            inputs=[text_in, use_lg, max_span, use_gguf],
            outputs=[out_refined, out_spks, out_stats],
        )

  return demo


if __name__ == "__main__":
  app = build_demo()
  app.launch()
