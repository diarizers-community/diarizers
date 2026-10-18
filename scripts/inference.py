"""CLI script for end-to-end Diarizers inference on WAV or JSON files."""

import argparse
import json
import os

from diarizers import configs
from diarizers import llm
from diarizers import pipeline


_PRESET_MAP = {
    "telephony": configs.FISHER_TELEPHONY_PRESET,
    "fisher": configs.FISHER_TELEPHONY_PRESET,
    "callhome": configs.CALLHOME_TELEPHONY_PRESET,
    "ami": configs.AMI_MEETING_PRESET,
    "meeting": configs.MEETING_MULTISPK_PRESET,
}


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run end-to-end Diarizers inference on audio or JSON."
  )
  parser.add_argument("--audio", default="", help="Path to input WAV file.")
  parser.add_argument(
      "--input_json", default="", help="Path to input JSON dataset."
  )
  parser.add_argument(
      "--output_json", default="", help="Path to save output JSON."
  )
  parser.add_argument("--output_rttm", default="", help="Optional RTTM path.")
  parser.add_argument("--output_srt", default="", help="Optional SRT path.")
  parser.add_argument("--output_vtt", default="", help="Optional VTT path.")
  parser.add_argument(
      "--preset",
      default="telephony",
      choices=["telephony", "fisher", "callhome", "ami", "meeting"],
      help="Acoustic diarization preset.",
  )
  parser.add_argument(
      "--asr_model",
      default="openai/whisper-large-v3-turbo",
      help="Open-source Whisper ASR model ID or path.",
  )
  parser.add_argument(
      "--segmentation_model",
      default="pyannote/segmentation-3.0",
      help=(
          "Neural speaker segmentation model ID or path "
          "('none' for sliding-window only)."
      ),
  )
  parser.add_argument(
      "--llm_model",
      default=None,
      help="Optional DiarizationLM GGUF or HF model path/repo.",
  )
  parser.add_argument(
      "--gguf_file",
      default=None,
      help="Optional GGUF filename in Hugging Face repo.",
  )
  parser.add_argument(
      "--llm_backend",
      default="auto",
      choices=["auto", "gguf", "transformers"],
      help="LLM inference backend.",
  )
  parser.add_argument("--num_speakers", type=int, default=None)
  parser.add_argument(
      "--no_locality_gate",
      "--no-locality-gate",
      dest="no_locality_gate",
      action="store_true",
      help="Disable LG-TPST locality gate and use Raw-TPST.",
  )
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  use_lg = not args.no_locality_gate
  if args.input_json:
    with open(args.input_json, "r", encoding="utf-8") as f:
      data = json.load(f)
    refiner = llm.DiarizationLMRefiner(
        model_path_or_repo=args.llm_model,
        gguf_filename=args.gguf_file,
        backend=args.llm_backend,
        prompt_config=configs.PromptConfig(use_locality_gate=use_lg),
        device=args.device,
    )
    out_data = refiner.refine_json_dict(data)
    if args.output_json:
      os.makedirs(
          os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
      )
      with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Processed {len(out_data.get('utterances', []))} utterances.")
    return

  if not args.audio:
    raise ValueError("Provide --audio or --input_json.")
  preset = _PRESET_MAP.get(
      args.preset.lower(), configs.FISHER_TELEPHONY_PRESET
  )
  pipe = pipeline.DiarizersPipeline.from_pretrained(
      asr_model=args.asr_model,
      diarizer_preset=preset,
      segmentation_model=args.segmentation_model,
      llm_model_or_path=args.llm_model,
      llm_gguf_filename=args.gguf_file,
      llm_backend=args.llm_backend,
      device=args.device,
      use_locality_gate=use_lg,
  )
  utt_id = os.path.splitext(os.path.basename(args.audio))[0]
  transcript = pipe(
      args.audio, utterance_id=utt_id, num_speakers=args.num_speakers
  )
  print(transcript.to_text(use_llm_speaker=True))

  if args.output_json:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
    )
    with open(args.output_json, "w", encoding="utf-8") as f:
      f.write(transcript.to_json())
  if args.output_rttm:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_rttm)), exist_ok=True
    )
    with open(args.output_rttm, "w", encoding="utf-8") as f:
      f.write(transcript.to_rttm() + "\n")
  if args.output_srt:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_srt)), exist_ok=True
    )
    with open(args.output_srt, "w", encoding="utf-8") as f:
      f.write(transcript.to_srt() + "\n")
  if args.output_vtt:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_vtt)), exist_ok=True
    )
    with open(args.output_vtt, "w", encoding="utf-8") as f:
      f.write(transcript.to_vtt() + "\n")


if __name__ == "__main__":
  main()
