"""Command-line interface for the unified Diarizers v1.0.1 package."""

import argparse
import json
import os
import sys
from typing import List, Optional

from diarizers import configs
from diarizers import dataprep
from diarizers import evaluation
from diarizers import export
from diarizers import llm
from diarizers import pipeline
from diarizers import train

_PRESET_MAP = {
    "telephony": configs.FISHER_TELEPHONY_PRESET,
    "fisher": configs.FISHER_TELEPHONY_PRESET,
    "callhome": configs.CALLHOME_TELEPHONY_PRESET,
    "ami": configs.AMI_MEETING_PRESET,
    "meeting": configs.MEETING_MULTISPK_PRESET,
}


def _cmd_prepare_data(args: argparse.Namespace) -> None:
  if args.generate_synthetic:
    raw_data = dataprep.generate_synthetic_dataset(
        num_utterances=args.num_synthetic, seed=args.seed
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
    )
    with open(args.output_json, "w", encoding="utf-8") as f:
      json.dump(raw_data, f, indent=2, ensure_ascii=False)
    print(f"Generated synthetic dataset at {args.output_json}")
    return

  if not args.input_json:
    raise ValueError("Provide --input_json or --generate_synthetic.")
  with open(args.input_json, "r", encoding="utf-8") as f:
    data = json.load(f)
  sft_rows = dataprep.build_sft_dataset(
      data.get("utterances", []),
      include_deg2ref=args.flavor in ("both", "deg2ref"),
      use_locality_oracle=not args.no_locality_oracle,
      max_chunk_chars=args.max_chunk_chars,
      flavor=args.flavor,
  )
  os.makedirs(
      os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
  )
  with open(args.output_json, "w", encoding="utf-8") as f:
    json.dump(sft_rows, f, indent=2, ensure_ascii=False)
  print(f"Wrote {len(sft_rows)} SFT examples to {args.output_json}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
  with open(args.input_json, "r", encoding="utf-8") as f:
    data = json.load(f)
  utts = data.get("utterances", [])
  res: dict = {"utterances": []}
  if utts and args.hyp_spk_field not in utts[0] and "hyp_spk" in utts[0]:
    hyp_spk_field = "hyp_spk"
  else:
    hyp_spk_field = args.hyp_spk_field
  res = evaluation.evaluate_json_dict(
      data,
      ref_text_field=args.ref_text_field,
      hyp_text_field=args.hyp_text_field,
      ref_spk_field=args.ref_spk_field,
      hyp_spk_field=hyp_spk_field,
  )
  summary = {k: v for k, v in res.items() if k != "utterances"}
  if args.bootstrap_samples > 0 and utts:
    base_eval = evaluation.evaluate_json_dict(
        data,
        ref_text_field=args.ref_text_field,
        hyp_text_field=args.hyp_text_field,
        ref_spk_field=args.ref_spk_field,
        hyp_spk_field="hyp_spk",
    )
    summary["bootstrap_ci"] = evaluation.bootstrap_paired_ci(
        res["utterances"],
        base_eval["utterances"],
        num_resamples=args.bootstrap_samples,
    )
    summary["bootstrap_ci"]["comparison"] = f"{hyp_spk_field} - hyp_spk"
  print(json.dumps(summary, indent=2))
  if args.output_json:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
    )
    with open(args.output_json, "w", encoding="utf-8") as f:
      json.dump(res, f, indent=2)


def _cmd_infer(args: argparse.Namespace) -> None:
  if args.input_json:
    with open(args.input_json, "r", encoding="utf-8") as f:
      data = json.load(f)
    refiner = llm.DiarizationLMRefiner(
        model_path_or_repo=args.llm_model,
        gguf_filename=args.gguf_file,
        backend=args.llm_backend,
        device=args.device,
    )
    refined = refiner.refine_json_dict(data)
    if args.output_json:
      os.makedirs(
          os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
      )
      with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2, ensure_ascii=False)
    print(f"Refined {len(refined.get('utterances', []))} utterances.")
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
  )
  transcript = pipe(args.audio, num_speakers=args.num_speakers)
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


def _cmd_train(args: argparse.Namespace) -> None:
  records = train.prepare_sft_records_from_json(args.train_json)
  print(f"Loaded {len(records)} SFT training records from {args.train_json}")
  cfg = train.TrainingConfig(
      model_name_or_path=args.model_name_or_path,
      output_dir=args.output_dir,
      lora_rank=args.lora_rank,
      lora_alpha=args.lora_rank,
      learning_rate=args.learning_rate,
      batch_size=args.batch_size,
      max_steps=args.max_steps,
      use_unsloth=args.use_unsloth,
  )
  if args.dry_run:
    os.makedirs(args.output_dir, exist_ok=True)
    plan_path = os.path.join(args.output_dir, "training_plan_summary.json")
    with open(plan_path, "w", encoding="utf-8") as f:
      json.dump(
          {
              "num_records": len(records),
              "model_name_or_path": cfg.model_name_or_path,
              "lora_rank": cfg.lora_rank,
              "learning_rate": cfg.learning_rate,
              "max_steps": cfg.max_steps,
          },
          f,
          indent=2,
      )
    return
  res = train.run_lora_sft_training(records, cfg=cfg)
  print(json.dumps(res, indent=2))


def _cmd_export(args: argparse.Namespace) -> None:
  out_dir = export.export_merged_safetensors(
      adapter_or_model_dir=args.model_dir,
      output_dir=args.output_dir,
      base_model_name_or_path=args.base_model,
  )
  result: dict = {"safetensors_dir": out_dir}
  out_gguf = args.output_gguf
  if not out_gguf and getattr(args, "export_gguf", False):
    out_gguf = os.path.join(out_dir, "model-q4_k_m.gguf")
  if out_gguf:
    result["gguf"] = export.export_gguf_q4_k_m(
        hf_model_dir=out_dir,
        output_gguf_path=out_gguf,
        llama_cpp_dir=args.llama_cpp_dir,
        quant_type="Q4_K_M",
    )
  print(json.dumps(result, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
  """Main CLI entrypoint for diarizers."""
  parser = argparse.ArgumentParser(
      prog="diarizers",
      description=(
          "Unified Open-Source ASR + Speaker Diarization + DiarizationLM"
      ),
  )
  sub = parser.add_subparsers(dest="command", required=True)

  for cmd_name in ("prepare-data", "prepare-sft-data"):
    p_prep = sub.add_parser(cmd_name, help="Prepare SFT dataset JSON")
    p_prep.add_argument(
        "--input_json",
        "--input-json",
        dest="input_json",
        default="",
        help="Input utterances JSON",
    )
    p_prep.add_argument(
        "--output_json",
        "--output-json",
        dest="output_json",
        required=True,
        help="Output SFT or synthetic JSON",
    )
    p_prep.add_argument(
        "--generate_synthetic",
        "--generate-synthetic",
        dest="generate_synthetic",
        action="store_true",
        help="Generate synthetic multi-speaker dataset",
    )
    p_prep.add_argument(
        "--num_synthetic",
        "--num-synthetic",
        dest="num_synthetic",
        type=int,
        default=8,
    )
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.add_argument(
        "--flavor",
        default="both",
        choices=["both", "hyp2ora", "deg2ref"],
        help="SFT training prompt flavor",
    )
    p_prep.add_argument(
        "--max_chunk_chars",
        "--max-chunk-chars",
        dest="max_chunk_chars",
        type=int,
        default=4000,
    )
    p_prep.add_argument(
        "--no_locality_oracle",
        "--no-locality-oracle",
        dest="no_locality_oracle",
        action="store_true",
    )
    p_prep.set_defaults(func=_cmd_prepare_data)

  p_eval = sub.add_parser("evaluate", help="Compute WER, WDER, cpWER, MAE")
  p_eval.add_argument(
      "--input_json", "--input-json", dest="input_json", required=True
  )
  p_eval.add_argument(
      "--output_json", "--output-json", dest="output_json", default=""
  )
  p_eval.add_argument(
      "--ref_text_field",
      "--ref-text-field",
      dest="ref_text_field",
      default="ref_text",
  )
  p_eval.add_argument(
      "--hyp_text_field",
      "--hyp-text-field",
      dest="hyp_text_field",
      default="hyp_text",
  )
  p_eval.add_argument(
      "--ref_spk_field",
      "--ref-spk-field",
      dest="ref_spk_field",
      default="ref_spk",
  )
  p_eval.add_argument(
      "--hyp_spk_field",
      "--hyp-spk-field",
      dest="hyp_spk_field",
      default="hyp_spk_llm",
  )
  p_eval.add_argument(
      "--bootstrap_samples",
      "--bootstrap-samples",
      dest="bootstrap_samples",
      type=int,
      default=0,
  )
  p_eval.set_defaults(func=_cmd_evaluate)

  for cmd_name in ("infer", "run"):
    p_inf = sub.add_parser(
        cmd_name, help="Run end-to-end diarization inference"
    )
    p_inf.add_argument("--audio", default="", help="Path to input WAV file")
    p_inf.add_argument(
        "--input_json",
        "--input-json",
        dest="input_json",
        default="",
        help="Path to input JSON",
    )
    p_inf.add_argument(
        "--output_json",
        "--output-json",
        dest="output_json",
        default="",
        help="Path to output JSON",
    )
    p_inf.add_argument(
        "--output_rttm",
        "--output-rttm",
        dest="output_rttm",
        default="",
        help="Path to output RTTM",
    )
    p_inf.add_argument(
        "--output_srt",
        "--output-srt",
        dest="output_srt",
        default="",
        help="Path to output SRT subtitles",
    )
    p_inf.add_argument(
        "--output_vtt",
        "--output-vtt",
        dest="output_vtt",
        default="",
        help="Path to output VTT subtitles",
    )
    p_inf.add_argument(
        "--preset",
        default="telephony",
        choices=["telephony", "fisher", "callhome", "ami", "meeting"],
        help="Acoustic diarization preset",
    )
    p_inf.add_argument(
        "--asr_model",
        "--asr-model",
        dest="asr_model",
        default="openai/whisper-large-v3-turbo",
    )
    p_inf.add_argument(
        "--segmentation_model",
        "--segmentation-model",
        dest="segmentation_model",
        default="pyannote/segmentation-3.0",
    )
    p_inf.add_argument(
        "--llm_model", "--llm-model", dest="llm_model", default=None
    )
    p_inf.add_argument(
        "--gguf_file",
        "--gguf-file",
        dest="gguf_file",
        default=None,
        help="GGUF filename in Hugging Face repo",
    )
    p_inf.add_argument(
        "--llm_backend",
        "--llm-backend",
        dest="llm_backend",
        default="auto",
        choices=["auto", "gguf", "transformers"],
        help="LLM inference backend",
    )
    p_inf.add_argument(
        "--num_speakers",
        "--num-speakers",
        dest="num_speakers",
        type=int,
        default=None,
    )
    p_inf.add_argument("--device", default="cpu")
    p_inf.set_defaults(func=_cmd_infer)

  p_train = sub.add_parser("train", help="Fine-tune DiarizationLM with LoRA")
  p_train.add_argument(
      "--train_json", "--train-json", dest="train_json", required=True
  )
  p_train.add_argument(
      "--model_name_or_path",
      "--model-name-or-path",
      dest="model_name_or_path",
      default="google/gemma-4-E4B",
  )
  p_train.add_argument(
      "--output_dir",
      "--output-dir",
      dest="output_dir",
      default="./checkpoints/diarizationlm_lora",
  )
  p_train.add_argument(
      "--lora_rank", "--lora-rank", dest="lora_rank", type=int, default=256
  )
  p_train.add_argument(
      "--learning_rate",
      "--learning-rate",
      dest="learning_rate",
      type=float,
      default=1.5e-4,
  )
  p_train.add_argument(
      "--batch_size", "--batch-size", dest="batch_size", type=int, default=4
  )
  p_train.add_argument(
      "--max_steps", "--max-steps", dest="max_steps", type=int, default=10000
  )
  p_train.add_argument(
      "--use_unsloth",
      "--use-unsloth",
      dest="use_unsloth",
      action="store_true",
  )
  p_train.add_argument(
      "--dry_run", "--dry-run", dest="dry_run", action="store_true"
  )
  p_train.set_defaults(func=_cmd_train)

  p_exp = sub.add_parser(
      "export", help="Merge LoRA Safetensors and quantize Q4_K_M GGUF"
  )
  p_exp.add_argument(
      "--model_dir",
      "--model-dir",
      "--model_path",
      "--model-path",
      dest="model_dir",
      required=True,
  )
  p_exp.add_argument(
      "--output_dir", "--output-dir", dest="output_dir", required=True
  )
  p_exp.add_argument(
      "--base_model", "--base-model", dest="base_model", default=None
  )
  p_exp.add_argument(
      "--output_gguf", "--output-gguf", dest="output_gguf", default=""
  )
  p_exp.add_argument(
      "--export_gguf",
      "--export-gguf",
      dest="export_gguf",
      action="store_true",
  )
  p_exp.add_argument(
      "--llama_cpp_dir",
      "--llama-cpp-dir",
      dest="llama_cpp_dir",
      default="./llama.cpp",
  )
  p_exp.set_defaults(func=_cmd_export)

  parsed = parser.parse_args(argv if argv is not None else sys.argv[1:])
  parsed.func(parsed)


if __name__ == "__main__":
  main()
