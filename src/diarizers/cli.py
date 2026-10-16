"""Command-line interface for the unified Diarizers v1.0.0 package."""

import argparse
import json
import os
import sys
from typing import List, Optional

from diarizers import dataprep
from diarizers import evaluation
from diarizers import llm
from diarizers import pipeline


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
      include_deg2ref=True,
      use_locality_oracle=not args.no_locality_oracle,
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
        base_eval["utterances"],
        res["utterances"],
        num_resamples=args.bootstrap_samples,
    )
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
  pipe = pipeline.DiarizersPipeline.from_pretrained(
      asr_model=args.asr_model,
      segmentation_model=args.segmentation_model,
      llm_model_or_path=args.llm_model,
      device=args.device,
  )
  transcript = pipe(args.audio, num_speakers=args.num_speakers)
  print(transcript.to_text(use_llm_speaker=True))
  if args.output_json:
    with open(args.output_json, "w", encoding="utf-8") as f:
      f.write(transcript.to_json())


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
        "--num_speakers",
        "--num-speakers",
        dest="num_speakers",
        type=int,
        default=None,
    )
    p_inf.add_argument("--device", default="cpu")
    p_inf.set_defaults(func=_cmd_infer)

  parsed = parser.parse_args(argv if argv is not None else sys.argv[1:])
  parsed.func(parsed)


if __name__ == "__main__":
  main()

