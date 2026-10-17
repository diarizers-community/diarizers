"""CLI script for evaluating WER, WDER, cpWER, SpkCntMAE, and bootstrap CIs."""

import argparse
import json
import os

from diarizers import evaluation


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Compute official Hungarian WER, WDER, cpWER, and SpkCntMAE."
  )
  parser.add_argument("--input_json", required=True, help="Decoded JSON path.")
  parser.add_argument("--output_json", default="", help="Report JSON path.")
  parser.add_argument("--ref_text_field", default="ref_text")
  parser.add_argument("--hyp_text_field", default="hyp_text")
  parser.add_argument("--ref_spk_field", default="ref_spk")
  parser.add_argument("--hyp_spk_field", default="hyp_spk_llm")
  parser.add_argument(
      "--compare_baseline",
      action="store_true",
      help="Also evaluate hyp_spk baseline and compute paired bootstrap CIs.",
  )
  parser.add_argument("--bootstrap_resamples", type=int, default=10000)
  args = parser.parse_args()

  res_llm = evaluation.evaluate_json_file(
      args.input_json,
      ref_text_field=args.ref_text_field,
      hyp_text_field=args.hyp_text_field,
      ref_spk_field=args.ref_spk_field,
      hyp_spk_field=args.hyp_spk_field,
  )

  report = {
      "evaluated_field": args.hyp_spk_field,
      "metrics": {k: v for k, v in res_llm.items() if k != "utterances"},
  }

  if args.compare_baseline:
    res_base = evaluation.evaluate_json_file(
        args.input_json,
        ref_text_field=args.ref_text_field,
        hyp_text_field=args.hyp_text_field,
        ref_spk_field=args.ref_spk_field,
        hyp_spk_field="hyp_spk",
    )
    report["baseline_metrics"] = {
        k: v for k, v in res_base.items() if k != "utterances"
    }
    report["paired_bootstrap_ci"] = evaluation.bootstrap_paired_ci(
        res_llm["utterances"],
        res_base["utterances"],
        num_resamples=args.bootstrap_resamples,
    )
    report["paired_bootstrap_ci"]["comparison"] = (
        f"{args.hyp_spk_field} - hyp_spk"
    )

  print(json.dumps(report, indent=2))
  if args.output_json:
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
    )
    with open(args.output_json, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)


if __name__ == "__main__":
  main()
