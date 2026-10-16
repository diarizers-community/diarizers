"""CLI script for preparing dual-flavor and Locality-Preserving SFT data."""

import argparse
import json
import os

from diarizers import dataprep


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Prepare DiarizationLM SFT dataset or synthetic dataset."
  )
  parser.add_argument(
      "--input_json", default="", help="Input JSON with utterances list."
  )
  parser.add_argument(
      "--output_json", required=True, help="Output JSON path."
  )
  parser.add_argument(
      "--generate_synthetic",
      action="store_true",
      help="Generate a synthetic multi-speaker dataset for quickstart/testing.",
  )
  parser.add_argument(
      "--num_synthetic", type=int, default=8, help="Synthetic utterance count."
  )
  parser.add_argument(
      "--seed", type=int, default=42, help="Random seed."
  )
  parser.add_argument(
      "--no_locality_oracle",
      action="store_true",
      help="Disable Locality-Preserving Oracle filtering.",
  )
  args = parser.parse_args()

  os.makedirs(
      os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
  )
  if args.generate_synthetic:
    synth = dataprep.generate_synthetic_dataset(
        num_utterances=args.num_synthetic, seed=args.seed
    )
    with open(args.output_json, "w", encoding="utf-8") as f:
      json.dump(synth, f, indent=2, ensure_ascii=False)
    print(f"Wrote synthetic dataset ({args.num_synthetic} utts) to "
          f"{args.output_json}")
    return

  if not args.input_json:
    raise ValueError("Specify --input_json or --generate_synthetic.")
  with open(args.input_json, "r", encoding="utf-8") as f:
    raw = json.load(f)
  sft_rows = dataprep.build_sft_dataset(
      raw.get("utterances", []),
      include_deg2ref=True,
      use_locality_oracle=not args.no_locality_oracle,
  )
  with open(args.output_json, "w", encoding="utf-8") as f:
    json.dump(sft_rows, f, indent=2, ensure_ascii=False)
  print(f"Wrote {len(sft_rows)} SFT examples to {args.output_json}")


if __name__ == "__main__":
  main()
