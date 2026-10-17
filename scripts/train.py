"""CLI script for training DiarizationLM LoRA / SFT checkpoints."""

import argparse

from diarizers import train


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Fine-tune a DiarizationLM model with LoRA / SFT."
  )
  parser.add_argument(
      "--train_json", required=True, help="Input SFT or utterances JSON."
  )
  parser.add_argument(
      "--model_name_or_path",
      default="google/gemma-4-E4B",
      help="Base Hugging Face model ID or local path.",
  )
  parser.add_argument(
      "--output_dir",
      default="./checkpoints/diarizationlm_lora",
      help="Directory to save LoRA checkpoint.",
  )
  parser.add_argument("--lora_rank", type=int, default=256)
  parser.add_argument("--learning_rate", type=float, default=1.5e-4)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--max_steps", type=int, default=10000)
  parser.add_argument("--use_unsloth", action="store_true")
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Verify dataset formatting without launching training loop.",
  )
  args = parser.parse_args()

  records = train.prepare_sft_records_from_json(args.train_json)
  print(f"Loaded {len(records)} SFT training records from {args.train_json}")
  if args.dry_run:
    return

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
  res = train.run_lora_sft_training(records, cfg=cfg)
  print(f"Training completed: {res}")


if __name__ == "__main__":
  main()
