"""CLI script for exporting DiarizationLM models to Safetensors and Q4_K_M."""

import argparse
import json

from diarizers import export


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Export DiarizationLM checkpoints to Safetensors and GGUF."
  )
  parser.add_argument(
      "--model_dir", required=True, help="Input model or LoRA adapter dir."
  )
  parser.add_argument(
      "--output_dir", required=True, help="Output directory for Safetensors."
  )
  parser.add_argument(
      "--base_model", default=None, help="Optional base model ID for LoRA."
  )
  parser.add_argument(
      "--output_gguf",
      default="",
      help="Optional output path for Q4_K_M GGUF quantization.",
  )
  parser.add_argument(
      "--llama_cpp_dir",
      default="./llama.cpp",
      help="Path to llama.cpp checkout for GGUF quantization.",
  )
  args = parser.parse_args()

  out_dir = export.export_merged_safetensors(
      adapter_or_model_dir=args.model_dir,
      output_dir=args.output_dir,
      base_model_name_or_path=args.base_model,
  )
  result = {"safetensors_dir": out_dir}
  if args.output_gguf:
    gguf_res = export.export_gguf_q4_k_m(
        hf_model_dir=out_dir,
        output_gguf_path=args.output_gguf,
        llama_cpp_dir=args.llama_cpp_dir,
        quant_type="Q4_K_M",
    )
    result["gguf"] = gguf_res
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
