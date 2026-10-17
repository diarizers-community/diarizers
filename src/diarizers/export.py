"""Model export utilities for Safetensors and GGUF (Q4_K_M) quantization."""

import json
import os
import subprocess
from typing import Any, Dict, Optional


def _load_causal_or_gemma4_model(
    model_id_or_path: str,
    torch_dtype: Any,
) -> Any:
  """Loads a CausalLM or Gemma-4 conditional generation model."""
  import transformers

  try:
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_id_or_path, torch_dtype=torch_dtype
    )
  except ValueError:
    cls_cand = getattr(
        transformers,
        "Gemma4ForConditionalGeneration",
        getattr(transformers, "AutoModelForImageTextToText", None),
    )
    if cls_cand is None:
      raise
    return cls_cand.from_pretrained(
        model_id_or_path, torch_dtype=torch_dtype
    )


def export_merged_safetensors(
    adapter_or_model_dir: str,
    output_dir: str,
    base_model_name_or_path: Optional[str] = None,
) -> str:
  """Merges a PEFT LoRA adapter into BF16/FP16 Safetensors weights."""
  os.makedirs(output_dir, exist_ok=True)
  import peft
  import torch
  import transformers

  adapter_cfg_path = os.path.join(adapter_or_model_dir, "adapter_config.json")
  if os.path.isfile(adapter_cfg_path):
    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
      adapter_cfg = json.load(f)
    base_id = base_model_name_or_path or adapter_cfg.get(
        "base_model_name_or_path"
    )
    base_model = _load_causal_or_gemma4_model(
        str(base_id), torch_dtype=torch.bfloat16
    )
    peft_model = peft.PeftModel.from_pretrained(
        base_model, adapter_or_model_dir
    )
    merged = peft_model.merge_and_unload()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        adapter_or_model_dir
    )
  else:
    merged = _load_causal_or_gemma4_model(
        adapter_or_model_dir, torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        adapter_or_model_dir
    )

  merged.save_pretrained(output_dir, safe_serialization=True)
  tokenizer.save_pretrained(output_dir)
  return output_dir


def export_gguf_q4_k_m(
    hf_model_dir: str,
    output_gguf_path: str,
    llama_cpp_dir: str = "./llama.cpp",
    quant_type: str = "Q4_K_M",
) -> Dict[str, Any]:
  """Converts a Hugging Face model directory to Q4_K_M GGUF via llama.cpp."""
  os.makedirs(os.path.dirname(os.path.abspath(output_gguf_path)), exist_ok=True)
  convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
  quant_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
  if output_gguf_path.endswith(".gguf"):
    f16_gguf = output_gguf_path[:-5] + ".f16.gguf"
  else:
    f16_gguf = output_gguf_path + ".f16.gguf"

  if not os.path.isfile(convert_script):
    return {
        "status": "skipped",
        "reason": f"llama.cpp converter not found at {convert_script}",
        "target_gguf_path": output_gguf_path,
        "quant_type": quant_type,
    }

  subprocess.run(
      [
          "python3",
          convert_script,
          hf_model_dir,
          "--outfile",
          f16_gguf,
          "--outtype",
          "f16",
      ],
      check=True,
  )
  if os.path.isfile(quant_bin):
    subprocess.run(
        [quant_bin, f16_gguf, output_gguf_path, quant_type], check=True
    )
    if os.path.isfile(f16_gguf):
      os.remove(f16_gguf)
  else:
    os.replace(f16_gguf, output_gguf_path)

  return {
      "status": "completed",
      "target_gguf_path": output_gguf_path,
      "quant_type": quant_type,
  }
