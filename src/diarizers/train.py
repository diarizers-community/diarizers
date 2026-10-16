"""SFT and LoRA training utilities for DiarizationLM models."""

import dataclasses
import inspect
import json
import os
from typing import Any, Dict, List, Optional

from diarizers import dataprep


@dataclasses.dataclass
class TrainingConfig:
  """Configuration for DiarizationLM LoRA / SFT fine-tuning."""

  model_name_or_path: str = "google/gemma-4-E4B"
  output_dir: str = "./checkpoints/diarizationlm_lora"
  max_seq_length: int = 4096
  lora_rank: int = 256
  lora_alpha: int = 256
  lora_dropout: float = 0.0
  learning_rate: float = 1.5e-4
  batch_size: int = 4
  gradient_accumulation_steps: int = 4
  max_steps: int = 10000
  warmup_steps: int = 500
  save_steps: int = 1000
  logging_steps: int = 25
  seed: int = 3407
  use_unsloth: bool = False
  use_locality_oracle: bool = True
  completion_only_loss: bool = True


def prepare_sft_records_from_json(
    input_json_path: str,
    use_locality_oracle: bool = True,
    max_chunk_chars: int = 4000,
    flavor: str = "both",
) -> List[Dict[str, str]]:
  """Loads utterances from JSON and builds dual-flavor SFT training records."""
  with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  if isinstance(data, list):
    return data
  utts = data.get("utterances", [])
  return dataprep.build_sft_dataset(
      utts,
      include_deg2ref=flavor in ("both", "deg2ref"),
      use_locality_oracle=use_locality_oracle,
      max_chunk_chars=max_chunk_chars,
      flavor=flavor,
  )


def run_lora_sft_training(
    train_records: List[Dict[str, str]],
    cfg: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
  """Runs LoRA SFT fine-tuning using Unsloth or HF Transformers + PEFT."""
  if cfg is None:
    cfg = TrainingConfig()
  os.makedirs(cfg.output_dir, exist_ok=True)

  import datasets
  import peft
  import torch
  import transformers
  import trl

  formatted_rows = []
  for r in train_records:
    prompt = r.get("prompt", "")
    resp = r.get("response", r.get("target", ""))
    if not resp.endswith("<turn|>"):
      resp_with_eot = f"{resp}<turn|>"
    else:
      resp_with_eot = resp
    prompt_turn = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    formatted_rows.append({
        "prompt": prompt_turn,
        "completion": resp_with_eot,
        "response": resp_with_eot,
        "text": prompt_turn + resp_with_eot,
    })

  ds = datasets.Dataset.from_list(formatted_rows)
  target_modules = [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj",
  ]

  if cfg.use_unsloth:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name_or_path,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model_name_or_path
    )
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
      model = transformers.AutoModelForCausalLM.from_pretrained(
          cfg.model_name_or_path,
          torch_dtype=dtype,
      )
    except ValueError:
      cls_cand = getattr(
          transformers,
          "Gemma4ForConditionalGeneration",
          getattr(transformers, "AutoModelForImageTextToText", None),
      )
      if cls_cand is None:
        raise
      model = cls_cand.from_pretrained(
          cfg.model_name_or_path,
          torch_dtype=dtype,
      )
    lora_cfg = peft.LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = peft.get_peft_model(model, lora_cfg)

  sft_args = transformers.TrainingArguments(
      output_dir=cfg.output_dir,
      per_device_train_batch_size=cfg.batch_size,
      gradient_accumulation_steps=cfg.gradient_accumulation_steps,
      learning_rate=cfg.learning_rate,
      max_steps=cfg.max_steps,
      warmup_steps=cfg.warmup_steps,
      save_steps=cfg.save_steps,
      logging_steps=cfg.logging_steps,
      seed=cfg.seed,
      fp16=False,
      bf16=torch.cuda.is_available(),
      report_to="none",
  )

  sft_sig = inspect.signature(trl.SFTTrainer.__init__).parameters
  trainer_kwargs: Dict[str, Any] = {
      "model": model,
      "train_dataset": ds,
      "args": sft_args,
  }
  if cfg.completion_only_loss and hasattr(
      trl, "DataCollatorForCompletionOnlyLM"
  ):
    trainer_kwargs["data_collator"] = trl.DataCollatorForCompletionOnlyLM(
        response_template="<|turn>model\n",
        tokenizer=tokenizer,
    )
  if "dataset_text_field" in sft_sig:
    trainer_kwargs["dataset_text_field"] = "text"
  if "max_seq_length" in sft_sig:
    trainer_kwargs["max_seq_length"] = cfg.max_seq_length
  if "tokenizer" in sft_sig:
    trainer_kwargs["tokenizer"] = tokenizer
  elif "processing_class" in sft_sig:
    trainer_kwargs["processing_class"] = tokenizer

  trainer = trl.SFTTrainer(**trainer_kwargs)
  train_res = trainer.train()
  model.save_pretrained(cfg.output_dir)
  tokenizer.save_pretrained(cfg.output_dir)
  return {"output_dir": cfg.output_dir, "metrics": train_res.metrics}
