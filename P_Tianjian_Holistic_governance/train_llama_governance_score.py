#!/usr/bin/env python3
"""LoRA fine-tuning script for governance report scoring.

The default dataset is built from:
  - Report/评价清单_20250828_185420.md -> 90
  - Report/评价清单_20250827_110611.md -> 60

Example:
  python train_llama_governance_score.py --do_train
  python train_llama_governance_score.py --predict Report/评价清单_20250828_185420.md
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL_PATH = "/data/public_models/Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/llama32_1b_governance_score_lora"
DEFAULT_REPORT_DIR = "Report"
DEFAULT_SCORE_BY_REPORT_ID = {
    "185420": 90,
    "110611": 60,
}


SYSTEM_PROMPT = (
    "你是政府治理报告评估专家。请阅读治理报告，并只输出一个0到100之间的整数分数。"
    "不要输出解释、单位或其他文字。"
)


def build_user_prompt(report_text: str) -> str:
    return (
        "请根据以下治理报告的完整性、处置规范性、证据链完整性、协同闭环程度和整改复核情况打分。\n\n"
        "治理报告：\n"
        f"{report_text.strip()}\n\n"
        "输出要求：只输出一个0到100之间的整数分数。"
    )


def load_score_mapping(path: str | None) -> dict[str, int]:
    if not path:
        return DEFAULT_SCORE_BY_REPORT_ID
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def infer_report_id(path: Path) -> str | None:
    match = re.search(r"_(\d+)\.md$", path.name)
    return match.group(1) if match else None


def load_examples(report_dir: str, score_mapping: dict[str, int]) -> list[dict[str, Any]]:
    report_paths = sorted(Path(report_dir).glob("*.md"))
    examples: list[dict[str, Any]] = []
    for report_path in report_paths:
        report_id = infer_report_id(report_path)
        if report_id is None or report_id not in score_mapping:
            continue
        report_text = report_path.read_text(encoding="utf-8")
        examples.append(
            {
                "report_id": report_id,
                "path": str(report_path),
                "report": report_text,
                "score": int(score_mapping[report_id]),
            }
        )

    if not examples:
        known_ids = ", ".join(sorted(score_mapping))
        raise ValueError(f"No reports matched the score mapping. Known report ids: {known_ids}")
    return examples


def make_messages(report_text: str, score: int | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(report_text)},
    ]
    if score is not None:
        messages.append({"role": "assistant", "content": str(int(score))})
    return messages


class GovernanceScoreDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int):
        self.features = [self._tokenize(example, tokenizer, max_length) for example in examples]

    def _tokenize(
        self,
        example: dict[str, Any],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        prompt_messages = make_messages(example["report"])
        full_messages = make_messages(example["report"], example["score"])

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        if len(full_ids) > max_length:
            overflow = len(full_ids) - max_length
            prompt_ids = prompt_ids[overflow:]
            full_ids = full_ids[overflow:]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        labels = labels[: len(full_ids)]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.features[idx]


@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(feature["input_ids"].shape[0] for feature in features)
        pad_id = int(self.tokenizer.pad_token_id)

        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - feature["input_ids"].shape[0]
            batch["input_ids"].append(
                torch.nn.functional.pad(feature["input_ids"], (0, pad_len), value=pad_id)
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(feature["attention_mask"], (0, pad_len), value=0)
            )
            batch["labels"].append(
                torch.nn.functional.pad(feature["labels"], (0, pad_len), value=-100)
            )

        return {key: torch.stack(value) for key, value in batch.items()}


def load_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_path: str, use_4bit: bool) -> AutoModelForCausalLM:
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    examples = load_examples(args.report_dir, load_score_mapping(args.score_map))

    if len(examples) < 10:
        print(
            f"Warning: only {len(examples)} training examples found. "
            "This can memorize the examples but will not generalize reliably."
        )
    print("Training examples:")
    for example in examples:
        print(f"  {example['path']} -> {example['score']}")

    dataset = GovernanceScoreDataset(examples, tokenizer, args.max_length)
    model = load_base_model(args.model_path, args.use_4bit)

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForCausalLM(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer files to: {args.output_dir}")


def parse_score(text: str) -> int | None:
    match = re.search(r"\b(?:100|[1-9]?\d)\b", text)
    return int(match.group(0)) if match else None


@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.model_path)
    base_model = load_base_model(args.model_path, use_4bit=False)
    model = PeftModel.from_pretrained(base_model, args.output_dir)
    model.eval()

    report_text = Path(args.predict).read_text(encoding="utf-8")
    input_ids = tokenizer.apply_chat_template(
        make_messages(report_text),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=8,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    score = parse_score(text)
    print(f"raw_output: {text}")
    print(f"score: {score if score is not None else 'N/A'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-1B-Instruct for report scoring.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report_dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--score_map", default=None, help="Optional JSON mapping report id to score.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true", help="Use QLoRA. Requires bitsandbytes.")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--predict", default=None, help="Path to a Markdown report for prediction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.do_train and not args.predict:
        raise SystemExit("Please pass --do_train and/or --predict REPORT.md")
    if args.do_train:
        train(args)
    if args.predict:
        predict(args)


if __name__ == "__main__":
    main()
