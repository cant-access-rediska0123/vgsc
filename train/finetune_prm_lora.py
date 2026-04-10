#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import create_repo
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

REWARD_TOKEN = "<extra_0>"
STEP_SEPARATOR = "\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--hf-dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--hf-output-dir", type=str, default=None)
    parser.add_argument("--eval-size", type=float, default=0.1)

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--label-mode", type=str, default="correctness",
        choices=["correctness", "step_continuation"])

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true")

    return parser.parse_args()


def get_step_labels_for_example(example: dict[str, Any], label_mode: str) -> list[float]:
    steps = example["steps"]
    if label_mode == "correctness":
        return build_step_correctness_labels(steps, example["first_incorrect_step"])
    elif label_mode == "step_continuation":
        return build_step_continuation_labels(steps, example["step_continuation_correct"])
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")


def get_visible_step_labels(
    step_labels: list[float],
    n_visible: int,
    truncation_side: str,
) -> list[float]:
    if n_visible == 0:
        return []
    if truncation_side == "right":
        return step_labels[:n_visible]
    elif truncation_side == "left":
        return step_labels[-n_visible:]
    else:
        raise ValueError(f"Unsupported truncation_side={truncation_side}")


def filter_dataset_for_label_mode(
    ds: Dataset,
    label_mode: str,
    tokenizer=None,
    max_length: int | None = None,
    split_name: str | None = None,
) -> Dataset:
    before_total = len(ds)
    before_one_step = sum(1 for ex in ds if len(ex["steps"]) == 1)

    if label_mode == "step_continuation":
        ds = ds.filter(
            lambda example: (
                len(example["steps"]) >= 2 and
                not any(x is None for x in example["step_continuation_correct"])
            ),
            desc="Filtering step_continuation examples with <2 steps",
        )

    reward_token_id = get_reward_token_id(tokenizer)

    def _has_visible_supervised_marker(example: dict[str, Any]) -> bool:
        text = build_training_text(example["question"], example["steps"])
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        n_visible = sum(tok_id == reward_token_id for tok_id in input_ids)
        if n_visible == 0:
            return False

        step_labels = get_step_labels_for_example(example, label_mode)
        visible_labels = get_visible_step_labels(
            step_labels,
            n_visible,
            tokenizer.truncation_side,
        )
        return any(lbl >= 0.0 for lbl in visible_labels)

    ds = ds.filter(
        _has_visible_supervised_marker,
        desc="Filtering examples with no visible supervised reward marker after truncation",
    )

    after_total = len(ds)
    after_one_step = sum(1 for ex in ds if len(ex["steps"]) == 1)

    removed_total = before_total - after_total
    removed_one_step = before_one_step - after_one_step
    removed_pct = 100.0 * removed_total / before_total if before_total else 0.0

    prefix = f"[{split_name}] " if split_name else ""
    print(
        f"{prefix}filtered dataset: removed {removed_total}/{before_total} "
        f"({removed_pct:.2f}%), one-step removed {removed_one_step}/{before_one_step}"
    )

    return ds


def load_input_dataset(
    dataset_path: str | None,
    hf_dataset_path: str | None,
    eval_size: float,
) -> tuple[Dataset, Dataset | None]:
    if dataset_path is not None and hf_dataset_path is not None:
        raise ValueError("Provide only one of dataset_path or hf_dataset_path")

    if dataset_path is None and hf_dataset_path is None:
        raise ValueError("Provide one of dataset_path or hf_dataset_path")

    if not (0.0 <= eval_size < 1.0):
        raise ValueError(f"eval_size must be in [0.0, 1.0), got {eval_size}")

    if dataset_path is not None:
        ds = load_from_disk(dataset_path)["train"]
    else:
        ds = load_dataset(hf_dataset_path, split="train")

    if eval_size == 0.0:
        return ds, None

    split_ds = ds.train_test_split(test_size=eval_size, seed=42, shuffle=True)
    return split_ds["train"], split_ds["test"]


def build_step_correctness_labels(steps: list[str], first_incorrect_step: int) -> list[float]:
    n_steps = len(steps)
    labels: list[float] = []
    for step_idx_1based in range(1, n_steps + 1):
        if step_idx_1based < first_incorrect_step:
            labels.append(1.0)
        elif step_idx_1based == first_incorrect_step:
            labels.append(0.0)
        else:
            labels.append(-100.0)
    return labels

def build_step_continuation_labels(steps: list[str], step_continuation_correct: list[float]) -> list[float]:
    n_steps = len(steps)
    if len(step_continuation_correct) != n_steps:
        raise ValueError(
            f"step_continuation_correct length must equal len(steps), "
            f"got {len(step_continuation_correct)} vs {n_steps}"
        )
    labels: list[float] = []
    for i, value in enumerate(step_continuation_correct):
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"step_continuation_correct[{i}] must be in [0,1], got {value}"
            )
        if i == len(step_continuation_correct) - 1:
            labels.append(-100)
        else:
            labels.append(value)
    return labels


def build_training_text(question: str, steps: list[str]) -> str:
    parts = [question]
    for step in steps:
        parts.append(step)
        parts.append(REWARD_TOKEN)
    return STEP_SEPARATOR.join(parts)


def get_reward_token_id(tokenizer) -> int:
    reward_token_ids = tokenizer.encode(REWARD_TOKEN, add_special_tokens=False)
    if len(reward_token_ids) != 1:
        raise ValueError(
            f"{REWARD_TOKEN!r} does not tokenize to a single token. Got ids={reward_token_ids}. "
            "This script expects exactly one reward marker token."
        )
    return reward_token_ids[0]


def preprocess_dataset(ds: Dataset, tokenizer, max_length: int, label_mode: str) -> Dataset:
    reward_token_id = get_reward_token_id(tokenizer)

    def _map_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        try:
            question: str = example["question"]
            steps: list[str] = example["steps"]
        except Exception as e:
            print(example)
            print('ERROR on example with keys', list(example.keys()))
            raise e

        if len(steps) == 0:
            raise ValueError(f"Row {idx}: empty 'steps' list is not supported")

        if label_mode == "correctness":
            step_labels = build_step_correctness_labels(steps, example["first_incorrect_step"])
        elif label_mode == "step_continuation":
            step_labels = build_step_continuation_labels(steps, example["step_continuation_correct"])
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")
        text = build_training_text(question, steps)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        token_labels = [-100.0] * len(input_ids)
        marker_positions = [pos for pos, tok_id in enumerate(input_ids) if tok_id == reward_token_id]

        if len(marker_positions) == 0:
            raise ValueError(
                f"Row {idx}: no {REWARD_TOKEN} markers remained after tokenization/truncation. "
                f"Increase --max-length. question_len={len(question)} n_steps={len(steps)}"
            )

        n_visible = len(marker_positions)
        visible_labels = step_labels[:n_visible]

        if len(visible_labels) != n_visible:
            raise RuntimeError(
                f"Row {idx}: mismatch between visible markers ({n_visible}) "
                f"and visible labels ({len(visible_labels)})"
            )

        for pos, label in zip(marker_positions, visible_labels):
            token_labels[pos] = float(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_labels": token_labels,
        }

    return ds.map(
        _map_fn,
        with_indices=True,
        remove_columns=ds.column_names,
        desc="Tokenizing PRM dataset",
    )


@dataclass
class PRMDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        token_labels = [f["token_labels"] for f in features]
        model_features = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for labels in token_labels:
            if len(labels) > max_len:
                raise RuntimeError(f"token_labels length {len(labels)} exceeds padded max_len {max_len}")
            padded = labels + ([-100.0] * (max_len - len(labels)))
            padded_labels.append(padded)

        batch["token_labels"] = torch.tensor(padded_labels, dtype=torch.float32)
        return batch


class PRMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_labels = inputs.pop("token_labels")
        outputs = model(**inputs)

        if not hasattr(outputs, "logits"):
            raise AttributeError("Model output has no .logits field")

        logits = outputs.logits
        if logits.ndim != 3:
            raise ValueError(f"Expected logits shape [batch, seq, num_labels], got {tuple(logits.shape)}")
        if logits.shape[-1] != 2:
            raise ValueError(f"Expected num_labels=2 for PRM token classification, got {logits.shape[-1]}")

        valid_mask = token_labels >= 0.0
        if not torch.any(valid_mask):
            raise RuntimeError("No valid PRM supervision positions found in batch")

        valid_logits = logits[valid_mask]
        valid_labels = token_labels[valid_mask]

        binary_logits = valid_logits[:, 1] - valid_logits[:, 0]
        loss = F.binary_cross_entropy_with_logits(binary_logits, valid_labels)

        if return_outputs:
            return loss, outputs
        return loss


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray], label_mode: str) -> dict[str, float]:
    logits, token_labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    if logits.ndim != 3:
        raise ValueError(f"Expected eval logits with shape [batch, seq, 2], got {logits.shape}")
    if token_labels.ndim != 2:
        raise ValueError(f"Expected eval labels with shape [batch, seq], got {token_labels.shape}")

    valid_mask = token_labels >= 0.0
    if not np.any(valid_mask):
        raise RuntimeError("No valid labels found during evaluation")

    valid_logits = logits[valid_mask]
    valid_labels = token_labels[valid_mask].astype(np.float32)

    binary_logits = valid_logits[:, 1] - valid_logits[:, 0]
    probs = 1.0 / (1.0 + np.exp(-binary_logits))

    eps = 1e-8
    bce = -np.mean(
        valid_labels * np.log(probs + eps) +
        (1.0 - valid_labels) * np.log(1.0 - probs + eps)
    )

    metrics = {
        "step_bce": float(bce),
        "mean_target": float(np.mean(valid_labels)),
        "mean_positive_prob": float(np.mean(probs)),
        "mse": float(np.mean((probs - valid_labels) ** 2)),
        "mae": float(np.mean(np.abs(probs - valid_labels))),
    }

    if label_mode == "correctness":
        preds = (probs >= 0.5).astype(np.float32)
        acc = float(np.mean(preds == valid_labels))
        metrics["step_accuracy"] = acc

    return metrics



def get_quantization_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_device_map(load_in_4bit: bool) -> dict[str, int] | None:
    if not load_in_4bit:
        return None

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        return {"": local_rank}
    return {"": 0}


def print_trainable_parameters(model) -> None:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    ratio = 100.0 * trainable / total
    print(f"trainable params: {trainable:,} | total params: {total:,} | trainable%: {ratio:.4f}")


def upload_merged_model_to_hub(
    *,
    base_model_name: str,
    adapter_dir: str,
    repo_id: str,
    tokenizer,
) -> None:
    """
    Reload base model in bf16, load trained adapter, merge, and push the full merged model.
    No fallbacks, no silent recovery.
    """
    create_repo(repo_id, private=False, exist_ok=True)

    base_model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    merged_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False).merge_and_unload()

    merged_model.push_to_hub(repo_id, private=False, safe_serialization=True)
    tokenizer.push_to_hub(repo_id, private=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_ds, eval_ds = load_input_dataset(args.dataset_path, args.hf_dataset_path, args.eval_size)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = filter_dataset_for_label_mode(
        train_ds,
        args.label_mode,
        tokenizer=tokenizer,
        max_length=args.max_length,
        split_name="train",
    )
    if eval_ds is not None:
        eval_ds = filter_dataset_for_label_mode(
            eval_ds,
            args.label_mode,
            tokenizer=tokenizer,
            max_length=args.max_length,
            split_name="eval",
        )

    quantization_config = get_quantization_config(args.load_in_4bit)
    device_map = get_device_map(args.load_in_4bit)

    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    train_ds = preprocess_dataset(train_ds, tokenizer, args.max_length, args.label_mode)
    if eval_ds is not None:
        eval_ds = preprocess_dataset(eval_ds, tokenizer, args.max_length, args.label_mode)

    data_collator = PRMDataCollator(tokenizer=tokenizer)

    do_eval = eval_ds is not None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["token_labels"],
        ddp_find_unused_parameters=False,
        report_to='wandb',
        eval_strategy="epoch" if do_eval else "no",
        save_strategy="epoch" if do_eval else "no",
        logging_strategy="steps",
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False if do_eval else None,
        save_total_limit=1 if do_eval else None,
        seed=args.seed,
    )

    trainer = PRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, label_mode=args.label_mode) if eval_ds is not None else None,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.hf_output_dir is not None:
        upload_merged_model_to_hub(
            base_model_name=args.model_name,
            adapter_dir=args.output_dir,
            repo_id=args.hf_output_dir,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()