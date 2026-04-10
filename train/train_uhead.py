#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.nn as nn
import torch.nn.init as init
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    logging as transformers_logging,
    set_seed,
)

from luh import AutoUncertaintyHead
from luh.feature_extractors.basic_hidden_states import FeatureExtractorBasicHiddenStates
from luh.feature_extractors.combined import FeatureExtractorCombined

try:
    from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator
except Exception as e:
    raise ImportError(
        "This script requires speculators.data_generation.vllm_hidden_states_generator.VllmHiddenStatesGenerator"
    ) from e


transformers_logging.set_verbosity_info()
transformers_logging.enable_default_handler()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

STEP_SEPARATOR = "\n\n"
IGNORE_INDEX = -100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data / I/O
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--hf-dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--hf-output-dir", type=str, default=None)
    parser.add_argument("--eval-size", type=float, default=0.1)

    # Base model / tokenizer
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--vllm-model-path", type=str, default=None)
    parser.add_argument("--token", type=str, default="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")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--model-torch-dtype", type=str, default="float16")
    parser.add_argument("--tokenizer-model-max-length", type=int, default=2400)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--truncation-side", type=str, default="right", choices=["left", "right"])

    # Task / labels
    parser.add_argument(
        "--label-mode",
        type=str,
        default="correctness",
        choices=["correctness", "step_continuation"],
    )

    # Training arguments (kept close to the PRM CLI surface)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # UHead architecture / init
    parser.add_argument("--uhead-path", type=str, default=None)
    parser.add_argument("--feature-layers", nargs="+", type=int, default=[-1])
    parser.add_argument("--uhead-head-dim", type=int, default=768)
    parser.add_argument("--uhead-n-layers", type=int, default=2)
    parser.add_argument("--uhead-n-heads", type=int, default=8)
    parser.add_argument("--uhead-dropout", type=float, default=0.1)
    parser.add_argument(
        "--uhead-mask-future-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--uhead-device", type=str, default="cuda:0")

    # vLLM / speculators hidden-state extraction
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)

    # Misc infra
    parser.add_argument("--report-to", type=str, default="wandb", choices=["wandb", "none"])

    return parser.parse_args()


def flatten_for_wandb(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}/{k}" if prefix else str(k)
            out.update(flatten_for_wandb(v, new_prefix))
    elif isinstance(obj, (list, tuple)):
        return {}
    else:
        try:
            out[prefix] = float(obj)
        except Exception:
            pass
    return out


def maybe_init_wandb(args: argparse.Namespace, output_dir: Path):
    if args.report_to != "wandb":
        return None
    try:
        import wandb
    except Exception as e:
        raise ImportError("report_to=wandb but wandb is not installed.") from e

    os.environ.setdefault("WANDB_DIR", str(output_dir))
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        name=os.environ.get("WANDB_NAME"),
        entity=os.environ.get("WANDB_ENTITY"),
        group=os.environ.get("WANDB_GROUP"),
        dir=str(output_dir),
        config=vars(args),
    )
    return run


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=max(args.tokenizer_model_max_length, args.max_length),
        padding_side="left",
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = args.truncation_side
    return tokenizer


def load_input_dataset(
    dataset_path: str | None,
    hf_dataset_path: str | None,
    eval_size: float,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    if dataset_path is not None and hf_dataset_path is not None:
        raise ValueError("Provide only one of dataset_path or hf_dataset_path")
    if dataset_path is None and hf_dataset_path is None:
        raise ValueError("Provide one of dataset_path or hf_dataset_path")
    if not (0.0 <= eval_size < 1.0):
        raise ValueError(f"eval_size must be in [0.0, 1.0), got {eval_size}")

    if dataset_path is not None:
        loaded = load_from_disk(dataset_path)
        if isinstance(loaded, DatasetDict):
            ds = loaded["train"]
        else:
            ds = loaded
    else:
        ds = load_dataset(hf_dataset_path, split="train")

    if eval_size == 0.0:
        return ds, None

    split_ds = ds.train_test_split(test_size=eval_size, seed=seed, shuffle=True)
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
            labels.append(IGNORE_INDEX)
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
            raise ValueError(f"step_continuation_correct[{i}] must be in [0,1], got {value}")
        if i == n_steps - 1:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(value)
    return labels


def get_step_labels_for_example(example: dict[str, Any], label_mode: str) -> list[float]:
    steps = example["steps"]
    if label_mode == "correctness":
        return build_step_correctness_labels(steps, int(example["first_incorrect_step"]))
    if label_mode == "step_continuation":
        return build_step_continuation_labels(steps, example["step_continuation_correct"])
    raise ValueError(f"Unknown label_mode: {label_mode}")


def build_training_text(question: str, steps: list[str]) -> str:
    return STEP_SEPARATOR.join([question, *steps])


def build_training_text_with_step_spans(question: str, steps: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text = question
    cursor = len(question)
    step_spans: list[tuple[int, int]] = []
    for step in steps:
        text += STEP_SEPARATOR
        cursor += len(STEP_SEPARATOR)
        start = cursor
        text += step
        cursor += len(step)
        step_spans.append((start, cursor))
    return text, step_spans


def tokenize_with_step_last_token_positions(
    question: str,
    steps: list[str],
    tokenizer,
    max_length: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    text, step_spans = build_training_text_with_step_spans(question, steps)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    visible_step_positions: list[tuple[int, int]] = []
    for step_idx, (step_start, step_end) in enumerate(step_spans):
        overlapping_positions = [
            pos
            for pos, (tok_start, tok_end) in enumerate(offsets)
            if tok_end > tok_start and tok_start < step_end and tok_end > step_start
        ]
        if overlapping_positions:
            visible_step_positions.append((step_idx, overlapping_positions[-1]))

    return input_ids, visible_step_positions


def filter_dataset_for_label_mode(
    ds: Dataset,
    label_mode: str,
    tokenizer,
    max_length: int,
    split_name: str | None = None,
) -> Dataset:
    before_total = len(ds)
    before_one_step = sum(1 for ex in ds if len(ex["steps"]) == 1)

    if label_mode == "step_continuation":
        ds = ds.filter(
            lambda example: (
                len(example["steps"]) >= 2
                and not any(x is None for x in example["step_continuation_correct"])
            ),
            desc="Filtering invalid step_continuation examples",
        )

    def _has_visible_supervised_step_end(example: dict[str, Any]) -> bool:
        step_labels = get_step_labels_for_example(example, label_mode)
        _, visible_step_positions = tokenize_with_step_last_token_positions(
            example["question"],
            example["steps"],
            tokenizer,
            max_length,
        )
        if not visible_step_positions:
            return False
        return any(step_labels[step_idx] >= 0.0 for step_idx, _ in visible_step_positions)

    ds = ds.filter(
        _has_visible_supervised_step_end,
        desc="Filtering examples with no visible supervised step-end token after truncation",
    )

    after_total = len(ds)
    after_one_step = sum(1 for ex in ds if len(ex["steps"]) == 1)
    removed_total = before_total - after_total
    removed_one_step = before_one_step - after_one_step
    removed_pct = 100.0 * removed_total / before_total if before_total else 0.0
    prefix = f"[{split_name}] " if split_name else ""
    log.info(
        "%sfiltered dataset: removed %d/%d (%.2f%%), one-step removed %d/%d",
        prefix,
        removed_total,
        before_total,
        removed_pct,
        removed_one_step,
        before_one_step,
    )
    return ds


def preprocess_dataset_for_uhead(ds: Dataset, tokenizer, max_length: int, label_mode: str) -> Dataset:
    def _map_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        question = example["question"]
        steps = example["steps"]
        if len(steps) == 0:
            raise ValueError(f"Row {idx}: empty 'steps' list is not supported")

        step_labels = get_step_labels_for_example(example, label_mode)
        input_ids, visible_step_positions = tokenize_with_step_last_token_positions(
            question,
            steps,
            tokenizer,
            max_length,
        )
        if not visible_step_positions:
            raise ValueError(
                f"Row {idx}: no step-end tokens remained visible after tokenization/truncation. "
                f"Increase --max-length. question_len={len(question)} n_steps={len(steps)}"
            )

        feature_positions: list[int] = []
        kept_labels: list[float] = []
        for step_idx, token_pos in visible_step_positions:
            feature_positions.append(int(token_pos))
            kept_labels.append(float(step_labels[step_idx]))

        if not any(lbl >= 0.0 for lbl in kept_labels):
            raise ValueError(f"Row {idx}: no supervised step labels remain after truncation")

        return {
            "input_ids": input_ids,
            "feature_positions": feature_positions,
            "step_labels": kept_labels,
        }

    return ds.map(
        _map_fn,
        with_indices=True,
        remove_columns=ds.column_names,
        desc="Tokenizing uhead step dataset",
    )


@dataclass
class VLLMStepMarkerCollator:
    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids_list: list[list[int]] = []
        claims_list: list[torch.Tensor] = []
        label_rows: list[list[float]] = []

        for e in examples:
            input_ids = list(e["input_ids"])
            feature_positions = list(e["feature_positions"])
            step_labels = [float(x) for x in e["step_labels"]]
            if len(feature_positions) != len(step_labels):
                raise RuntimeError(
                    f"feature_positions length {len(feature_positions)} does not match step_labels length {len(step_labels)}"
                )

            feat_len = len(input_ids)
            input_ids_list.append(input_ids)

            claim_masks: list[torch.Tensor] = []
            for pos in feature_positions:
                mask = torch.zeros(feat_len, dtype=torch.long)
                if 0 <= pos < feat_len:
                    mask[pos] = 1
                claim_masks.append(mask)

            if claim_masks:
                claims_tensor = torch.stack(claim_masks, dim=0)
            else:
                claims_tensor = torch.zeros((0, feat_len), dtype=torch.long)
            claims_list.append(claims_tensor)
            label_rows.append(step_labels)

        max_steps = max((len(row) for row in label_rows), default=0)
        padded_labels = torch.full((len(label_rows), max_steps), IGNORE_INDEX, dtype=torch.float32)
        for i, row in enumerate(label_rows):
            if row:
                padded_labels[i, : len(row)] = torch.tensor(row, dtype=torch.float32)

        return {
            "input_ids_list": input_ids_list,
            "claims": claims_list,
            "labels": padded_labels,
        }


class VLLMHiddenStateProvider:
    def __init__(self, args: argparse.Namespace, uhead):
        self.args = args
        self.uhead = uhead
        self.num_hidden_layers = self._get_num_hidden_layers()
        self.layer_ids = self._resolve_vllm_layer_ids(uhead)
        # Initialize vLLM engine with seed for reproducibility
        if args.token is not None:
            prev_token = os.environ.get("HF_TOKEN", None)
            os.environ["HF_TOKEN"] = args.token
        self.hs_generator = VllmHiddenStatesGenerator(
            layer_ids=self.layer_ids,
            model_path=args.vllm_model_path or args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
        )
        if args.token is not None:
            del os.environ["HF_TOKEN"]
            if prev_token is not None:
                os.environ["HF_TOKEN"] = prev_token
        log.info("vLLM hidden-state layers: %s", self.layer_ids)

    def _get_num_hidden_layers(self) -> int:
        cfg = AutoConfig.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
        )
        num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
        if num_hidden_layers is None:
            num_hidden_layers = getattr(getattr(cfg, "text_config", {}), "num_hidden_layers", None)
        if num_hidden_layers is None:
            raise ValueError(f"Could not determine num_hidden_layers for model {self.args.model_name!r}")
        return int(num_hidden_layers)

    def _hf_to_vllm_layer_id(self, layer_num: int) -> int:
        if layer_num > 0:
            vllm_layer_id = layer_num - 1
        else:
            vllm_layer_id = self.num_hidden_layers + layer_num

        if not (0 <= vllm_layer_id < self.num_hidden_layers):
            raise ValueError(
                f"Layer index {layer_num} resolves to vLLM layer {vllm_layer_id}, "
                f"which is out of bounds [0, {self.num_hidden_layers - 1}]"
            )
        return vllm_layer_id

    def _resolve_vllm_layer_ids(self, uhead) -> list[int]:
        fe = getattr(uhead, "feature_extractor")
        feature_extractors = getattr(fe, "_feature_extractors", [fe]) if isinstance(fe, FeatureExtractorCombined) else [fe]

        layer_ids: list[int] = []
        unsupported: list[str] = []
        for f in feature_extractors:
            if isinstance(f, FeatureExtractorBasicHiddenStates):
                layer_nums = getattr(f, "_layer_nums")
                raw_layer_nums = layer_nums if isinstance(layer_nums, (list, tuple)) else [layer_nums]
                layer_ids.extend([self._hf_to_vllm_layer_id(int(x)) for x in raw_layer_nums])
            else:
                unsupported.append(f.__class__.__name__)

        if unsupported:
            raise NotImplementedError(
                "This training script currently supports hidden-state-only uheads. "
                f"Unsupported feature extractors: {unsupported}"
            )
        if not layer_ids:
            raise ValueError("Could not infer hidden-state layer ids from uhead.feature_extractor")

        return list(dict.fromkeys(layer_ids))

    @staticmethod
    def _as_2d_float_cpu(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().cpu()
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        else:
            t = t.float()
        if t.ndim != 2:
            raise ValueError(f"Expected [T, H] or [1, T, H], got {tuple(t.shape)}")
        return t.contiguous()

    @torch.no_grad()
    def get_features(self, input_ids_list: list[list[int]], device: torch.device):
        payloads = self.hs_generator.generate(input_ids_list)

        per_sample_feats: list[torch.Tensor] = []
        lengths: list[int] = []
        feat_dim: int | None = None

        for seq_ids, payload in zip(input_ids_list, payloads):
            if payload is None or "hidden_states" not in payload or not payload["hidden_states"]:
                raise RuntimeError("Hidden-state generator returned an empty payload")

            hs_layers = [self._as_2d_float_cpu(h) for h in payload["hidden_states"]]
            seq_len = hs_layers[0].shape[0]
            for h in hs_layers[1:]:
                if h.shape[0] != seq_len:
                    raise RuntimeError("Returned hidden-state layers have different sequence lengths")

            sample = torch.cat(hs_layers, dim=-1)
            expected = len(seq_ids)
            if sample.shape[0] > expected:
                sample = sample[:expected]
            if sample.shape[0] < expected:
                raise RuntimeError(
                    f"Recovered hidden states are shorter than the input sequence: got={sample.shape[0]} expected={expected}"
                )

            per_sample_feats.append(sample)
            lengths.append(sample.shape[0])
            feat_dim = sample.shape[-1]

        max_len = max(lengths, default=0)
        batch_size = len(per_sample_feats)
        if feat_dim is None:
            raise RuntimeError("No features were generated for this batch")

        feats = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, sample in enumerate(per_sample_feats):
            T = sample.shape[0]
            if T > 0:
                feats[i, :T] = sample
                attn_mask[i, :T] = 1

        return feats.to(device, non_blocking=True), attn_mask.to(device, non_blocking=True)


class VLLMTrainableUHeadModel(nn.Module):
    def __init__(self, uhead, feature_provider, device: torch.device, pos_weight: float = 1.0):
        super().__init__()
        self.uhead = uhead.to(device)
        self.feature_provider = feature_provider
        self.device_ = device
        self.pos_weight = torch.tensor([pos_weight], device=device, dtype=torch.float32)

    def forward_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, attention_mask = self.feature_provider.get_features(batch["input_ids_list"], self.device_)
        claims = [c.to(self.device_) for c in batch["claims"]]
        labels = batch["labels"].to(self.device_)

        logits = self.uhead.forward_from_features(
            features=features,
            attention_mask=attention_mask,
            claims=claims,
        ).squeeze(-1)

        if logits.shape != labels.shape:
            raise RuntimeError(f"Logits shape {tuple(logits.shape)} does not match labels shape {tuple(labels.shape)}")

        mask = labels != IGNORE_INDEX
        if mask.sum() == 0:
            loss = logits.sum() * 0.0
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask],
                labels[mask],
                pos_weight=self.pos_weight,
            )
        return loss, logits, labels


def compute_step_metrics_from_flat(logits: np.ndarray, labels: np.ndarray, label_mode: str) -> dict[str, float]:
    if logits.size == 0 or labels.size == 0:
        out = {
            "step_bce": 0.0,
            "mean_target": 0.0,
            "mean_positive_prob": 0.0,
            "mse": 0.0,
            "mae": 0.0,
        }
        if label_mode == "correctness":
            out["step_accuracy"] = 0.0
        return out

    probs = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-8
    bce = -np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps))

    metrics = {
        "step_bce": float(bce),
        "mean_target": float(np.mean(labels)),
        "mean_positive_prob": float(np.mean(probs)),
        "mse": float(np.mean((probs - labels) ** 2)),
        "mae": float(np.mean(np.abs(probs - labels))),
    }
    if label_mode == "correctness":
        preds = (probs >= 0.5).astype(np.float32)
        metrics["step_accuracy"] = float(np.mean(preds == labels))
    return metrics


@torch.no_grad()
def evaluate_model(model: VLLMTrainableUHeadModel, dataloader: DataLoader, label_mode: str, desc: str) -> dict[str, Any]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        loss, logits, labels = model.forward_batch(batch)
        total_loss += float(loss.item())
        steps += 1

        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = labels_np != IGNORE_INDEX
        if np.any(mask_np):
            all_logits.append(logits_np[mask_np].astype(np.float32, copy=False))
            all_labels.append(labels_np[mask_np].astype(np.float32, copy=False))

    flat_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,), dtype=np.float32)
    flat_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.float32)
    metrics = compute_step_metrics_from_flat(flat_logits, flat_labels, label_mode=label_mode)
    metrics["loss"] = total_loss / max(steps, 1)
    return {"metrics": metrics, "logits": flat_logits, "labels": flat_labels}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_reinit_uhead(uhead):
    def reinitialize_weights(module):
        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.ndim >= 2:
                init.xavier_uniform_(module.weight)
            else:
                init.uniform_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            init.zeros_(module.bias)

    uhead.apply(reinitialize_weights)
    return uhead


def _torch_dtype_from_name(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)


def build_uhead_config(args: argparse.Namespace):
    layer_nums: int | list[int]
    if len(args.feature_layers) == 1:
        layer_nums = args.feature_layers[0]
    else:
        layer_nums = list(args.feature_layers)

    return OmegaConf.create(
        {
            "head_type": "step_reasoning",
            "feature_extractor": [
                {
                    "name": "luh.feature_extractors.basic_hidden_states",
                    "layer_nums": layer_nums,
                }
            ],
            "uncertainty_head": {
                "head_dim": args.uhead_head_dim,
                "n_layers": args.uhead_n_layers,
                "n_heads": args.uhead_n_heads,
                "dropout": args.uhead_dropout,
                "mask_future_tokens": args.uhead_mask_future_tokens,
            },
        }
    )


def build_uhead(args: argparse.Namespace):
    log.info("Initializing uncertainty head via AutoUncertaintyHead")
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype_from_name(args.model_torch_dtype),
        "trust_remote_code": True,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    base_model = argparse.Namespace(config=AutoConfig.from_pretrained(args.model_name, **model_kwargs))
    try:
        if args.uhead_path:
            uhead = AutoUncertaintyHead.from_pretrained(args.uhead_path, base_model)
        else:
            uhead_cfg = build_uhead_config(args)
            uhead = AutoUncertaintyHead.from_config(uhead_cfg, base_model)
            uhead = maybe_reinit_uhead(uhead)
    finally:
        del base_model
        gc.collect()
    return uhead


def build_dataloaders(args: argparse.Namespace, train_ds: Dataset, eval_ds: Dataset | None):
    collator = VLLMStepMarkerCollator()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.per_device_eval_batch_size or args.per_device_train_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, eval_loader


def save_artifacts(output_dir: Path, tokenizer, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    save_json(output_dir / "train_args.json", vars(args))


def main() -> None:
    args = parse_args()

    for _mp in (mp, tmp):
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_args.json", vars(args))

    wandb_run = maybe_init_wandb(args, output_dir)
    set_all_seeds(args.seed)

    if os.environ.get("CUDNN_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_device = torch.device(args.uhead_device if torch.cuda.is_available() else "cpu")
    log.info("Training uhead on device: %s", train_device)

    train_ds_raw, eval_ds_raw = load_input_dataset(
        args.dataset_path,
        args.hf_dataset_path,
        args.eval_size,
        args.seed,
    )

    tokenizer = load_tokenizer(args)

    train_ds_raw = filter_dataset_for_label_mode(
        train_ds_raw,
        args.label_mode,
        tokenizer=tokenizer,
        max_length=args.max_length,
        split_name="train",
    )
    if eval_ds_raw is not None:
        eval_ds_raw = filter_dataset_for_label_mode(
            eval_ds_raw,
            args.label_mode,
            tokenizer=tokenizer,
            max_length=args.max_length,
            split_name="eval",
        )

    train_ds = preprocess_dataset_for_uhead(train_ds_raw, tokenizer, args.max_length, args.label_mode)
    eval_ds = None
    if eval_ds_raw is not None:
        eval_ds = preprocess_dataset_for_uhead(eval_ds_raw, tokenizer, args.max_length, args.label_mode)

    save_artifacts(output_dir, tokenizer, args)

    uhead = build_uhead(args)
    feature_provider = VLLMHiddenStateProvider(args, uhead)
    model = VLLMTrainableUHeadModel(
        uhead=uhead,
        feature_provider=feature_provider,
        device=train_device,
        pos_weight=args.pos_weight,
    )

    train_loader, eval_loader = build_dataloaders(args, train_ds, eval_ds)

    optimizer = AdamW(
        model.uhead.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_epochs = int(args.num_train_epochs)
    grad_accum = int(args.gradient_accumulation_steps)
    total_train_steps = int(np.ceil(len(train_loader) / max(grad_accum, 1)) * num_epochs)
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    best_eval_loss = float("inf")
    no_improve_epochs = 0
    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(pbar, start=1):
            loss, _, _ = model.forward_batch(batch)
            raw_loss = float(loss.item())
            (loss / grad_accum).backward()
            running_loss += raw_loss

            should_step = (step % grad_accum == 0)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.uhead.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    mean_train_loss = running_loss / step
                    log.info(
                        "epoch=%d global_step=%d train_loss=%.6f lr=%.8f",
                        epoch + 1,
                        global_step,
                        mean_train_loss,
                        scheduler.get_last_lr()[0],
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": mean_train_loss,
                                "train/lr": scheduler.get_last_lr()[0],
                                "epoch": epoch + 1,
                            },
                            step=global_step,
                        )

            pbar.set_postfix(loss=f"{running_loss / step:.4f}")

        if len(train_loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.uhead.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        epoch_summary: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
        }

        if eval_loader is not None:
            eval_out = evaluate_model(model, eval_loader, label_mode=args.label_mode, desc="eval")
            epoch_summary["eval"] = eval_out["metrics"]
            current_eval_loss = float(eval_out["metrics"]["loss"])

            if current_eval_loss < best_eval_loss:
                best_eval_loss = current_eval_loss
                no_improve_epochs = 0

                best_model_dir = output_dir / "model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                model.uhead.save(best_model_dir)

                save_json(output_dir / "metrics" / "best_metrics.json", epoch_summary)
                log.info("Saved new best uhead to %s", best_model_dir)
            else:
                no_improve_epochs += 1
        else:
            current_eval_loss = float("nan")

        history.append(epoch_summary)
        save_json(output_dir / "metrics" / f"epoch_{epoch + 1}.json", epoch_summary)
        log.info(json.dumps(epoch_summary, indent=2))

        if wandb_run is not None:
            wb_payload = flatten_for_wandb(
                {
                    "train": {"loss": epoch_summary["train_loss"]},
                    "eval": epoch_summary.get("eval", {}),
                    "epoch": epoch + 1,
                    "lr": scheduler.get_last_lr()[0] if scheduler is not None else float("nan"),
                }
            )
            wandb_run.log(wb_payload, step=global_step)
            if eval_loader is not None:
                wandb_run.summary["best_eval_loss_so_far"] = min(
                    float(wandb_run.summary.get("best_eval_loss_so_far", float("inf"))),
                    current_eval_loss,
                )

        if eval_loader is not None and no_improve_epochs >= args.early_stopping_patience:
            log.info("Early stopping at epoch %d", epoch + 1)
            break

    if eval_loader is not None:
        final_eval = evaluate_model(model, eval_loader, label_mode=args.label_mode, desc="final-eval")
        save_json(output_dir / "metrics" / "final_eval.json", final_eval["metrics"])
        log.info("Final evaluation:\n%s", json.dumps(final_eval["metrics"], indent=2))
        if wandb_run is not None:
            wandb_run.log(flatten_for_wandb({"final_eval": final_eval["metrics"]}), step=global_step)
            for k, v in flatten_for_wandb({"final_eval": final_eval["metrics"]}).items():
                wandb_run.summary[k] = v

    final_model_dir = output_dir / "model"
    log.info("Saved best uhead to %s", final_model_dir)

    if args.hf_output_dir is not None:
        if eval_loader is not None:
            best_for_hub = AutoUncertaintyHead.from_pretrained(
                final_model_dir, argparse.Namespace(config=AutoConfig.from_pretrained(args.model_name)))
            best_for_hub.push_to_hub(args.hf_output_dir)
            log.info("Pushed best uhead to HF: %s", args.hf_output_dir)
        else:
            log.warning("No eval set: no best checkpoint exists, so nothing was pushed to HF.")

    save_json(output_dir / "metrics" / "history.json", {"history": history})

    if wandb_run is not None:
        try:
            wandb_run.summary["output_dir"] = str(output_dir)
            wandb_run.summary["history_len"] = len(history)
        finally:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
