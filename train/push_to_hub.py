#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import create_repo
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    return parser.parse_args()


def get_base_model_name(local_dir: Path) -> str:
    adapter_config_path = local_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Could not find {adapter_config_path}. "
            "This script expects a PEFT/LoRA adapter directory."
        )
    with adapter_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_model_name = cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("adapter_config.json does not contain 'base_model_name_or_path'")
    return base_model_name


def main() -> None:
    args = parse_args()

    base_model_name = get_base_model_name(Path(args.local_dir))
    print(f"Base model: {base_model_name}")

    create_repo(args.repo_id, private=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.local_dir, trust_remote_code=True)

    print("Loading base model...")
    base_model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("Loading adapter and merging...")
    merged_model = (
        PeftModel.from_pretrained(base_model, args.local_dir, is_trainable=False)
        .merge_and_unload()
    )

    print(f"Pushing merged model to {args.repo_id} ...")
    merged_model.push_to_hub(args.repo_id, private=False, safe_serialization=True)

    print(f"Pushing tokenizer to {args.repo_id} ...")
    tokenizer.push_to_hub(args.repo_id, private=False)

    print("Done.")


if __name__ == "__main__":
    main()