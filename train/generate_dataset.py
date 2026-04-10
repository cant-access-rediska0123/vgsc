#!/usr/bin/env python3
"""Generate a sampled answer dataset from rediska0123/math-from-prm800k."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llm_tts.step_boundary_detectors import ThinkingMarkerDetector

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_thinking_marker_detector(cfg) -> ThinkingMarkerDetector:
    custom_markers = None
    if "custom_words" in cfg.strategy and cfg.strategy.custom_words is not None:
        custom_markers = list(cfg.strategy.custom_words)

    return ThinkingMarkerDetector(
        min_step_tokens=cfg.strategy.min_step_tokens,
        max_step_tokens=cfg.strategy.max_step_tokens,
        use_sequence=cfg.strategy.use_sequence,
        use_conclusion=cfg.strategy.use_conclusion,
        use_thinking=cfg.strategy.use_thinking,
        use_verification=cfg.strategy.use_verification,
        use_structure=cfg.strategy.use_structure,
        use_reasoning=cfg.strategy.use_reasoning,
        use_correction=cfg.strategy.use_correction,
        custom_markers=custom_markers,
    )


def build_chat_prompt(
    tokenizer: AutoTokenizer,
    *,
    question: str,
    system_prompt: str,
    prompt_template: Optional[str],
) -> str:
    if prompt_template:
        user_text = prompt_template.format(question=question)
    else:
        user_text = question

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def flatten_generations(
    prompts: list[str],
    request_rows: list[dict[str, Any]],
    outputs,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for prompt, meta, request_output in zip(prompts, request_rows, outputs):
        prompt_token_ids = list(request_output.prompt_token_ids)

        for out in request_output.outputs:
            completion_token_ids = list(out.token_ids)

            row = dict(meta)
            row["prompt"] = prompt
            row["prompt_token_ids"] = prompt_token_ids
            row["completion"] = out.text
            row["completion_token_ids"] = completion_token_ids
            rows.append(row)

    return rows


def generate_full_samples(
    llm: LLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    cfg,
) -> list[dict[str, Any]]:
    sampling_params = SamplingParams(
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        top_k=cfg.generation.top_k,
        repetition_penalty=cfg.generation.repetition_penalty,
        max_tokens=cfg.generation.max_new_tokens,
        stop_token_ids=list(getattr(cfg.strategy, "stop_token_ids", [])),
        n=cfg.num_samples_per_question,
        seed=cfg.seed,
    )

    prompts: list[str] = []
    metas: list[dict[str, Any]] = []

    question_field = getattr(cfg, "question_field", "question")
    answer_field = getattr(cfg, "answer_field", "answer")
    system_prompt = getattr(cfg, "system_prompt", "")
    prompt_template = getattr(cfg, "prompt_template", None)

    for ds_index, item in enumerate(tqdm(dataset, desc="Building prompts")):
        question = item[question_field]
        answer = item.get(answer_field)

        prompt = build_chat_prompt(
            tokenizer,
            question=question,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
        )
        prompts.append(prompt)
        metas.append(
            {
                "dataset_index": ds_index,
                "question": question,
                "gold_answer": answer,
            }
        )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    return flatten_generations(prompts, metas, outputs)


def add_step_columns(
    rows: list[dict[str, Any]],
    detector: ThinkingMarkerDetector,
    tokenizer: AutoTokenizer,
) -> None:
    for row in tqdm(rows, desc="Splitting steps"):
        completion = row["completion"]
        completion_token_ids: list[int] = row["completion_token_ids"]

        steps = detector.detect_steps(
            completion,
            normalize=True,
            use_stop_tokens=False,
        )
        row["steps"] = steps

        cumulative_end_positions: list[int] = []
        running_text = ""
        for step in steps:
            running_text += step
            prefix_token_ids = tokenizer.encode(
                running_text,
                add_special_tokens=False,
            )
            end_pos = min(len(prefix_token_ids), len(completion_token_ids))
            cumulative_end_positions.append(end_pos)

        step_token_spans: list[tuple[int, int]] = []
        prev_end = 0
        for end_pos in cumulative_end_positions:
            step_token_spans.append((prev_end, end_pos))
            prev_end = end_pos

        # If tokenization mismatch leaves leftover tokens, attach them to the last step.
        if prev_end < len(completion_token_ids):
            if step_token_spans:
                last_start, _ = step_token_spans[-1]
                step_token_spans[-1] = (last_start, len(completion_token_ids))
            elif completion_token_ids:
                step_token_spans = [(0, len(completion_token_ids))]

        row["step_token_spans"] = step_token_spans


def add_step_continuations(
    llm: LLM,
    rows: list[dict[str, Any]],
    cfg,
) -> None:
    sampling_params = SamplingParams(
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        top_k=cfg.generation.top_k,
        repetition_penalty=cfg.generation.repetition_penalty,
        max_tokens=cfg.generation.max_new_tokens,
        stop_token_ids=list(cfg.strategy.stop_token_ids),
        n=cfg.continuations_per_step,
        seed=cfg.seed,
    )

    max_model_len = llm.llm_engine.model_config.max_model_len
    inputs: list[dict[str, Any]] = []
    request_map: list[tuple[int, int]] = []

    for row_idx, row in enumerate(tqdm(rows, desc="Preparing step continuations")):
        prompt_token_ids: list[int] = row.get("prompt_token_ids", [])
        completion_token_ids: list[int] = row.get("completion_token_ids", [])
        step_token_spans: list[tuple[int, int]] = row.get("step_token_spans", [])
        if not step_token_spans:
            row["step_continuations"] = None
            continue
        row["step_continuations"] = [None for _ in range(len(step_token_spans))]
        for step_idx, (_, end_pos) in enumerate(step_token_spans):
            prefix_ids = prompt_token_ids + completion_token_ids[:end_pos]
            if len(prefix_ids) >= max_model_len:
                continue
            inputs.append({"prompt_token_ids": prefix_ids})
            request_map.append((row_idx, step_idx))
    
    if not inputs:
        return
    outputs = llm.generate(inputs, sampling_params, use_tqdm=True)
    for out, (row_idx, step_idx) in zip(outputs, request_map):
        rows[row_idx]["step_continuations"][step_idx] = [x.text for x in out.outputs]


def save_rows(
    rows: list[dict[str, Any]],
    cfg,
) -> None:
    ds = Dataset.from_list(rows)

    output = Path(cfg.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output))
    print(f"Saved {len(rows)} rows to {cfg.output_path}")

    if getattr(cfg, "hf_output_path", None):
        ds.push_to_hub(cfg.hf_output_path, private=True)
        print(f"Pushed Dataset to hub repo {cfg.hf_output_path}")


@hydra.main(version_base=None, config_name="config")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    set_random_seed(cfg.seed)

    if cfg.get("token", None) is not None:
        prev_token = os.environ.get("HF_TOKEN", None)
        os.environ["HF_TOKEN"] = cfg.token
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    llm = LLM(
        model=cfg.model_name_or_path,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=cfg.trust_remote_code,
        seed=cfg.seed,
        enforce_eager=cfg.enforce_eager,
    )
    if cfg.get("token", None) is not None:
        del os.environ["HF_TOKEN"]
        if prev_token is not None:
            os.environ["HF_TOKEN"] = prev_token

    if getattr(cfg, "dataset_subset", None):
        dataset = load_dataset(cfg.dataset_name, cfg.dataset_subset, split=cfg.dataset_split)
    else:
        dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    offset = getattr(cfg, "offset", 0) or 0
    if offset > 0:
        dataset = dataset.select(range(offset, len(dataset)))

    max_questions = getattr(cfg, "max_questions", None)
    if max_questions is not None:
        dataset = dataset.select(range(min(max_questions, len(dataset))))

    rows = generate_full_samples(llm, tokenizer, dataset, cfg)

    detector = create_thinking_marker_detector(cfg)
    add_step_columns(rows, detector=detector, tokenizer=tokenizer)
    
    save_rows(rows, cfg)

    if cfg.enable_step_continuations:
        add_step_continuations(llm, rows, cfg)
        save_rows(rows, cfg)


if __name__ == "__main__":
    main()