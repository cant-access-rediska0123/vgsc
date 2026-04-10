from __future__ import annotations

import argparse
import logging
import re
import traceback
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import multiprocessing as mp
from pathlib import Path
from typing import Any
import warnings

from datasets import Dataset, load_dataset, load_from_disk
from tqdm.auto import tqdm

from llm_tts.evaluation import EvaluatorExactMatch
from lm_polygraph.utils.openai_chat import OpenAIChat

from llm_tts.utils import extract_answer

log = logging.getLogger(__name__)

FIRST_INCORRECT_STEP_PROMPT = r"""
You will be given:
- a <Problem>
- a full proposed <Solution>
- the same <Solution Steps>, split into numbered steps
- the <Gold Answer>

Your task is to determine the index of the first incorrect step.

Rules:
- A step is incorrect if it contains a mathematical/logical error that makes the reasoning invalid.
- If a step is correct but incomplete, do not mark it incorrect.
- If all steps are correct, output the value {all_correct_index}.
- Use the numbered steps exactly as provided.

Respond using the exact format below and nothing else:

<start of response>
Comments:
... brief reasoning ...
<First Incorrect Step>: {step_number}
<end of response>

<Problem>: {problem}

<Solution>: {solution}

<Solution Steps>:
{steps_text}

<Gold Answer>: {gold_answer}
"""


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_input_dataset(
        input_path: str | None,
        hf_input_path: str | None,
        split: str,
) -> Dataset:
    if input_path is not None:
        log.info("Loading dataset from local path: %s [%s]", input_path, split)
        return load_from_disk(input_path, split=split)
    if hf_input_path is not None:
        log.info("Loading dataset from HF Hub: %s [%s]", hf_input_path, split)
        return load_dataset(hf_input_path, split=split)
    raise ValueError("Provide either --input_path or --hf_input_path")


def format_steps(steps: list[str]) -> str:
    if not steps:
        return "[No steps provided]"
    return "\n\n".join(f"<STEP {i}>: {step.strip()}" for i, step in enumerate(steps, start=1))


def parse_first_incorrect_step(response: str, fallback_value: int) -> int:
    match = re.search(r"<First Incorrect Step>:\s*(\d+)", response)
    if match:
        return int(match.group(1))
    return fallback_value


class FirstIncorrectStepJudge:
    def __init__(
            self,
            cache_path: str,
            model: str = "gpt-5-nano",
            base_url: str = "https://api.openai.com/v1",
            n_threads: int = 1,
    ):
        self.chat = OpenAIChat(cache_path=cache_path, openai_model=model, base_url=base_url)
        self.n_threads = n_threads

    def _score_single(self, inp: tuple[int, dict[str, Any]]) -> tuple[int, int, str]:
        idx, row = inp
        question = row["question"]
        solution = row["completion"]
        gold_answer = row["gold_answer"]
        steps = row.get("steps") or []

        all_correct_index = len(steps) + 1
        steps_text = format_steps(steps)

        prompt = FIRST_INCORRECT_STEP_PROMPT.format(
            problem=question,
            solution=solution,
            steps_text=steps_text,
            gold_answer=gold_answer,
            all_correct_index=all_correct_index,
            step_number="{step_number}",
        )

        try:
            response = self.chat.ask(prompt)
        except AttributeError as e:
            log.error(e)
            response = ""
        parsed_step = parse_first_incorrect_step(response, fallback_value=all_correct_index)
        return idx, parsed_step, response

    def __call__(self, rows: list[dict[str, Any]]) -> list[int]:
        inputs = list(enumerate(rows))
        results: list[tuple[int, int, str]] = []

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(self._score_single, item) for item in inputs]
            for future in tqdm(futures, desc="Judging first incorrect steps"):
                results.append(future.result())

        results.sort(key=lambda x: x[0])
        steps = [x[1] for x in results]
        return steps


def annotate_first_incorrect_step(
        rows: list[dict[str, Any]],
        judge: FirstIncorrectStepJudge,
) -> None:
    first_steps = judge(rows)
    for row, first_step in zip(rows, first_steps):
        row["first_incorrect_step"] = first_step


def _chunk_ranges(n_items: int, n_chunks: int) -> list[tuple[int, int]]:
    if n_items <= 0:
        return []
    n_chunks = max(1, min(n_chunks, n_items))
    base, rem = divmod(n_items, n_chunks)
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        end = start + size
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def _score_rows_batch(
        rows_batch: list[dict[str, Any]],
        evaluator: EvaluatorExactMatch,
) -> None:
    """
    Score one mini-batch of rows in-place using a single evaluator call.
    """
    flat_problems: list[str] = []
    flat_solutions: list[str] = []
    flat_gold_answers: list[str] = []
    group_spans: list[tuple[int, int, int, int]] = []

    for row_idx, row in enumerate(rows_batch):
        continuations = row.get("step_continuations")
        if continuations is None:
            row["step_continuation_correct"] = None
            continue
        row["step_continuation_correct"] = [None] * len(continuations)
        gold_answer = extract_answer(row["gold_answer"] or "")
        if not gold_answer:
            gold_answer = row["gold_answer"]
        question = row["question"]
        for step_idx, step_conts in enumerate(continuations):
            if not step_conts:
                continue
            start = len(flat_solutions)
            n = len(step_conts)
            flat_problems.extend([question] * n)
            flat_solutions.extend(step_conts)
            flat_gold_answers.extend([gold_answer] * n)
            group_spans.append((row_idx, step_idx, start, n))

    if not flat_solutions:
        return
    flat_scores = evaluator(
        problems=flat_problems,
        solutions=flat_solutions,
        gold_answers=flat_gold_answers,
    )
    for row_idx, step_idx, start, n in group_spans:
        scores = flat_scores[start: start + n]
        n_correct = sum(1 for s in scores if s == 1.0)
        rows_batch[row_idx]["step_continuation_correct"][step_idx] = n_correct / n


def _needs_step_scoring(row: dict[str, Any]) -> bool:
    conts = row.get("step_continuations")
    if conts is None:
        return False

    existing = row.get("step_continuation_correct")
    if existing is None:
        return True

    if len(existing) != len(conts):
        return True

    return any(v is None for v in existing)


def _row_work_size(row: dict[str, Any]) -> int:
    conts = row.get("step_continuations") or []
    total = 0
    for step_conts in conts:
        if step_conts:
            total += len(step_conts)
    return max(total, 1)


def _make_step_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "step_continuations": row.get("step_continuations"),
    }


def _build_tasks(
    rows: list[dict[str, Any]],
    max_rows_per_task: int = 8,
    max_work_per_task: int = 256,
) -> list[tuple[list[int], list[dict[str, Any]]]]:
    tasks: list[tuple[list[int], list[dict[str, Any]]]] = []
    cur_indices: list[int] = []
    cur_payload: list[dict[str, Any]] = []
    cur_work = 0

    for i, row in enumerate(rows):
        if not _needs_step_scoring(row):
            continue

        row_work = _row_work_size(row)

        if cur_indices and (
            len(cur_indices) >= max_rows_per_task
            or cur_work + row_work > max_work_per_task
        ):
            tasks.append((cur_indices, cur_payload))
            cur_indices = []
            cur_payload = []
            cur_work = 0

        cur_indices.append(i)
        cur_payload.append(_make_step_payload(row))
        cur_work += row_work

    if cur_indices:
        tasks.append((cur_indices, cur_payload))

    return tasks


def _split_task_to_single_rows(
    task: tuple[list[int], list[dict[str, Any]]]
) -> list[tuple[list[int], list[dict[str, Any]]]]:
    indices, payload = task
    return [([i], [p]) for i, p in zip(indices, payload)]


def _score_rows_task(
    indices: list[int],
    payload_rows: list[dict[str, Any]],
    answer_format: str,
) -> list[tuple[int, Any]]:
    warnings.filterwarnings(
        "ignore",
        message=r".*object is not callable; perhaps you missed a comma\?",
        category=SyntaxWarning,
    )

    evaluator = EvaluatorExactMatch(
        dataset_answer_format=answer_format,
        data_name="math",
        silent=True,
    )

    _score_rows_batch(payload_rows, evaluator)

    out: list[tuple[int, Any]] = []
    for idx, row in zip(indices, payload_rows):
        out.append((idx, row.get("step_continuation_correct")))
    return out


def _kill_executor(executor: ProcessPoolExecutor) -> None:
    procs_dict = getattr(executor, "_processes", None) or {}
    procs = list(procs_dict.values())

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass

    for p in procs:
        try:
            p.join(timeout=1)
        except Exception:
            pass

    for p in procs:
        try:
            if p.is_alive():
                p.kill()
        except Exception:
            pass

    for p in procs:
        try:
            p.join(timeout=1)
        except Exception:
            pass


def annotate_step_continuations(
    rows: list[dict[str, Any]],
    answer_format: str,
    n_workers: int,
    task_timeout_s: int = 900,
    max_rows_per_task: int = 8,
    max_work_per_task: int = 256,
    checkpoint_every_tasks: int = 100,
    checkpoint_path: str | None = None,
) -> None:
    if not rows:
        return

    for row in rows:
        if row.get("step_continuations") is None and "step_continuation_correct" not in row:
            row["step_continuation_correct"] = None

    pending_tasks = _build_tasks(
        rows,
        max_rows_per_task=max_rows_per_task,
        max_work_per_task=max_work_per_task,
    )
    if not pending_tasks:
        return

    ctx = mp.get_context("spawn")
    pbar = tqdm(total=len(pending_tasks), desc="Exact-match tasks", dynamic_ncols=True)
    completed_tasks = 0

    while pending_tasks:
        retry_tasks: list[tuple[list[int], list[dict[str, Any]]]] = []
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            max_tasks_per_child=8,
        )
        pool_killed = False

        try:
            in_flight: dict[Any, tuple[float, tuple[list[int], list[dict[str, Any]]]]] = {}

            def submit_one(task: tuple[list[int], list[dict[str, Any]]]) -> None:
                fut = executor.submit(_score_rows_task, task[0], task[1], answer_format)
                in_flight[fut] = (time.monotonic(), task)

            while pending_tasks and len(in_flight) < n_workers:
                submit_one(pending_tasks.pop())

            while in_flight:
                done, _ = wait(
                    list(in_flight.keys()),
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )

                now = time.monotonic()

                for fut in list(done):
                    _, task = in_flight.pop(fut)

                    try:
                        result = fut.result()
                    except Exception as e:
                        log.exception("Task failed for rows %s: %s", task[0], e)

                        if len(task[0]) == 1:
                            row_idx = task[0][0]
                            rows[row_idx]["step_continuation_correct"] = None
                            rows[row_idx]["step_continuation_error"] = (
                                f"{type(e).__name__}: {e}"
                            )
                            completed_tasks += 1
                            pbar.update(1)
                        else:
                            retry_tasks.extend(_split_task_to_single_rows(task))
                            pbar.total += len(task[0]) - 1
                            pbar.refresh()

                        if pending_tasks:
                            submit_one(pending_tasks.pop())
                        continue

                    for row_idx, scores in result:
                        rows[row_idx]["step_continuation_correct"] = scores

                    completed_tasks += 1
                    pbar.update(1)

                    if checkpoint_path is not None and completed_tasks % checkpoint_every_tasks == 0:
                        save_dataset(rows, checkpoint_path, None)

                    if pending_tasks:
                        submit_one(pending_tasks.pop())

                overdue: list[tuple[Any, tuple[list[int], list[dict[str, Any]]]]] = []
                for fut, (started_at, task) in list(in_flight.items()):
                    if now - started_at > task_timeout_s:
                        overdue.append((fut, task))

                if overdue:
                    log.warning("Detected %d overdue task(s); restarting pool", len(overdue))

                    for fut, task in overdue:
                        in_flight.pop(fut, None)

                        if len(task[0]) == 1:
                            row_idx = task[0][0]
                            rows[row_idx]["step_continuation_correct"] = None
                            rows[row_idx]["step_continuation_error"] = "timeout"
                            completed_tasks += 1
                            pbar.update(1)
                        else:
                            retry_tasks.extend(_split_task_to_single_rows(task))
                            pbar.total += len(task[0]) - 1

                    retry_tasks.extend(task for _, task in in_flight.values())
                    retry_tasks.extend(pending_tasks)
                    pbar.refresh()

                    _kill_executor(executor)
                    pool_killed = True
                    break

        finally:
            if not pool_killed:
                executor.shutdown(wait=True, cancel_futures=False)

        pending_tasks = retry_tasks

    pbar.close()

    if checkpoint_path is not None:
        save_dataset(rows, checkpoint_path, None)


def save_dataset(
        rows: list[dict[str, Any]],
        output_path: str | None,
        hf_output_path: str | None,
) -> None:
    ds = Dataset.from_list(rows)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output))
        log.info("Saved annotated dataset to %s", output_path)

    if hf_output_path is not None:
        ds.push_to_hub(hf_output_path, private=True)
        log.info("Pushed annotated dataset to HF Hub: %s", hf_output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--hf-input-path", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--size", type=int, default=-1)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--answer-format", type=str, default="numeric")

    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--hf-output-path", type=str, default=None)

    parser.add_argument("--judge-cache-path", type=str, required=True)
    parser.add_argument("--judge-model", type=str, default="gpt-5-nano")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--n-threads", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    ds = load_input_dataset(
        input_path=args.input_path,
        hf_input_path=args.hf_input_path,
        split=args.split,
    )
    if args.size == -1:
        args.size = len(ds)
    rows = ds.to_list()[args.offset:args.offset + args.size]
    log.info("Loaded %d rows", len(rows))

    judge = FirstIncorrectStepJudge(
        cache_path=args.judge_cache_path,
        model=args.judge_model,
        n_threads=args.n_threads,
        base_url=args.base_url,
    )

    while True:
        try:
            annotate_first_incorrect_step(rows, judge)
            annotate_step_continuations(rows, args.answer_format, args.n_threads)
            break
        except Exception as e:
            log.error(f'Error during annotation: {e}.')
            log.error(traceback.format_exc())
            log.error(f'Resuming...')

    save_dataset(
        rows=rows,
        output_path=args.output_path,
        hf_output_path=args.hf_output_path,
    )


if __name__ == "__main__":
    main()
