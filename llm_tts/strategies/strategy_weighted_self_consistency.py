import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from llm_tts.generators import convert_trajectory_to_string
from llm_tts.generators.base import StepCandidate
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase

from evaluation.grader import math_equal_batched

if TYPE_CHECKING:
    from llm_tts.generators import (
        StepCandidateGeneratorThroughAPI,
        StepCandidateGeneratorThroughHuggingface,
    )

log = logging.getLogger(__name__)

STRATEGY_NAME = "weighted_self_consistency_stepwise"
NO_ANSWER = "no_answer"
TOKEN_MARGIN = 200
ALLOWED_AGGS = {
    "sum",
    "mean",
    "product",
    "product_correctness",
    "max",
    "min",
    "last",
}


@dataclass
class _PathState:
    steps: List[StepCandidate] = field(default_factory=list)
    generated_tokens_total: int = 0
    completed: bool = False

    # Filled after full-path scoring
    validity_scores: List[float] = field(default_factory=list)
    path_weight: float = 0.0

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def avg_validity(self) -> float:
        if not self.validity_scores:
            return 0.0
        return float(np.mean(self.validity_scores).item())


class StrategyWeightedSelfConsistencyStepwise(StrategyBase):
    """
    Stepwise self-consistency baseline with post-hoc weighting.

    Behavior:
    - Generate num_paths trajectories step-by-step.
    - No rejection / regeneration based on validity.
    - After ALL generations are finished, score the full trajectories with the scorer.
    - Aggregate each trajectory's scores into a scalar weight using `agg`.
    - Perform weighted self-consistency over extracted final answers.

    Notes:
    - Uses the same step_generator APIs as the other stepwise strategies.
    - In batch mode, generation proceeds in rounds: one next-step generation for every
      still-active path, with one batched generator call per round.
    """

    def __init__(
        self,
        step_generator: (
            "StepCandidateGeneratorThroughAPI | StepCandidateGeneratorThroughHuggingface"
        ),
        scorer: Optional[Any],
        num_paths: int = 10,
        max_steps: int = 10,
        agg: str = "product_correctness",
        batch_generation: bool = True,
        max_trajectory_tokens: int = 30000,
        alpha: float = 1.0,
    ):
        self.step_generator = step_generator
        self.scorer = scorer
        self.num_paths = num_paths
        self.max_steps = max_steps
        self.agg = agg
        self.batch_generation = batch_generation
        self.max_trajectory_tokens = max_trajectory_tokens
        self.alpha = alpha

        if self.agg not in ALLOWED_AGGS:
            raise ValueError(
                f"Unknown agg={self.agg}. Allowed: {sorted(ALLOWED_AGGS)}"
            )

        mode = "batched" if batch_generation else "sequential"
        log.info(
            "StrategyWeightedSelfConsistencyStepwise initialized: "
            f"num_paths={num_paths}, max_steps={max_steps}, agg={agg}, mode={mode}"
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def generate_trajectory(
        self,
        request: List[Dict[str, str]],
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Single-sample entrypoint.

        If batch_generation=True, delegates to generate_trajectories_batch([request])
        so single-sample and multi-sample logic stay aligned.
        """
        if self.batch_generation:
            return self.generate_trajectories_batch(
                requests=[request],
                sample_indices=[sample_idx],
            )[0]

        log.info(
            f"[WeightedSCStepwise] Sequential single-sample start: "
            f"sample_idx={sample_idx}, num_paths={self.num_paths}"
        )

        paths: List[_PathState] = []
        total_generated_tokens = 0

        for _ in range(self.num_paths):
            path = self._run_single_path_sequential(request=request)
            paths.append(path)
            total_generated_tokens += path.generated_tokens_total

        self._score_paths_after_generation(
            requests=[request],
            sample_indices=[sample_idx],
            paths_per_sample={0: paths},
        )

        result = self._finalize_sample_result(
            sample_idx=sample_idx,
            paths=paths,
            total_generated_tokens=total_generated_tokens,
        )

        log.info(
            f"[WeightedSCStepwise] sample {sample_idx}: "
            f"generated {self.num_paths} paths | "
            f"total generated tokens={total_generated_tokens}"
        )
        return result

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batched stepwise generation across multiple samples.

        In each round:
        - collect all active paths from all samples
        - generate exactly one next-step candidate per active path in a single batched call
        - accept that candidate immediately
        - stop paths that complete / hit max_steps / hit token budget

        After generation for ALL samples is done:
        - score all full trajectories
        - aggregate weights
        - do weighted self-consistency per sample
        """
        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        num_samples = len(requests)
        if num_samples == 0:
            return []

        log.info(
            f"[WeightedSCStepwise] Batch start: "
            f"{num_samples} samples x {self.num_paths} paths | "
            f"max_steps={self.max_steps}, agg={self.agg}"
        )

        paths_per_sample: Dict[int, List[_PathState]] = {
            sid: [_PathState() for _ in range(self.num_paths)]
            for sid in range(num_samples)
        }

        total_generated_tokens_all_samples = 0
        generated_tokens_per_sample: Dict[int, int] = {
            sid: 0 for sid in range(num_samples)
        }
        num_generate_calls = 0
        round_idx = 0

        while True:
            round_idx += 1

            active_items, batch_requests, batch_trajectories = self._prepare_batch_round(
                paths_per_sample=paths_per_sample,
                requests=requests,
            )

            if not active_items:
                break

            log.info(
                f"[WeightedSCStepwise] round {round_idx}: "
                f"expanding {len(active_items)} active paths"
            )

            all_candidates = self.step_generator.generate_step_candidates_batched_requests(
                batch_requests,
                trajectories=batch_trajectories,
                candidates_per_step=1,
            )
            num_generate_calls += 1

            if len(all_candidates) != len(active_items):
                raise ValueError(
                    "Batched step generator returned unexpected number of candidate lists: "
                    f"got {len(all_candidates)}, expected {len(active_items)}"
                )

            for (sid, _pid, path), cand_list in zip(active_items, all_candidates):
                cand = cand_list[0] if cand_list else None
                tok_count = self._apply_candidate_to_path(path, cand)

                generated_tokens_per_sample[sid] += tok_count
                total_generated_tokens_all_samples += tok_count

        self._score_paths_after_generation(
            requests=requests,
            sample_indices=sample_indices,
            paths_per_sample=paths_per_sample,
        )

        results: List[Dict[str, Any]] = []
        for sid in range(num_samples):
            result = self._finalize_sample_result(
                sample_idx=sample_indices[sid],
                paths=paths_per_sample[sid],
                total_generated_tokens=generated_tokens_per_sample[sid],
            )
            results.append(result)

        avg_tokens_per_sample = (
            total_generated_tokens_all_samples / num_samples if num_samples > 0 else 0.0
        )
        log.info(
            f"[WeightedSCStepwise] TOTAL generated tokens: {total_generated_tokens_all_samples} | "
            f"avg per sample: {avg_tokens_per_sample:.2f} | "
            f"num generate calls: {num_generate_calls}"
        )

        return results

    def cleanup(self):
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()

    # ---------------------------------------------------------------------
    # Generation helpers
    # ---------------------------------------------------------------------

    def _run_single_path_sequential(
        self,
        request: List[Dict[str, str]],
    ) -> _PathState:
        path = _PathState()

        while not self._should_stop_before_generating(path):
            cand_list = self.step_generator(
                request,
                trajectory=path.steps,
                candidates_per_step=1,
            )
            cand = cand_list[0] if cand_list else None
            self._apply_candidate_to_path(path, cand)

        return path

    def _prepare_batch_round(
        self,
        paths_per_sample: Dict[int, List[_PathState]],
        requests: List[List[Dict[str, str]]],
    ) -> Tuple[
        List[Tuple[int, int, _PathState]],
        List[List[Dict[str, str]]],
        List[List[StepCandidate]],
    ]:
        active_items: List[Tuple[int, int, _PathState]] = []
        batch_requests: List[List[Dict[str, str]]] = []
        batch_trajectories: List[List[StepCandidate]] = []

        for sid, paths in paths_per_sample.items():
            for pid, path in enumerate(paths):
                if self._should_stop_before_generating(path):
                    continue

                active_items.append((sid, pid, path))
                batch_requests.append(requests[sid])
                batch_trajectories.append(path.steps)

        return active_items, batch_requests, batch_trajectories

    def _should_stop_before_generating(self, path: _PathState) -> bool:
        if path.completed:
            return True

        if path.depth >= self.max_steps:
            path.completed = True
            return True

        if path.generated_tokens_total >= self.max_trajectory_tokens - TOKEN_MARGIN:
            if path.steps:
                path.steps[-1].is_trajectory_complete = True
            path.completed = True
            return True

        return False

    def _apply_candidate_to_path(
        self,
        path: _PathState,
        cand: Optional[StepCandidate],
    ) -> int:
        """
        Apply one candidate to a path.

        This intentionally preserves the previous generation logic:
        - None candidate => complete path
        - token count comes from len(cand.token_ids or [])
        - if token budget is now exceeded, mark the candidate complete
        - append candidate
        - complete path if candidate says trajectory complete or max_steps reached
        """
        if cand is None:
            path.completed = True
            return 0

        tok_ids = getattr(cand, "token_ids", None) or []
        tok_count = len(tok_ids)
        path.generated_tokens_total += tok_count

        if (
            path.generated_tokens_total >= self.max_trajectory_tokens
            and not cand.is_trajectory_complete
        ):
            cand.is_trajectory_complete = True

        path.steps.append(cand)

        if cand.is_trajectory_complete or path.depth >= self.max_steps:
            path.completed = True

        return tok_count

    # ---------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------

    def _score_paths_after_generation(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int],
        paths_per_sample: Dict[int, List[_PathState]],
    ) -> None:
        """
        Score all generated full trajectories after generation is complete.

        Populates:
            - path.validity_scores
            - path.path_weight
        """
        flat_requests: List[List[Dict[str, str]]] = []
        flat_trajectories: List[List[StepCandidate]] = []
        flat_sample_ids: List[int] = []
        flat_map: List[Tuple[int, int]] = []

        for sid, paths in paths_per_sample.items():
            for pid, path in enumerate(paths):
                flat_requests.append(requests[sid])
                flat_trajectories.append(path.steps)
                flat_sample_ids.append(sample_indices[sid])
                flat_map.append((sid, pid))

        if not flat_map:
            return

        if self.scorer is None:
            for sid, pid in flat_map:
                path = paths_per_sample[sid][pid]
                path.validity_scores = []
                path.path_weight = 1.0
            return

        raw_outputs = self._score_all_trajectories(
            flat_requests=flat_requests,
            flat_trajectories=flat_trajectories,
            flat_sample_ids=flat_sample_ids,
        )

        if len(raw_outputs) != len(flat_map):
            raise ValueError(
                f"Scorer returned {len(raw_outputs)} outputs for {len(flat_map)} trajectories"
            )

        for (sid, pid), raw in zip(flat_map, raw_outputs):
            path = paths_per_sample[sid][pid]
            path.validity_scores = self._normalize_scores(raw)
            path.path_weight = self._aggregate_scores(path.validity_scores)

    def _score_all_trajectories(
        self,
        flat_requests: List[List[Dict[str, str]]],
        flat_trajectories: List[List[StepCandidate]],
        flat_sample_ids: List[int],
    ) -> List[Any]:
        scorer_obj = self.scorer

        if hasattr(scorer_obj, "score_trajectories_batch"):
            return scorer_obj.score_trajectories_batch(
                flat_requests,
                flat_trajectories,
                sample_ids=flat_sample_ids,
                trajectory_ids=list(range(len(flat_trajectories))),
            )

        raw_outputs: List[Any] = []
        for req, traj in zip(flat_requests, flat_trajectories):
            if hasattr(scorer_obj, "score_trajectory"):
                raw = scorer_obj.score_trajectory(req, traj)
            else:
                raise AttributeError(
                    "Scorer must implement either "
                    "`score_trajectories_batch(...)` or `score_trajectory(...)`."
                )
            raw_outputs.append(raw)
        return raw_outputs

    def _normalize_scores(self, raw: Any) -> List[float]:
        if isinstance(raw, (list, tuple)):
            return [self._clip_score(float(x)) for x in raw]
        return [self._clip_score(float(raw))]

    def _aggregate_scores(self, scores: List[float]) -> float:
        if len(scores) == 0:
            return 0.0
        if self.agg == "sum":
            return float(sum(scores))
        if self.agg == "mean":
            return float(np.mean(scores).item())
        if self.agg == "product":
            return 1 - float(np.prod([1 - s for s in scores]).item())
        if self.agg == "product_correctness":
            return float(np.prod(scores).item())
        if self.agg == "max":
            return float(np.max(scores).item())
        if self.agg == "min":
            return float(np.min(scores).item())
        if self.agg == "last":
            return float(scores[-1])
        raise ValueError(f"Unknown agg={self.agg}")

    # ---------------------------------------------------------------------
    # Voting / finalization
    # ---------------------------------------------------------------------

    def _finalize_sample_result(
        self,
        sample_idx: int,
        paths: List[_PathState],
        total_generated_tokens: int,
    ) -> Dict[str, Any]:
        if not paths:
            return self._empty_result()

        path_texts = [convert_trajectory_to_string(path.steps) for path in paths]
        answers = [self._safe_extract_answer(text) for text in path_texts]

        voting_indices = self._get_voting_indices(answers)
        voting_answers = [answers[i] for i in voting_indices]

        cluster_ids_local, representatives, members_local = self._group_answers(
            voting_answers
        )

        if not members_local:
            return self._empty_result()

        members = {
            cid: [voting_indices[j] for j in idxs]
            for cid, idxs in members_local.items()
        }

        cluster_ids = [-1] * len(paths)
        for local_j, original_i in enumerate(voting_indices):
            cluster_ids[original_i] = cluster_ids_local[local_j]

        cluster_weights, cluster_counts = self._compute_cluster_stats(members, paths)
        total_weight = float(sum(cluster_weights.values()))

        selected_cluster = max(
            members.keys(),
            key=lambda cid: self._cluster_key(
                cid=cid,
                members=members,
                cluster_weights=cluster_weights,
                cluster_counts=cluster_counts,
                paths=paths,
            ),
        )
        selected_idx = self._select_best_index_within_cluster(
            members[selected_cluster],
            paths,
        )

        best_path = paths[selected_idx]
        best_text = path_texts[selected_idx]
        best_answer = answers[selected_idx]

        if total_weight > 0:
            consensus_score = cluster_weights[selected_cluster] / total_weight
        else:
            consensus_score = cluster_counts[selected_cluster] / max(len(voting_indices), 1)

        answer_distribution = {
            representatives[cid]: {
                "count": cluster_counts[cid],
                "weight": cluster_weights[cid],
            }
            for cid in members
        }

        all_traces = self._build_all_traces(
            paths=paths,
            path_texts=path_texts,
            answers=answers,
            cluster_ids=cluster_ids,
            representatives=representatives,
            cluster_weights=cluster_weights,
            cluster_counts=cluster_counts,
            selected_idx=selected_idx,
        )

        avg_num_steps = sum(len(path.steps) for path in paths) / max(len(paths), 1)

        log.info(
            f"[WeightedSCStepwise] sample {sample_idx}: "
            f"best_answer={best_answer}, weighted_consensus={consensus_score:.3f}, "
            f"answer_distribution={answer_distribution}"
        )

        return {
            "trajectory": best_text,
            "steps": best_path.steps,
            "validity_scores": list(best_path.validity_scores),
            "completed": True,
            "strategy": STRATEGY_NAME,
            "extracted_answer": best_answer,
            "answer_distribution": answer_distribution,
            "consensus_score": float(consensus_score),
            "all_answers": answers,
            "all_traces": all_traces,
            "total_tokens": int(total_generated_tokens),
            "token_stats": {
                "generated_output_tokens_total": int(total_generated_tokens),
                "selected_path_generated_tokens": int(best_path.generated_tokens_total),
                "avg_generated_tokens_per_path": (
                    float(total_generated_tokens) / max(len(paths), 1)
                ),
            },
            "thinking_num_steps": avg_num_steps,
            "response_num_steps": avg_num_steps,
        }

    def _get_voting_indices(self, answers: List[str]) -> List[int]:
        answered_indices = [
            i for i, ans in enumerate(answers)
            if ans not in ("", NO_ANSWER, None)
        ]
        return answered_indices if answered_indices else list(range(len(answers)))

    def _compute_cluster_stats(
        self,
        members: Dict[int, List[int]],
        paths: List[_PathState],
    ) -> Tuple[Dict[int, float], Dict[int, int]]:
        cluster_weights = {
            cid: float(sum((paths[i].path_weight ** self.alpha) for i in idxs))
            for cid, idxs in members.items()
        }
        cluster_counts = {
            cid: len(idxs)
            for cid, idxs in members.items()
        }
        return cluster_weights, cluster_counts

    def _cluster_key(
        self,
        cid: int,
        members: Dict[int, List[int]],
        cluster_weights: Dict[int, float],
        cluster_counts: Dict[int, int],
        paths: List[_PathState],
    ) -> Tuple[float, int, float, float, int]:
        best_idx = self._select_best_index_within_cluster(members[cid], paths)
        return (
            cluster_weights[cid],
            cluster_counts[cid],
            paths[best_idx].path_weight,
            paths[best_idx].avg_validity,
            -best_idx,
        )

    def _build_all_traces(
        self,
        paths: List[_PathState],
        path_texts: List[str],
        answers: List[str],
        cluster_ids: List[int],
        representatives: List[str],
        cluster_weights: Dict[int, float],
        cluster_counts: Dict[int, int],
        selected_idx: int,
    ) -> List[Dict[str, Any]]:
        all_traces: List[Dict[str, Any]] = []

        for i, (path, text, answer) in enumerate(zip(paths, path_texts, answers)):
            cid = cluster_ids[i]
            all_traces.append(
                {
                    "text": text,
                    # "steps": path.steps,
                    "num_steps": len(path.steps),
                    "generated_tokens_total": path.generated_tokens_total,
                    "validity_scores": list(path.validity_scores),
                    "avg_validity": path.avg_validity,
                    "path_weight": path.path_weight,
                    "answer": answer,
                    "answer_cluster_id": cid,
                    "cluster_representative": representatives[cid] if cid != -1 else None,
                    "cluster_weight": cluster_weights[cid] if cid != -1 else 0.0,
                    "cluster_count": cluster_counts[cid] if cid != -1 else 0,
                    "selected": i == selected_idx,
                }
            )

        return all_traces

    def _group_answers(
        self,
        answers: List[str],
    ) -> Tuple[List[int], List[str], Dict[int, List[int]]]:
        cluster_ids: List[int] = [-1] * len(answers)
        representatives: List[str] = []
        members: Dict[int, List[int]] = {}

        for i, ans in enumerate(answers):
            assigned_cluster = None

            if representatives:
                try:
                    matches = math_equal_batched(
                        predictions=[ans] * len(representatives),
                        references=representatives,
                        show_progress=False,
                    )
                    for cid, is_match in enumerate(matches):
                        if is_match:
                            assigned_cluster = cid
                            break
                except Exception:
                    for cid, rep in enumerate(representatives):
                        if str(ans).strip() == str(rep).strip():
                            assigned_cluster = cid
                            break

            if assigned_cluster is None:
                assigned_cluster = len(representatives)
                representatives.append(ans)
                members[assigned_cluster] = []

            cluster_ids[i] = assigned_cluster
            members.setdefault(assigned_cluster, []).append(i)

        return cluster_ids, representatives, members

    def _select_best_index_within_cluster(
        self,
        cluster_member_indices: List[int],
        paths: List[_PathState],
    ) -> int:
        return max(
            cluster_member_indices,
            key=lambda i: (
                paths[i].path_weight,
                paths[i].avg_validity,
                -i,
            ),
        )

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def _safe_extract_answer(self, text: str) -> str:
        try:
            ans = extract_answer(text)
            if ans is None or ans == "":
                return NO_ANSWER
            return str(ans)
        except Exception:
            return NO_ANSWER

    def _clip_score(self, score: float) -> float:
        if score is None:
            return 0.0
        return float(np.clip(score, 0.0, 1.0))

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "trajectory": "",
            "steps": [],
            "validity_scores": [],
            "completed": False,
            "strategy": STRATEGY_NAME,
            "extracted_answer": "",
            "answer_distribution": {},
            "consensus_score": 0.0,
            "all_answers": [],
            "all_traces": [],
            "total_tokens": 0,
            "token_stats": {},
            "thinking_num_steps": 0,
            "response_num_steps": 0,
        }