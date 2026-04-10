import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from evaluation.grader import math_equal_batched
from llm_tts.generators import convert_trajectory_to_string
from llm_tts.generators.base import StepCandidate
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase

if TYPE_CHECKING:
    from llm_tts.generators import (
        StepCandidateGeneratorThroughAPI,
        StepCandidateGeneratorThroughHuggingface,
    )

log = logging.getLogger(__name__)


@dataclass
class _SampledPathState:
    steps: List[StepCandidate] = field(default_factory=list)
    generated_tokens_total: int = 0
    completed: bool = False

    combined_step_scores: List[float] = field(default_factory=list)
    main_step_scores: List[float] = field(default_factory=list)
    heuristic_step_scores: List[float] = field(default_factory=list)
    sampled_step_probs: List[float] = field(default_factory=list)

    sampled_candidate_indices: List[int] = field(default_factory=list)
    candidate_main_scores_history: List[List[float]] = field(default_factory=list)
    candidate_heuristic_scores_history: List[List[float]] = field(default_factory=list)
    candidate_combined_scores_history: List[List[float]] = field(default_factory=list)
    candidate_probs_history: List[List[float]] = field(default_factory=list)

    finished_voting_traces: List["_VotingTraceState"] = field(default_factory=list)
    finished_voting_trace_keys: Set[str] = field(default_factory=set)

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def avg_combined_score(self) -> float:
        return float(np.mean(self.combined_step_scores).item()) if self.combined_step_scores else 0.0

    @property
    def avg_main_score(self) -> float:
        return float(np.mean(self.main_step_scores).item()) if self.main_step_scores else 0.0

    @property
    def avg_heuristic_score(self) -> float:
        return float(np.mean(self.heuristic_step_scores).item()) if self.heuristic_step_scores else 0.0


@dataclass
class _VotingTraceState:
    steps: List[StepCandidate] = field(default_factory=list)
    generated_tokens_total: int = 0

    combined_step_scores: List[float] = field(default_factory=list)
    main_step_scores: List[float] = field(default_factory=list)
    heuristic_step_scores: List[float] = field(default_factory=list)

    source_path_idx: int = -1
    trace_kind: str = "sampled_path"

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def avg_combined_score(self) -> float:
        return float(np.mean(self.combined_step_scores).item()) if self.combined_step_scores else 0.0

    @property
    def avg_main_score(self) -> float:
        return float(np.mean(self.main_step_scores).item()) if self.main_step_scores else 0.0

    @property
    def avg_heuristic_score(self) -> float:
        return float(np.mean(self.heuristic_step_scores).item()) if self.heuristic_step_scores else 0.0


class StrategyVerifierGuidedSelfConsistencyStepwise(StrategyBase):
    """
    Simplified version with fixed behavior:

    - answer extraction always uses extract_answer
    - voting always uses math_equal_batched (with exact-string fallback)
    - finished candidates are always collected into the voting pool
    - self-consistency voting is always score-weighted
    - voting weights always come from main_step_scores
    """

    def __init__(
        self,
        step_generator: (
            "StepCandidateGeneratorThroughAPI | StepCandidateGeneratorThroughHuggingface"
        ),
        scorer: Optional[Any],
        scorer_heuristic: Optional[Any] = None,
        num_paths: int = 5,
        candidates_per_step: int = 3,
        max_steps: int = 20,
        agg: str = "product_correctness",
        batch_generation: bool = True,
        max_trajectory_tokens: int = 30000,
        seed: Optional[int] = None,
        alpha: float = 1.0,
        self_consistency_score_agg: str = "product_correctness",
    ):
        self.step_generator = step_generator
        self.scorer = scorer
        self.scorer_heuristic = scorer_heuristic

        self.num_paths = num_paths
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.agg = agg
        self.batch_generation = batch_generation
        self.max_trajectory_tokens = max_trajectory_tokens
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.self_consistency_score_agg = self_consistency_score_agg

        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        allowed_agg = {
            "sum",
            "mean",
            "product",
            "product_correctness",
            "max",
            "min",
            "last",
        }
        if self.agg not in allowed_agg:
            raise ValueError(f"Unknown agg={self.agg}. Allowed: {sorted(allowed_agg)}")
        if self.self_consistency_score_agg not in allowed_agg:
            raise ValueError(
                "Unknown self_consistency_score_agg="
                f"{self.self_consistency_score_agg}. Allowed: {sorted(allowed_agg)}"
            )

        mode = "batched" if batch_generation else "sequential"
        log.info(
            "StrategyVerifierGuidedSelfConsistencyStepwise initialized: "
            f"num_paths={num_paths}, candidates_per_step={candidates_per_step}, "
            f"max_steps={max_steps}, agg={agg}, mode={mode}, "
            f"heuristic={'on' if scorer_heuristic is not None else 'off'}, "
            f"seed={seed}, alpha={self.alpha}, "
            f"self_consistency_score_agg={self.self_consistency_score_agg}"
        )

    # ---------------------------------------------------------------------
    # Score aggregation helpers
    # ---------------------------------------------------------------------

    def _aggregate_scores(self, scores: List[float], agg: str) -> float:
        if not scores:
            return 1.0

        arr = np.asarray([self._clip_score(s) for s in scores], dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)

        if agg == "sum":
            return float(arr.sum())
        if agg == "mean":
            return float(arr.mean())
        if agg == "product":
            return 1.0 - float(np.prod(1.0 - arr).item())
        if agg == "product_correctness":
            return float(np.prod(arr).item())
        if agg == "max":
            return float(arr.max())
        if agg == "min":
            return float(arr.min())
        if agg == "last":
            return float(arr[-1])

        raise ValueError(f"Unknown agg={agg}")

    def _combine_main_and_heuristic_scores(
        self,
        main_score: Optional[float],
        heuristic_score: Optional[float],
    ) -> float:
        parts = []
        if main_score is not None:
            parts.append(self._clip_score(main_score))
        if heuristic_score is not None:
            parts.append(self._clip_score(heuristic_score))

        if not parts:
            return 1.0
        if len(parts) == 1:
            return parts[0]

        return self._aggregate_scores(parts, self.agg)

    def _trace_vote_weight(self, trace: _VotingTraceState) -> float:
        return max(
            0.0,
            float(self._aggregate_scores(trace.main_step_scores, self.self_consistency_score_agg)),
        )

    # ---------------------------------------------------------------------
    # Scorer helpers
    # ---------------------------------------------------------------------

    def _score_candidates_for_path(
        self,
        scorer_obj: Optional[Any],
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates: List[StepCandidate],
        sample_idx: int,
        *,
        allow_score_candidates: bool,
    ) -> List[float]:
        if not candidates:
            return []

        if scorer_obj is None:
            return [1.0] * len(candidates)

        if allow_score_candidates and hasattr(scorer_obj, "score_candidates"):
            scores = scorer_obj.score_candidates(
                request,
                candidates,
                trajectory=trajectory,
            )
            return [self._clip_score(float(s)) for s in scores]

        full_trajectories = [trajectory + [cand] for cand in candidates]

        if hasattr(scorer_obj, "score_trajectories_batch"):
            raw_outputs = scorer_obj.score_trajectories_batch(
                [request] * len(full_trajectories),
                full_trajectories,
                sample_ids=[sample_idx] * len(full_trajectories),
                trajectory_ids=list(range(len(full_trajectories))),
            )
            return [self._extract_last_score(raw) for raw in raw_outputs]

        if hasattr(scorer_obj, "score_trajectory"):
            return [
                self._extract_last_score(scorer_obj.score_trajectory(request, full_traj))
                for full_traj in full_trajectories
            ]

        scorer_name = scorer_obj.__class__.__name__
        if allow_score_candidates:
            raise AttributeError(
                f"{scorer_name} must implement `score_candidates(...)`, "
                "`score_trajectories_batch(...)`, or `score_trajectory(...)`."
            )
        raise AttributeError(
            f"{scorer_name} must implement `score_trajectories_batch(...)` "
            "or `score_trajectory(...)`."
        )

    def _score_candidates_batched(
        self,
        scorer_obj: Optional[Any],
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int],
        active_items: List[Tuple[int, int, _SampledPathState]],
        all_candidates: List[List[StepCandidate]],
        *,
        allow_score_candidates: bool,
    ) -> List[List[float]]:
        if not all_candidates:
            return []

        if scorer_obj is None:
            return [[1.0] * len(cands) for cands in all_candidates]

        if hasattr(scorer_obj, "score_trajectories_batch"):
            flat_requests: List[List[Dict[str, str]]] = []
            flat_trajectories: List[List[StepCandidate]] = []
            flat_sample_ids: List[int] = []
            candidate_map: List[Tuple[int, int]] = []

            for item_idx, ((sid, _pid, path), candidates) in enumerate(zip(active_items, all_candidates)):
                for cand_idx, cand in enumerate(candidates):
                    flat_requests.append(requests[sid])
                    flat_trajectories.append(path.steps + [cand])
                    flat_sample_ids.append(sample_indices[sid])  # important fix
                    candidate_map.append((item_idx, cand_idx))

            if not flat_trajectories:
                return [[] for _ in all_candidates]

            raw_outputs = scorer_obj.score_trajectories_batch(
                flat_requests,
                flat_trajectories,
                sample_ids=flat_sample_ids,
                trajectory_ids=list(range(len(flat_trajectories))),
            )

            scores_by_item: List[List[Optional[float]]] = [
                [None] * len(cands) for cands in all_candidates
            ]
            for (item_idx, cand_idx), raw in zip(candidate_map, raw_outputs):
                scores_by_item[item_idx][cand_idx] = self._extract_last_score(raw)

            return [
                [1.0 if s is None else float(s) for s in item_scores]
                for item_scores in scores_by_item
            ]

        return [
            self._score_candidates_for_path(
                scorer_obj=scorer_obj,
                request=requests[sid],
                trajectory=path.steps,
                candidates=candidates,
                sample_idx=sample_indices[sid],
                allow_score_candidates=allow_score_candidates,
            )
            for (sid, _pid, path), candidates in zip(active_items, all_candidates)
        ]

    # ---------------------------------------------------------------------
    # Voting pool helpers
    # ---------------------------------------------------------------------

    def _path_to_voting_trace(
        self,
        path: _SampledPathState,
        source_path_idx: int,
        trace_kind: str,
    ) -> _VotingTraceState:
        return _VotingTraceState(
            steps=list(path.steps),
            generated_tokens_total=int(path.generated_tokens_total),
            combined_step_scores=list(path.combined_step_scores),
            main_step_scores=list(path.main_step_scores),
            heuristic_step_scores=list(path.heuristic_step_scores),
            source_path_idx=int(source_path_idx),
            trace_kind=trace_kind,
        )

    def _maybe_collect_finished_candidates_for_path(
        self,
        path: _SampledPathState,
        candidates: List[StepCandidate],
        main_scores: List[float],
        heuristic_scores: List[float],
        combined_scores: List[float],
        source_path_idx: int,
    ) -> None:
        if not candidates:
            return

        prefix_steps = list(path.steps)
        prefix_main_scores = list(path.main_step_scores)
        prefix_heuristic_scores = list(path.heuristic_step_scores)
        prefix_combined_scores = list(path.combined_step_scores)
        prefix_generated_tokens_total = int(path.generated_tokens_total)

        for cand, ms, hs, cs in zip(candidates, main_scores, heuristic_scores, combined_scores):
            if not getattr(cand, "is_trajectory_complete", False):
                continue

            tok_count = len(getattr(cand, "token_ids", None) or [])

            trace = _VotingTraceState(
                steps=prefix_steps + [cand],
                generated_tokens_total=prefix_generated_tokens_total + tok_count,
                combined_step_scores=prefix_combined_scores + [self._clip_score(cs)],
                main_step_scores=prefix_main_scores + [self._clip_score(ms)],
                heuristic_step_scores=prefix_heuristic_scores + [self._clip_score(hs)],
                source_path_idx=int(source_path_idx),
                trace_kind="collected_finished_chain",
            )

            key = convert_trajectory_to_string(trace.steps)
            if key in path.finished_voting_trace_keys:
                continue

            path.finished_voting_trace_keys.add(key)
            path.finished_voting_traces.append(trace)

    def _maybe_collect_current_sampled_path_as_finished(
        self,
        path: _SampledPathState,
        source_path_idx: int,
    ) -> None:
        if not path.steps:
            return
        if not getattr(path.steps[-1], "is_trajectory_complete", False):
            return

        trace = self._path_to_voting_trace(
            path=path,
            source_path_idx=source_path_idx,
            trace_kind="sampled_path",
        )
        key = convert_trajectory_to_string(trace.steps)

        if key in path.finished_voting_trace_keys:
            return

        path.finished_voting_trace_keys.add(key)
        path.finished_voting_traces.append(trace)

    def _build_weighted_voting_pool(
        self,
        paths: List[_SampledPathState],
    ) -> Tuple[List[_VotingTraceState], List[float]]:
        if not paths:
            return [], []

        traces: List[_VotingTraceState] = []
        for path_idx, path in enumerate(paths):
            if path.finished_voting_traces:
                traces.extend(path.finished_voting_traces)
            else:
                traces.append(
                    self._path_to_voting_trace(
                        path=path,
                        source_path_idx=path_idx,
                        trace_kind="sampled_path_fallback",
                    )
                )

        weights = [self._trace_vote_weight(trace) for trace in traces]
        if any(weight > 0.0 for weight in weights):
            return traces, weights

        log.warning(
            "All self-consistency weights were zero; falling back to uniform voting weights."
        )
        return traces, [1.0] * len(traces)

    # ---------------------------------------------------------------------
    # Sampling helpers
    # ---------------------------------------------------------------------

    def _sample_candidate_index_with_finished_handling(
        self,
        candidates: List[StepCandidate],
        scores: List[float],
    ) -> Tuple[Optional[int], List[float]]:
        if len(candidates) != len(scores):
            raise ValueError(
                f"Candidates/scores length mismatch: {len(candidates)} vs {len(scores)}"
            )

        if not candidates:
            return None, []

        unfinished_indices = [
            i for i, cand in enumerate(candidates)
            if not getattr(cand, "is_trajectory_complete", False)
        ]

        if not unfinished_indices:
            return None, [0.0] * len(candidates)

        unfinished_scores = [scores[i] for i in unfinished_indices]
        selected_idx_local, unfinished_probs = self._sample_candidate_index(unfinished_scores)

        probs_full = [0.0] * len(candidates)
        for orig_idx, p in zip(unfinished_indices, unfinished_probs):
            probs_full[orig_idx] = float(p)

        return unfinished_indices[selected_idx_local], probs_full

    def _sample_candidate_index(self, scores: List[float]) -> Tuple[int, List[float]]:
        probs = self._normalize_scores(scores)
        if len(probs) == 0:
            return 0, []
        if len(probs) == 1:
            return 0, probs.tolist()
        idx = int(self.rng.choice(len(probs), p=probs))
        return idx, probs.tolist()

    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        if len(scores) == 0:
            return np.asarray([], dtype=float)

        arr = np.asarray(scores, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)

        if self.alpha == 0.0:
            positive = arr > 0.0
            if positive.any():
                out = np.zeros_like(arr, dtype=float)
                out[positive] = 1.0 / positive.sum()
                return out
            return np.ones_like(arr, dtype=float) / len(arr)

        arr = np.power(arr, self.alpha)

        total = float(arr.sum())
        if total <= 0.0:
            return np.ones_like(arr, dtype=float) / len(arr)

        return arr / total

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def generate_trajectory(
        self,
        request: List[Dict[str, str]],
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        if self.batch_generation:
            return self.generate_trajectories_batch(
                requests=[request],
                sample_indices=[sample_idx],
            )[0]

        paths: List[_SampledPathState] = []
        total_generated_tokens = 0

        for path_idx in range(self.num_paths):
            path = self._run_single_path_sequential(
                request=request,
                sample_idx=sample_idx,
                path_idx=path_idx,
            )
            paths.append(path)
            total_generated_tokens += path.generated_tokens_total

        return self._finalize_sample_result(
            sample_idx=sample_idx,
            paths=paths,
            total_generated_tokens=total_generated_tokens,
        )

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        num_samples = len(requests)
        if num_samples == 0:
            return []

        paths_per_sample: Dict[int, List[_SampledPathState]] = {
            sid: [_SampledPathState() for _ in range(self.num_paths)]
            for sid in range(num_samples)
        }
        generated_tokens_per_sample: Dict[int, int] = {sid: 0 for sid in range(num_samples)}

        while True:
            active_items: List[Tuple[int, int, _SampledPathState]] = []
            batch_requests: List[List[Dict[str, str]]] = []
            batch_trajectories: List[List[StepCandidate]] = []

            for sid in range(num_samples):
                for pid, path in enumerate(paths_per_sample[sid]):
                    if path.completed:
                        continue

                    if path.depth >= self.max_steps:
                        path.completed = True
                        continue

                    if path.generated_tokens_total >= self.max_trajectory_tokens - 200:
                        if path.steps:
                            path.steps[-1].is_trajectory_complete = True
                            self._maybe_collect_current_sampled_path_as_finished(path, pid)
                        path.completed = True
                        continue

                    active_items.append((sid, pid, path))
                    batch_requests.append(requests[sid])
                    batch_trajectories.append(path.steps)

            if not active_items:
                break

            all_candidates = self.step_generator.generate_step_candidates_batched_requests(
                batch_requests,
                trajectories=batch_trajectories,
                candidates_per_step=self.candidates_per_step,
            )

            if len(all_candidates) != len(active_items):
                raise ValueError(
                    "Batched step generator returned unexpected number of candidate lists: "
                    f"got {len(all_candidates)}, expected {len(active_items)}"
                )

            all_main_scores = self._score_candidates_batched(
                scorer_obj=self.scorer,
                requests=requests,
                sample_indices=sample_indices,
                active_items=active_items,
                all_candidates=all_candidates,
                allow_score_candidates=True,
            )
            all_heuristic_scores = self._score_candidates_batched(
                scorer_obj=self.scorer_heuristic,
                requests=requests,
                sample_indices=sample_indices,
                active_items=active_items,
                all_candidates=all_candidates,
                allow_score_candidates=False,
            )

            for (sid, pid, path), cand_list, main_scores, heuristic_scores in zip(
                active_items,
                all_candidates,
                all_main_scores,
                all_heuristic_scores,
            ):
                if not cand_list:
                    path.completed = True
                    continue

                combined_scores = [
                    self._combine_main_and_heuristic_scores(ms, hs)
                    for ms, hs in zip(main_scores, heuristic_scores)
                ]

                self._maybe_collect_finished_candidates_for_path(
                    path=path,
                    candidates=cand_list,
                    main_scores=main_scores,
                    heuristic_scores=heuristic_scores,
                    combined_scores=combined_scores,
                    source_path_idx=pid,
                )

                sampled_idx, probs = self._sample_candidate_index_with_finished_handling(
                    candidates=cand_list,
                    scores=combined_scores,
                )

                if sampled_idx is None:
                    path.completed = True
                    continue

                cand = cand_list[sampled_idx]
                tok_count = len(getattr(cand, "token_ids", None) or [])

                path.generated_tokens_total += tok_count
                generated_tokens_per_sample[sid] += tok_count

                if path.generated_tokens_total >= self.max_trajectory_tokens and not cand.is_trajectory_complete:
                    cand.is_trajectory_complete = True

                selected_main = self._clip_score(main_scores[sampled_idx])
                selected_heur = self._clip_score(heuristic_scores[sampled_idx])
                selected_combined = self._clip_score(combined_scores[sampled_idx])

                path.steps.append(cand)
                path.main_step_scores.append(selected_main)
                path.heuristic_step_scores.append(selected_heur)
                path.combined_step_scores.append(selected_combined)
                path.sampled_step_probs.append(float(probs[sampled_idx]))
                path.sampled_candidate_indices.append(int(sampled_idx))
                path.candidate_main_scores_history.append([float(x) for x in main_scores])
                path.candidate_heuristic_scores_history.append([float(x) for x in heuristic_scores])
                path.candidate_combined_scores_history.append([float(x) for x in combined_scores])
                path.candidate_probs_history.append([float(x) for x in probs])

                self._maybe_collect_current_sampled_path_as_finished(path, pid)

                if cand.is_trajectory_complete or path.depth >= self.max_steps:
                    path.completed = True

        return [
            self._finalize_sample_result(
                sample_idx=sample_indices[sid],
                paths=paths_per_sample[sid],
                total_generated_tokens=generated_tokens_per_sample[sid],
            )
            for sid in range(num_samples)
        ]

    def cleanup(self):
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
        if self.scorer_heuristic is not None and hasattr(self.scorer_heuristic, "cleanup"):
            self.scorer_heuristic.cleanup()

    # ---------------------------------------------------------------------
    # Sequential helper
    # ---------------------------------------------------------------------

    def _run_single_path_sequential(
        self,
        request: List[Dict[str, str]],
        sample_idx: int,
        path_idx: int,
    ) -> _SampledPathState:
        path = _SampledPathState()

        while not path.completed:
            if path.depth >= self.max_steps:
                path.completed = True
                break

            if path.generated_tokens_total >= self.max_trajectory_tokens - 200:
                if path.steps:
                    path.steps[-1].is_trajectory_complete = True
                    self._maybe_collect_current_sampled_path_as_finished(path, path_idx)
                path.completed = True
                break

            cand_list = self.step_generator(
                request,
                trajectory=path.steps,
                candidates_per_step=self.candidates_per_step,
            )
            if not cand_list:
                path.completed = True
                break

            main_scores = self._score_candidates_for_path(
                scorer_obj=self.scorer,
                request=request,
                trajectory=path.steps,
                candidates=cand_list,
                sample_idx=sample_idx,
                allow_score_candidates=True,
            )
            heuristic_scores = self._score_candidates_for_path(
                scorer_obj=self.scorer_heuristic,
                request=request,
                trajectory=path.steps,
                candidates=cand_list,
                sample_idx=sample_idx,
                allow_score_candidates=False,
            )

            combined_scores = [
                self._combine_main_and_heuristic_scores(ms, hs)
                for ms, hs in zip(main_scores, heuristic_scores)
            ]

            self._maybe_collect_finished_candidates_for_path(
                path=path,
                candidates=cand_list,
                main_scores=main_scores,
                heuristic_scores=heuristic_scores,
                combined_scores=combined_scores,
                source_path_idx=path_idx,
            )

            sampled_idx, probs = self._sample_candidate_index_with_finished_handling(
                candidates=cand_list,
                scores=combined_scores,
            )
            if sampled_idx is None:
                path.completed = True
                break

            cand = cand_list[sampled_idx]
            tok_count = len(getattr(cand, "token_ids", None) or [])
            path.generated_tokens_total += tok_count

            if path.generated_tokens_total >= self.max_trajectory_tokens and not cand.is_trajectory_complete:
                cand.is_trajectory_complete = True

            selected_main = self._clip_score(main_scores[sampled_idx])
            selected_heur = self._clip_score(heuristic_scores[sampled_idx])
            selected_combined = self._clip_score(combined_scores[sampled_idx])

            path.steps.append(cand)
            path.main_step_scores.append(selected_main)
            path.heuristic_step_scores.append(selected_heur)
            path.combined_step_scores.append(selected_combined)
            path.sampled_step_probs.append(float(probs[sampled_idx]))
            path.sampled_candidate_indices.append(int(sampled_idx))
            path.candidate_main_scores_history.append([float(x) for x in main_scores])
            path.candidate_heuristic_scores_history.append([float(x) for x in heuristic_scores])
            path.candidate_combined_scores_history.append([float(x) for x in combined_scores])
            path.candidate_probs_history.append([float(x) for x in probs])

            self._maybe_collect_current_sampled_path_as_finished(path, path_idx)

            if cand.is_trajectory_complete or path.depth >= self.max_steps:
                path.completed = True

        return path

    # ---------------------------------------------------------------------
    # Voting / finalization
    # ---------------------------------------------------------------------

    def _finalize_sample_result(
        self,
        sample_idx: int,
        paths: List[_SampledPathState],
        total_generated_tokens: int,
    ) -> Dict[str, Any]:
        if not paths:
            return self._empty_result()

        sampled_path_texts = [convert_trajectory_to_string(p.steps) for p in paths]
        sampled_answers = [self._safe_extract_answer(t) for t in sampled_path_texts]

        sampled_answered_indices = [
            i for i, ans in enumerate(sampled_answers)
            if ans not in ("", "no_answer", None)
        ]
        sampled_voting_indices = (
            sampled_answered_indices if sampled_answered_indices else list(range(len(paths)))
        )

        sampled_voting_answers = [sampled_answers[i] for i in sampled_voting_indices]
        sampled_cluster_ids_local, sampled_representatives, sampled_members_local = (
            self._group_answers(sampled_voting_answers)
        )

        sampled_members = {
            cid: [sampled_voting_indices[j] for j in idxs]
            for cid, idxs in sampled_members_local.items()
        }
        sampled_cluster_ids = [-1] * len(paths)
        for local_j, original_i in enumerate(sampled_voting_indices):
            sampled_cluster_ids[original_i] = sampled_cluster_ids_local[local_j]

        sampled_cluster_counts = {
            cid: len(idxs)
            for cid, idxs in sampled_members.items()
        }

        voting_traces, voting_weights = self._build_weighted_voting_pool(paths)
        if not voting_traces:
            return self._empty_result()

        voting_texts = [convert_trajectory_to_string(t.steps) for t in voting_traces]
        voting_answers = [self._safe_extract_answer(t) for t in voting_texts]

        answered_voting_indices = [
            i for i, ans in enumerate(voting_answers)
            if ans not in ("", "no_answer", None)
        ]
        voting_indices = (
            answered_voting_indices if answered_voting_indices else list(range(len(voting_traces)))
        )

        voting_answers_for_clustering = [voting_answers[i] for i in voting_indices]
        cluster_ids_local, representatives, members_local = self._group_answers(
            voting_answers_for_clustering
        )

        if not members_local:
            return self._empty_result()

        members = {
            cid: [voting_indices[j] for j in idxs]
            for cid, idxs in members_local.items()
        }

        cluster_ids = [-1] * len(voting_traces)
        for local_j, original_i in enumerate(voting_indices):
            cluster_ids[original_i] = cluster_ids_local[local_j]

        cluster_counts = {
            cid: len(idxs)
            for cid, idxs in members.items()
        }
        cluster_weights = {
            cid: float(sum(voting_weights[i] for i in idxs))
            for cid, idxs in members.items()
        }

        def cluster_key(cid: int):
            best_idx = self._select_best_index_within_cluster(members[cid], voting_traces)
            return (
                cluster_weights[cid],
                cluster_counts[cid],
                voting_traces[best_idx].avg_combined_score,
                voting_traces[best_idx].avg_main_score,
                voting_traces[best_idx].avg_heuristic_score,
                -best_idx,
            )

        selected_cluster = max(members.keys(), key=cluster_key)
        selected_idx = self._select_best_index_within_cluster(
            members[selected_cluster],
            voting_traces,
        )

        best_trace = voting_traces[selected_idx]
        best_text = voting_texts[selected_idx]
        best_answer = voting_answers[selected_idx]

        total_voting_weight = float(sum(voting_weights[i] for i in voting_indices))
        consensus_score = (
            cluster_weights[selected_cluster] / total_voting_weight
            if total_voting_weight > 0.0
            else 0.0
        )

        answer_distribution = {
            representatives[cid]: {
                "count": cluster_counts[cid],
                "weight": cluster_weights[cid],
            }
            for cid in members
        }

        per_path_total_voting_weight = {
            i: float(sum(w for t, w in zip(voting_traces, voting_weights) if t.source_path_idx == i))
            for i in range(len(paths))
        }

        all_traces = []
        for i, (path, text, answer) in enumerate(zip(paths, sampled_path_texts, sampled_answers)):
            cid = sampled_cluster_ids[i]
            all_traces.append(
                {
                    "text": text,
                    "num_steps": len(path.steps),
                    "generated_tokens_total": path.generated_tokens_total,
                    "combined_step_scores": list(path.combined_step_scores),
                    "main_step_scores": list(path.main_step_scores),
                    "heuristic_step_scores": list(path.heuristic_step_scores),
                    "sampled_step_probs": list(path.sampled_step_probs),
                    "sampled_candidate_indices": list(path.sampled_candidate_indices),
                    "candidate_main_scores_history": [list(x) for x in path.candidate_main_scores_history],
                    "candidate_heuristic_scores_history": [list(x) for x in path.candidate_heuristic_scores_history],
                    "candidate_combined_scores_history": [list(x) for x in path.candidate_combined_scores_history],
                    "candidate_probs_history": [list(x) for x in path.candidate_probs_history],
                    "avg_combined_score": path.avg_combined_score,
                    "avg_main_score": path.avg_main_score,
                    "avg_heuristic_score": path.avg_heuristic_score,
                    "answer": answer,
                    "answer_cluster_id": cid,
                    "cluster_representative": sampled_representatives[cid] if cid != -1 else None,
                    "cluster_count": sampled_cluster_counts[cid] if cid != -1 else 0,
                    "num_finished_chains_for_voting": len(path.finished_voting_traces),
                    "total_voting_weight_from_this_path": per_path_total_voting_weight[i],
                    "selected": (
                        i == best_trace.source_path_idx
                        and text == best_text
                    ),
                }
            )

        voting_trace_records = []
        for i, (trace, text, answer, weight) in enumerate(
            zip(voting_traces, voting_texts, voting_answers, voting_weights)
        ):
            cid = cluster_ids[i]
            voting_trace_records.append(
                {
                    "text": text,
                    "num_steps": len(trace.steps),
                    "generated_tokens_total": trace.generated_tokens_total,
                    "combined_step_scores": list(trace.combined_step_scores),
                    "main_step_scores": list(trace.main_step_scores),
                    "heuristic_step_scores": list(trace.heuristic_step_scores),
                    "avg_combined_score": trace.avg_combined_score,
                    "avg_main_score": trace.avg_main_score,
                    "avg_heuristic_score": trace.avg_heuristic_score,
                    "answer": answer,
                    "answer_cluster_id": cid,
                    "cluster_representative": representatives[cid] if cid != -1 else None,
                    "cluster_count": cluster_counts[cid] if cid != -1 else 0,
                    "cluster_weight": cluster_weights[cid] if cid != -1 else 0.0,
                    "voting_weight": float(weight),
                    "source_path_idx": int(trace.source_path_idx),
                    "trace_kind": trace.trace_kind,
                    "selected": i == selected_idx,
                }
            )

        avg_num_steps = sum(len(p.steps) for p in paths) / max(len(paths), 1)

        log.info(
            f"[VerifierGuidedSC] sample {sample_idx}: "
            f"best_answer={best_answer}, weighted_majority={consensus_score:.3f}, "
            f"answer_distribution={answer_distribution}"
        )

        return {
            "trajectory": best_text,
            "steps": best_trace.steps,
            "validity_scores": list(best_trace.combined_step_scores),
            "completed": True,
            "strategy": "verifier_guided_self_consistency_stepwise",
            "extracted_answer": best_answer,
            "answer_distribution": answer_distribution,
            "consensus_score": float(consensus_score),
            "all_answers": sampled_answers,
            "voting_answers": voting_answers,
            "all_traces": all_traces,
            "voting_traces": voting_trace_records,
            "total_tokens": int(total_generated_tokens),
            "token_stats": {
                "generated_output_tokens_total": int(total_generated_tokens),
                "selected_path_generated_tokens": int(best_trace.generated_tokens_total),
                "selected_trace_generated_tokens": int(best_trace.generated_tokens_total),
                "avg_generated_tokens_per_path": (
                    float(total_generated_tokens) / max(len(paths), 1)
                ),
            },
            "thinking_num_steps": avg_num_steps,
            "response_num_steps": avg_num_steps,
        }

    def _group_answers(
        self,
        answers: List[str],
    ) -> Tuple[List[int], List[str], Dict[int, List[int]]]:
        cluster_ids: List[int] = [-1] * len(answers)
        representatives: List[str] = []
        members: Dict[int, List[int]] = {}

        for i, ans in enumerate(answers):
            assigned_cluster = None

            if representatives and math_equal_batched is not None:
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
                    assigned_cluster = None

            if assigned_cluster is None:
                stripped_ans = str(ans).strip()
                for cid, rep in enumerate(representatives):
                    if stripped_ans == str(rep).strip():
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
        traces: List[Any],
    ) -> int:
        return max(
            cluster_member_indices,
            key=lambda i: (
                traces[i].avg_combined_score,
                traces[i].avg_main_score,
                traces[i].avg_heuristic_score,
                -i,
            ),
        )

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def _extract_last_score(self, raw: Any) -> float:
        if isinstance(raw, np.ndarray):
            if raw.size == 0:
                return 0.0
            return self._clip_score(float(raw.reshape(-1)[-1]))
        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                return 0.0
            return self._clip_score(float(raw[-1]))
        return self._clip_score(float(raw))

    def _safe_extract_answer(self, text: str) -> str:
        try:
            ans = extract_answer(text)
            if ans is None or ans == "":
                return "no_answer"
            return str(ans)
        except Exception:
            return "no_answer"

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
            "strategy": "verifier_guided_self_consistency_stepwise",
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