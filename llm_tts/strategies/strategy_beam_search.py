"""
Beam Search strategy for LLM reasoning with optional batched generation.

Two modes:
- Sequential (default): Process one sample at a time, good for debugging
- Batched: Process ALL samples in parallel with synchronized beam search steps
  - One vLLM call per step (not per sample×beam)
  - Reduces calls from O(samples × steps × beams) to O(steps)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from vllm import SamplingParams

from llm_tts.generators import (
    StepCandidateGeneratorThroughAPI,
    StepCandidateGeneratorThroughHuggingface,
    convert_trajectory_to_string,
)
from llm_tts.generators.base import StepCandidate
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase

try:
    from evaluation.grader import math_equal_batched
except Exception:
    math_equal_batched = None

log = logging.getLogger(__name__)


class StrategyBeamSearch(StrategyBase):
    """
    Beam Search strategy for LLM reasoning.
    ---------------------------------------

    Keeps a beam of top-N reasoning chains at each step based on a scoring function.
    This balances exploration (multiple reasoning branches) and exploitation
    (keeping only the highest-scoring paths).

    Supports two modes:
    - Sequential: Process samples one at a time (generate_trajectory)
    - Batched: Process ALL samples in parallel (generate_trajectories_batch)
      - One vLLM call per step for all samples × beams
      - Reduces calls from O(samples × steps × beams) to O(steps)

    Args:
        step_generator: Generator for candidate steps (vLLM, API, or HuggingFace).
        scorer: Scoring function for ranking step candidates (PRM, entropy, perplexity).
        beam_size: Number of top beams to keep at each step.
        candidates_per_beam: Number of candidates to generate for each beam per step.
        max_steps: Maximum reasoning steps.
        aggregation: How to aggregate scores across steps ("mean", "sum", "min", "max", "product").
        batch_generation: If True, use batched mode in generate_trajectories_batch.
    """

    def __init__(
            self,
            step_generator: (
                    StepCandidateGeneratorThroughAPI | StepCandidateGeneratorThroughHuggingface
            ),
            scorer: Any,
            beam_size: int = 5,
            candidates_per_beam: int = 3,
            max_steps: int = 10,
            aggregation: str = "mean",
            batch_generation: bool = True,
            self_consistency: bool = False,
            use_math_equal_for_voting: bool = True,
    ):
        self.step_generator = step_generator
        self.scorer = scorer
        self.beam_size = beam_size
        self.candidates_per_beam = candidates_per_beam
        self.max_steps = max_steps
        self.aggregation = aggregation
        self.batch_generation = batch_generation
        self.self_consistency = self_consistency
        self.use_math_equal_for_voting = use_math_equal_for_voting

        # Get stop tokens info for logging
        stop_tokens = getattr(step_generator, "stop_tokens", [])
        eos_token_ids = getattr(step_generator, "eos_token_ids", [151645, 151643])
        min_step_tokens = getattr(step_generator, "min_step_tokens", 50)

        log.info(
            f"StrategyBeamSearch initialized: beam_size={beam_size}, "
            f"candidates_per_beam={candidates_per_beam}, max_steps={max_steps}, "
            f"aggregation={aggregation}, batch_generation={batch_generation}, "
            f"self_consistency={self_consistency}, "
            f"use_math_equal_for_voting={use_math_equal_for_voting}, "
            f"stop_tokens={len(stop_tokens)}, min_step_tokens={min_step_tokens}, "
            f"eos_token_ids={eos_token_ids}"
        )

    def _safe_extract_answer(self, text: str) -> str:
        try:
            ans = extract_answer(text)
            if ans is None or ans == "":
                return "no_answer"
            return str(ans)
        except Exception:
            return "no_answer"

    def _beam_score(self, beam: Dict[str, Any]) -> float:
        return float(self._aggregate_scores(beam.get("scores", [])))

    def _group_answers(
            self,
            answers: List[str],
    ) -> tuple[List[int], List[str], Dict[int, List[int]]]:
        """
        Group answers for self-consistency voting.

        If use_math_equal_for_voting=True and math_equal_batched is available,
        cluster by mathematical equivalence. Otherwise cluster by exact stripped string.
        """
        if (
                self.use_math_equal_for_voting
                and math_equal_batched is not None
                and len(answers) > 0
        ):
            return self._group_answers_by_math_equivalence(answers)

        rep_to_cid: Dict[str, int] = {}
        representatives: List[str] = []
        members: Dict[int, List[int]] = {}
        cluster_ids: List[int] = []

        for i, ans in enumerate(answers):
            key = str(ans).strip()
            if key not in rep_to_cid:
                cid = len(representatives)
                rep_to_cid[key] = cid
                representatives.append(ans)
                members[cid] = []
            cid = rep_to_cid[key]
            members[cid].append(i)
            cluster_ids.append(cid)

        return cluster_ids, representatives, members

    def _group_answers_by_math_equivalence(
            self,
            answers: List[str],
    ) -> tuple[List[int], List[str], Dict[int, List[int]]]:
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
            beams: List[Dict[str, Any]],
    ) -> int:
        return max(
            cluster_member_indices,
            key=lambda i: (
                self._beam_score(beams[i]),
                len(beams[i].get("steps", [])),
                -i,
            ),
        )

    def _select_beam_with_self_consistency(
            self,
            beams: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Select a final beam by self-consistency over completed beams.

        Voting rule:
          1) cluster completed beams by extracted answer
          2) choose the cluster with the largest count
          3) break ties by summed non-negative beam scores
          4) inside the winning cluster, return the best-scoring beam
        """
        if not beams:
            return {"steps": [], "scores": [], "total_tokens": 0}, {}

        trajectory_texts = [
            convert_trajectory_to_string(beam.get("steps", []))
            for beam in beams
        ]
        answers = [self._safe_extract_answer(text) for text in trajectory_texts]

        answered_indices = [
            i for i, ans in enumerate(answers)
            if ans not in ("", "no_answer", None)
        ]
        voting_indices = answered_indices if answered_indices else list(range(len(beams)))
        voting_answers = [answers[i] for i in voting_indices]

        cluster_ids_local, representatives, members_local = self._group_answers(voting_answers)

        if not members_local:
            best_beam = self._select_best_beam(beams)
            return best_beam, {}

        members = {
            cid: [voting_indices[j] for j in idxs]
            for cid, idxs in members_local.items()
        }

        cluster_ids = [-1] * len(beams)
        for local_j, original_i in enumerate(voting_indices):
            cluster_ids[original_i] = cluster_ids_local[local_j]

        cluster_counts = {
            cid: len(idxs)
            for cid, idxs in members.items()
        }

        # Non-negative support as tie-break; keeps majority-vote semantics stable
        cluster_score_sums = {
            cid: float(sum(max(0.0, self._beam_score(beams[i])) for i in idxs))
            for cid, idxs in members.items()
        }

        def cluster_key(cid: int):
            best_idx = self._select_best_index_within_cluster(members[cid], beams)
            return (
                cluster_counts[cid],
                cluster_score_sums[cid],
                self._beam_score(beams[best_idx]),
                -best_idx,
            )

        selected_cluster = max(members.keys(), key=cluster_key)
        selected_idx = self._select_best_index_within_cluster(
            members[selected_cluster],
            beams,
        )
        selected_beam = beams[selected_idx]

        consensus_score = cluster_counts[selected_cluster] / max(len(voting_indices), 1)

        answer_distribution = {
            representatives[cid]: {
                "count": cluster_counts[cid],
                "score_sum": cluster_score_sums[cid],
            }
            for cid in members
        }

        metadata = {
            "selection_method": "self_consistency",
            "consensus_score": float(consensus_score),
            "answer_distribution": answer_distribution,
            "all_answers": answers,
        }

        log.info(
            f"Self-consistency selected answer={answers[selected_idx]!r}, "
            f"consensus={consensus_score:.3f}, "
            f"answer_distribution={answer_distribution}"
        )

        return selected_beam, metadata

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "trajectory": "",
            "steps": [],
            "validity_scores": [],
            "completed": False,
            "extracted_answer": "",
        }

    def _finalize_sample(
            self,
            steps: List[StepCandidate],
            scores: List[float],
            completed: bool = True,
            extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create final result dict for a completed or partially completed sample."""
        trajectory_text = convert_trajectory_to_string(steps)
        extracted = extract_answer(trajectory_text)

        result = {
            "trajectory": trajectory_text,
            "steps": steps,
            "validity_scores": scores,
            "completed": completed,
            "extracted_answer": extracted,
        }

        if extra:
            result.update(extra)

        return result

    def _init_search_trace(self):
        self._last_search_trace = {
            "nodes": {},  # node_id -> metadata
            "edges": [],  # (parent_id, child_id)
            "completed_ids": [],
            "opened_ids": [],  # nodes actually expanded
            "best_final_node_id": None,
            "mode": None,  # "sequential" or "batch"
        }

    def _record_node(
            self,
            node_id: int,
            parent_id: Optional[int],
            steps: List[StepCandidate],
            scores: List[float],
            total_tokens: int = 0,
            sample_id: Optional[int] = None,
    ):
        if not hasattr(self, "_last_search_trace") or self._last_search_trace is None:
            return

        step_text = ""
        is_complete = False
        if steps:
            last_step = steps[-1]
            step_text = getattr(last_step, "text", "") or ""
            is_complete = bool(getattr(last_step, "is_trajectory_complete", False))

        self._last_search_trace["nodes"][node_id] = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": len(steps),
            "total_tokens": total_tokens,
            "f_value": float(self._aggregate_scores(scores)) if scores else 0.0,
            "h_value": None,
            "priority": float(self._aggregate_scores(scores)) if scores else 0.0,
            "scores": list(scores),
            "step_text": step_text,
            "is_complete": is_complete,
            "sample_id": sample_id,
        }

        if parent_id is not None:
            self._last_search_trace["edges"].append((parent_id, node_id))

    def _record_opened_node(self, node_id: int):
        if not hasattr(self, "_last_search_trace") or self._last_search_trace is None:
            return
        self._last_search_trace["opened_ids"].append(node_id)

    def _finalize_search_trace(
            self,
            completed_ids: List[int],
            best_final_node_id: Optional[int],
            mode: str,
    ):
        if not hasattr(self, "_last_search_trace") or self._last_search_trace is None:
            return

        self._last_search_trace["completed_ids"] = list(dict.fromkeys(completed_ids))
        self._last_search_trace["opened_ids"] = list(dict.fromkeys(self._last_search_trace["opened_ids"]))
        self._last_search_trace["best_final_node_id"] = best_final_node_id
        self._last_search_trace["mode"] = mode

    def get_last_search_trace(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "_last_search_trace", None)

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate a reasoning trajectory using beam search.

        Args:
            request: Input chat or prompt context.
            sample_idx: Index of current sample (for logging).

        Returns:
            Dictionary with trajectory, steps, score, and metadata.
        """

        self._init_search_trace()

        node_id_ctr = 0
        completed_node_ids = []

        beams = [{"steps": [], "scores": [], "node_id": node_id_ctr, "total_tokens": 0}]
        self._record_node(
            node_id=node_id_ctr,
            parent_id=None,
            steps=[],
            scores=[],
            total_tokens=0,
            sample_id=sample_idx,
        )
        node_id_ctr += 1

        completed_beams = []

        for step in range(self.max_steps):
            log.info(f"\n=== Beam Search Step {step} ===")
            new_beams = []

            # Expand each current beam
            for beam_idx, beam in enumerate(beams):
                self._record_opened_node(beam["node_id"])
                if beam_idx == 0:
                    log.info(
                        f"Expanding beam {beam_idx} with score "
                        f"{self._aggregate_scores(beam['scores']):.3f}"
                    )

                candidates = self.step_generator(
                    request,
                    trajectory=beam["steps"],
                    candidates_per_step=self.candidates_per_beam,
                )

                if not candidates:
                    log.info(f"  No candidates for beam {beam_idx}, skipping")
                    continue

                # Pass trajectory context so PRM can score candidate in context of previous steps
                candidate_scores = self.scorer.score_candidates(
                    request, candidates, trajectory=beam["steps"]
                )

                # Expand with new candidates
                for cand, score in zip(candidates, candidate_scores):
                    child_scores = beam["scores"] + [score]
                    child_steps = beam["steps"] + [cand]
                    child_total_tokens = beam.get("total_tokens", 0) + len(getattr(cand, "token_ids", []) or [])

                    child_node_id = node_id_ctr
                    node_id_ctr += 1

                    new_beams.append({
                        "steps": child_steps,
                        "scores": child_scores,
                        "node_id": child_node_id,
                        "total_tokens": child_total_tokens,
                    })

                    self._record_node(
                        node_id=child_node_id,
                        parent_id=beam["node_id"],
                        steps=child_steps,
                        scores=child_scores,
                        total_tokens=child_total_tokens,
                        sample_id=sample_idx,
                    )

                    log.info(
                        f"    Candidate: score={score:.3f}, aggregated score={self._aggregate_scores(child_scores):.3f}, text='{cand.text[:80]}'"
                    )

            if not new_beams:
                log.info("No new beams generated, stopping early.")
                break

            # Sort and prune to top-k beams
            new_beams.sort(
                key=lambda b: self._aggregate_scores(b["scores"]),
                reverse=True,
            )
            beams = new_beams[: self.beam_size]
            log.info(f"Kept top {len(beams)} beams for next step")

            # Separate completed beams
            done, active = self._split_completed(beams)
            completed_beams.extend(done)
            beams = active
            completed_node_ids.extend([b["node_id"] for b in done])

            # Stop if all beams are completed
            if not beams:
                log.info("All beams completed early.")
                break

        # Choose final beam:
        # - default behavior unchanged
        # - optional self-consistency over COMPLETED beams only
        sc_metadata: Dict[str, Any] = {}

        if self.self_consistency and completed_beams:
            best_beam, sc_metadata = self._select_beam_with_self_consistency(completed_beams)
        else:
            best_beam = self._select_best_beam(completed_beams or beams)

        best_final_node_id = best_beam.get("node_id")
        self._finalize_search_trace(
            completed_ids=completed_node_ids,
            best_final_node_id=best_final_node_id,
            mode="sequential",
        )

        return self._finalize_sample(
            steps=best_beam["steps"],
            scores=best_beam["scores"],
            completed=bool(completed_beams),
            extra=sc_metadata,
        )

    def _score_step_candidates(
            self,
            request: List[Dict[str, str]],
            trajectory: List[StepCandidate],
            candidates: List[StepCandidate],
    ) -> List[float]:
        """Score candidates exactly the same way in all modes."""
        if not candidates:
            return []
        return self.scorer.score_candidates(request, candidates, trajectory=trajectory)

    def _make_child_beam(
            self,
            parent_beam: Dict[str, Any],
            cand: StepCandidate,
            score: float,
    ) -> Dict[str, Any]:
        """Create a child beam from a parent beam and one candidate."""
        return {
            "steps": parent_beam["steps"] + [cand],
            "scores": parent_beam["scores"] + [float(score)],
            "total_tokens": parent_beam.get("total_tokens", 0)
                            + len(getattr(cand, "token_ids", []) or []),
        }

    def generate_trajectories_batch(
            self,
            requests: List[List[Dict[str, str]]],
            sample_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate trajectories for ALL samples in parallel using batched beam search.

        This implementation is intentionally aligned with generate_trajectory():

        Sequential logic per step:
          1) expand each active beam
          2) score candidates in context of the parent trajectory
          3) create child beams
          4) sort all child beams by aggregated score
          5) keep top beam_size
          6) split completed vs active
          7) repeat

        Batched version keeps the same semantics, but batches:
          - step generation via step_generator.generate_step_candidates_batched_requests(...)
          - PRM scoring via scorer.score_trajectories_batch(...) when available
        """

        total_generated_tokens = 0
        num_generate_calls = 0

        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        M = len(requests)
        log.info(
            f"Batched Beam Search: {M} samples, beam_size={self.beam_size}, "
            f"candidates_per_beam={self.candidates_per_beam}, max_steps={self.max_steps}"
        )

        # Active beams only, exactly like sequential mode conceptually.
        sample_beams: Dict[int, List[Dict[str, Any]]] = {
            i: [{"steps": [], "scores": [], "total_tokens": 0}] for i in range(M)
        }

        # Completed beams are stored separately so we can finalize with the best one.
        completed_beams_by_sample: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(M)
        }

        completed_results: Dict[int, Dict[str, Any]] = {}
        active_samples = set(range(M))

        # Keep this only if you want the extra guard in batch mode.
        # Note: sequential mode does not currently enforce this.
        max_trajectory_tokens = 3000

        for step_num in range(self.max_steps):
            if not active_samples:
                log.info(f"All samples completed at step {step_num}")
                break

            log.info(
                f"\n{'=' * 60}\n"
                f"Beam Search Step {step_num}: {len(active_samples)} active samples\n"
                f"{'=' * 60}"
            )

            # Build the flat batch of (sample, beam) pairs to expand this step.
            prompt_metadata: List[tuple[int, int, Dict[str, Any]]] = []
            batch_requests: List[List[Dict[str, str]]] = []
            batch_trajectories: List[List[StepCandidate]] = []

            # Beams skipped due to context limit (optional behavior retained from old batch code)
            skipped_beams_by_sample: Dict[int, List[Dict[str, Any]]] = {
                i: [] for i in active_samples
            }

            for sample_id in active_samples:
                for beam_idx, beam in enumerate(sample_beams[sample_id]):
                    # Match sequential behavior: only active beams are expanded.
                    if beam["steps"] and beam["steps"][-1].is_trajectory_complete:
                        continue

                    beam_tokens = beam.get("total_tokens", 0)
                    if beam_tokens >= max_trajectory_tokens - 200:
                        skipped_beam = {
                            "steps": list(beam["steps"]),
                            "scores": list(beam["scores"]),
                            "total_tokens": beam["total_tokens"],
                        }
                        if skipped_beam["steps"]:
                            skipped_beam["steps"][-1].is_trajectory_complete = True
                        skipped_beams_by_sample[sample_id].append(skipped_beam)
                        log.info(
                            f"Sample {sample_id}: Skipping beam {beam_idx} "
                            f"(tokens: {beam_tokens} >= limit)"
                        )
                        continue

                    prompt_metadata.append((sample_id, beam_idx, beam))
                    batch_requests.append(requests[sample_id])
                    batch_trajectories.append(beam["steps"])

            if not prompt_metadata and not any(skipped_beams_by_sample.values()):
                log.info("No prompts to process, stopping")
                break

            log.info(
                f"Generating candidates for {len(prompt_metadata)} (sample, beam) pairs"
            )

            # 1) Generate step candidates using the SAME extractor logic as sequential mode.
            if prompt_metadata:
                num_generate_calls += 1
                batched_candidates: List[List[StepCandidate]] = (
                    self.step_generator.generate_step_candidates_batched_requests(
                        requests=batch_requests,
                        trajectories=batch_trajectories,
                        candidates_per_step=self.candidates_per_beam,
                        compute_uncertainty=True,
                    )
                )

                # Count every token actually generated across all candidates in this call
                total_generated_tokens += sum(
                    len(getattr(cand, "token_ids", []) or [])
                    for candidate_list in batched_candidates
                    for cand in candidate_list
                )
            else:
                batched_candidates = []

            # 2) Score candidates with the same semantics as sequential mode.
            #    Prefer batched trajectory scoring when available.
            batched_scores = self._score_candidates_batched_if_possible(
                requests=requests,
                prompt_metadata=prompt_metadata,
                batched_candidates=batched_candidates,
            )

            # 3) Build new child beams per sample.
            new_sample_beams: Dict[int, List[Dict[str, Any]]] = {
                i: [] for i in active_samples
            }

            for (sample_id, beam_idx, parent_beam), candidates, candidate_scores in zip(
                    prompt_metadata, batched_candidates, batched_scores
            ):
                if not candidates:
                    log.info(f"Sample {sample_id}, beam {beam_idx}: no candidates")
                    continue

                for cand, score in zip(candidates, candidate_scores):
                    child_beam = self._make_child_beam(parent_beam, cand, score)

                    # Optional context-limit completion retained from previous batch logic.
                    if (
                            child_beam["total_tokens"] >= max_trajectory_tokens
                            and child_beam["steps"]
                            and not child_beam["steps"][-1].is_trajectory_complete
                    ):
                        child_beam["steps"][-1].is_trajectory_complete = True
                        log.info(
                            f"Sample {sample_id}: Trajectory marked complete "
                            f"(tokens: {child_beam['total_tokens']} >= {max_trajectory_tokens})"
                        )

                    new_sample_beams[sample_id].append(child_beam)

                    if sample_id == 0:
                        log.info(
                            f"Sample {sample_id}, beam {beam_idx}: "
                            f"candidate score={score:.3f}, "
                            f"aggregated={self._aggregate_scores(child_beam['scores']):.3f}, "
                            f"text='{cand.text[:80]}'"
                        )

            # 4) Add skipped beams back as completed beams for this step.
            for sample_id in active_samples:
                new_sample_beams[sample_id].extend(skipped_beams_by_sample[sample_id])

            # 5) Per sample, apply the SAME prune order as generate_trajectory:
            #    sort -> top beam_size -> split completed/active
            samples_to_remove = []

            for sample_id in active_samples:
                beams = new_sample_beams[sample_id]

                if not beams:
                    log.warning(
                        f"Sample {sample_indices[sample_id]}: No beams generated"
                    )
                    completed_results[sample_id] = self._empty_result()
                    samples_to_remove.append(sample_id)
                    continue

                beams.sort(
                    key=lambda b: self._aggregate_scores(b["scores"]),
                    reverse=True,
                )

                # IMPORTANT: prune BEFORE splitting completed vs active,
                # matching generate_trajectory().
                beams = beams[: self.beam_size]

                completed, active = self._split_completed(beams)
                completed_beams_by_sample[sample_id].extend(completed)
                sample_beams[sample_id] = active

                if not active:
                    sc_metadata: Dict[str, Any] = {}

                    if self.self_consistency and completed_beams_by_sample[sample_id]:
                        best_beam, sc_metadata = self._select_beam_with_self_consistency(
                            completed_beams_by_sample[sample_id]
                        )
                    else:
                        best_beam = self._select_best_beam(completed_beams_by_sample[sample_id])

                    completed_results[sample_id] = self._finalize_sample(
                        steps=best_beam["steps"],
                        scores=best_beam["scores"],
                        completed=bool(completed_beams_by_sample[sample_id]),
                        extra=sc_metadata,
                    )
                    samples_to_remove.append(sample_id)

                    scores_str = ", ".join(f"{s:.3f}" for s in best_beam["scores"])
                    log.info(
                        f"Sample {sample_indices[sample_id]}: Completed with "
                        f"{len(best_beam['steps'])} steps, "
                        f"score={self._aggregate_scores(best_beam['scores']):.3f}, "
                        f"scores=[{scores_str}]"
                    )

            for sample_id in samples_to_remove:
                active_samples.discard(sample_id)

            log.info(
                f"Step {step_num} complete: {len(active_samples)} samples still active, "
                f"{len(completed_results)} completed"
            )

        # Finalize any remaining active samples, preferring completed beams if any exist.
        for sample_id in active_samples:
            active = sample_beams[sample_id]
            completed_beams = completed_beams_by_sample[sample_id]
            sc_metadata: Dict[str, Any] = {}

            if self.self_consistency and completed_beams:
                best_beam, sc_metadata = self._select_beam_with_self_consistency(
                    completed_beams
                )
            else:
                candidates = completed_beams or active
                best_beam = self._select_best_beam(candidates)

            completed_results[sample_id] = self._finalize_sample(
                steps=best_beam["steps"],
                scores=best_beam["scores"],
                completed=bool(completed_beams),
                extra=sc_metadata,
            )

            scores_str = ", ".join(f"{s:.3f}" for s in best_beam["scores"])
            log.info(
                f"Sample {sample_indices[sample_id]}: Reached max_steps with "
                f"{len(best_beam['steps'])} steps, "
                f"score={self._aggregate_scores(best_beam['scores']):.3f}, "
                f"scores=[{scores_str}]"
            )

        avg_generated_tokens_per_sample = total_generated_tokens / M if M > 0 else 0.0

        log.info(
            f"[BFBS] TOTAL generated tokens: {total_generated_tokens} | "
            f"avg per sample: {avg_generated_tokens_per_sample:.2f} | "
            f"num generate calls: {num_generate_calls}"
        )

        return [completed_results[i] for i in range(M)]

    def _score_candidates_batched_if_possible(
            self,
            requests: List[List[Dict[str, str]]],
            prompt_metadata: List[tuple[int, int, Dict[str, Any]]],
            batched_candidates: List[List[StepCandidate]],
    ) -> List[List[float]]:
        """
        Score candidates for each (sample, beam) pair.

        Preferred path:
          - if scorer supports score_trajectories_batch(...), batch score all
            full trajectories and take the last-step score for each candidate

        Fallback:
          - use scorer.score_candidates(request, candidates, trajectory=...)
            exactly like generate_trajectory()
        """
        if not batched_candidates:
            return []

        can_batch_score = hasattr(self.scorer, "score_trajectories_batch")

        if can_batch_score:
            flat_chats: List[List[Dict[str, str]]] = []
            full_trajectories: List[List[StepCandidate]] = []
            candidate_map: List[tuple[int, int]] = []
            sample_ids_for_scoring: List[int] = []
            trajectory_ids_for_scoring: List[int] = []

            trajectory_counter = 0

            for prompt_idx, ((sample_id, _beam_idx, parent_beam), candidates) in enumerate(
                    zip(prompt_metadata, batched_candidates)
            ):
                for cand_idx, cand in enumerate(candidates):
                    full_traj = parent_beam["steps"] + [cand]

                    flat_chats.append(requests[sample_id])
                    full_trajectories.append(full_traj)
                    candidate_map.append((prompt_idx, cand_idx))
                    sample_ids_for_scoring.append(sample_id)
                    trajectory_ids_for_scoring.append(trajectory_counter)

                    trajectory_counter += 1

            if not full_trajectories:
                return [[] for _ in batched_candidates]

            log.info(f"Batch scoring {len(full_trajectories)} candidate trajectories")

            all_scores = self.scorer.score_trajectories_batch(
                flat_chats,
                full_trajectories,
                sample_ids=sample_ids_for_scoring,
                trajectory_ids=trajectory_ids_for_scoring,
            )

            scores_by_prompt: List[List[Optional[float]]] = [
                [None] * len(cands) for cands in batched_candidates
            ]

            for (prompt_idx, cand_idx), traj_scores in zip(candidate_map, all_scores):
                score = traj_scores[-1] if traj_scores else 0.0
                scores_by_prompt[prompt_idx][cand_idx] = float(score)

            return [
                [0.0 if s is None else float(s) for s in prompt_scores]
                for prompt_scores in scores_by_prompt
            ]

        # Fallback to exact sequential-style candidate scoring.
        all_candidate_scores: List[List[float]] = []

        for (sample_id, _beam_idx, parent_beam), candidates in zip(
                prompt_metadata, batched_candidates
        ):
            if not candidates:
                all_candidate_scores.append([])
                continue

            scores = self.scorer.score_candidates(
                requests[sample_id],
                candidates,
                trajectory=parent_beam["steps"],
            )
            all_candidate_scores.append([float(s) for s in scores])

        return all_candidate_scores

    def _build_prompt_with_history(
        self,
        request: List[Dict[str, str]],
        trajectory_text: str,
        max_prompt_tokens: int = 3500,
    ) -> str:
        """Build a prompt string with chat template and trajectory history.

        Args:
            request: Chat messages
            trajectory_text: Previous steps as text
            max_prompt_tokens: Maximum tokens for prompt (leave room for generation)
        """
        # Apply chat template to get base prompt
        if getattr(self.step_generator, "disable_thinking_mode", False):
            prompt = self.step_generator._apply_chat_template(
                request, enable_thinking=False
            )
        else:
            prompt = self.step_generator.tokenizer.apply_chat_template(
                request, tokenize=False, add_generation_prompt=True
            )

        # Append trajectory history if any
        if trajectory_text:
            full_prompt = prompt + trajectory_text

            # Check if prompt is too long and truncate trajectory if needed
            tokenizer = self.step_generator.tokenizer
            prompt_tokens = len(tokenizer.encode(full_prompt))

            if prompt_tokens > max_prompt_tokens:
                # Truncate trajectory from the beginning (keep recent steps)
                base_tokens = len(tokenizer.encode(prompt))
                available_for_traj = max_prompt_tokens - base_tokens - 100  # Buffer

                if available_for_traj > 0:
                    # Truncate trajectory text to fit
                    traj_tokens = tokenizer.encode(trajectory_text)
                    if len(traj_tokens) > available_for_traj:
                        # Keep last N tokens of trajectory
                        truncated_tokens = traj_tokens[-available_for_traj:]
                        trajectory_text = tokenizer.decode(truncated_tokens)
                        log.warning(
                            f"Truncated trajectory from {len(traj_tokens)} to "
                            f"{len(truncated_tokens)} tokens to fit context"
                        )

                full_prompt = prompt + trajectory_text

            return full_prompt

        return prompt

    def _batch_score_with_prm(
        self,
        requests: List[List[Dict[str, str]]],
        all_candidates_data: List[List[Dict]],
        prompt_metadata: List[tuple],
    ) -> List[List[Dict]]:
        """
        Batch score all candidates using PRM scorer.

        Args:
            requests: Original requests for each sample
            all_candidates_data: List of candidate lists (one per prompt)
            prompt_metadata: List of (sample_id, beam_idx, parent_beam) tuples

        Returns:
            all_candidates_data with 'prm_score' added to each candidate
        """
        # Build full trajectories (parent steps + new candidate) for batch scoring
        full_trajectories = []
        flat_chats = []
        candidate_map = []  # (prompt_idx, cand_idx)
        sample_ids_for_scoring = []  # Track sample IDs for logging
        traj_ids_for_scoring = []  # Track trajectory IDs within each sample

        for prompt_idx, candidates in enumerate(all_candidates_data):
            sample_id, beam_idx, parent_beam = prompt_metadata[prompt_idx]
            request = requests[sample_id]

            for cand_idx, cand_data in enumerate(candidates):
                # Create StepCandidate for the new candidate
                new_step = StepCandidate(
                    text=cand_data["text"],
                    token_ids=cand_data.get("token_ids", []),
                    is_complete=True,
                    is_trajectory_complete=cand_data.get("is_complete", False),
                )
                # Build full trajectory: parent steps + new step
                full_traj = parent_beam["steps"] + [new_step]
                full_trajectories.append(full_traj)
                flat_chats.append(request)
                candidate_map.append((prompt_idx, cand_idx))
                sample_ids_for_scoring.append(sample_id)
                traj_ids_for_scoring.append(len(candidate_map) - 1)

        if not full_trajectories:
            return all_candidates_data

        log.info(f"Batch PRM scoring {len(full_trajectories)} candidates")

        # Score all trajectories in batch using score_trajectories_batch
        if hasattr(self.scorer, "score_trajectories_batch"):
            # Returns list of score lists (one per trajectory)
            all_scores = self.scorer.score_trajectories_batch(
                flat_chats,
                full_trajectories,
                sample_ids=sample_ids_for_scoring,
                trajectory_ids=traj_ids_for_scoring,
            )
            # Extract the last step score (the new candidate's score)
            scores = []
            for traj_scores in all_scores:
                if traj_scores:
                    scores.append(traj_scores[-1])  # Last step is the new candidate
                else:
                    scores.append(0.0)
        else:
            # Fallback: score one by one (less efficient)
            scores = []
            for chat, traj in zip(flat_chats, full_trajectories):
                score_list = self.scorer.score_trajectory(chat, traj)
                if score_list:
                    scores.append(score_list[-1])  # Last step score
                else:
                    scores.append(0.0)

        # Map scores back to candidates
        for (prompt_idx, cand_idx), score in zip(candidate_map, scores):
            all_candidates_data[prompt_idx][cand_idx]["prm_score"] = score

        return all_candidates_data

    def _aggregate_scores(self, scores: list[float]) -> float:
        """Aggregate scores across steps according to selected strategy."""
        if len(scores) == 0:
            return 0
        if self.aggregation == "sum": # sum of uncertainty
            return 1 - sum([1 - s for s in scores])
        elif self.aggregation == "mean":
            return np.mean(scores).item()
        elif self.aggregation == "product": # product of uncertainties
            return 1 - np.prod([1 - s for s in scores]).item()
        elif self.aggregation == "product_correctness":
            return np.prod(scores).item()
        elif self.aggregation == "max": # max of uncertainty = min of correctness
            return np.min(scores).item()
        elif self.aggregation == "min": # min of uncertainty = max of correctness
            return np.max(scores).item()
        elif self.aggregation == "last":
            return float(scores[-1])
        else:
            raise Exception(f"Unknown aggregation {self.aggregation}")

    def _split_completed(self, beams: List[Dict]) -> tuple:
        """Split beams into completed and active."""
        completed = []
        active = []
        for b in beams:
            if b["steps"] and b["steps"][-1].is_trajectory_complete:
                completed.append(b)
            else:
                active.append(b)
        return completed, active

    def _select_best_beam(self, beams: List[Dict]) -> Dict:
        """Select the highest scoring beam."""
        if not beams:
            return {"steps": [], "scores": [], "total_tokens": 0}
        return max(beams, key=lambda b: self._aggregate_scores(b["scores"]))

    def cleanup(self):
        """Clean up scorer resources if necessary."""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
