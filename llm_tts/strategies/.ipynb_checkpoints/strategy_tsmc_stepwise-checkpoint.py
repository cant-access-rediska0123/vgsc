import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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

STRATEGY_NAME = "tsmc_stepwise"
NO_ANSWER = "no_answer"
TOKEN_MARGIN = 200


@dataclass
class _ParticleState:
    steps: List[StepCandidate] = field(default_factory=list)
    generated_tokens_total: int = 0
    completed: bool = False

    # Prefix values V_theta(x_{1:t}) after each generated step.
    value_scores: List[float] = field(default_factory=list)

    # Incremental TSMC weights used for the current step. For non-terminal steps,
    # these are the resampling weights. For terminal steps, the last entry equals
    # the terminal voting weight.
    incremental_weights: List[float] = field(default_factory=list)

    # Final voting weight used at answer aggregation time.
    terminal_weight: float = 0.0

    # Optional diagnostics.
    resample_history: List[int] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def avg_value(self) -> float:
        return float(np.mean(self.value_scores).item()) if self.value_scores else 0.0

    @property
    def last_value(self) -> float:
        return float(self.value_scores[-1]) if self.value_scores else 1.0

    @property
    def last_incremental_weight(self) -> float:
        return float(self.incremental_weights[-1]) if self.incremental_weights else 1.0

    def clone(self) -> "_ParticleState":
        return _ParticleState(
            steps=deepcopy(self.steps),
            generated_tokens_total=int(self.generated_tokens_total),
            completed=bool(self.completed),
            value_scores=list(self.value_scores),
            incremental_weights=list(self.incremental_weights),
            terminal_weight=float(self.terminal_weight),
            resample_history=list(self.resample_history),
        )


class StrategyTSMCStepwise(StrategyBase):
    """
    Twisted Sequential Monte Carlo (TSMC) baseline adapted to the existing
    stepwise generator / scorer interfaces.

    Core paper-faithful behavior:
    - proposal q is the base generator p
    - exactly one new step is sampled per particle at each round
    - non-terminal incremental weights use sqrt(V_t / V_{t-1})
    - terminal weights use V_T / sqrt(V_{T-1})
    - resampling happens after each non-terminal round
    - final answer is chosen by weighted voting over terminal particles

    Practical adaptations for variable-length reasoning traces:
    - particles can terminate early via StepCandidate.is_trajectory_complete
    - completed particles are carried forward and are NOT resampled again
    - only unfinished particles are resampled within each sample
    - if generation stops before a new terminal step can be produced (e.g.
      token / step budget reached), we finalize the particle using its most
      recent incremental weight as an engineering fallback

    Parameters
    ----------
    step_generator:
        Step-wise generator. Must support the same APIs as the existing
        stepwise strategies.
    scorer:
        Value function estimator V_theta. This should approximate the expected
        future correctness of the current prefix under the base generator.
    num_particles:
        Number of particles (paths) per sample.
    max_steps:
        Maximum number of reasoning steps per particle.
    batch_generation:
        If True, generate one step for all active particles in a single batched
        call. Otherwise fall back to a sequential single-sample loop.
    max_trajectory_tokens:
        Per-particle output token budget.
    seed:
        Random seed for resampling.
    epsilon:
        Numerical floor used in ratio computations.
    initial_value:
        Value used for the empty prefix. The paper sets psi_0 = 1; using an
        initial value of 1.0 yields the expected first-step ratio.
    allow_score_candidates:
        If True and the scorer exposes score_candidates(...), use it for the
        current step. Otherwise score the full updated prefixes.
    """

    def __init__(
        self,
        step_generator: (
            "StepCandidateGeneratorThroughAPI | StepCandidateGeneratorThroughHuggingface"
        ),
        scorer: Optional[Any],
        scorer_heuristic: Optional[Any] = None,
        num_particles: Optional[int] = None,
        num_paths: Optional[int] = None,
        max_steps: int = 20,
        batch_generation: bool = True,
        max_trajectory_tokens: int = 30000,
        seed: Optional[int] = None,
        epsilon: float = 1e-8,
        initial_value: float = 1.0,
        allow_score_candidates: bool = True,
    ):
        self.step_generator = step_generator
        self.scorer = scorer_heuristic if scorer_heuristic is not None else scorer

        resolved_num_particles = num_particles
        if resolved_num_particles is None:
            resolved_num_particles = num_paths if num_paths is not None else 10
        self.num_particles = int(resolved_num_particles)
        self.max_steps = int(max_steps)
        self.batch_generation = bool(batch_generation)
        self.max_trajectory_tokens = int(max_trajectory_tokens)
        self.rng = np.random.default_rng(seed)
        self.epsilon = float(max(epsilon, 1e-12))
        self.initial_value = float(max(initial_value, self.epsilon))
        self.allow_score_candidates = bool(allow_score_candidates)

        log.info(
            "StrategyTSMCStepwise initialized: "
            f"num_particles={self.num_particles}, max_steps={max_steps}, "
            f"mode={'batched' if batch_generation else 'sequential'}, "
            f"max_trajectory_tokens={max_trajectory_tokens}, seed={seed}, "
            f"epsilon={self.epsilon}, scorer={type(self.scorer).__name__ if self.scorer is not None else 'None'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        particles, total_generated_tokens = self._run_single_sample_sequential(
            request=request,
            sample_idx=sample_idx,
        )
        return self._finalize_sample_result(
            sample_idx=sample_idx,
            particles=particles,
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

        particles_per_sample: Dict[int, List[_ParticleState]] = {
            sid: [_ParticleState() for _ in range(self.num_particles)]
            for sid in range(num_samples)
        }
        generated_tokens_per_sample: Dict[int, int] = {sid: 0 for sid in range(num_samples)}

        round_idx = 0
        while True:
            round_idx += 1

            # Finalize particles that cannot be extended anymore before entering
            # the next generation round.
            for sid in range(num_samples):
                for particle in particles_per_sample[sid]:
                    self._finalize_if_cannot_continue(particle)

            active_items: List[Tuple[int, int, _ParticleState]] = []
            batch_requests: List[List[Dict[str, str]]] = []
            batch_trajectories: List[List[StepCandidate]] = []

            for sid in range(num_samples):
                for pid, particle in enumerate(particles_per_sample[sid]):
                    if particle.completed:
                        continue
                    active_items.append((sid, pid, particle))
                    batch_requests.append(requests[sid])
                    batch_trajectories.append(particle.steps)

            if not active_items:
                break

            log.info(
                f"[TSMC] round {round_idx}: expanding {len(active_items)} active particles"
            )

            all_candidates = self.step_generator.generate_step_candidates_batched_requests(
                batch_requests,
                trajectories=batch_trajectories,
                candidates_per_step=1,
            )

            if len(all_candidates) != len(active_items):
                raise ValueError(
                    "Batched step generator returned unexpected number of candidate lists: "
                    f"got {len(all_candidates)}, expected {len(active_items)}"
                )

            updated_items: List[
                Tuple[int, int, _ParticleState, List[StepCandidate], Optional[StepCandidate]]
            ] = []

            for (sid, pid, particle), cand_list in zip(active_items, all_candidates):
                prev_steps = list(particle.steps)
                cand = cand_list[0] if cand_list else None
                tok_count = self._apply_candidate_to_particle(particle, cand)
                generated_tokens_per_sample[sid] += tok_count
                updated_items.append((sid, pid, particle, prev_steps, cand))

            current_values = self._score_updated_particles_batched(
                requests=requests,
                sample_indices=sample_indices,
                updated_items=updated_items,
            )

            if len(current_values) != len(updated_items):
                raise ValueError(
                    f"Value scorer returned {len(current_values)} scores for {len(updated_items)} particles"
                )

            for (sid, _pid, particle, _prev_steps, _cand), current_value in zip(
                updated_items,
                current_values,
            ):
                prev_value = self._previous_prefix_value(particle)
                current_value = self._clip_value(current_value)
                particle.value_scores.append(current_value)

                if particle.completed:
                    weight = self._terminal_weight(prev_value=prev_value, current_value=current_value)
                    particle.terminal_weight = float(weight)
                    particle.incremental_weights.append(float(weight))
                else:
                    weight = self._nonterminal_weight(prev_value=prev_value, current_value=current_value)
                    particle.incremental_weights.append(float(weight))

            for sid in range(num_samples):
                particles_per_sample[sid] = self._resample_unfinished_particles(
                    particles_per_sample[sid]
                )

        return [
            self._finalize_sample_result(
                sample_idx=sample_indices[sid],
                particles=self._finalize_remaining_particles(particles_per_sample[sid]),
                total_generated_tokens=generated_tokens_per_sample[sid],
            )
            for sid in range(num_samples)
        ]

    def cleanup(self):
        if self.scorer is not None and hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()

    # ------------------------------------------------------------------
    # Sequential single-sample helper
    # ------------------------------------------------------------------

    def _run_single_sample_sequential(
        self,
        request: List[Dict[str, str]],
        sample_idx: int,
    ) -> Tuple[List[_ParticleState], int]:
        particles = [_ParticleState() for _ in range(self.num_particles)]
        total_generated_tokens = 0

        round_idx = 0
        while True:
            round_idx += 1
            for particle in particles:
                self._finalize_if_cannot_continue(particle)

            active_particles = [p for p in particles if not p.completed]
            if not active_particles:
                break

            updated_particles: List[Tuple[_ParticleState, List[StepCandidate], Optional[StepCandidate]]] = []
            current_values: List[float] = []

            for particle in active_particles:
                prev_steps = list(particle.steps)
                cand_list = self.step_generator(
                    request,
                    trajectory=particle.steps,
                    candidates_per_step=1,
                )
                cand = cand_list[0] if cand_list else None
                total_generated_tokens += self._apply_candidate_to_particle(particle, cand)
                updated_particles.append((particle, prev_steps, cand))

            current_values = self._score_updated_particles_sequential(
                request=request,
                sample_idx=sample_idx,
                updated_items=updated_particles,
            )

            for (particle, _prev_steps, _cand), current_value in zip(updated_particles, current_values):
                prev_value = self._previous_prefix_value(particle)
                current_value = self._clip_value(current_value)
                particle.value_scores.append(current_value)

                if particle.completed:
                    weight = self._terminal_weight(prev_value=prev_value, current_value=current_value)
                    particle.terminal_weight = float(weight)
                    particle.incremental_weights.append(float(weight))
                else:
                    weight = self._nonterminal_weight(prev_value=prev_value, current_value=current_value)
                    particle.incremental_weights.append(float(weight))

            particles = self._resample_unfinished_particles(particles)

        return self._finalize_remaining_particles(particles), total_generated_tokens

    # ------------------------------------------------------------------
    # Particle / generation helpers
    # ------------------------------------------------------------------

    def _finalize_if_cannot_continue(self, particle: _ParticleState) -> None:
        if particle.completed:
            return

        if particle.depth >= self.max_steps:
            if particle.steps:
                particle.steps[-1].is_trajectory_complete = True
            particle.completed = True
            if particle.terminal_weight <= 0.0:
                particle.terminal_weight = float(max(particle.last_incremental_weight, self.epsilon))
            return

        if particle.generated_tokens_total >= self.max_trajectory_tokens - TOKEN_MARGIN:
            if particle.steps:
                particle.steps[-1].is_trajectory_complete = True
            particle.completed = True
            if particle.terminal_weight <= 0.0:
                particle.terminal_weight = float(max(particle.last_incremental_weight, self.epsilon))

    def _apply_candidate_to_particle(
        self,
        particle: _ParticleState,
        cand: Optional[StepCandidate],
    ) -> int:
        if cand is None:
            particle.completed = True
            if particle.terminal_weight <= 0.0:
                particle.terminal_weight = float(max(particle.last_incremental_weight, self.epsilon))
            return 0

        tok_ids = getattr(cand, "token_ids", None) or []
        tok_count = len(tok_ids)
        particle.generated_tokens_total += tok_count

        if (
            particle.generated_tokens_total >= self.max_trajectory_tokens
            and not getattr(cand, "is_trajectory_complete", False)
        ):
            cand.is_trajectory_complete = True

        particle.steps.append(cand)

        if getattr(cand, "is_trajectory_complete", False) or particle.depth >= self.max_steps:
            particle.completed = True

        return tok_count

    def _finalize_remaining_particles(
        self,
        particles: List[_ParticleState],
    ) -> List[_ParticleState]:
        for particle in particles:
            if not particle.completed:
                particle.completed = True
                if particle.steps:
                    particle.steps[-1].is_trajectory_complete = True
            if particle.terminal_weight <= 0.0:
                # Engineering fallback for particles that stop before an explicit
                # terminal step can be scored.
                particle.terminal_weight = float(max(particle.last_incremental_weight, self.epsilon))
        return particles

    def _resample_unfinished_particles(
        self,
        particles: List[_ParticleState],
    ) -> List[_ParticleState]:
        completed_particles = [p for p in particles if p.completed]
        unfinished_particles = [p for p in particles if not p.completed]

        if len(unfinished_particles) <= 1:
            return completed_particles + unfinished_particles

        weights = np.asarray(
            [max(self.epsilon, p.last_incremental_weight) for p in unfinished_particles],
            dtype=float,
        )
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / total

        selected = self.rng.choice(len(unfinished_particles), size=len(unfinished_particles), replace=True, p=probs)

        cloned_particles: List[_ParticleState] = []
        for idx in selected.tolist():
            clone = unfinished_particles[idx].clone()
            clone.resample_history.append(int(idx))
            cloned_particles.append(clone)

        return completed_particles + cloned_particles

    # ------------------------------------------------------------------
    # Value scoring helpers
    # ------------------------------------------------------------------

    def _score_updated_particles_batched(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int],
        updated_items: List[Tuple[int, int, _ParticleState, List[StepCandidate], Optional[StepCandidate]]],
    ) -> List[float]:
        if not updated_items:
            return []

        if self.scorer is None:
            return [1.0] * len(updated_items)

        if hasattr(self.scorer, "score_trajectories_batch"):
            flat_requests: List[List[Dict[str, str]]] = []
            flat_trajectories: List[List[StepCandidate]] = []
            flat_sample_ids: List[int] = []

            for sid, _pid, particle, _prev_steps, _cand in updated_items:
                flat_requests.append(requests[sid])
                flat_trajectories.append(particle.steps)
                flat_sample_ids.append(sample_indices[sid])

            raw_outputs = self.scorer.score_trajectories_batch(
                flat_requests,
                flat_trajectories,
                sample_ids=flat_sample_ids,
                trajectory_ids=list(range(len(flat_trajectories))),
            )
            return [self._extract_last_score(raw) for raw in raw_outputs]

        if hasattr(self.scorer, "score_trajectory"):
            out: List[float] = []
            for sid, _pid, particle, _prev_steps, _cand in updated_items:
                raw = self.scorer.score_trajectory(requests[sid], particle.steps)
                out.append(self._extract_last_score(raw))
            return out

        if self.allow_score_candidates and hasattr(self.scorer, "score_candidates"):
            out = []
            for sid, _pid, _particle, prev_steps, cand in updated_items:
                if cand is None:
                    out.append(0.0)
                    continue
                raw = self.scorer.score_candidates(requests[sid], [cand], trajectory=prev_steps)
                out.append(self._extract_last_score(raw))
            return out

        raise AttributeError(
            f"{self.scorer.__class__.__name__} must implement score_trajectories_batch(...), "
            "score_trajectory(...), or score_candidates(...)."
        )

    def _score_updated_particles_sequential(
        self,
        request: List[Dict[str, str]],
        sample_idx: int,
        updated_items: List[Tuple[_ParticleState, List[StepCandidate], Optional[StepCandidate]]],
    ) -> List[float]:
        if not updated_items:
            return []

        if self.scorer is None:
            return [1.0] * len(updated_items)

        if hasattr(self.scorer, "score_trajectories_batch"):
            raw_outputs = self.scorer.score_trajectories_batch(
                [request] * len(updated_items),
                [particle.steps for particle, _prev_steps, _cand in updated_items],
                sample_ids=[sample_idx] * len(updated_items),
                trajectory_ids=list(range(len(updated_items))),
            )
            return [self._extract_last_score(raw) for raw in raw_outputs]

        if hasattr(self.scorer, "score_trajectory"):
            return [
                self._extract_last_score(self.scorer.score_trajectory(request, particle.steps))
                for particle, _prev_steps, _cand in updated_items
            ]

        if self.allow_score_candidates and hasattr(self.scorer, "score_candidates"):
            out = []
            for _particle, prev_steps, cand in updated_items:
                if cand is None:
                    out.append(0.0)
                    continue
                raw = self.scorer.score_candidates(request, [cand], trajectory=prev_steps)
                out.append(self._extract_last_score(raw))
            return out

        raise AttributeError(
            f"{self.scorer.__class__.__name__} must implement score_trajectories_batch(...), "
            "score_trajectory(...), or score_candidates(...)."
        )

    # ------------------------------------------------------------------
    # TSMC weight helpers
    # ------------------------------------------------------------------

    def _previous_prefix_value(self, particle: _ParticleState) -> float:
        # The particle already contains the newly appended current prefix value in
        # neither sequential nor batched mode when this is called. Therefore, the
        # previous prefix value is the last stored value score if any; otherwise
        # use the empty-prefix initialization.
        if not particle.value_scores:
            return self.initial_value
        return float(max(self.epsilon, particle.value_scores[-1]))

    def _nonterminal_weight(self, prev_value: float, current_value: float) -> float:
        prev_value = max(self.epsilon, prev_value)
        current_value = max(self.epsilon, current_value)
        return float(np.sqrt(current_value / prev_value).item())

    def _terminal_weight(self, prev_value: float, current_value: float) -> float:
        prev_value = max(self.epsilon, prev_value)
        current_value = max(self.epsilon, current_value)
        return float((current_value / np.sqrt(prev_value)).item())

    def _clip_value(self, value: float) -> float:
        if value is None:
            return self.epsilon
        value = float(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0))
        return float(np.clip(value, self.epsilon, 1.0))

    # ------------------------------------------------------------------
    # Voting / finalization
    # ------------------------------------------------------------------

    def _finalize_sample_result(
        self,
        sample_idx: int,
        particles: List[_ParticleState],
        total_generated_tokens: int,
    ) -> Dict[str, Any]:
        if not particles:
            return self._empty_result()

        particle_texts = [convert_trajectory_to_string(p.steps) for p in particles]
        answers = [self._safe_extract_answer(text) for text in particle_texts]
        vote_weights = [max(self.epsilon, float(p.terminal_weight)) for p in particles]

        voting_indices = self._get_voting_indices(answers)
        voting_answers = [answers[i] for i in voting_indices]

        cluster_ids_local, representatives, members_local = self._group_answers(voting_answers)
        if not members_local:
            return self._empty_result()

        members = {
            cid: [voting_indices[j] for j in idxs]
            for cid, idxs in members_local.items()
        }

        cluster_ids = [-1] * len(particles)
        for local_j, original_i in enumerate(voting_indices):
            cluster_ids[original_i] = cluster_ids_local[local_j]

        cluster_counts = {cid: len(idxs) for cid, idxs in members.items()}
        cluster_weights = {
            cid: float(sum(vote_weights[i] for i in idxs))
            for cid, idxs in members.items()
        }

        def cluster_key(cid: int) -> Tuple[float, int, float, float, int]:
            best_idx = self._select_best_index_within_cluster(members[cid], particles)
            return (
                cluster_weights[cid],
                cluster_counts[cid],
                particles[best_idx].terminal_weight,
                particles[best_idx].avg_value,
                -best_idx,
            )

        selected_cluster = max(members.keys(), key=cluster_key)
        selected_idx = self._select_best_index_within_cluster(members[selected_cluster], particles)

        best_particle = particles[selected_idx]
        best_text = particle_texts[selected_idx]
        best_answer = answers[selected_idx]

        total_weight = float(sum(vote_weights[i] for i in voting_indices))
        consensus_score = (
            cluster_weights[selected_cluster] / total_weight
            if total_weight > 0.0
            else cluster_counts[selected_cluster] / max(len(voting_indices), 1)
        )

        answer_distribution = {
            representatives[cid]: {
                "count": cluster_counts[cid],
                "weight": cluster_weights[cid],
            }
            for cid in members
        }

        all_traces: List[Dict[str, Any]] = []
        for i, (particle, text, answer, weight) in enumerate(zip(particles, particle_texts, answers, vote_weights)):
            cid = cluster_ids[i]
            all_traces.append(
                {
                    "text": text,
                    "num_steps": len(particle.steps),
                    "generated_tokens_total": particle.generated_tokens_total,
                    "value_scores": list(particle.value_scores),
                    "incremental_weights": list(particle.incremental_weights),
                    "avg_value": particle.avg_value,
                    "terminal_weight": float(weight),
                    "answer": answer,
                    "answer_cluster_id": cid,
                    "cluster_representative": representatives[cid] if cid != -1 else None,
                    "cluster_weight": cluster_weights[cid] if cid != -1 else 0.0,
                    "cluster_count": cluster_counts[cid] if cid != -1 else 0,
                    "selected": i == selected_idx,
                }
            )

        avg_num_steps = sum(len(p.steps) for p in particles) / max(len(particles), 1)

        log.info(
            f"[TSMC] sample {sample_idx}: best_answer={best_answer}, "
            f"consensus={consensus_score:.3f}, answer_distribution={answer_distribution}"
        )

        return {
            "trajectory": best_text,
            "steps": best_particle.steps,
            "validity_scores": list(best_particle.value_scores),
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
                "selected_path_generated_tokens": int(best_particle.generated_tokens_total),
                "selected_particle_generated_tokens": int(best_particle.generated_tokens_total),
                "avg_generated_tokens_per_particle": (
                    float(total_generated_tokens) / max(len(particles), 1)
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
        particles: List[_ParticleState],
    ) -> int:
        return max(
            cluster_member_indices,
            key=lambda i: (
                particles[i].terminal_weight,
                particles[i].avg_value,
                -i,
            ),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_last_score(self, raw: Any) -> float:
        if isinstance(raw, np.ndarray):
            if raw.size == 0:
                return self.epsilon
            return self._clip_value(float(raw.reshape(-1)[-1]))
        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                return self.epsilon
            return self._clip_value(float(raw[-1]))
        return self._clip_value(float(raw))

    def _safe_extract_answer(self, text: str) -> str:
        try:
            ans = extract_answer(text)
            if ans is None or ans == "":
                return NO_ANSWER
            return str(ans)
        except Exception:
            return NO_ANSWER

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
