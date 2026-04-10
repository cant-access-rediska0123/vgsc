import logging
from typing import Dict, List

import numpy as np

from vllm import SamplingParams

from .step_scorer_reward_base import StepScorerRewardBase

log = logging.getLogger(__name__)


class StepScorerToTValue(StepScorerRewardBase):
    """
    Tree-of-Thoughts value scorer.

    Scores each candidate or trajectory by asking:
    "Can this partial reasoning still lead to a correct answer?"
    """

    def __init__(
        self,
        model,
        n_samples: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 64,
        value_prompt: str | None = None,
    ):
        """
        Args:
            model: Whitebox model or step generator (must expose .llm or be a vLLM LLM)
            n_samples: Number of evaluation samples per state
            temperature: Sampling temperature
            max_tokens: Max tokens for evaluation output
            value_prompt: Optional custom prompt template
        """
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.value_prompt = value_prompt or self._default_value_prompt()

        # Monotonic value mapping (ToT-style)
        self.value_map = {
            "sure": 10.0,
            "correct": 10.0,
            "likely": 5.0,
            "unlikely": 1.0,
            "impossible": 0.1,
            "incorrect": 0.1,
        }

    # ------------------------------------------------------------------
    # Required by StepScorerRewardBase (used during beam expansion)
    # ------------------------------------------------------------------

    def compute_claim_rewards(
        self,
        chat: List[Dict[str, str]],
        candidates: List[str],
        trajectory: List = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Score each candidate by evaluating the full trajectory
        (previous steps + candidate).
        """
        if not candidates:
            return []

        trajectory = trajectory or []
        question = self._extract_question(chat)

        states = []
        for cand in candidates:
            cand_text = cand.text if hasattr(cand, "text") else cand
            steps = [
                step.text if hasattr(step, "text") else str(step)
                for step in trajectory
            ]
            steps.append(cand_text)
            states.append("\n".join(steps))

        scores = self._score_states(question, states)

        # Return one "claim reward" per candidate
        return [[float(score)] for score in scores]

    # ------------------------------------------------------------------
    # Used by StrategyBeamSearch for rescoring
    # ------------------------------------------------------------------

    def score_trajectory(
        self,
        chat: List[Dict[str, str]],
        trajectory: List,
        **kwargs,
    ) -> List[float]:
        """
        Score a single trajectory.

        Returns one score per step (broadcasted).
        """
        if not trajectory:
            return []

        question = self._extract_question(chat)
        steps = [
            step.text if hasattr(step, "text") else str(step)
            for step in trajectory
        ]
        state = "\n".join(steps)

        score = self._score_states(question, [state])[0]

        return [float(score)] * len(trajectory)

    def score_trajectories_batch(
        self,
        chats: List[List[Dict[str, str]]],
        trajectories: List[List],
        sample_ids: List[int] = None,
        trajectory_ids: List[int] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Batch score multiple trajectories.

        Each trajectory is evaluated once and its value
        is broadcast to all steps.
        """
        if not trajectories:
            return []

        questions = []
        states = []
        valid_indices = []

        for i, (chat, traj) in enumerate(zip(chats, trajectories)):
            if not traj:
                continue

            question = self._extract_question(chat)
            steps = [
                step.text if hasattr(step, "text") else str(step)
                for step in traj
            ]

            questions.append(question)
            states.append("\n".join(steps))
            valid_indices.append(i)

        scores = self._score_states_batch(questions, states)

        results = [[] for _ in trajectories]
        for idx, score in zip(valid_indices, scores):
            results[idx] = [float(score)] * len(trajectories[idx])

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_states(self, question: str, states: List[str]) -> List[float]:
        """
        Sequential scoring fallback.
        """
        return self._score_states_batch(
            [question] * len(states),
            states,
        )

    def _score_states_batch(
        self,
        questions: List[str],
        states: List[str],
    ) -> List[float]:
        """
        Core vLLM-based batched ToT value scoring.
        """
        if not states:
            return []

        prompts = [
            self._build_prompt(q, s) for q, s in zip(questions, states)
        ]

        # Get raw vLLM LLM (same pattern as strategy_baseline.py)
        raw_llm = getattr(self.model, "llm", self.model)

        sampling_params = SamplingParams(
            n=self.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        log.info(f"ToT value scoring {len(prompts)} states (n={self.n_samples})")

        outputs = raw_llm.generate(prompts, sampling_params)

        scores = []
        for output in outputs:
            texts = []
            for out in output.outputs:
                texts.append(out.text)

            parsed = [self._parse_output(t) for t in texts]
            score = float(np.mean(parsed)) if parsed else 1.0
            scores.append(score)

        return scores

    def _parse_output(self, text: str) -> float:
        """
        Parse LLM output into a numeric value.
        """
        text = text.lower()
        for label in sorted(self.value_map.keys(), key=len, reverse=True):
            if label in text:
                return self.value_map[label]
        return 1.0

    def _build_prompt(self, question: str, state: str) -> str:
        return self.value_prompt.format(
            problem=question.strip(),
            state=state.strip(),
        )

    @staticmethod
    def _extract_question(chat: List[Dict[str, str]]) -> str:
        for msg in chat:
            if msg["role"] == "user":
                return msg["content"]
        return chat[-1]["content"]

    @staticmethod
    def _default_value_prompt() -> str:
        return """You are evaluating a partial reasoning process.

Problem:
{problem}

Partial reasoning so far:
{state}

Question:
Can this reasoning still lead to a correct final answer?

Answer with ONE word only:
- sure
- likely
- unlikely
- impossible

Rating:"""
