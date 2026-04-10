import logging
from typing import Dict, List

import numpy as np
import torch

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
        alpha: float = 0.5,
        n_samples: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 64,
        value_prompt: str | None = None,
    ):
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.alpha = alpha

        self.value_prompt = value_prompt or self._default_value_prompt()

        self.value_map = {
            "sure": 1.0,
            "correct": 1.0,
            "likely": 0.5,
            "unlikely": 0.01,
            "impossible": 0.01,
            "incorrect": 0.01,
        }

        self.prm_model = "mock_prm_model"

    # ================================================================
    # StepScorerRewardBase API
    # ================================================================

    def compute_claim_rewards(
        self,
        chat: List[Dict[str, str]],
        candidates: List[str],
        trajectory: List = None,
        **kwargs,
    ) -> List[List[float]]:
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
        return [[float(s)] for s in scores]

    # ================================================================
    # Beam-search rescoring API
    # ================================================================

    def score_trajectory(
        self,
        chat: List[Dict[str, str]],
        trajectory: List,
        **kwargs,
    ) -> List[float]:
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
        if not trajectories:
            return []

        questions, states, valid_indices = [], [], []

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

        # ---- ToT logging (batch only) ----
        for local_i, score in enumerate(scores):
            idx = valid_indices[local_i]
            log.info(
                f"[ToT] Sample {sample_ids[idx] if sample_ids else '?'} "
                f"Traj {trajectory_ids[idx] if trajectory_ids else idx}: "
                f"num_steps={len(trajectories[idx])}, value={score:.3f}"
            )
        # ---------------------------------

        results = [[] for _ in trajectories]
        for idx, score in zip(valid_indices, scores):
            results[idx] = [float(score)] * len(trajectories[idx])

        return results

    def _get_tokenizer(self, raw_llm):
        # vLLM LLM
        if hasattr(raw_llm, "get_tokenizer"):
            return raw_llm.get_tokenizer()
        # Sometimes vLLM wrapper exposes tokenizer directly
        if hasattr(raw_llm, "tokenizer"):
            return raw_llm.tokenizer
        # Fall back to whatever the pipeline has
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer
        if hasattr(self.model, "step_generator") and hasattr(self.model.step_generator, "tokenizer"):
            return self.model.step_generator.tokenizer
        return None

    def _get_max_model_len(self, raw_llm, default=4096):
        # Try common vLLM internals (best-effort)
        for path in [
            ("llm_engine", "model_config", "max_model_len"),
            ("engine", "model_config", "max_model_len"),
        ]:
            obj = raw_llm
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and isinstance(obj, int):
                return obj
        return default

    def _truncate_state_for_prompt(
        self,
        tokenizer,
        question: str,
        state: str,
        max_model_len: int,
        max_new_tokens: int,
        margin: int = 32,
    ) -> str:
        """
        Truncate ONLY the state so that the final prompt fits in max_model_len.

        Keeps the tail of the state (most recent reasoning).
        """
        # Build prompt with empty state to measure overhead
        empty_prompt = self.value_prompt.format(problem=question.strip(), state="")
        overhead_ids = tokenizer.encode(empty_prompt)

        # How many tokens can we spend on {state}?
        budget = max_model_len - max_new_tokens - margin - len(overhead_ids)

        if budget <= 0:
            # Degenerate case: prompt template itself too large
            return ""

        state_ids = tokenizer.encode(state)

        if len(state_ids) <= budget:
            return state

        # Keep the most recent part of the state
        truncated_ids = state_ids[-budget:]
        truncated_state = tokenizer.decode(truncated_ids)

        return truncated_state


    # ================================================================
    # Core scoring logic (FIXED)
    # ================================================================

    def _score_states(self, question: str, states: List[str]) -> List[float]:
        return self._score_states_batch([question] * len(states), states)

    def _score_states_batch(
            self,
            questions: List[str],
            states: List[str],
    ) -> List[float]:
        if not states:
            return []

        # Resolve raw vLLM LLM robustly
        raw_model = getattr(self.model, "model", self.model)
        raw_llm = getattr(raw_model, "llm", raw_model)

        tokenizer = self._get_tokenizer(raw_llm)
        if tokenizer is None:
            raise RuntimeError("ToT value scorer: could not resolve a tokenizer for truncation.")

        max_model_len = self._get_max_model_len(raw_llm, default=4096)

        sampling_params = SamplingParams(
            n=self.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Truncate states BEFORE building prompts
        truncated_prompts = []
        for q, s in zip(questions, states):
            s_trunc = self._truncate_state_for_prompt(
                tokenizer=tokenizer,
                question=q,
                state=s,
                max_model_len=max_model_len,
                max_new_tokens=self.max_tokens,
                margin=32,
            )
            prompt = self.value_prompt.format(problem=q.strip(), state=s_trunc.strip())
            truncated_prompts.append(prompt)

        log.info(
            f"ToT value scoring {len(truncated_prompts)} states "
            f"(n={self.n_samples}, max_model_len={max_model_len}, max_new_tokens={self.max_tokens})"
        )

        outputs = raw_llm.generate(truncated_prompts, sampling_params)

        scores = []
        for output in outputs:
            texts = [o.text for o in output.outputs]
            parsed = [self._parse_output(t) for t in texts]
            scores.append(float(np.quantile(parsed, self.alpha)) if parsed else 0.5)

        return scores

    # ================================================================
    # Utilities
    # ================================================================

    def _parse_output(self, text: str) -> float:
        text = text.lower()
        for label in sorted(self.value_map.keys(), key=len, reverse=True):
            if label in text:
                return self.value_map[label]
        return 0.5

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
