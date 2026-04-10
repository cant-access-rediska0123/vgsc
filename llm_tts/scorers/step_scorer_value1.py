import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from vllm import SamplingParams

from .step_scorer_reward_base import StepScorerRewardBase

log = logging.getLogger(__name__)


class StepScorerToTValue(StepScorerRewardBase):
    """
    Tree-of-Thoughts value scorer (deterministic, low-memory).

    Key idea:
      - Ask the model to output ONE character: 1/2/3/4 (single-token friendly).
      - Read the *completion* logprobs for that first generated token (NOT prompt_logprobs).
      - Convert digit distribution -> label distribution -> scalar score via alpha:
          alpha == 0.5 : expected value (mean)
          alpha < 0.5  : lower-tail weighted quantile (risk-averse)
          alpha > 0.5  : upper-tail weighted quantile (risk-seeking)

    This avoids:
      - Sampling variance (temperature=0)
      - prompt_logprobs memory blowups
      - 4x prompt batching (one prompt per state)
    """

    LABELS = ("sure", "likely", "unlikely", "impossible")
    DIGITS = ("1", "2", "3", "4")  # 1->sure, 2->likely, 3->unlikely, 4->impossible

    def __init__(
        self,
        model,
        alpha: float = 0.5,
        margin_tokens: int = 32,
        logprobs_k: int = 20,
        value_prompt: Optional[str] = None,
    ):
        self.model = model

        self.alpha = float(alpha)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.margin_tokens = int(margin_tokens)
        self.logprobs_k = int(logprobs_k)

        self.value_prompt = value_prompt or self._default_digit_prompt()

        # Scalar mapping (keep identical to your prior implementations)
        self.value_map = {
            "sure": 1.0,
            "likely": 0.5,
            "unlikely": 0.01,
            "impossible": 0.01,
        }

        # expected by your codebase
        self.prm_model = "mock_prm_model"

        # token accounting (best-effort)
        self._last_token_stats = {
            "total_prompt_tokens_last_call": 0,
            "avg_prompt_tokens_per_sample_last_call": 0.0,
            "num_generate_calls_last_call": 0,
        }

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

        states: List[str] = []
        for cand in candidates:
            cand_text = cand.text if hasattr(cand, "text") else cand
            steps = [step.text if hasattr(step, "text") else str(step) for step in trajectory]
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
        steps = [step.text if hasattr(step, "text") else str(step) for step in trajectory]
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

        questions: List[str] = []
        states: List[str] = []
        valid_indices: List[int] = []

        for i, (chat, traj) in enumerate(zip(chats, trajectories)):
            if not traj:
                continue
            question = self._extract_question(chat)
            steps = [step.text if hasattr(step, "text") else str(step) for step in traj]
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

        ts = getattr(self, "_last_token_stats", None)
        if ts:
            log.info(
                f"[StepScorerToTValue] prompt_tokens(last_call)={ts['total_prompt_tokens_last_call']} | "
                f"avg_prompt_tokens/sample(last_call)={ts['avg_prompt_tokens_per_sample_last_call']:.2f} | "
                f"generate_calls(last_call)={ts['num_generate_calls_last_call']}"
            )
        # ---------------------------------

        results: List[List[float]] = [[] for _ in trajectories]
        for idx, score in zip(valid_indices, scores):
            results[idx] = [float(score)] * len(trajectories[idx])

        return results

    # ================================================================
    # vLLM / tokenizer helpers
    # ================================================================

    def _get_tokenizer(self, raw_llm):
        # vLLM LLM
        if hasattr(raw_llm, "get_tokenizer"):
            return raw_llm.get_tokenizer()
        if hasattr(raw_llm, "tokenizer"):
            return raw_llm.tokenizer
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer
        if hasattr(self.model, "step_generator") and hasattr(self.model.step_generator, "tokenizer"):
            return self.model.step_generator.tokenizer
        return None

    def _get_max_model_len(self, raw_llm, default: int = 4096) -> int:
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

    def _encode_text(self, tokenizer, text: str) -> List[int]:
        try:
            return tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(text)

    def _truncate_state_for_prompt(
        self,
        tokenizer,
        question: str,
        state: str,
        max_model_len: int,
        reserved_tokens: int,
        margin: int = 32,
    ) -> str:
        """
        Truncate ONLY the state so the final prompt fits in max_model_len.

        reserved_tokens: tokens reserved for generation (here: 1 token).
        Keeps the tail of the state (most recent reasoning).
        """
        empty_prompt = self._build_prompt(question, "")
        overhead_ids = self._encode_text(tokenizer, empty_prompt)

        budget = max_model_len - reserved_tokens - margin - len(overhead_ids)
        if budget <= 0:
            return ""

        state_ids = self._encode_text(tokenizer, state)
        if len(state_ids) <= budget:
            return state

        truncated_ids = state_ids[-budget:]
        return tokenizer.decode(truncated_ids)

    # ================================================================
    # Distribution -> scalar score (alpha)
    # ================================================================

    def _weighted_quantile_continuous(
        self,
        values: np.ndarray,
        probs: np.ndarray,
        q: float,
    ) -> float:
        """
        Continuous weighted quantile via linear interpolation on mid-CDF points.
        For values [0, 1] with probs [0.5, 0.5], q=0.5 -> 0.5.
        """
        v = np.asarray(values, dtype=np.float64)
        w = np.asarray(probs, dtype=np.float64)

        if v.size == 0:
            return 0.0

        w_sum = float(w.sum())
        if w_sum <= 0:
            return float(v.mean())

        idx = np.argsort(v)
        v = v[idx]
        w = w[idx] / w_sum

        # merge duplicates
        merged_v: List[float] = []
        merged_w: List[float] = []
        for vi, wi in zip(v, w):
            if merged_v and np.isclose(vi, merged_v[-1]):
                merged_w[-1] += float(wi)
            else:
                merged_v.append(float(vi))
                merged_w.append(float(wi))

        v = np.asarray(merged_v, dtype=np.float64)
        w = np.asarray(merged_w, dtype=np.float64)

        mids = np.cumsum(w) - 0.5 * w
        q = float(np.clip(q, 0.0, 1.0))
        return float(np.interp(q, mids, v, left=v[0], right=v[-1]))

    def _distribution_to_score(self, label_probs: np.ndarray) -> float:
        values = np.array([self.value_map[lbl] for lbl in self.LABELS], dtype=np.float64)
        probs = np.asarray(label_probs, dtype=np.float64)

        s = float(probs.sum())
        if s <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / s

        mean = float(np.dot(probs, values))

        if np.isclose(self.alpha, 0.5):
            return mean

        return self._weighted_quantile_continuous(values, probs, self.alpha)

    # ================================================================
    # Logprobs extraction (completion, first generated token)
    # ================================================================

    def _coerce_logprob(self, obj) -> Optional[float]:
        if obj is None:
            return None
        if isinstance(obj, (float, int)):
            return float(obj)
        if hasattr(obj, "logprob"):
            return float(obj.logprob)
        if isinstance(obj, dict) and "logprob" in obj:
            return float(obj["logprob"])
        return None

    def _extract_first_token_logprob_dict(self, req_out) -> Optional[dict]:
        """
        Return dict-like mapping token_id -> (logprob object/float) for the FIRST generated token.
        Handles common vLLM output shapes.
        """
        # Common: RequestOutput.outputs[0].logprobs -> list[dict] (one dict per generated position)
        if hasattr(req_out, "outputs") and req_out.outputs:
            out0 = req_out.outputs[0]
            lps = getattr(out0, "logprobs", None)
            if isinstance(lps, list) and lps:
                if isinstance(lps[0], dict):
                    return lps[0]
        # Alternate: RequestOutput.logprobs -> list[dict]
        lps2 = getattr(req_out, "logprobs", None)
        if isinstance(lps2, list) and lps2:
            if isinstance(lps2[0], dict):
                return lps2[0]
        return None

    def _extract_generated_text_first_char(self, req_out) -> str:
        if hasattr(req_out, "outputs") and req_out.outputs:
            txt = getattr(req_out.outputs[0], "text", "") or ""
            return txt.strip()[:1]
        return ""

    # ================================================================
    # Core scoring logic (single prompt per state)
    # ================================================================

    def _score_states(self, question: str, states: List[str]) -> List[float]:
        return self._score_states_batch([question] * len(states), states)

    def _score_states_batch(
        self,
        questions: List[str],
        states: List[str],
    ) -> List[float]:
        if not states:
            self._last_token_stats = {
                "total_prompt_tokens_last_call": 0,
                "avg_prompt_tokens_per_sample_last_call": 0.0,
                "num_generate_calls_last_call": 0,
            }
            return []

        raw_model = getattr(self.model, "model", self.model)
        raw_llm = getattr(raw_model, "llm", raw_model)

        tokenizer = self._get_tokenizer(raw_llm)
        if tokenizer is None:
            raise RuntimeError("ToT value scorer: could not resolve a tokenizer for truncation.")

        max_model_len = self._get_max_model_len(raw_llm, default=4096)

        prompts: List[str] = []
        prompt_token_total = 0

        for q, s in zip(questions, states):
            s_trunc = self._truncate_state_for_prompt(
                tokenizer=tokenizer,
                question=q,
                state=s,
                max_model_len=max_model_len,
                reserved_tokens=1,  # only generating one token (the digit)
                margin=self.margin_tokens,
            )
            p = self._build_prompt(q, s_trunc)
            prompts.append(p)
            prompt_token_total += len(self._encode_text(tokenizer, p))

        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=1,
            logprobs=max(1, self.logprobs_k),
        )

        log.info(
            f"ToT value scoring {len(prompts)} states (digit classifier) "
            f"(max_model_len={max_model_len}, logprobs_k={self.logprobs_k})"
        )

        outs = raw_llm.generate(prompts, sampling_params)

        scores: List[float] = []
        for req_out in outs:
            lp_dict = self._extract_first_token_logprob_dict(req_out)
            print('req_out logprobs:', lp_dict)

            if not lp_dict:
                raise Exception('No logprobs found!')

            # Build logprob vector over digits
            lps: List[list[float]] = [[], [], [], []]
            for v in lp_dict.values():
                tok = str(v.decoded_token).strip()
                if tok in self.DIGITS:
                    lps[int(tok) - 1].append(v.logprob)
            lps = [np.logaddexp.reduce(l) for l in lps]

            lps_arr = np.asarray(lps, dtype=np.float64)
            m = float(np.max(lps_arr))
            exps = np.exp(lps_arr - m)
            z = float(exps.sum())
            probs = exps / z if z > 0 else (np.ones(4, dtype=np.float64) / 4.0)

            scores.append(float(self._distribution_to_score(probs)))

        num_states = max(len(states), 1)
        self._last_token_stats = {
            "total_prompt_tokens_last_call": int(prompt_token_total),
            "avg_prompt_tokens_per_sample_last_call": float(prompt_token_total / num_states),
            "num_generate_calls_last_call": 1,
        }

        return scores

    # ================================================================
    # Utilities
    # ================================================================

    def _build_prompt(self, question: str, state: str) -> str:
        return self.value_prompt.format(
            problem=question.strip(),
            state=state.strip(),
        )

    @staticmethod
    def _extract_question(chat: List[Dict[str, str]]) -> str:
        for msg in chat:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return chat[-1].get("content", "") if chat else ""

    @staticmethod
    def _default_digit_prompt() -> str:
        return """You are evaluating a partial reasoning process.

Problem:
{problem}

Partial reasoning so far:
{state}

Question:
Can this reasoning still lead to a correct final answer?

Answer with ONE character only: 1, 2, 3, or 4.
1 = sure
2 = likely
3 = unlikely
4 = impossible

Rating: """