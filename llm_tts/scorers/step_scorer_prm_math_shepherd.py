import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from llm_tts.utils import get_torch_dtype

from .step_scorer_reward_base import StepScorerRewardBase

# Optional vLLM import
try:
    from vllm import LLM
    from vllm.config import PoolerConfig

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    PoolerConfig = None

from transformers import AutoTokenizer, AutoModelForCausalLM

log = logging.getLogger(__name__)


class StepScorerPRMMathShepherd:
    """
    Direct PRM scorer for peiyi9979/math-shepherd-mistral-7b-prm.

    Same interface as StepScorerPRM:
        prm_model_path: str
        device: str
        batch_size: int
        torch_dtype: str
        use_vllm: bool = True
        gpu_memory_utilization: float = 0.9
    """

    STEP_TAG_TEXT = "ки"
    GOOD_TOKEN_TEXT = "+"
    BAD_TOKEN_TEXT = "-"

    def __init__(
        self,
        prm_model_path: str,
        device: str,
        batch_size: int,
        torch_dtype: str,
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.9,
    ):
        self.prm_model_path = prm_model_path
        self.device = device
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_vllm = use_vllm and VLLM_AVAILABLE

        self.prm_model = None
        self.prm_tokenizer = None

        # Cached token ids (computed after tokenizer is available)
        self._step_tag_id: Optional[int] = None
        self._plus_id: Optional[int] = None
        self._minus_id: Optional[int] = None

        if use_vllm and not VLLM_AVAILABLE:
            log.warning("vLLM requested but not available, falling back to HuggingFace")

        self.prepare_model()

    # ----------------------------
    # Model loading / cleanup
    # ----------------------------

    def prepare_model(self):
        """Load PRM model and tokenizer using selected backend."""
        if self.use_vllm:
            self._prepare_vllm_model()
        else:
            self._prepare_hf_model()

        self._init_token_ids()

    def _init_token_ids(self):
        """Compute and cache token ids needed for scoring."""
        if self.prm_tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        # step tag id must be the ID of "ки" token (take last to be robust)
        step_ids = self.prm_tokenizer.encode(self.STEP_TAG_TEXT, add_special_tokens=False)
        if not step_ids:
            raise RuntimeError(f"Could not tokenize step tag '{self.STEP_TAG_TEXT}'")
        self._step_tag_id = step_ids[-1]

        plus_ids = self.prm_tokenizer.encode(self.GOOD_TOKEN_TEXT, add_special_tokens=False)
        minus_ids = self.prm_tokenizer.encode(self.BAD_TOKEN_TEXT, add_special_tokens=False)
        if not plus_ids or not minus_ids:
            raise RuntimeError("Could not tokenize '+' or '-' tokens")

        # Prefer first id; these should be single tokens in Mistral tokenizers
        self._plus_id = plus_ids[0]
        self._minus_id = minus_ids[0]

        log.info(
            f"Math-Shepherd token ids: step_tag_id={self._step_tag_id}, "
            f"plus_id={self._plus_id}, minus_id={self._minus_id}"
        )

    def _prepare_vllm_model(self):
        """Load PRM model using vLLM backend."""
        import os

        # Parse device to get GPU ID
        if "cuda:" in self.device:
            gpu_id = int(self.device.split(":")[1])
        else:
            gpu_id = 0

        # Get current CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_gpus = [int(x) for x in cuda_visible.split(",") if x.strip()]

        # Determine the actual physical GPU ID
        if visible_gpus:
            if gpu_id < len(visible_gpus):
                physical_gpu = visible_gpus[gpu_id]
            else:
                physical_gpu = visible_gpus[0]
                log.warning(
                    f"Requested GPU {gpu_id} not in CUDA_VISIBLE_DEVICES={cuda_visible}, "
                    f"using GPU {physical_gpu}"
                )
        else:
            physical_gpu = gpu_id

        log.info(
            f"Loading Math-Shepherd PRM from {self.prm_model_path} (vLLM backend) on GPU {physical_gpu}"
        )

        # Temporarily set CUDA_VISIBLE_DEVICES to only the target GPU
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

        try:
            # IMPORTANT: override_pooler_config requires token ids -> we need a tokenizer first.
            # vLLM can give us a tokenizer after LLM init, but pooler config is required at init.
            # Workaround: load HF tokenizer quickly to compute ids, then pass to vLLM.
            tmp_tok = AutoTokenizer.from_pretrained(self.prm_model_path, trust_remote_code=True)
            step_tag_id = tmp_tok.encode(self.STEP_TAG_TEXT, add_special_tokens=False)[-1]
            plus_id = tmp_tok.encode(self.GOOD_TOKEN_TEXT, add_special_tokens=False)[0]
            minus_id = tmp_tok.encode(self.BAD_TOKEN_TEXT, add_special_tokens=False)[0]

            pooler_cfg = PoolerConfig(
                pooling_type="STEP",
                step_tag_id=step_tag_id,
                returned_token_ids=[plus_id, minus_id],
            )

            self.prm_model = LLM(
                model=self.prm_model_path,
                task="reward",
                trust_remote_code=True,
                dtype=self.torch_dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=True,
                max_model_len=4096,
                runner="pooling",
                pooler_config=pooler_cfg,
            )
            self.prm_tokenizer = self.prm_model.get_tokenizer()
            log.info("vLLM Math-Shepherd PRM loaded successfully")
        finally:
            # Restore original CUDA_VISIBLE_DEVICES
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def _prepare_hf_model(self):
        """Load PRM model using HuggingFace backend."""
        log.info(f"Loading Math-Shepherd PRM from {self.prm_model_path} (HuggingFace backend)")
        self.prm_tokenizer = AutoTokenizer.from_pretrained(
            self.prm_model_path,
            trust_remote_code=True,
        )
        self.prm_model = AutoModelForCausalLM.from_pretrained(
            self.prm_model_path,
            device_map=self.device,
            torch_dtype=get_torch_dtype(self.torch_dtype),
            trust_remote_code=True,
        ).eval()

    def cleanup(self):
        """Free PRM model memory."""
        if self.prm_model is not None:
            del self.prm_model
            self.prm_model = None
        if self.prm_tokenizer is not None:
            del self.prm_tokenizer
            self.prm_tokenizer = None
        torch.cuda.empty_cache()

    # ----------------------------
    # Public scoring API (same names)
    # ----------------------------

    def compute_claim_rewards(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Compute reward scores for claims in each candidate.

        Same behavior as your vLLM path in StepScorerPRM:
        - vLLM: treats each candidate as a *single new step* appended to trajectory
        - HF: splits candidate by newlines into steps and returns per-step rewards
        """
        if not candidates:
            return []

        trajectory = kwargs.get("trajectory", None)

        if self.use_vllm:
            return self._compute_rewards_vllm(chat, candidates, trajectory=trajectory)
        else:
            return self._compute_rewards_hf(chat, candidates)

    def _compute_rewards_hf(
        self, chat: List[Dict[str, str]], candidates: List[str]
    ) -> List[List[float]]:
        all_rewards: List[List[float]] = []
        for candidate in candidates:
            candidate_text = candidate.text if hasattr(candidate, "text") else candidate
            rewards = self._score_single_candidate_hf(chat, candidate_text)
            all_rewards.append(rewards)
            torch.cuda.empty_cache()
        return all_rewards

    def _compute_rewards_vllm(
        self, chat: List[Dict[str, str]], candidates: List[str], trajectory: List = None
    ) -> List[List[float]]:
        """
        vLLM scoring (batched):
        - trajectory steps are used as-is (each step.text is one PRM step)
        - candidate appended as one PRM step
        - returns only the last-step score per candidate for backward compatibility
        """
        if not candidates:
            return []

        question = self._extract_question(chat)

        trajectory_steps: List[str] = []
        if trajectory:
            trajectory_steps = [
                step.text if hasattr(step, "text") else str(step) for step in trajectory
            ]

        # Build prompts: question + (traj steps + candidate) each with step tag "ки"
        all_prompts: List[str] = []
        for candidate in candidates:
            candidate_text = candidate.text if hasattr(candidate, "text") else candidate
            steps = trajectory_steps + [candidate_text]
            prompt = self._format_prm_prompt(question, steps)
            all_prompts.append(prompt)

        # Optional truncation (token-based, left-truncation not implemented here)
        max_tokens = 4000
        truncated_prompts: List[str] = []
        for prompt in all_prompts:
            ids = self.prm_tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > max_tokens:
                log.warning(f"Truncating PRM prompt from {len(ids)} to {max_tokens} tokens")
                ids = ids[:max_tokens]
                prompt = self.prm_tokenizer.decode(ids)
            truncated_prompts.append(prompt)

        num_traj_steps = len(trajectory_steps)
        log.info(
            f"Math-Shepherd PRM scoring {len(truncated_prompts)} candidates "
            f"(trajectory has {num_traj_steps} steps)"
        )

        outputs = self.prm_model.reward(truncated_prompts, use_tqdm=True)

        # Parse outputs:
        # With STEP pooling + returned_token_ids=[plus_id, minus_id],
        # vLLM should return per-step vectors of length 2, but be defensive.
        all_last_scores: List[List[float]] = []
        for cand_idx, output in enumerate(outputs):
            all_step_rewards: List[float] = []
            if hasattr(output, "outputs") and hasattr(output.outputs, "data"):
                data = output.outputs.data
                raw = data.tolist() if hasattr(data, "tolist") else (data if isinstance(data, list) else [data])

                for s in raw:
                    if isinstance(s, (list, tuple)) and len(s) >= 1:
                        # s is expected like [p_plus, p_minus] (or [logit_plus, logit_minus] depending on activation)
                        all_step_rewards.append(float(s[0]))
                    else:
                        all_step_rewards.append(float(s))

            last_score = all_step_rewards[-1] if all_step_rewards else 0.0

            traj_scores = all_step_rewards[:num_traj_steps] if num_traj_steps > 0 else []
            cand_scores = all_step_rewards[num_traj_steps:] if len(all_step_rewards) > num_traj_steps else all_step_rewards
            log.info(
                f"Candidate {cand_idx}: traj_scores={[f'{x:.3f}' for x in traj_scores]}, "
                f"cand_score={[f'{x:.3f}' for x in cand_scores]}"
            )

            all_last_scores.append([last_score])

        return all_last_scores

    def score_trajectory(
        self, chat: List[Dict[str, str]], trajectory: List, **kwargs
    ) -> List[float]:
        """Score a complete trajectory and return scores for ALL steps in a single forward pass."""
        if not trajectory:
            return []

        question = self._extract_question(chat)
        step_texts = [step.text if hasattr(step, "text") else str(step) for step in trajectory]
        prompt = self._format_prm_prompt(question, step_texts)

        max_tokens = 4000
        ids = self.prm_tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_tokens:
            log.warning(f"Truncating PRM prompt from {len(ids)} to {max_tokens} tokens")
            ids = ids[:max_tokens]
            prompt = self.prm_tokenizer.decode(ids)

        if self.use_vllm:
            outputs = self.prm_model.reward([prompt], use_tqdm=True)
            return self._parse_vllm_step_scores(outputs[0], expected_steps=len(step_texts))
        else:
            return self._score_prompt_hf(prompt)

    def score_trajectories_batch(
        self,
        chats: List[List[Dict[str, str]]],
        trajectories: List[List],
        sample_ids: List[int] = None,
        trajectory_ids: List[int] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Score multiple trajectories in a single batched vLLM call."""
        if not trajectories:
            return []

        if not self.use_vllm:
            log.info("HuggingFace backend: falling back to sequential scoring")
            return [
                self.score_trajectory(chat, traj, **kwargs)
                for chat, traj in zip(chats, trajectories)
            ]

        # Build prompts + metadata
        all_prompts: List[str] = []
        meta: List[Tuple[int, int, Optional[int], Optional[int]]] = []

        for i, (chat, traj) in enumerate(zip(chats, trajectories)):
            if not traj:
                all_prompts.append("")
                meta.append((i, 0, sample_ids[i] if sample_ids else None, trajectory_ids[i] if trajectory_ids else i))
                continue

            question = self._extract_question(chat)
            step_texts = [s.text if hasattr(s, "text") else str(s) for s in traj]
            prompt = self._format_prm_prompt(question, step_texts)

            max_tokens = 4000
            ids = self.prm_tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > max_tokens:
                log.warning(f"Truncating PRM prompt {i} from {len(ids)} to {max_tokens} tokens")
                ids = ids[:max_tokens]
                prompt = self.prm_tokenizer.decode(ids)

            all_prompts.append(prompt)
            meta.append((i, len(step_texts), sample_ids[i] if sample_ids else None, trajectory_ids[i] if trajectory_ids else i))

        valid_idx = [i for i, p in enumerate(all_prompts) if p]
        valid_prompts = [all_prompts[i] for i in valid_idx]

        log.info(f"Math-Shepherd PRM batch scoring {len(valid_prompts)} trajectories in single call")
        outputs = self.prm_model.reward(valid_prompts, use_tqdm=True) if valid_prompts else []

        results: List[List[float]] = [[] for _ in trajectories]
        out_i = 0
        for i, (traj_i, n_steps, sample_id, traj_id) in enumerate(meta):
            if i not in valid_idx:
                results[traj_i] = []
                continue

            scores = self._parse_vllm_step_scores(outputs[out_i], expected_steps=n_steps)
            out_i += 1
            results[traj_i] = scores

            sample_str = f"Sample {sample_id}" if sample_id is not None else ""
            log.info(
                f"{sample_str} Traj {traj_id}: {n_steps} steps, scores={[f'{s:.3f}' for s in scores]}"
            )

        return results

    # ----------------------------
    # Formatting + parsing helpers
    # ----------------------------

    def _extract_question(self, chat: List[Dict[str, str]]) -> str:
        for msg in chat:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return chat[-1].get("content", "") if chat else ""

    def _ensure_step_tagged(self, step: str) -> str:
        """
        Ensure a step ends with the step tag token.
        We add a space before the tag to help tokenization.
        """
        s = (step or "").rstrip()
        # If it already ends with the tag (possibly with punctuation), don't double-tag.
        if s.endswith(self.STEP_TAG_TEXT):
            return s
        return f"{s} {self.STEP_TAG_TEXT}".rstrip()

    def _format_prm_prompt(self, question: str, step_texts: List[str]) -> str:
        """
        Math-Shepherd expects: question + step-by-step solution, where each step ends with 'ки'.
        We do not use chat templates here (keeps behavior closer to the model’s intended usage).
        """
        tagged_steps = [self._ensure_step_tagged(s) for s in step_texts]
        solution = "\n".join(tagged_steps)
        # Keep a space between question and solution to avoid token gluing
        return f"{question}\n{solution}"

    def _parse_vllm_step_scores(self, output: Any, expected_steps: int) -> List[float]:
        step_scores: List[float] = []
        if hasattr(output, "outputs") and hasattr(output.outputs, "data"):
            data = output.outputs.data
            raw = data.tolist() if hasattr(data, "tolist") else (data if isinstance(data, list) else [data])
            for s in raw:
                if isinstance(s, (list, tuple)) and len(s) >= 1:
                    step_scores.append(float(s[0]))  # p_plus (or corresponding first returned token)
                else:
                    step_scores.append(float(s))

        # Pad/truncate to expected length
        while len(step_scores) < expected_steps:
            step_scores.append(0.0)
        return step_scores[:expected_steps]

    # ----------------------------
    # HuggingFace scoring internals
    # ----------------------------

    def _score_single_candidate_hf(
        self, chat: List[Dict[str, str]], candidate: str
    ) -> List[float]:
        """
        HF behavior: split candidate into newline steps and score each step.
        """
        question = self._extract_question(chat)

        # Split into steps by newline; remove empties
        raw_steps = [s.strip() for s in (candidate or "").split("\n")]
        steps = [s for s in raw_steps if s]

        if not steps:
            log.debug(f"No steps extracted from candidate: {candidate[:50]}...")
            return [0.0]

        prompt = self._format_prm_prompt(question, steps)
        scores = self._score_prompt_hf(prompt)

        log.info(f"Math-Shepherd PRM rewards for {len(steps)} steps: {scores}")
        return scores if scores else [0.0]

    def _score_prompt_hf(self, prompt: str) -> List[float]:
        """
        Score a fully formatted prompt (question + tagged steps) in HF:
        Return P('+') at each position where token == step_tag_id ('ки').
        """
        if self.prm_model is None or self.prm_tokenizer is None:
            raise RuntimeError("HF model/tokenizer not initialized")

        assert self._step_tag_id is not None
        assert self._plus_id is not None
        assert self._minus_id is not None

        input_ids = self.prm_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(self.prm_model.device)

        with torch.no_grad():
            out = self.prm_model(input_ids=input_ids, use_cache=False)
            logits = out.logits  # [1, T, V]

        # Take logits for + and - only: [1, T, 2]
        two = torch.stack(
            [logits[..., self._plus_id], logits[..., self._minus_id]],
            dim=-1,
        )
        probs = F.softmax(two, dim=-1)[..., 0]  # P('+') -> [1, T]

        # Select positions where token == step_tag_id
        mask = (input_ids == self._step_tag_id)  # [1, T]
        step_scores = probs[mask].detach().float().cpu().tolist()

        return step_scores
