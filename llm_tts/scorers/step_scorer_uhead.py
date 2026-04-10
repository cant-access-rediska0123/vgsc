import gc
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoConfig

from llm_tts.generators.base import StepCandidate

from .step_scorer_reward_base import StepScorerRewardBase

try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd().parent))
    from luh import AutoUncertaintyHead
except Exception as e:
    print(f"Couldn't import AutoUncertaintyHead: {e}")
    print(traceback.format_exc())
    AutoUncertaintyHead = None

try:
    from speculators.data_generation.vllm_hidden_states_generator import (
        VllmHiddenStatesGenerator,
    )
except Exception as e:  # pragma: no cover
    print(f"Couldn't import VllmHiddenStatesGenerator: {e}")
    print(traceback.format_exc())
    VllmHiddenStatesGenerator = None

log = logging.getLogger(__name__)


class StepScorerUHead(StepScorerRewardBase):
    """
    UHead-based step scorer.

    Design:
      - reuses the same prompt formatting as the generation-time VLLMStepGenerator
      - does NOT insert synthetic reward tokens
      - uses exact StepCandidate.token_ids for every generated step
      - scores each step from the hidden state at its final token position
      - extracts hidden states with VllmHiddenStatesGenerator on separate GPU(s)
      - runs the uncertainty head on a dedicated scoring GPU
    """

    def __init__(
        self,
        uhead_path: str,
        step_generator: Any,
        uhead_device: str,
        model_name_or_path: str,
        feature_layers: Optional[Sequence[int]] = None,
        batch_size: int = 32,
        hs_device: Optional[Union[str, Sequence[int], int]] = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        hf_token: Optional[str] = None,
        hf_cache: Optional[str] = None,
        hs_generator_kwargs: Optional[Dict[str, Any]] = None,
        progress_bar: bool = True,
    ):
        if AutoUncertaintyHead is None:
            raise ImportError("StepScorerUHead requires the `luh` package.")
        if VllmHiddenStatesGenerator is None:
            raise ImportError(
                "StepScorerUHead requires "
                "speculators.data_generation.vllm_hidden_states_generator.VllmHiddenStatesGenerator"
            )

        self.uhead_path = uhead_path
        self.step_generator = step_generator
        self.tokenizer = step_generator.tokenizer
        self.uhead_device = torch.device(
            uhead_device if torch.cuda.is_available() else "cpu"
        )
        self.model_name_or_path = model_name_or_path
        self.batch_size = int(batch_size)
        self.hs_device = hs_device
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = int(max_model_len)
        self.hf_token = hf_token
        self.hf_cache = hf_cache
        self.hs_generator_kwargs = dict(hs_generator_kwargs or {})
        self.progress_bar = bool(progress_bar)

        self.uhead = None
        self.hs_generator = None
        self.base_model_config = None
        self.feature_layers = list(feature_layers) if feature_layers is not None else None
        self.resolved_vllm_layer_ids: List[int] = []
        self.feature_dim: Optional[int] = None
        self._hs_visible_devices: List[int] = []
        self._hs_build_kwargs: Dict[str, Any] = {}

        self._prefix_cache: Dict[Tuple[Tuple[str, str], ...], List[int]] = {}

        self.prepare_model()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def prepare_model(self) -> None:
        self.base_model_config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            token=self.hf_token,
            cache_dir=self.hf_cache,
        )

        num_hidden_layers = self._get_num_hidden_layers(self.base_model_config)
        hidden_size = self._get_hidden_size(self.base_model_config)

        if self.feature_layers is None:
            self.feature_layers = self._infer_feature_layers_from_uhead_config(self.uhead_path)

        self.resolved_vllm_layer_ids = self._resolve_vllm_layer_ids(
            self.feature_layers,
            num_hidden_layers=num_hidden_layers,
        )
        self.feature_dim = hidden_size * len(self.resolved_vllm_layer_ids)

        log.info(
            "Loading UHead from %s | feature_layers=%s | resolved_vllm_layers=%s | feature_dim=%s",
            self.uhead_path,
            self.feature_layers,
            self.resolved_vllm_layer_ids,
            self.feature_dim,
        )

        self.uhead = AutoUncertaintyHead.from_pretrained(
            self.uhead_path,
            base_model=None,
            feature_dim=self.feature_dim,
        ).to(self.uhead_device)
        self.uhead.eval()

        hs_kwargs = {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
        }
        hs_kwargs.update(self.hs_generator_kwargs)

        self._hs_visible_devices = self._normalize_hs_devices(self.hs_device)
        if "tensor_parallel_size" not in hs_kwargs:
            hs_kwargs["tensor_parallel_size"] = len(self._hs_visible_devices)
        self._hs_build_kwargs = hs_kwargs

        self.hs_generator = self._build_hidden_state_generator()

    def _build_hidden_state_generator(self):
        log.info(
            "Initializing hidden-state generator on CUDA_VISIBLE_DEVICES=%s with tensor_parallel_size=%s",
            ",".join(map(str, self._hs_visible_devices)),
            self._hs_build_kwargs["tensor_parallel_size"],
        )
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self._hs_visible_devices))
        try:
            return VllmHiddenStatesGenerator(
                layer_ids=self.resolved_vllm_layer_ids,
                model_path=self.model_name_or_path,
                **self._hs_build_kwargs,
            )
        finally:
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def _rebuild_hs_generator(self) -> None:
        log.warning("Reinitializing hidden-state generator")
        try:
            if self.hs_generator is not None:
                del self.hs_generator
        finally:
            self.hs_generator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.hs_generator = self._build_hidden_state_generator()

    @staticmethod
    def _parse_single_gpu_token(token: Union[str, int]) -> int:
        if isinstance(token, int):
            return int(token)
        s = str(token).strip()
        if not s:
            raise ValueError("Encountered an empty GPU token while parsing hs_device")
        if s.startswith("cuda:"):
            s = s.split(":", 1)[1].strip()
        return int(s)

    @classmethod
    def _normalize_hs_devices(
        cls,
        hs_device: Optional[Union[str, Sequence[int], int]],
    ) -> List[int]:
        if hs_device is None:
            env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if env:
                raw_tokens = [x.strip() for x in env.split(",") if x.strip()]
                devices = [cls._parse_single_gpu_token(x) for x in raw_tokens]
                if devices:
                    return devices
            return [0]

        if isinstance(hs_device, int):
            return [int(hs_device)]

        if isinstance(hs_device, str):
            raw = hs_device.strip()
            if not raw:
                raise ValueError("hs_device was provided as an empty string")
            normalized = raw.replace(" ", ",")
            raw_tokens = [x.strip() for x in normalized.split(",") if x.strip()]
            devices = [cls._parse_single_gpu_token(x) for x in raw_tokens]
        else:
            devices = [cls._parse_single_gpu_token(x) for x in hs_device]

        if not devices:
            raise ValueError("hs_device did not resolve to any GPU ids")

        seen = set()
        ordered = []
        for d in devices:
            if d < 0:
                raise ValueError(f"hs_device entries must be non-negative GPU ids, got {d}")
            if d not in seen:
                seen.add(d)
                ordered.append(d)
        return ordered

    @staticmethod
    def _get_num_hidden_layers(config: Any) -> int:
        if hasattr(config, "num_hidden_layers"):
            return int(config.num_hidden_layers)
        if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
            return int(config.text_config.num_hidden_layers)
        raise ValueError("Could not infer num_hidden_layers from model config")

    @staticmethod
    def _get_hidden_size(config: Any) -> int:
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        raise ValueError("Could not infer hidden_size from model config")

    @staticmethod
    def _resolve_cfg_path(
        pretrained_path: str,
        revision: str = "main",
        use_auth_token: Optional[str] = None,
    ) -> str:
        if os.path.isdir(pretrained_path):
            return os.path.join(pretrained_path, "config.yaml")
        return hf_hub_download(
            repo_id=pretrained_path,
            filename="config.yaml",
            revision=revision,
            use_auth_token=use_auth_token,
        )

    def _infer_feature_layers_from_uhead_config(self, uhead_path: str) -> List[int]:
        cfg_path = self._resolve_cfg_path(
            uhead_path,
            revision="main",
            use_auth_token=self.hf_token,
        )
        cfg = OmegaConf.load(cfg_path)
        fe_cfg = getattr(cfg, "feature_extractor", None)
        if fe_cfg is None:
            raise ValueError("Could not find feature_extractor in UHead config")

        layers: List[int] = []

        def _collect(obj: Any) -> None:
            if obj is None:
                return
            if OmegaConf.is_list(obj) or isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)
                return
            if isinstance(obj, dict) or OmegaConf.is_dict(obj):
                name = obj.get("name", "")
                if name == "luh.feature_extractors.basic_hidden_states":
                    vals = obj.get("layer_nums", [])
                    if not (OmegaConf.is_list(vals) or isinstance(vals, (list, tuple))):
                        vals = [vals]
                    for x in vals:
                        layers.append(int(x))
                return

        _collect(fe_cfg)
        if not layers:
            raise ValueError(
                "Could not infer feature layers from UHead config. Please pass feature_layers explicitly."
            )
        return layers

    @staticmethod
    def _resolve_vllm_layer_ids(
        feature_layers: Sequence[int],
        num_hidden_layers: int,
    ) -> List[int]:
        resolved: List[int] = []
        for layer in feature_layers:
            layer = int(layer)
            if layer < 0:
                idx = num_hidden_layers + layer
            else:
                idx = layer - 1
            if not (0 <= idx < num_hidden_layers):
                raise ValueError(
                    f"Layer index {idx} out of bounds [0, {num_hidden_layers - 1}]"
                )
            resolved.append(idx)

        deduped: List[int] = []
        seen = set()
        for x in resolved:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return deduped

    # ------------------------------------------------------------------
    # Prompt / token construction
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze_request_key(request: List[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
        return tuple((str(m.get("role", "")), str(m.get("content", ""))) for m in request)

    def _build_scoring_prefix_text(self, request: List[Dict[str, str]]) -> str:
        if hasattr(self.step_generator, "build_scoring_prefix_text"):
            return self.step_generator.build_scoring_prefix_text(request)
        if getattr(self.step_generator, "thinking_mode", False):
            return self.step_generator._build_prompt(request, [])
        return self.step_generator._apply_chat_template(request, enable_thinking=False)

    def _build_scoring_prefix_token_ids(self, request: List[Dict[str, str]]) -> List[int]:
        key = self._freeze_request_key(request)
        cached = self._prefix_cache.get(key)
        if cached is not None:
            return list(cached)
        if hasattr(self.step_generator, "build_scoring_prefix_token_ids"):
            prefix_ids = list(
                map(int, self.step_generator.build_scoring_prefix_token_ids(request))
            )
        else:
            prefix_text = self._build_scoring_prefix_text(request)
            prefix_ids = list(map(int, self.tokenizer.encode(prefix_text)))
        self._prefix_cache[key] = list(prefix_ids)
        return prefix_ids

    def _as_step_candidate(self, obj: Any) -> StepCandidate:
        if isinstance(obj, StepCandidate):
            return obj
        if hasattr(obj, "token_ids") and hasattr(obj, "text"):
            return obj
        text = obj.text if hasattr(obj, "text") else str(obj)
        log.warning(
            "StepScorerUHead received a non-StepCandidate object; falling back to tokenizer.encode(text). "
            "For exact alignment, pass StepCandidate objects with token_ids."
        )
        token_ids = list(map(int, self.tokenizer.encode(text, add_special_tokens=False)))
        return StepCandidate(
            text=text,
            token_ids=token_ids,
            is_complete=True,
            is_trajectory_complete=False,
        )

    def _left_truncate_to_hs_limit(
        self,
        input_ids: List[int],
        claims: torch.Tensor,
        validity: List[bool],
    ) -> Tuple[List[int], torch.Tensor, List[bool]]:
        max_input_len = max(1, self.max_model_len - 1)
        if len(input_ids) <= max_input_len:
            return input_ids, claims, validity

        offset = len(input_ids) - max_input_len
        truncated_ids = input_ids[offset:]
        truncated_claims = claims[:, offset:].clone()

        new_validity: List[bool] = []
        for i, was_valid in enumerate(validity):
            still_valid = bool(was_valid and truncated_claims[i].sum().item() > 0)
            new_validity.append(still_valid)

        log.warning(
            "UHead scorer truncated left context from %s to %s tokens; some early step scores may become invalid.",
            len(input_ids),
            max_input_len,
        )
        return truncated_ids, truncated_claims, new_validity

    def _build_full_input_and_claims(
        self,
        request: List[Dict[str, str]],
        trajectory: List[Any],
    ) -> Tuple[List[int], torch.Tensor, List[bool]]:
        prefix_ids = self._build_scoring_prefix_token_ids(request)
        full_ids = list(prefix_ids)
        claim_end_positions: List[Optional[int]] = []
        validity: List[bool] = []

        for step in trajectory:
            step_cand = self._as_step_candidate(step)
            step_ids = list(map(int, getattr(step_cand, "token_ids", []) or []))
            if len(step_ids) == 0:
                claim_end_positions.append(None)
                validity.append(False)
                continue
            full_ids.extend(step_ids)
            claim_end_positions.append(len(full_ids) - 1)
            validity.append(True)

        seq_len = len(full_ids)
        num_claims = len(trajectory)
        claims = torch.zeros((num_claims, seq_len), dtype=torch.long)
        for i, pos in enumerate(claim_end_positions):
            if pos is not None:
                claims[i, pos] = 1

        return self._left_truncate_to_hs_limit(full_ids, claims, validity)

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _as_2d_float_cpu(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().cpu()
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        else:
            t = t.float()
        if t.ndim != 2:
            raise ValueError(f"Expected [T, H] or [1, T, H], got {tuple(t.shape)}")
        return t.contiguous()

    def _payload_seq_len(self, payload) -> int:
        if payload is None or "hidden_states" not in payload or not payload["hidden_states"]:
            return 0
        t0 = payload["hidden_states"][0]
        if t0.ndim == 3 and t0.shape[0] == 1:
            t0 = t0[0]
        return int(t0.shape[0])

    def _reset_hs_prefix_cache(self) -> bool:
        try:
            sch = getattr(self.hs_generator, "scheduler", None)
            if sch is None:
                return False
            if hasattr(sch, "reset_prefix_cache"):
                return bool(sch.reset_prefix_cache())
            kvm = getattr(sch, "kv_cache_manager", None)
            if kvm is not None and hasattr(kvm, "reset_prefix_cache"):
                return bool(kvm.reset_prefix_cache())
        except Exception as e:
            log.warning("Could not reset HS generator prefix cache: %s", e)
        return False

    def _payloads_to_padded_features(
        self,
        input_ids_list: List[List[int]],
        payloads: List[dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        per_sample_feats: List[torch.Tensor] = []
        lengths: List[int] = []
        feat_dim = None

        for seq_ids, payload in zip(input_ids_list, payloads):
            if payload is None or "hidden_states" not in payload or not payload["hidden_states"]:
                raise RuntimeError("Hidden-state generator returned an empty payload")

            hs_layers = [self._as_2d_float_cpu(h) for h in payload["hidden_states"]]
            seq_len = hs_layers[0].shape[0]
            for h in hs_layers[1:]:
                if h.shape[0] != seq_len:
                    raise RuntimeError("Returned hidden-state layers have different sequence lengths")

            sample = torch.cat(hs_layers, dim=-1)
            expected = len(seq_ids)
            if sample.shape[0] > expected:
                sample = sample[:expected]
            if sample.shape[0] < expected:
                raise RuntimeError(
                    f"Recovered hidden states are shorter than input sequence even after recovery: "
                    f"got={sample.shape[0]}, expected={expected}"
                )

            per_sample_feats.append(sample)
            lengths.append(sample.shape[0])
            feat_dim = sample.shape[-1]

        max_len = max(lengths, default=0)
        batch_size = len(per_sample_feats)
        feats = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, sample in enumerate(per_sample_feats):
            T = sample.shape[0]
            if T > 0:
                feats[i, :T] = sample
                attn_mask[i, :T] = 1

        return (
            feats.to(self.uhead_device, non_blocking=True),
            attn_mask.to(self.uhead_device, non_blocking=True),
        )

    @torch.no_grad()
    def _get_features(self, input_ids_list: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        payloads = None
        try:
            payloads = self.hs_generator.generate(input_ids_list)
        except Exception as e:
            log.warning("Initial hidden-state generation failed: %s", e)

        if payloads is None:
            payloads = [None] * len(input_ids_list)

        unresolved: List[int] = []
        for i, (seq_ids, payload) in enumerate(zip(input_ids_list, payloads)):
            got = self._payload_seq_len(payload)
            expected = len(seq_ids)
            if got < expected:
                unresolved.append(i)

        if unresolved:
            log.warning(
                "Hidden states shorter than inputs for %d/%d sequences; resetting HS prefix cache and recomputing individually.",
                len(unresolved),
                len(input_ids_list),
            )
            self._reset_hs_prefix_cache()

            for i in unresolved:
                seq_ids = input_ids_list[i]
                one_payload = None
                try:
                    out = self.hs_generator.generate([seq_ids])
                    one_payload = out[0] if out else None
                except Exception as e:
                    log.warning("Single-sequence HS recompute failed for idx=%d: %s", i, e)

                got = self._payload_seq_len(one_payload)
                if got < len(seq_ids):
                    log.warning(
                        "Single-sequence HS recompute still short for idx=%d (got=%d expected=%d); rebuilding generator and retrying once.",
                        i,
                        got,
                        len(seq_ids),
                    )
                    self._rebuild_hs_generator()
                    out = self.hs_generator.generate([seq_ids])
                    one_payload = out[0] if out else None
                    got = self._payload_seq_len(one_payload)

                if got < len(seq_ids):
                    raise RuntimeError(
                        "Recovered hidden states are shorter than input sequence: "
                        f"got={got}, expected={len(seq_ids)}"
                    )

                payloads[i] = one_payload

        return self._payloads_to_padded_features(input_ids_list, payloads)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_batch_from_trajectories(
        self,
        chats: List[List[Dict[str, str]]],
        trajectories: List[List[Any]],
    ) -> List[List[float]]:
        prepared: List[Tuple[List[int], torch.Tensor, List[bool], int]] = []
        for chat, trajectory in zip(chats, trajectories):
            full_ids, claims, validity = self._build_full_input_and_claims(chat, trajectory)
            prepared.append((full_ids, claims, validity, len(trajectory)))

        all_results: List[List[float]] = []
        batch_starts = list(range(0, len(prepared), self.batch_size))
        total_batches = len(batch_starts)
        iterator = batch_starts
        if self.progress_bar and total_batches > 1:
            iterator = tqdm(
                batch_starts,
                total=total_batches,
                desc=f"Scoring with UHead (batch_size={self.batch_size})",
                leave=False,
            )

        for start in iterator:
            chunk = prepared[start : start + self.batch_size]
            input_ids_list = [x[0] for x in chunk]
            raw_claims_list = [x[1] for x in chunk]
            validities = [x[2] for x in chunk]
            traj_lengths = [x[3] for x in chunk]

            features, attention_mask = self._get_features(input_ids_list)
            batch_seq_len = int(features.shape[1])
            claims_list = []
            for claims in raw_claims_list:
                if claims.shape[1] > batch_seq_len:
                    raise RuntimeError(
                        f"Claim mask length {claims.shape[1]} exceeds feature sequence length {batch_seq_len}"
                    )
                if claims.shape[1] < batch_seq_len:
                    pad = torch.zeros(
                        (claims.shape[0], batch_seq_len - claims.shape[1]),
                        dtype=claims.dtype,
                    )
                    claims = torch.cat([claims, pad], dim=1)
                claims_list.append(claims.to(self.uhead_device))

            logits = self.uhead.forward_from_features(
                features=features,
                attention_mask=attention_mask,
                claims=claims_list,
            ).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            for row_probs, validity, traj_len in zip(probs, validities, traj_lengths):
                out: List[float] = []
                for idx in range(traj_len):
                    is_valid = validity[idx] if idx < len(validity) else False
                    if not is_valid:
                        out.append(0.0)
                        continue
                    val = float(row_probs[idx])
                    if val < 0.0 or val > 1.0 or np.isnan(val):
                        out.append(0.0)
                    else:
                        out.append(val)
                all_results.append(out)

        return all_results

    def compute_claim_rewards(
        self,
        chat: List[Dict[str, str]],
        candidates: List[Any],
        **kwargs,
    ) -> List[List[float]]:
        if not candidates:
            return []

        trajectory = kwargs.get("trajectory", []) or []
        full_trajectories = [list(trajectory) + [self._as_step_candidate(c)] for c in candidates]
        chats = [chat for _ in full_trajectories]
        scores = self._score_batch_from_trajectories(chats, full_trajectories)
        return [[traj_scores[-1]] if traj_scores else [0.0] for traj_scores in scores]

    def score_trajectory(
        self,
        chat: List[Dict[str, str]],
        trajectory: List[Any],
        **kwargs,
    ) -> List[float]:
        del kwargs
        if not trajectory:
            return []
        return self._score_batch_from_trajectories([chat], [trajectory])[0]

    def score_trajectories_batch(
        self,
        chats: List[List[Dict[str, str]]],
        trajectories: List[List[Any]],
        sample_ids: Optional[List[int]] = None,
        trajectory_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> List[List[float]]:
        del sample_ids, trajectory_ids, kwargs
        if len(chats) != len(trajectories):
            raise ValueError(
                f"Expected len(chats)==len(trajectories), got {len(chats)} and {len(trajectories)}"
            )
        if not trajectories:
            return []
        return self._score_batch_from_trajectories(chats, trajectories)

    def cleanup(self) -> None:
        if self.uhead is not None:
            del self.uhead
            self.uhead = None
        if self.hs_generator is not None:
            del self.hs_generator
            self.hs_generator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
