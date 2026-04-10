from .heads.saplma_head import SaplmaHead
from .heads.uncertainty_head import UncertaintyHead
from .heads.uncertainty_head_claim import UncertaintyHeadClaim
from .heads.uncertainty_head_steps_reasoning import UncertaintyHeadStepReasoning
from .heads.linear_head import LinearHead
from .heads.linear_head_claim import LinearHeadClaim
from .utils import load_feature_extractor

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import os
import torch


class CachedFeatureExtractorStub:
    """
    Minimal feature extractor used only to instantiate a head when training
    from cached hidden states. It is never used to compute features.
    """
    def __init__(self, feature_dim: int):
        self._feature_dim = feature_dim

    def feature_dim(self):
        return self._feature_dim


class AutoUncertaintyHead:
    DEFAULT_MODEL_TYPE = "luh"

    MODEL_MAPPING = {
        "saplma": SaplmaHead,
        DEFAULT_MODEL_TYPE: UncertaintyHead,
        "claim": UncertaintyHeadClaim,
        "step_reasoning": UncertaintyHeadStepReasoning,
        "linear": LinearHead,
        "linear_claim": LinearHeadClaim,
    }

    @staticmethod
    def _resolve_cfg_path(
        pretrained_path: str,
        revision: str = "main",
        use_auth_token: str = None,
    ):
        if os.path.isdir(pretrained_path):
            print(f"Loading uncertainty head from {pretrained_path}")
            return os.path.join(pretrained_path, "config.yaml")
        else:
            print(f"Loading uncertainty head from huggingface {pretrained_path}")
            return hf_hub_download(
                repo_id=pretrained_path,
                filename="config.yaml",
                revision=revision,
                use_auth_token=use_auth_token,
            )

    @staticmethod
    def _load_state_dict(
        pretrained_path: str,
        revision: str = "main",
        use_auth_token: str = None,
    ):
        candidates = ["pytorch_model.bin", "model.safetensors", "ue_layer.pth", "weights.pth"]

        for filename in candidates:
            try:
                if os.path.isdir(pretrained_path):
                    path = os.path.join(pretrained_path, filename)
                    if not os.path.exists(path):
                        continue
                else:
                    path = hf_hub_download(
                        repo_id=pretrained_path,
                        filename=filename,
                        revision=revision,
                        use_auth_token=use_auth_token,
                    )

                if filename.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    return load_file(path)
                return torch.load(path, map_location="cpu")
            except Exception:
                continue

        raise FileNotFoundError(
            f"Could not find a supported weights file in {pretrained_path}. "
            f"Tried: {candidates}"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        base_model=None,
        revision: str = "main",
        use_auth_token: str = None,
        feature_dim: int = None,
    ):
        """
        Backward compatible:
          - old usage still works: from_pretrained(path, base_model=...)
          - new cached-HS usage also works: from_pretrained(path, base_model=None, feature_dim=...)
        """
        cfg_path = cls._resolve_cfg_path(pretrained_path, revision, use_auth_token)
        cfg = OmegaConf.load(cfg_path)

        model_class = (
            cls.MODEL_MAPPING[cfg.head_type.lower()]
            if hasattr(cfg, "head_type")
            else cls.MODEL_MAPPING[cls.DEFAULT_MODEL_TYPE]
        )

        # Old path: keep exact previous behavior.
        if base_model is not None:
            return model_class.from_pretrained(
                pretrained_path, base_model, revision, use_auth_token
            )

        # New cached-HS path: instantiate without base model, then load weights.
        if feature_dim is None:
            raise ValueError(
                "feature_dim must be provided when base_model=None."
            )

        head = cls.from_config(cfg, base_model=None, feature_dim=feature_dim)
        state_dict = cls._load_state_dict(pretrained_path, revision, use_auth_token)
        missing, unexpected = head.load_state_dict(state_dict, strict=False)

        if len(missing) > 0 or len(unexpected) > 0:
            print(f"load_state_dict missing={missing}, unexpected={unexpected}")

        return head

    @classmethod
    def from_config(cls, config, base_model=None, feature_dim: int = None):
        """
        Backward compatible:
          - old usage still works: from_config(config, base_model=...)
          - cached-HS usage: from_config(config, base_model=None, feature_dim=...)
        """
        uq_head_type = cls.MODEL_MAPPING[config.head_type]

        if base_model is None:
            if feature_dim is None:
                raise ValueError(
                    "feature_dim must be provided when base_model=None."
                )
            feature_extractor = CachedFeatureExtractorStub(feature_dim)
        else:
            feature_extractor = load_feature_extractor(
                config.feature_extractor, base_model
            )

        ue_head_cfg = dict() if config.uncertainty_head is None else config.uncertainty_head
        uq_head = uq_head_type(
            feature_extractor,
            cfg=config,
            **ue_head_cfg,
        )

        return uq_head