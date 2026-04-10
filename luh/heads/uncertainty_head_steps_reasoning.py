import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty_head_base import UncertaintyHeadBase

import logging


log = logging.getLogger()


class UncertaintyHeadStepReasoning(UncertaintyHeadBase):
    """
    Step-level uncertainty head optimized for sequential reasoning traces.

    Compared with UncertaintyHeadClaim:
    - runs the transformer encoder ONCE per sequence, not once per claim
    - assumes claims are subsequent contiguous spans
    - marks claim-end / separator positions with a learned embedding
    - predicts uncertainty from the hidden state at each claim end position

    Expected claims format:
        claims is a list of length B
        claims[i] is a tensor of shape [num_claims_i, T]
        each row is a binary mask for one claim span

    Structural assumptions within one sample:
        - each claim is a contiguous non-empty span
        - claims are ordered
        - claims are non-overlapping
        - claims are subsequent:
              start_{k+1} == end_k + 1
    """

    def __init__(
        self,
        feature_extractor,
        head_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        cfg=None,
        mask_future_tokens: bool = False,
    ):
        super().__init__(feature_extractor, cfg=cfg, model_type="step_reasoning")

        self.mask_future_tokens = mask_future_tokens
        self.feature_extractor = feature_extractor

        log.info(f"Feature size: {feature_extractor.feature_dim()}")

        self.proj = nn.Sequential(
            nn.Linear(feature_extractor.feature_dim(), head_dim * 2),
            nn.LayerNorm(head_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim * 2, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
        )

        # 0 = regular token, 1 = claim-end / separator token
        self.step_end_embedding = nn.Embedding(2, head_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=n_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_dim, 1),
        )

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total number of parameters {total_params}")

    def _extract_span_boundaries(self, entity_mask: torch.Tensor):
        """
        Convert binary claim masks [C, T] into span boundaries.

        Returns:
            starts: LongTensor [C_valid]
            ends: LongTensor [C_valid]
            valid_mask: BoolTensor [C]   # True for non-empty claims
        """
        assert entity_mask.ndim == 2, f"Expected entity_mask [C, T], got {entity_mask.shape}"

        C, T = entity_mask.shape
        if C == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=entity_mask.device)
            empty_bool = torch.empty(0, dtype=torch.bool, device=entity_mask.device)
            return empty_long, empty_long, empty_bool

        if entity_mask.dtype == torch.bool:
            mask = entity_mask
        else:
            unique_vals = torch.unique(entity_mask)
            assert torch.all((unique_vals == 0) | (unique_vals == 1)), (
                f"Claims must be binary 0/1 masks, got values {unique_vals.tolist()}"
            )
            mask = entity_mask.bool()

        token_counts = mask.sum(dim=1)
        valid_mask = token_counts > 0

        # No valid claims at all
        if not torch.any(valid_mask):
            empty_long = torch.empty(0, dtype=torch.long, device=entity_mask.device)
            return empty_long, empty_long, valid_mask

        valid_claims = mask[valid_mask]
        # valid_counts = token_counts[valid_mask]

        starts = torch.argmax(valid_claims.int(), dim=1)

        reversed_mask = torch.flip(valid_claims, dims=[1])
        ends = T - 1 - torch.argmax(reversed_mask.int(), dim=1)

        # expected_lengths = ends - starts + 1
        # assert torch.all(valid_counts == expected_lengths), (
        #     "Each non-empty claim must be a single contiguous span"
        # )
        #
        # if valid_claims.shape[0] > 1:
        #     assert torch.all(starts[1:] > starts[:-1]), (
        #         "Claims must be ordered by start position"
        #     )
        #     assert torch.all(starts[1:] == ends[:-1] + 1), (
        #         "Non-empty claims must be subsequent contiguous spans "
        #         "(each next claim must start exactly after previous ends)"
        #     )

        return starts.long(), ends.long(), valid_mask

    def _build_causal_step_mask(self, T: int, device: torch.device):
        """
        Build a standard causal attention mask:
        token i cannot attend to tokens j > i.

        Returns:
            Bool tensor [T, T], where True means "masked out".
        """
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        claims = llm_inputs["claims"]

        # [B, T, H_in] -> [B, T, H_head]
        features = self.proj(X)

        # True = padding
        src_key_padding_mask = (X_attn_mask == 0)

        results = []
        batch_size = len(claims)
        seq_len = features.shape[1]

        for i in range(batch_size):
            entity_mask = claims[i]

            assert entity_mask.ndim == 2, (
                f"claims[{i}] must have shape [num_claims_i, T], got {entity_mask.shape}"
            )
            assert entity_mask.shape[1] == seq_len, (
                f"claims[{i}] sequence length {entity_mask.shape[1]} does not match "
                f"feature sequence length {seq_len}"
            )

            num_claims = entity_mask.shape[0]
            if num_claims == 0:
                results.append(features.new_full((0, 1), -100.0))
                continue

            starts, ends, valid_mask = self._extract_span_boundaries(entity_mask)

            # Pre-fill all claims with ignore value
            sample_out = features.new_full((num_claims, 1), -100.0)

            # If all claims are empty, keep ignore-only output
            if valid_mask.sum() == 0:
                results.append(sample_out)
                continue

            # Build one token-level end-marker mask using only valid claims
            end_mask = torch.zeros(seq_len, dtype=torch.long, device=features.device)
            end_mask[ends] = 1

            seq_features = features[i] + self.step_end_embedding(end_mask)
            seq_features = seq_features.unsqueeze(0)
            seq_padding = src_key_padding_mask[i].unsqueeze(0)

            attn_mask = None
            if self.mask_future_tokens:
                attn_mask = self._build_causal_step_mask(seq_len, seq_features.device)

            seq_out = self.transformer_encoder(
                seq_features,
                mask=attn_mask,
                src_key_padding_mask=seq_padding,
            ).squeeze(0)

            end_representations = seq_out[ends]  # [num_valid_claims, H]
            valid_out = self.classifier(end_representations)  # [num_valid_claims, 1]

            # Put predictions back into original claim slots
            sample_out[valid_mask] = valid_out
            results.append(sample_out)

        max_entities_per_batch = max([o.shape[0] for o in results], default=1)
        padded_results = [
            F.pad(o, (0, 0, 0, max_entities_per_batch - o.shape[0]), value=-100)
            for o in results
        ]

        if len(padded_results) == 0:
            return torch.zeros(0, device=X.device, dtype=features.dtype)

        assert len(results) == batch_size, (
            f"Expected one output tensor per batch item, got {len(results)} for batch size {batch_size}"
        )

        return torch.stack(padded_results, dim=0)

    def forward_from_features(self, features, attention_mask, claims):
        """
        Use precomputed token features directly, bypassing self.feature_extractor.

        Args:
            features: Tensor [B, T, H]
            attention_mask: Tensor [B, T]
            claims: list of length B, each tensor [num_claims_i, T]

        Returns:
            Tensor [B, max_claims_in_batch, 1]
        """
        return self._compute_tensors(
            llm_inputs={"claims": claims},
            X=features,
            X_attn_mask=attention_mask,
        )