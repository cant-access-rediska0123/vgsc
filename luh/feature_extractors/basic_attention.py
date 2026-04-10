import torch

from .feature_extractor_base import FeatureExtractorBase
from .lookback_lens import process_attentions
from .utils import get_layer_nums
import logging
import time

log = logging.getLogger(__name__)


def move_to_cpu(obj):
    """Recursively move all tensors in a nested structure to CPU."""
    # Case 1: If it's a tensor, move to CPU
    if torch.is_tensor(obj):
        return obj.cpu()

    # Case 2: If it's a list or tuple, recursively process each element
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(x) for x in obj)

    # Case 3: If it's a dict, recursively process each value
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}

    # Case 4: If it's anything else (e.g., int, float, string), just return as is
    else:
        return obj


class FeatureExtractorBasicAttention(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums, attn_history_sz, pool, **kwargs):
        """layer_nums = {all, [-1, -2, -5, ...]}"""

        self.pool = pool
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)
        self._head_num = orig_base_model.config.num_attention_heads
        self._input_size = (
            self._head_num * len(self._layer_nums)
            if not pool
            else self._head_num
        )
        self._attn_history_sz = attn_history_sz

    def feature_dim(self):
        return self._input_size * self._attn_history_sz

    def __call__(self, llm_inputs, llm_outputs):
        # tm = time.time()
        attentions_all: list[tuple] = process_attentions(
            llm_outputs.attentions, llm_inputs['attention_mask'],
            is_training=not hasattr(llm_outputs, "sequences"),
            stack_layers=True,
        )
        # print(f'Processed attentions in {round(time.time() - tm, 2)} sec')
        # tm = time.time()
        batch_size, seq_len = llm_inputs['attention_mask'].shape
        all_features = []  # [seq_len](batch_sz, attn_hist, heads, layers)
        for i in range(len(attentions_all)):
            cur_attn = attentions_all[i][self._layer_nums]  # (layers, batch_sz, num_heads, 1, prev_seq_len)
            attn_index = torch.arange(
                cur_attn.shape[-1] - 1,
                cur_attn.shape[-1] - self._attn_history_sz - 1,
                -1,
                device=cur_attn.device,
            )
            cur_features = cur_attn[:, :, :, 0, attn_index.clamp(0)].permute(1, 3, 2, 0)  # (batch_sz, attn_hist, heads, layers)
            if cur_attn.shape[-1] - self._attn_history_sz - 1 < 0:
                cur_features[:, attn_index < 0, :, :] = 0.0
            all_features.append(cur_features)
        # print(f'Done all iterations in {round(time.time() - tm, 2)} sec')
        # tm = time.time()
        all_features = torch.stack(all_features, dim=1)  # (batch_sz, seq_len, attn_hist, heads, layers)
        if self.pool:
            all_features = torch.amax(all_features, dim=-1)  # (batch_sz, seq_len, attn_hist, heads)
        # print(f'Done stacking stuff in {round(time.time() - tm, 2)} sec')

        return all_features.reshape(batch_size, len(attentions_all), -1)

    def input_size(self):
        return self._input_size * self._attn_history_sz

    def output_attention(self):
        return True


def load_extractor(config, base_model):
    return FeatureExtractorBasicAttention(base_model, **config)