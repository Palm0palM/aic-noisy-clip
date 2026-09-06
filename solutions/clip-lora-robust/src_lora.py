"""LoRA: 在冻结的 Linear 层上注入低秩可训练旁路。"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包装一个冻结的 nn.Linear，叠加低秩增量 (alpha/r) * x @ A^T @ B^T。"""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        delta = self.drop(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scale


def apply_lora_to_clip_vision(vision_model, rank=8, alpha=16.0, dropout=0.05, targets=("q_proj", "v_proj")):
    """对 CLIP vision transformer 每层的指定线性层注入 LoRA，返回注入层数。

    targets 中的 q_proj/k_proj/v_proj/out_proj 定位到 self_attn，fc1/fc2 定位到 mlp。
    """
    count = 0
    for layer in vision_model.encoder.layers:
        for name in targets:
            if name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                parent = layer.self_attn
            elif name in ("fc1", "fc2"):
                parent = layer.mlp
            else:
                continue
            lin = getattr(parent, name, None)
            if isinstance(lin, nn.Linear):
                setattr(parent, name, LoRALinear(lin, rank, alpha, dropout))
                count += 1
    return count
