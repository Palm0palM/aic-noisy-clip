"""噪声鲁棒组件：GMM 干净样本识别、EMA 教师、鲁棒损失。"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


def gmm_clean_prob(losses: np.ndarray) -> np.ndarray:
    """
    双组分 GMM 拟合 per-sample loss；记忆效应下低 loss 组分为干净样本。
    返回每个样本属于干净组分的后验概率。
    """
    x = losses.reshape(-1, 1).astype(np.float64)
    gmm = GaussianMixture(n_components=2, max_iter=50, tol=1e-3, reg_covar=1e-4,
                          n_init=2, random_state=0)
    gmm.fit(x)
    clean_comp = int(np.argmin(gmm.means_.flatten()))
    prob = gmm.predict_proba(x)[:, clean_comp]
    return prob


class EMA:
    """模型参数的指数滑动平均，作为自训练教师。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


def build_teacher(model: nn.Module, ema: EMA = None):
    """深拷贝当前模型为教师；若提供 ema 则载入影子权重。教师全程 eval、无梯度。"""
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    if ema is not None:
        ema.copy_to(teacher)
    teacher.eval()
    return teacher


def generalized_cross_entropy(logits, targets, q: float = 0.7):
    """GCE (Zhang & Sabuncu 2018)，对标签噪声鲁棒的 warm-up 损失。"""
    p = F.softmax(logits, dim=-1)
    p_y = p.gather(1, targets.unsqueeze(1)).clamp(min=1e-7).squeeze(1)
    return ((1.0 - p_y.pow(q)) / q).mean()


def soft_cross_entropy(logits, soft_targets):
    """学生 logits 与教师软标签之间的交叉熵。"""
    log_p = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_p).sum(dim=-1).mean()
