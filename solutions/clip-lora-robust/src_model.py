"""模型封装：CLIP ViT-B/32 视觉编码器(冻结) + LoRA + 可学习原型分类头。"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from .lora import apply_lora_to_clip_vision


def _interpolate_pos_encoding(vision_model, img_size: int):
    """将 CLIP vision 的 patch 位置嵌入从 7x7 双三次插值到目标网格（class 位不变）。"""
    emb = vision_model.embeddings
    patch = vision_model.config.patch_size
    gs_new = img_size // patch
    n_new = gs_new * gs_new + 1
    pe = emb.position_embedding.weight.detach()  # (50, 768)
    n_old, dim = pe.shape
    # 同步分辨率元数据：embeddings 前向会校验输入边长，config 亦保持一致
    emb.image_size = img_size
    vision_model.config.image_size = img_size
    if n_old == n_new:
        return
    gs_old = int((n_old - 1) ** 0.5)
    cls_pe, patch_pe = pe[:1], pe[1:]
    patch_pe = patch_pe.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(patch_pe, size=(gs_new, gs_new), mode="bicubic",
                             align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(n_new - 1, dim)
    emb.position_embedding = nn.Embedding(n_new, dim)
    emb.position_embedding.weight = nn.Parameter(torch.cat([cls_pe, patch_pe], 0))
    emb.register_buffer("position_ids", torch.arange(n_new).unsqueeze(0), persistent=False)


class CLIPProtoClassifier(nn.Module):
    """
    特征: CLIP vision tower -> visual_projection -> L2 归一化 (512 维)
    分类: logits = exp(logit_scale) * (f @ normalize(W)^T)
    """

    def __init__(
        self,
        num_classes: int,
        clip_name: str = "openai/clip-vit-base-patch32",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets=("q_proj", "v_proj"),
        train_ln: bool = True,
        logit_scale_init: float = math.log(25.0),
        img_size: int = 224,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        del self.clip.text_model  # 纯视觉分类，释放文本塔显存
        if img_size != self.clip.config.vision_config.image_size:
            _interpolate_pos_encoding(self.clip.vision_model, img_size)
            print(f"[model] pos-embed interpolated to {img_size}px "
                  f"({(img_size // 32) ** 2 + 1} tokens)")
        for p in self.clip.parameters():
            p.requires_grad = False

        n_lora = apply_lora_to_clip_vision(
            self.clip.vision_model, lora_rank, lora_alpha, lora_dropout, lora_targets
        )
        print(f"[model] LoRA injected into {n_lora} linear layers "
              f"(rank={lora_rank}, alpha={lora_alpha}, targets={lora_targets})")

        if train_ln:
            for m in self.clip.vision_model.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

        feat_dim = self.clip.config.projection_dim
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))

    def features(self, pixel_values):
        out = self.clip.get_image_features(pixel_values=pixel_values)
        if torch.is_tensor(out):          # transformers v4: 直接返回张量
            f = out
        elif hasattr(out, "pooler_output"):  # transformers v5: 投影后写入 pooler_output
            f = out.pooler_output
        else:                             # tuple 形式
            f = out[0]
        return F.normalize(f, dim=-1)

    def forward(self, pixel_values):
        f = self.features(pixel_values)
        w = F.normalize(self.prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (f @ w.t()), f

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_reference_encoder(model: CLIPProtoClassifier):
    """
    深拷贝当前(未训练的) vision tower + projection 作为冻结参考编码器。
    因 LoRA 的 B 初始为 0、LN 初始为原始权重, 此时前向等价于 CLIP 零样本特征。
    """
    ref_vision = copy.deepcopy(model.clip.vision_model)
    ref_proj = copy.deepcopy(model.clip.visual_projection)
    for p in list(ref_vision.parameters()) + list(ref_proj.parameters()):
        p.requires_grad = False
    ref_vision.eval()
    ref_proj.eval()

    class RefEncoder(nn.Module):
        def __init__(self, vision, proj):
            super().__init__()
            self.vision = vision
            self.proj = proj

        @torch.no_grad()
        def forward(self, pixel_values):
            out = self.vision(pixel_values=pixel_values)
            f = self.proj(out.pooler_output)
            return F.normalize(f, dim=-1)

    return RefEncoder(ref_vision, ref_proj)
