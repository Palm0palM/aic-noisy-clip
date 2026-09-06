"""通过 hf-mirror 下载 CLIP ViT-B/32 官方权重（openai/clip-vit-base-patch32）。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"  # xet 后端绕过镜像会 401，强制走普通 HTTP

from huggingface_hub import snapshot_download

p = snapshot_download("openai/clip-vit-base-patch32")
print("downloaded to:", p)

from transformers import CLIPModel
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
n = sum(x.numel() for x in m.parameters())
print(f"CLIPModel loaded OK, params={n / 1e6:.1f}M")
