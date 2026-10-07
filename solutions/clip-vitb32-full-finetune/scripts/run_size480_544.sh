#!/usr/bin/env bash
# Inference-size check requested by the reviewer: V31 EMA weights, the exact B
# view composition and ratios, only the size changes (480 and 544 vs the
# existing 512). No other change, no cross-scanning.
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
TESTDIR=/root/autodl-tmp/data/aic-rematch/test

exec >> logs/size480_544.log 2>&1
echo "[size] start $(date)"
for S in 480 544; do
  V="center:${S}:1.0,flip:${S}:1.0,center:${S}:1.14,flip:${S}:1.14,center:${S}:1.28,center:${S}:1.4"
  V="$V,tl:${S}:1.14,tr:${S}:1.14,bl:${S}:1.14,br:${S}:1.14"
  $PY -m aic_clip.infer_ft --checkpoint checkpoints/v31/s3_576/last.pt --test-dir "$TESTDIR" \
      --views "$V" --weights ema --output-dir "artifacts/size_${S}_B"
done
echo "[size] done $(date)"
