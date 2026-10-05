#!/usr/bin/env bash
# Dump per-view probabilities at 512px for the two best weight files, so that
# spatial-coverage TTA candidates can be scored without re-running the model.
#
# All six current views are centre crops of a short-side resize: nothing covers
# the parts of a non-square image that a centre crop cuts off. This adds eight
# anchors, the four 1.14 five-crop corners and two whole-image padded views.
# Test images are only read here, never during training.
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
TESTDIR=/root/autodl-tmp/data/aic-rematch/test
V="center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,center:512:1.28,center:512:1.4"
V="$V,tl:512:1.0,tc:512:1.0,tr:512:1.0,ml:512:1.0,mr:512:1.0,bl:512:1.0,bc:512:1.0,br:512:1.0"
V="$V,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14,fullpad_edge:512:1.0,fullpad_gray:512:1.0"

exec >> logs/views512.log 2>&1
echo "[views] start $(date)"

run_one () {
  local tag="$1" ckpt="$2" weights="$3"
  if [ -f "artifacts/views512_${tag}/test_view_probs.npz" ]; then
    echo "[views] $tag already dumped"
  else
    echo "[views] $tag <- $ckpt ($weights) $(date)"
    $PY -m aic_clip.infer_ft --checkpoint "$ckpt" --test-dir "$TESTDIR" \
        --views "$V" --weights "$weights" --batch-size 160 \
        --output-dir "artifacts/views512_${tag}" --save-probs || return 1
  fi
  $PY scripts/combine_view_probs.py --npz "artifacts/views512_${tag}/test_view_probs.npz" \
      --out-dir "artifacts/views512_${tag}/recipes"
  echo "[views] $tag done $(date)"
}

run_one v16ema checkpoints/v16/s3_576/last.pt ema
run_one swa3 checkpoints/v16/s3_576/swa3.pt raw

echo "[views] done $(date)"
