#!/usr/bin/env bash
# Bounded screening for the L2-SP pretrained anchor, fold A only.
#
#   * trains fold A from the official initialisation with the v30_oof_a recipe
#     plus (anchor_lambda/2)||theta-theta0||^2 on the backbone
#   * dumps fold-B probabilities for the candidate on the raw images and on the
#     pre-registered degraded view (JPEG q45 + Gaussian blur 1.0), and the same
#     degraded dump for the control so the pair is like-for-like
#   * the control's raw dump already exists (artifacts/fast_v30_oof_a_probs.npy)
#
# Decision rule fixed in advance (GPT, 2026-10-06):
#   raw +~0.5pp and no regression on degraded views -> advance to a full ladder
#   raw flat but degraded clearly better            -> tentative, needs a second
#                                                      held-out check
#   everything within ~0.1pp                        -> stop this recipe
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/data/aic-rematch/train
VIEWS="center:448:1.14,flip:448:1.14"

exec >> logs/anchor_a.log 2>&1
echo "[anchor] start $(date)"
avail=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
echo "[anchor] free ${avail} MB"
if [ "$avail" -lt 8000 ]; then
  echo "[anchor] REFUSING: less than 8 GB free"
  exit 1
fi

$PY scripts/verify_anchor_grad.py || exit 1
echo "[anchor] gradient checks passed $(date)"

if [ ! -f checkpoints/anchor_a/last.pt ]; then
  $PY -m aic_clip.train_ft --config configs/anchor_a/s1_384.yaml \
      --train-on-all --drop-indices artifacts/oof_a_drop.npy || exit 1
  echo "[anchor] train exit=$? $(date)"
fi

$PY scripts/dump_teacher_probs.py --checkpoint checkpoints/anchor_a/last.pt \
    --cache "$CACHE" --views "$VIEWS" --weights ema --decode-cap 0 \
    --fold-file artifacts/folds_2.json --eval-fold 1 --only-fold 1 \
    --output artifacts/anchor_a_probs.npy || exit 1
echo "[anchor] candidate raw dump done $(date)"

$PY scripts/dump_teacher_probs.py --checkpoint checkpoints/anchor_a/last.pt \
    --cache "$CACHE" --views "$VIEWS" --weights ema --decode-cap 0 \
    --degrade jpeg45_blur1 \
    --fold-file artifacts/folds_2.json --eval-fold 1 --only-fold 1 \
    --output artifacts/anchor_a_probs_degraded.npy || exit 1
echo "[anchor] candidate degraded dump done $(date)"

if [ ! -f artifacts/fast_v30_oof_a_probs_degraded.npy ]; then
  $PY scripts/dump_teacher_probs.py --checkpoint checkpoints/v30_oof_a/last.pt \
      --cache "$CACHE" --views "$VIEWS" --weights ema --decode-cap 0 \
      --degrade jpeg45_blur1 \
      --fold-file artifacts/folds_2.json --eval-fold 1 --only-fold 1 \
      --output artifacts/fast_v30_oof_a_probs_degraded.npy || exit 1
  echo "[anchor] control degraded dump done $(date)"
fi

$PY scripts/compare_fold_probs.py --fold 1 \
    --reference artifacts/fast_v30_oof_a_probs.npy \
    --candidate artifacts/anchor_a_probs.npy \
    --labels-name "anchor_a raw" \
    --degraded-reference artifacts/fast_v30_oof_a_probs_degraded.npy \
    --degraded-candidate artifacts/anchor_a_probs_degraded.npy \
    --output artifacts/anchor_a_report.json || exit 1

echo "[anchor] done $(date)"
