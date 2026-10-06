#!/usr/bin/env bash
# Bounded screening for the corrected ELR regulariser, fold A only.
#
#   * trains fold A from the official initialisation with the v30_oof_a recipe
#     plus the fixed ELR term (lambda 3.0, beta 0.7)
#   * dumps the weak-view probabilities for fold B only (--only-fold 1), the
#     same protocol, cache and views used for the control dump
#     artifacts/fast_v30_oof_a_probs.npy, so the two agreements are comparable
#
# Decision rule fixed in advance: >= ~0.5pp improvement in fold-B agreement over
# the control advances to a full ladder; anything smaller ends the arm.
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/data/aic-rematch/train
VIEWS="center:448:1.14,flip:448:1.14"

exec >> logs/elr_a.log 2>&1
echo "[elr] start $(date)"
avail=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
echo "[elr] free ${avail} MB"
if [ "$avail" -lt 8000 ]; then
  echo "[elr] REFUSING: less than 8 GB free"
  exit 1
fi

$PY scripts/verify_elr_grad.py || exit 1
echo "[elr] gradient check passed $(date)"

if [ ! -f checkpoints/elr_a/last.pt ]; then
  $PY -m aic_clip.train_ft --config configs/elr_a/s1_384.yaml \
      --train-on-all --drop-indices artifacts/oof_a_drop.npy || exit 1
  echo "[elr] train exit=$? $(date)"
fi

$PY scripts/dump_teacher_probs.py --checkpoint checkpoints/elr_a/last.pt \
    --cache "$CACHE" --views "$VIEWS" --weights ema --decode-cap 0 \
    --fold-file artifacts/folds_2.json --eval-fold 1 --only-fold 1 \
    --output artifacts/elr_a_probs.npy || exit 1
echo "[elr] dump exit=$? $(date)"

echo "[elr] control (v30_oof_a, same protocol): 0.7181 fold-B agreement"
echo "[elr] done $(date)"
