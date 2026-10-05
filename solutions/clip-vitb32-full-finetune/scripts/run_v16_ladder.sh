#!/usr/bin/env bash
# V16: the production ladder, rebuilt with the two changes the proxy protocol
# actually validated today.
#
#   1. more epochs per stage (14 / 8 / 6 instead of 10 / 6 / 4). Evidence: B0 +10
#      extra epochs at a fixed resolution lifted Group-IID by 1.39pp and
#      Nuisance-OOD by 0.94pp, both with p=1.000 - the training budget, not the
#      tricks, was the bottleneck.
#   2. layer-wise LR decay, gamma 0.8. Evidence: b3_llrd was the only arm whose
#      shifted-domain gain (+0.71pp) clearly exceeded its in-domain gain (+0.52pp);
#      every other arm gained 0.24-0.32pp on OOD.
#
# Training pool: all 148,643 decodable official images minus the 4,071 samples in
# duplicate groups with contradictory labels. No test image, label or statistic is
# touched. Progress: logs/v16_ladder.log
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface
export TORCH_HOME=/root/autodl-tmp/torch-cache
export TMPDIR=/root/autodl-tmp/tmp
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
DROP="--train-on-all --drop-indices artifacts/dedup_drop.npy"

exec >> logs/v16_ladder.log 2>&1
echo "[v16] start $(date)"

while pgrep -f "run_f_[s]eries.sh" > /dev/null; do sleep 60; done
echo "[v16] gpu free $(date)"

echo "[v16] stage 1: 384px x14 $(date)"
$PY -m aic_clip.train_ft --config configs/v16/s1_384.yaml $DROP

echo "[v16] stage 2: 448px x8 $(date)"
$PY -m aic_clip.train_ft --config configs/v16/s2_448.yaml $DROP \
    --initialize checkpoints/v16/s1_384/last.pt --init-weights raw

echo "[v16] stage 3: 576px x6 $(date)"
$PY -m aic_clip.train_ft --config configs/v16/s3_576.yaml $DROP \
    --initialize checkpoints/v16/s2_448/last.pt --init-weights raw \
    --snapshot-dir checkpoints/v16/s3_576/snapshots --snapshot-keep 3

echo "[v16] inference: 576px six views $(date)"
$PY -m aic_clip.infer_ft \
    --checkpoint checkpoints/v16/s3_576/last.pt \
    --test-dir /root/autodl-tmp/data/aic-rematch/test \
    --views "center:576:1.0,flip:576:1.0,center:576:1.14,flip:576:1.14,center:576:1.28,center:576:1.4" \
    --weights raw \
    --output-dir artifacts/submission_v16

echo "[v16] done $(date)"
