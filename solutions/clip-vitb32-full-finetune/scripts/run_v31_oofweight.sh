#!/usr/bin/env bash
# V31: V16's ladder plus out-of-fold reliability weighting.
#
# The single addition is a per-sample weight: the 2,481 images where both weak
# views of the fold model that never saw them confidently predicted the same
# alternative class (margin >= 0.5), and which are not already in the duplicate
# drop list, train at half weight. The weight is applied to the target before
# mixup, so a mixed pair carries lambda*w_i and (1-lambda)*w_j.
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
DROP="--train-on-all --drop-indices artifacts/dedup_drop.npy"
WEIGHTS="--sample-weight-file artifacts/oof_weights_v2.npy"
TESTDIR=/root/autodl-tmp/data/aic-rematch/test
VIEWS6="center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,center:512:1.28,center:512:1.4"
VIEWS10="$VIEWS6,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14"

exec >> logs/v31_oofweight.log 2>&1
echo "[v31] start $(date)"
avail=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
echo "[v31] free ${avail} MB"

$PY -m aic_clip.train_ft --config configs/v31/s1_384.yaml $DROP $WEIGHTS
echo "[v31] stage1 exit=$? $(date)"
$PY -m aic_clip.train_ft --config configs/v31/s2_448.yaml $DROP $WEIGHTS \
    --initialize checkpoints/v31/s1_384/last.pt --init-weights raw
echo "[v31] stage2 exit=$? $(date)"
$PY -m aic_clip.train_ft --config configs/v31/s3_576.yaml $DROP $WEIGHTS \
    --initialize checkpoints/v31/s2_448/last.pt --init-weights raw \
    --snapshot-dir checkpoints/v31/s3_576/snapshots --snapshot-keep 3
echo "[v31] stage3 exit=$? $(date)"

$PY -m aic_clip.infer_ft --checkpoint checkpoints/v31/s3_576/last.pt --test-dir "$TESTDIR" \
    --views "$VIEWS6" --weights ema --output-dir artifacts/submission_v31_ema_base6
$PY -m aic_clip.infer_ft --checkpoint checkpoints/v31/s3_576/last.pt --test-dir "$TESTDIR" \
    --views "$VIEWS10" --weights ema --output-dir artifacts/submission_v31_ema_B
echo "[v31] done $(date)"
