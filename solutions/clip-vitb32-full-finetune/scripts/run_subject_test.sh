#!/usr/bin/env bash
# Fixed single-model B versus B+subject comparison; no training or label input.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp
export HF_HUB_OFFLINE=1
PY=/root/miniconda3/bin/python
TESTDIR=/root/autodl-tmp/data/aic-rematch/test
CHECKPOINT=checkpoints/v31/s3_576/last.pt
OUTPUT_DIR=${OUTPUT_DIR:-artifacts/subject_precision_$(date +%Y%m%d_%H%M%S)}
BASE_BATCH_SIZE=${BASE_BATCH_SIZE:-192}
SUBJECT_BATCH_SIZE=${SUBJECT_BATCH_SIZE:-16}
VIEWS10="center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,center:512:1.28,center:512:1.4,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14"
mkdir -p logs
exec >> logs/subject_precision.log 2>&1
echo "[subject-precision] start $(date) output=$OUTPUT_DIR"
if [ -e "$OUTPUT_DIR" ]; then
  echo "Refusing to reuse an existing output directory: $OUTPUT_DIR"
  exit 1
fi
$PY -m aic_clip.infer_ft --checkpoint "$CHECKPOINT" --test-dir "$TESTDIR" \
  --views "$VIEWS10" --weights ema --batch-size "$BASE_BATCH_SIZE" \
  --output-dir "$OUTPUT_DIR" --save-probs --probs-dtype float32
$PY scripts/subject_view_probs.py --checkpoint "$CHECKPOINT" --test-dir "$TESTDIR" \
  --size 512 --weights ema --batch-size "$SUBJECT_BATCH_SIZE" --output-dir "$OUTPUT_DIR"
$PY scripts/combine_subject_arm.py --views-npz "$OUTPUT_DIR/test_view_probs.npz" \
  --subject-npz "$OUTPUT_DIR/subject_view_probs.npz" --out-dir "$OUTPUT_DIR/arms"
echo "[subject-precision] completed $(date) output=$OUTPUT_DIR"
