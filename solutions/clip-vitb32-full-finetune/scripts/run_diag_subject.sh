#!/usr/bin/env bash
# Corrected, auditable fixed-checkpoint diagnosis on held-out training images.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp
export HF_HUB_OFFLINE=1
PY=/root/miniconda3/bin/python
OUTPUT_DIR=${OUTPUT_DIR:-artifacts/subject_v3_$(date +%Y%m%d_%H%M%S)}
mkdir -p logs
exec >> logs/diag_subject_v3.log 2>&1
echo "[subject-v3] start $(date) output=$OUTPUT_DIR"
$PY scripts/diag_subject_crop_v3.py --checkpoint checkpoints/v30_oof_a/last.pt \
  --images "${IMAGES:-6000}" --eval-fold 1 --output-dir "$OUTPUT_DIR"
echo "[subject-v3] completed $(date) output=$OUTPUT_DIR"
