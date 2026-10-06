#!/usr/bin/env bash
# V35 screening: mild augmentation on fold A only, then the same evaluations as
# its control (the fold-A out-of-fold teacher), so the comparison is controlled:
#   * held-out fold B agreement via a manifest-wide probability dump
#   * test-set inference with the six-view and B recipes
# Failure of any stage stops the script and is recorded.
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/data/aic-rematch/train
TESTDIR=/root/autodl-tmp/data/aic-rematch/test
VIEWS6="center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,center:512:1.28,center:512:1.4"
VIEWS10="$VIEWS6,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14"
WEAK="center:448:1.14,flip:448:1.14"

exec >> logs/v35_mildaub.log 2>&1
echo "[v35] start $(date)"

# last.pt is written every epoch, so its existence does NOT mean the run
# finished. Only a completed history (14 epochs) may be treated as done;
# otherwise retrain from scratch rather than adopting a half-trained model.
EPOCHS_DONE=$(grep -c '"epoch":' checkpoints/v35_mildaub/history.json 2>/dev/null || echo 0)
echo "[v35] completed epochs in history: ${EPOCHS_DONE}/14"
if [ "${EPOCHS_DONE}" -lt 14 ]; then
  $PY -m aic_clip.train_ft --config configs/v35/mild.yaml \
      --train-on-all --drop-indices artifacts/oof_a_drop.npy
  echo "[v35] train exit=$? $(date)"
else
  echo "[v35] training already complete, skipping"
fi

for tag in v35_mildaub v30_oof_a; do
  $PY scripts/dump_teacher_probs.py --checkpoint "checkpoints/${tag}/last.pt" \
      --cache "$CACHE" --views "$WEAK" --weights ema --decode-cap 0 \
      --fold-file artifacts/folds_2.json --eval-fold 1 \
      --output "artifacts/${tag}_probs.npy"
  echo "[v35] dump ${tag} exit=$? $(date)"
  $PY -m aic_clip.infer_ft --checkpoint "checkpoints/${tag}/last.pt" --test-dir "$TESTDIR" \
      --views "$VIEWS6" --weights ema --output-dir "artifacts/submission_${tag}_base6"
  $PY -m aic_clip.infer_ft --checkpoint "checkpoints/${tag}/last.pt" --test-dir "$TESTDIR" \
      --views "$VIEWS10" --weights ema --output-dir "artifacts/submission_${tag}_B"
  echo "[v35] test inference ${tag} exit=$? $(date)"
done

echo "[v35] done $(date)"
