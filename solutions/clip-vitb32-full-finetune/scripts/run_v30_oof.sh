#!/usr/bin/env bash
# V30: two out-of-fold models, one per duplicate-cluster-respecting fold.
#
# Each model trains on one fold only, so its predictions on the other fold are
# genuinely out of fold; those predictions are the reliability signal used later.
# The teacher weights are fixed in advance (last.pt + ema), never chosen on a
# validation set that contains the target fold, and the reported agreement is
# restricted to the held-out fold.
set -uo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/root/autodl-tmp/huggingface TORCH_HOME=/root/autodl-tmp/torch-cache TMPDIR=/root/autodl-tmp/tmp HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/data/aic-rematch/train_cache384
VIEWS="center:448:1.14,flip:448:1.14"

exec >> logs/v30_oof.log 2>&1
echo "[v30] start $(date)"

# merged drop lists: the other fold plus the duplicate drop list
$PY - <<'PYEOF'
import json
from pathlib import Path
import numpy as np
p = Path(".")
folds = np.array(json.loads((p / "artifacts/folds_2.json").read_text(encoding="utf-8"))["fold"], dtype=np.int64)
dedup = np.load(p / "artifacts/dedup_drop.npy")
for tag, fold_id in (("a", 0), ("b", 1)):
    other = np.nonzero(folds != fold_id)[0]
    merged = np.union1d(other, dedup)
    np.save(p / f"artifacts/oof_{tag}_drop.npy", merged)
    print(f"[v30] model {tag}: trains on fold {fold_id}, drops {len(merged)} rows")
PYEOF

for tag in a b; do
  case "$tag" in
    a) eval_fold=1 ;;
    b) eval_fold=0 ;;
  esac
  if [ ! -f "checkpoints/v30_oof_${tag}/last.pt" ]; then
    $PY -m aic_clip.train_ft --config "configs/v30/oof_${tag}.yaml" \
        --train-on-all --drop-indices "artifacts/oof_${tag}_drop.npy"
    echo "[v30] model ${tag} train exit=$? $(date)"
  else
    echo "[v30] model ${tag} already trained"
  fi
  $PY scripts/dump_teacher_probs.py --checkpoint "checkpoints/v30_oof_${tag}/last.pt" \
      --cache "$CACHE" --views "$VIEWS" --weights ema --decode-cap 0 \
      --fold-file artifacts/folds_2.json --eval-fold "$eval_fold" \
      --output "artifacts/oof_${tag}_probs.npy"
  echo "[v30] model ${tag} dump exit=$? $(date)"
done

md5sum checkpoints/v30_oof_a/last.pt checkpoints/v30_oof_b/last.pt > artifacts/v30_provenance.txt 2>&1
md5sum artifacts/folds_2.json artifacts/oof_a_probs_views.npz artifacts/oof_b_probs_views.npz >> artifacts/v30_provenance.txt 2>&1
echo "[v30] done $(date)"
