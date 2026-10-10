"""Freeze the round-2 continuation dose experiment before any outcome is seen."""
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

LOSS = Path('/root/autodl-tmp/aic-experiments/loss_transfer_20261009')
BATCH = Path('/root/autodl-tmp/aic-experiments/batch_test_20261009')
OUT = Path('/root/autodl-tmp/aic-experiments/round2_20261010')
HERE = Path(__file__).resolve().parent
LRB = BATCH/'loss_robust/last.pt'
LRB_SHA = 'e5f831411b7cadb56e131032cd35c0780fd2ef44705a61cb4de25abe4b7489e0'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert not (OUT/'protocol.json').exists(), OUT
    assert shutil.disk_usage('/root/autodl-tmp').free > 5*2**30
    prior = json.loads((LOSS/'protocol.json').read_text())
    SNAP = OUT/'frozen'
    copies = []
    for p in (LOSS/'frozen/src').rglob('*.py'):
        copies.append((p, SNAP/'src'/p.relative_to(LOSS/'frozen/src')))
    for name in ['subject_view_probs.py', 'subject_crop_utils.py']:
        copies.append((LOSS/'frozen/scripts'/name, SNAP/'scripts'/name))
    for name in ['train_manifest.csv', 'dedup_drop.npy', 'oof_weights_v2.npy',
                 'oof_a_drop.npy', 'folds_2.json']:
        copies.append((LOSS/'frozen/artifacts'/name, SNAP/'artifacts'/name))
    copies.append((LOSS/'frozen/configs/continuation.json', SNAP/'configs/continuation.json'))
    for source, dest in copies:
        assert sha(source) == prior['frozen'][str(source)], source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    for p in sorted(HERE.glob('*.py')):
        shutil.copy2(p, SNAP/'scripts'/p.name)
    assert sha(LRB) == LRB_SHA
    refs = {}
    for p in [BATCH/'loss_robust/history.json', BATCH/'loss_control/history.json',
              BATCH/'predictions/loss_robust/completion.json']:
        refs[str(p)] = sha(p)
    frozen = {str(p): sha(p) for p in SNAP.rglob('*') if p.is_file()}
    protocol = dict(
        experiment=OUT.name,
        frozen_at=datetime.datetime.now().astimezone().isoformat(),
        initial_checkpoints={'full': str(LRB)},
        initial_checkpoint_hashes={'full': LRB_SHA},
        initial_state_key='model (raw finals of loss_robust; no ema key present)',
        arms=['r2_robust', 'd6_robust'],
        arm_definitions={'r2_robust': 'robust3', 'd6_robust': 'robust6'},
        schedules={'robust3': [[.8, 0, .2], [.35, .35, .30], [.15, .70, .15]],
                   'robust6': [[.8, 0, .2], [.35, .35, .30], [.15, .70, .15]]*2},
        coefficient_order=['CE', 'LA-CE', 'GCE'],
        head_epochs={'robust3': [2], 'robust6': [2, 5]},
        common='same as the winning loss_robust recipe: full data, seed, sampler, augmentation, '
               'effective prior, reliability weights, batch80, fresh optimizer, fixed last raw; '
               'below-zero-if-worse candidates are dropped, nothing is tuned on test',
        bar='loss_robust soft50 = aa5e6e700370a7657e17ee998f7b3f1c36b1fb0381a5080859fd8af584d857f8 '
            '(29,266 / 78.1594); adopt a candidate only if clearly above',
        reference_results=refs,
        test_data_used=False,
        frozen=frozen)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (OUT/'logs').mkdir(exist_ok=True)
    print('BOOTSTRAP OK', len(frozen), 'frozen files,', len(refs), 'reference hashes')


if __name__ == '__main__':
    main()
