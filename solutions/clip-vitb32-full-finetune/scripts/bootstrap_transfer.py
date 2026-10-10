"""Freeze the paired continuation experiment before accessing its outcomes."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import yaml

OLD=Path('/root/autodl-tmp/aic-experiments/llrd_g09_20261008/frozen')
PROD=Path('/root/aic-experiments/long384_full_20261008/frozen')
OUT=Path('/root/autodl-tmp/aic-experiments/loss_transfer_20261009')
SNAP=OUT/'frozen'
ROOT=Path('/root/autodl-tmp/projects/AIC-Robust-CLIP-复赛')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not OUT.exists()
    assert shutil.disk_usage('/root/autodl-tmp').free>10*2**30
    old_protocol=json.loads((OLD.parent/'protocol.json').read_text())
    for p in (OLD/'src').rglob('*.py'):
        assert sha(p)==old_protocol['frozen'][str(p)]
    shutil.copytree(OLD/'src',SNAP/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (SNAP/'scripts').mkdir()
    for name in ['fold_eval.py','compare_fold_probs.py','dump_teacher_probs.py','subject_view_probs.py','subject_crop_utils.py']:
        p=OLD/'scripts'/name
        assert sha(p)==old_protocol['frozen'][str(p)]
        shutil.copy2(p,SNAP/'scripts'/name)
    shutil.copy2(OLD/'scripts/fold_eval.py',SNAP/'scripts/reference_fold_eval.py')
    # Same evaluation arithmetic; only checkpoint epoch/weight identity changes.
    q=SNAP/'scripts/fold_eval.py';s=q.read_text()
    replacements={"assert payload['epoch'] == 5":"assert payload['epoch'] == 2",
                  "and 'ema' in payload":"and 'model' in payload",
                  "model.load_state_dict(payload['ema'], strict=True)":"model.load_state_dict(payload['model'], strict=True)",
                  "weights='ema'":"weights='raw'"}
    for a,b in replacements.items():
        assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    q.write_text(s)
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p,SNAP/'scripts'/p.name)
    (SNAP/'artifacts').mkdir()
    for name in ['train_manifest.csv','oof_a_drop.npy','folds_2.json','fold_b_files.json']:
        p=OLD/'artifacts'/name
        assert sha(p)==old_protocol['frozen'][str(p)]
        shutil.copy2(p,SNAP/'artifacts'/name)
    assert sha(OLD/'artifacts/train_manifest.csv')==sha(PROD/'artifacts/train_manifest.csv')
    for name in ['dedup_drop.npy','oof_weights_v2.npy']:
        shutil.copy2(PROD/'artifacts'/name,SNAP/'artifacts'/name)
    cfg=yaml.safe_load((PROD/'configs/s3_576.yaml').read_text())
    cfg['paths']['project_root']=str(SNAP)
    cfg['train'].update(epochs=3,seed=20261009,lr_backbone=2e-6,lr_head=5e-5,
                        warmup_epochs=.2,min_lr_ratio=.02,llrd_gamma=.8)
    cfg['train'].pop('ema_decay',None)
    cfg['train'].pop('output_dir',None)
    cfg['data']['eval_size']=512
    cfg['data'].pop('split',None)
    (SNAP/'configs').mkdir()
    (SNAP/'configs/continuation.json').write_text(json.dumps(cfg,indent=2))
    ckpts=dict(fold='/root/aic-experiments/long384_a/s3_576/last.pt',
               full='/root/aic-experiments/long384_full_20261008/s3_576/last.pt')
    hashes=dict(fold='fb5fa19ef381a1a940cb04cf269ec7cb0cce10729644ab498e919c9fa54f4fbc',
                full='078c2e31ddcca4a13302f6c3299ab25965b3ca4f95ee3aec7c7c1d76c138c5c2')
    assert all(sha(ckpts[k])==hashes[k] for k in ckpts)
    extra=[OLD.parent/'eval/long/predictions.npz',OLD.parent/'eval/long/report.json']
    frozen={str(p):sha(p) for p in SNAP.rglob('*') if p.is_file()}
    frozen.update({str(p):sha(p) for p in extra})
    protocol=dict(experiment=OUT.name,frozen_at=datetime.datetime.now().astimezone().isoformat(),
        initial_checkpoints=ckpts,initial_checkpoint_hashes=hashes,frozen=frozen,
        train_rows=72771,heldout_rows=74135,full_train_rows=144572,
        initialization='same final epoch5 EMA; fresh optimizer; no checkpoint/epoch selection',
        common='three epochs 576, batch80, existing augmentation and sqrt-inverse sampler, same seed and LR; epoch3 head only',
        contrast='only objective coefficients differ; no QKV adapter, new filtering or head expansion',
        coefficients={'control':[[1,0,0]]*3,'robust':[[.8,0,.2],[.35,.35,.30],[.15,.70,.15]]},
        coefficient_order=['CE','LA-CE','GCE'],gce_q=.7,la_tau=1.,
        prior='normalized sum of actual sampler weight times reliability per class, then same 0.15 label smoothing',
        gce='expectation of per-class (1-p_c^q)/q under weighted mixed targets; not GCE of expected p',
        final_weights='fixed last epoch2 raw for both arms; no SWA/EMA averaging',
        selection='eligible if full10+subject delta>=0.3pp vs original, macro>0, tail>=-0.2pp, degraded2 acc>=0 and tail>=-0.2pp; select highest primary accuracy, tie control',
        paired_attribution='robust minus control is reported separately from gain against original',
        screen_timeout_hours=8,auto_full_refit_on_pass=True,
        full_predictions='predeclared subject and subject+fallback-letterbox, each raw and fixed soft50; all same single checkpoint',
        test_labels_on_server=False,test_images_used_in_training=False,
        adaptations_from_teammate=['Preserve our backbone/head and data cleaning','No KL term or QKV adapters','Same freeze schedule in CE control','Use effective smoothed supervision prior'])
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (ROOT/f'artifacts/{OUT.name}_preregistered.json').write_text(json.dumps(protocol,indent=2))
    print('FROZEN',OUT,flush=True)


if __name__=='__main__':main()
