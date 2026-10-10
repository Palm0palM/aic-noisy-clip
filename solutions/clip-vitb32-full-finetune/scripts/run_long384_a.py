"""Expand only the first-stage cosine schedule: 384x24 -> 448x8 -> 576x6."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/projects/AIC-Robust-CLIP-复赛')
TAG = 'long384_a'
OUT = Path('/root/aic-experiments') / TAG
CONFIG_DIR = Path('configs') / TAG
STAGES = [('s1_384',24,'configs/cutmix_seed20261007/control.yaml','train_ft'),
          ('s2_448',8,'configs/llrd_tail_a/c08_s2_448.yaml','train_llrd_tail'),
          ('s3_576',6,'configs/llrd_tail_a/c08_s3_576.yaml','train_llrd_tail')]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(suffix, obj):
    p = Path(f'artifacts/{TAG}_{suffix}.json')
    assert not p.exists(), p
    p.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def run(args):
    print('RUN', datetime.datetime.now().astimezone().isoformat(), json.dumps(args), flush=True)
    subprocess.run([sys.executable]+args, check=True)


def main():
    import fcntl
    import numpy as np
    import yaml
    os.chdir(ROOT)
    lock = open(f'artifacts/{TAG}.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and not CONFIG_DIR.exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'],text=True).strip()
    assert shutil.disk_usage('/root').free > 6*1024**3
    assert shutil.disk_usage('/root/autodl-tmp').free > 2*1024**3
    expected = {'src/aic_clip/train_ft.py':'9c8f56fb67ed0f78aaa7eca7e721c65e15779a78328b248502827c3071f146a2',
                'src/aic_clip/train_llrd_tail.py':'3ac5afff6e39391570b40ba7d1ca8a6c14a864a4e5980869db2e2416c9ddf3ca',
                'scripts/dump_teacher_probs.py':'f38840bd3a47bd26cd45f9326382f0f4e6885ce4b109b0ebb7c0e56aba78dd3c',
                'scripts/compare_fold_probs.py':'636d4ea9438cd54d68d42d44741c15a2a5f8588e46cb9ab27bbb62160d30955e'}
    assert all(sha(p)==v for p,v in expected.items())
    folds=np.array(json.loads(Path('artifacts/folds_2.json').read_text())['fold'])
    kept=np.setdiff1d(np.arange(len(folds)),np.load('artifacts/oof_a_drop.npy'))
    assert len(kept)==72771 and (folds[kept]==0).all() and (folds==1).sum()==74135
    reference_protocol=json.loads(Path('artifacts/sam_tail_a_512_preregistered.json').read_text())
    control_path='/root/aic-experiments/llrd_tail_a/c08/s3_576/last.pt'
    assert sha(control_path)==reference_protocol['hashes'][control_path]
    reference_report=json.loads(Path('artifacts/sam_tail_a_512_tailoff_report.json').read_text())
    assert all(reference_report[v]['n']==74135 for v in ['raw','degraded'])
    CONFIG_DIR.mkdir()
    files=list(expected)+['scripts/run_long384_a.py','artifacts/train_manifest.csv',
                          'artifacts/folds_2.json','artifacts/oof_a_drop.npy',
                          'artifacts/split95_seed20260926.json',control_path,
                          'artifacts/sam_tail_a_512_control_raw.npy',
                          'artifacts/sam_tail_a_512_control_degraded.npy']
    for stage,epochs,source,module in STAGES:
        cfg=yaml.safe_load(Path(source).read_text())
        assert cfg['train']['seed']==20261007 and cfg['train']['llrd_gamma']==.8
        assert cfg['train']['early_stop_patience']==0
        if stage=='s1_384': assert cfg['train']['epochs']==14
        else: assert cfg['train']['epochs']==epochs and cfg['train']['cutmix_policy']=='apply_preserve_rng'
        baseline=json.loads(json.dumps(cfg))
        cfg['train']['epochs']=epochs
        cfg['train']['output_dir']=str(OUT/stage)
        check=json.loads(json.dumps(cfg))
        check['train']['epochs']=baseline['train']['epochs']
        check['train']['output_dir']=baseline['train']['output_dir']
        assert check==baseline
        path=CONFIG_DIR/f'{stage}.yaml'
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        files.extend([source,str(path)])
    hashes={p:sha(p) for p in files}
    save('preregistered',dict(experiment=TAG,started_at=datetime.datetime.now().astimezone().isoformat(),
         intervention='Only first-stage budget/cosine horizon 14->24; later stages unchanged 8/6',
         control='Completed CutMix-on 14/8/6 matched ladder, seed20261007, foldA; reused512 dumps',
         initialization='Official CLIP init; no reuse of a 14-epoch endpoint for the 24-epoch cosine',
         training_samples=72771,heldout_samples=74135,seed=20261007,
         stage_handoff='Fixed last raw weights; fresh optimizer/scheduler/EMA each stage',
         evaluation='Fixed final epoch5 EMA; heldout foldB center/flip512:1.14; raw and jpeg45_blur1',
         selection='No early stopping or best checkpoint selection; internal val diagnostic only',
         gate='raw>=0.3pp; raw macro>0; degraded>=0; raw/degraded tails>=-0.2pp',
         test_data_used=False,auto_production=False,hashes=hashes))
    print('PREFLIGHT PASS: 24/8/6; train72771 heldout74135; same seed, code and recipe',flush=True)
    os.environ.update(HF_HOME='/root/autodl-tmp/huggingface',TORCH_HOME='/root/autodl-tmp/torch-cache',
                      TMPDIR='/root/autodl-tmp/tmp',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    previous=None
    for stage,epochs,source,module in STAGES:
        assert all(sha(p)==v for p,v in hashes.items()),'Frozen inputs changed'
        args=['-m',f'aic_clip.{module}','--config',str(CONFIG_DIR/f'{stage}.yaml'),
              '--train-on-all','--drop-indices','artifacts/oof_a_drop.npy']
        if previous is not None: args+=['--initialize',str(previous),'--init-weights','raw']
        run(args)
        previous=OUT/stage/'last.pt'
        h=json.loads((OUT/stage/'history.json').read_text());rows=h['epochs'] if isinstance(h,dict) else h
        assert len(rows)==epochs and rows[-1]['epoch']==epochs-1
        save(stage+'_checkpoint',dict(path=str(previous),sha256=sha(previous),epochs=epochs,
                                      completed_at=datetime.datetime.now().astimezone().isoformat()))
        print('STAGE DONE',stage,flush=True)
    for view in ['raw','degraded']:
        run(['scripts/dump_teacher_probs.py','--checkpoint',str(previous),
             '--cache','/root/autodl-tmp/data/aic-rematch/train',
             '--views','center:512:1.14,flip:512:1.14','--weights','ema','--decode-cap','0',
             '--fold-file','artifacts/folds_2.json','--eval-fold','1','--only-fold','1',
             '--output',f'artifacts/{TAG}_{view}.npy']+
            (['--degrade','jpeg45_blur1'] if view=='degraded' else []))
    run(['scripts/compare_fold_probs.py','--fold','1',
         '--reference','artifacts/sam_tail_a_512_control_raw.npy','--candidate',f'artifacts/{TAG}_raw.npy',
         '--degraded-reference','artifacts/sam_tail_a_512_control_degraded.npy',
         '--degraded-candidate',f'artifacts/{TAG}_degraded.npy',
         '--labels-name','384x24 vs384x14 complete ladder; fixed512',
         '--output',f'artifacts/{TAG}_report.json'])
    r=json.loads(Path(f'artifacts/{TAG}_report.json').read_text())
    assert all(r[v]['n']==74135 for v in r)
    delta={v:{k:100*(r[v]['candidate'][k]-r[v]['reference'][k])
              for k in ['accuracy','macro_accuracy','tail_accuracy']} for v in r}
    passed=(delta['raw']['accuracy']>=.3 and delta['raw']['macro_accuracy']>0
            and delta['degraded']['accuracy']>=0 and all(delta[v]['tail_accuracy']>=-.2 for v in delta))
    result=dict(deltas_pp=delta,screen_pass=passed,auto_production=False,
                completed_at=datetime.datetime.now().astimezone().isoformat())
    assert all(sha(p)==v for p,v in hashes.items()),'Frozen inputs changed'
    save('decision',result)
    print('DONE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
