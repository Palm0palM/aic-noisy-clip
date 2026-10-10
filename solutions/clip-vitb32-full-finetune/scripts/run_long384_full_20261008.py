"""V31 production refit with only the first-stage schedule expanded to 24 epochs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT=Path('/root/autodl-tmp/projects/AIC-Robust-CLIP-复赛')
TAG='long384_full_20261008'
OUT=Path('/root/aic-experiments')/TAG
SNAP=OUT/'frozen'
PRED=ROOT/'artifacts'/TAG
VIEWS='center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,center:512:1.28,center:512:1.4,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14'


def now():
    return datetime.datetime.now().astimezone().isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(suffix,obj):
    path=ROOT/f'artifacts/{TAG}_{suffix}.json'
    assert not path.exists(),path
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')


def status(phase,**details):
    path=ROOT/f'artifacts/{TAG}_status.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(phase=phase,updated_at=now(),**details),indent=2))
    temp.replace(path)
    print('STATUS',phase,json.dumps(details),flush=True)


def run(args,env):
    args=[str(x) for x in args]
    print('RUN',now(),json.dumps(args),flush=True)
    subprocess.run([sys.executable]+args,cwd=ROOT,env=env,check=True)


def main():
    import fcntl
    import numpy as np
    import yaml
    os.chdir(ROOT)
    lock=open(f'artifacts/{TAG}.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and not PRED.exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'],text=True).strip()
    assert shutil.disk_usage('/root').free>6*1024**3
    assert shutil.disk_usage('/root/autodl-tmp').free>3*1024**3
    assert json.loads(Path('artifacts/long384_a_decision.json').read_text())['screen_pass']
    expected={
        'src/aic_clip/train_ft.py':'9c8f56fb67ed0f78aaa7eca7e721c65e15779a78328b248502827c3071f146a2',
        'src/aic_clip/infer_ft.py':'4525c84f7af63f4512e196b7ac7e02c66aa5d47c389fe773dd42c1c71c72df1a',
        'scripts/subject_view_probs.py':'e8ad040694a6a0c2784074e778d88855562682c2bd76e63e70f636ce79251b67',
        'scripts/subject_crop_utils.py':'6b3c3a87968f295d482e42dab73ade597046a85a294568c1eb4ad2c03400d3f2',
        'scripts/combine_subject_arm.py':'19d31187e21f0bdb840d4f5ea8ba7bed136d99a15d09c24fbdf177005e78256f'}
    assert all(len(h)==64 for h in expected.values())
    assert all(sha(p)==h for p,h in expected.items()),'Previously audited source changed'
    drop=np.load('artifacts/dedup_drop.npy')
    weights=np.load('artifacts/oof_weights_v2.npy')
    assert len(weights)==148643 and len(drop)==len(np.unique(drop))==4071
    keep=np.ones(len(weights),bool);keep[drop]=False
    assert keep.sum()==144572
    assert np.isfinite(weights).all()
    assert np.array_equal(np.unique(weights[keep]),[.5,1.])
    assert (weights[keep]==.5).sum()==2481
    # Copy code, configs and small training manifests; later shared-file edits cannot affect this run.
    shutil.copytree(ROOT/'src/aic_clip',SNAP/'src/aic_clip',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (SNAP/'scripts').mkdir()
    for f in ['subject_view_probs.py','subject_crop_utils.py','combine_subject_arm.py']:
        shutil.copy2(ROOT/'scripts'/f,SNAP/'scripts'/f)
    shutil.copy2(Path(__file__),SNAP/'scripts/run_full.py')
    (SNAP/'artifacts').mkdir()
    for f in ['train_manifest.csv','split95_seed20260926.json','dedup_drop.npy','oof_weights_v2.npy']:
        shutil.copy2(ROOT/'artifacts'/f,SNAP/'artifacts'/f)
    (SNAP/'configs').mkdir()
    stage_defs=[('s1_384',24),('s2_448',8),('s3_576',6)]
    original_configs={}
    for stage,epochs in stage_defs:
        original=ROOT/f'configs/v31/{stage}.yaml'
        baseline=yaml.safe_load(original.read_text())
        original_configs[str(original)]=sha(original)
        assert baseline['train'].get('seed',20260926)==20260926
        assert baseline['train']['sample_weight_mode']=='target'
        assert baseline['train']['llrd_gamma']==.8 and baseline['train']['early_stop_patience']==0
        assert baseline['model']['revision']=='c237dc49a33fc61debc9276459120b7eac67e7ef'
        assert baseline['augment']['cutmix']==1. and baseline['augment']['mix_prob']==.8
        assert baseline['train']['epochs']==(14 if stage=='s1_384' else epochs)
        cfg=json.loads(json.dumps(baseline))
        cfg['paths']['project_root']=str(SNAP)
        cfg['train']['epochs']=epochs
        cfg['train']['seed']=20260926  # Explicitly preserve V31's effective default seed.
        cfg['train']['output_dir']=str(OUT/stage)
        check=json.loads(json.dumps(cfg))
        check['paths']=baseline['paths']
        check['train']['epochs']=baseline['train']['epochs']
        check['train']['output_dir']=baseline['train']['output_dir']
        if 'seed' not in baseline['train']: del check['train']['seed']
        assert check==baseline
        (SNAP/f'configs/{stage}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    frozen={str(p):sha(p) for p in SNAP.rglob('*') if p.is_file()}
    def verify():
        assert all(sha(p)==h for p,h in frozen.items()),'Frozen experiment input changed'
    env=os.environ.copy()
    env.update(PYTHONPATH=str(SNAP/'src'),HF_HOME='/root/autodl-tmp/huggingface',
               TORCH_HOME='/root/autodl-tmp/torch-cache',TMPDIR='/root/autodl-tmp/tmp',
               HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    run(['-c','import aic_clip.train_ft as m; from pathlib import Path; '
         f'assert Path(m.__file__).resolve()==Path({str(SNAP/"src/aic_clip/train_ft.py")!r}); '
         'print("ISOLATED TRAINER",m.__file__)'],env)
    protocol=dict(experiment=TAG,frozen_at=now(),basis='long384_a passed: raw+0.8255pp, degraded+0.9388pp',
        baseline='V31 full-data production recipe; only384 stage14->24',schedule=[24,8,6],
        seed=20260926,train_count=144572,dropped_count=4071,half_weight_count=2481,
        initialization='Official CLIP revision; fresh full-data training, fixed last raw stage handoffs',
        selection='No early stopping, no checkpoint selection using test; fixed final epoch5 EMA',
        test_data_in_training=False,test_labels_on_server=False,
        inference=dict(views=VIEWS.split(','),base_batch_size=192,subject_batch_size=16,
                       weights='ema',probabilities='float32',subject_size=512,tau=.6,margin=.15,
                       min_view_area=.02,max_view_area=.95,expected_images=37444),
        primary_prediction=str(PRED/'arms/B_subject.csv'),diagnostic_prediction=str(PRED/'arms/B.csv'),
        scoring='Local aggregate evaluation only after checkpoint, recipe and predictions frozen; no posthoc arm selection',
        submission='No leaderboard upload; no competition delivery unless user goal met',
        original_config_hashes=original_configs,frozen_inputs=frozen)
    save('preregistered',protocol)
    checkpoints={}
    previous=None
    for stage,epochs in stage_defs:
        verify()
        status('training',stage=stage,epochs=epochs)
        args=['-m','aic_clip.train_ft','--config',SNAP/f'configs/{stage}.yaml',
              '--train-on-all','--drop-indices',SNAP/'artifacts/dedup_drop.npy',
              '--sample-weight-file',SNAP/'artifacts/oof_weights_v2.npy']
        if previous is not None:args+=['--initialize',previous,'--init-weights','raw']
        run(args,env)
        previous=OUT/stage/'last.pt'
        rows=json.loads((OUT/stage/'history.json').read_text())
        assert len(rows)==epochs and rows[-1]['epoch']==epochs-1
        checkpoints[stage]=dict(path=str(previous),sha256=sha(previous),epochs=epochs,completed_at=now())
        save(stage+'_checkpoint',checkpoints[stage])
    verify()
    import torch
    payload=torch.load(previous,map_location='cpu',weights_only=False)
    assert payload['epoch']==5 and 'ema' in payload and payload['config']['train']['seed']==20260926
    assert all(torch.isfinite(v).all() for v in payload['ema'].values())
    del payload
    save('checkpoints',checkpoints)
    PRED.mkdir()
    freeze=dict(protocol=protocol,final_checkpoint=checkpoints['s3_576'],inference_started_at=now())
    (PRED/'frozen_protocol.json').write_text(json.dumps(freeze,indent=2))
    status('inference_base10',checkpoint=str(previous))
    run(['-m','aic_clip.infer_ft','--checkpoint',previous,
         '--test-dir','/root/autodl-tmp/data/aic-rematch/test','--views',VIEWS,
         '--weights','ema','--batch-size','192','--expected-count','37444',
         '--output-dir',PRED,'--save-probs','--probs-dtype','float32'],env)
    status('inference_subject')
    run([SNAP/'scripts/subject_view_probs.py','--project-root',ROOT,'--checkpoint',previous,
         '--test-dir','/root/autodl-tmp/data/aic-rematch/test','--weights','ema','--size','512',
         '--tau','.6','--margin','.15','--min-view-area','.02','--max-view-area','.95',
         '--batch-size','16','--output-dir',PRED],env)
    verify()
    assert sha(previous)==checkpoints['s3_576']['sha256']
    run([SNAP/'scripts/combine_subject_arm.py','--views-npz',PRED/'test_view_probs.npz',
         '--subject-npz',PRED/'subject_view_probs.npz','--out-dir',PRED/'arms'],env)
    base=json.loads((PRED/'report.json').read_text())
    subject=json.loads((PRED/'subject_view_report.json').read_text())
    combined=json.loads((PRED/'arms/combine_report.json').read_text())
    assert base['rows']==subject['images']==combined['images']==37444
    assert base['checkpoint_sha256']==subject['checkpoint_sha256']==sha(previous)
    assert base['weights']==subject['weights']=='ema' and combined['checkpoint_identity_verified']
    result=dict(completed_at=now(),checkpoint=checkpoints['s3_576'],images=37444,
                primary_prediction=protocol['primary_prediction'],
                primary_prediction_sha256=sha(PRED/'arms/B_subject.csv'),
                diagnostic_prediction_sha256=sha(PRED/'arms/B.csv'),
                final_score='pending local aggregate evaluation; no labels on server')
    (PRED/'completion.json').write_text(json.dumps(result,indent=2))
    with tarfile.open(PRED/'review_bundle.tar.gz','w:gz') as archive:
        for f in ['frozen_protocol.json','completion.json','report.json','subject_view_report.json',
                  'arms/combine_report.json','arms/B.csv','arms/B_subject.csv']:
            archive.add(PRED/f,arcname=f)
    save('completion',result)
    status('predictions_complete_pending_local_score',**result)
    print('DONE',json.dumps(result),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise
