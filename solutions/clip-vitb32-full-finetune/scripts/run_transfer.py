"""Paired screening, then automatic full-data continuation only after its fixed gate."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np

SNAP=Path(__file__).resolve().parents[1]
OUT=SNAP.parent
ROOT=Path('/root/autodl-tmp/projects/AIC-Robust-CLIP-复赛')
REFERENCE=Path('/root/autodl-tmp/aic-experiments/llrd_g09_20261008/eval/long')
DEADLINE=None


def now():return datetime.datetime.now().astimezone().isoformat()


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,d):
    t=Path(p).with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)


def status(phase,**kw):
    d=dict(phase=phase,updated_at=now(),pid=os.getpid(),output=str(OUT),**kw)
    write(OUT/'status.json',d);write(ROOT/f'artifacts/{OUT.name}_status.json',d)
    print('STATUS',json.dumps(d),flush=True)


def verify(p):
    for path,h in p['frozen'].items():assert sha(path)==h,path
    for key,path in p['initial_checkpoints'].items():assert sha(path)==p['initial_checkpoint_hashes'][key]


def run(name,args,env):
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU already in use'
    log=OUT/'logs'/f'{name}.log';assert not log.exists(),log
    with log.open('w') as f:
        print('RUN',name,[str(a) for a in args],flush=True)
        proc=subprocess.Popen([sys.executable]+[str(a) for a in args],cwd=SNAP,env=env,
                               stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=proc.wait(timeout=max(1,DEADLINE-time.monotonic()) if DEADLINE else None)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise RuntimeError('Preregistered execution time limit reached')
        if rc:raise RuntimeError(f'{name} failed with code {rc}; see {log}')


def train(arm,mode,env,p,smoke=False):
    name=('smoke_' if smoke else '')+mode+'_'+arm
    dest=OUT/name
    status('smoke_training' if smoke else 'training',arm=arm,mode=mode,log=str(OUT/'logs'/f'{name}.log'))
    args=[SNAP/'scripts/train_transfer.py','--arm',arm,'--mode',mode,
          '--checkpoint',p['initial_checkpoints'][mode],'--output',dest]
    if smoke:args+=['--smoke-steps','2']
    run(name,args,env)
    h=json.loads((dest/'history.json').read_text())
    assert len(h)==3 and h[-1]['epoch']==2 and h[-1]['head_only']
    assert json.loads((dest/'complete.json').read_text())['smoke']==smoke
    return dest/'last.pt'


def evaluate(arm,checkpoint,env):
    dest=OUT/'eval'/arm
    common=['--checkpoint',checkpoint,'--files-json',SNAP/'artifacts/fold_b_files.json','--output',dest]
    status('heldout_ten_views',arm=arm,log=str(OUT/'logs'/f'{arm}_base.log'))
    run(arm+'_base',[SNAP/'scripts/fold_eval.py','--mode','base']+common,env)
    status('heldout_subject',arm=arm,log=str(OUT/'logs'/f'{arm}_subject.log'))
    run(arm+'_subject',[SNAP/'scripts/subject_view_probs.py','--checkpoint',checkpoint,
        '--test-dir','/root/autodl-tmp/data/aic-rematch/train','--files-json',SNAP/'artifacts/fold_b_files.json',
        '--weights','raw','--size','512','--tau','.6','--margin','.15','--min-view-area','.02',
        '--max-view-area','.95','--batch-size','16','--output-dir',dest],env)
    run(arm+'_combine',[SNAP/'scripts/fold_eval.py','--mode','finish','--output',dest],env)
    status('heldout_degraded',arm=arm,log=str(OUT/'logs'/f'{arm}_degraded.log'))
    run(arm+'_degraded',[SNAP/'scripts/fold_eval.py','--mode','base','--degraded',
        '--checkpoint',checkpoint,'--files-json',SNAP/'artifacts/fold_b_files.json',
        '--output',OUT/'eval'/f'{arm}_degraded','--batch-size','128'],env)
    return dest


def compare(a,b,degraded=False):
    fn='base_summary.npz' if degraded else 'predictions.npz'
    x=np.load(a/fn);y=np.load(b/fn)
    for k in ['files','indices','labels']:assert np.array_equal(x[k],y[k]),k
    assert len(x['labels'])==74135
    ar=json.loads((a/('base_report.json' if degraded else 'report.json')).read_text())['metrics']
    br=json.loads((b/('base_report.json' if degraded else 'report.json')).read_text())['metrics']
    out={}
    for key in (['degraded_two'] if degraded else ['two','ten','subject']):
        pred='two_pred' if degraded else key+'_pred'
        u=x[pred]==x['labels'];v=y[pred]==y['labels'];w=int((~u&v).sum());z=int((u&~v).sum())
        out[key]=dict(reference=ar[key],candidate=br[key],wrong_to_correct=w,correct_to_wrong=z,net=w-z,
            deltas_pp={k:100*(br[key][k]-ar[key][k]) for k in ['accuracy','macro_accuracy','tail_accuracy']})
    return out


def main():
    global DEADLINE
    lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists()
    p=json.loads((OUT/'protocol.json').read_text());verify(p)
    (OUT/'logs').mkdir()
    env=os.environ.copy();env.update(PYTHONPATH=str(SNAP/'src'),HF_HOME='/root/autodl-tmp/huggingface',
        TORCH_HOME='/root/autodl-tmp/torch-cache',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',TMPDIR='/root/tmp')
    DEADLINE=time.monotonic()+8*3600
    status('verification');run('verify_transfer',[SNAP/'scripts/verify_transfer.py'],env)
    for arm in ['control','robust']:train(arm,'fold',env,p,smoke=True)
    sh=[json.loads((OUT/f'smoke_fold_{a}/history.json').read_text()) for a in ['control','robust']]
    assert [v['batch_indices_sha256'] for v in sh[0]]==[v['batch_indices_sha256'] for v in sh[1]],'Smoke sampling differs'
    checkpoints={}
    for arm in ['control','robust']:
        verify(p);checkpoints[arm]=train(arm,'fold',env,p)
    histories=[json.loads((OUT/f'fold_{a}/history.json').read_text()) for a in ['control','robust']]
    assert [v['batch_indices_sha256'] for v in histories[0]]==[v['batch_indices_sha256'] for v in histories[1]],'Paired sampling differs'
    status('reference_degraded')
    run('reference_degraded',[SNAP/'scripts/reference_fold_eval.py','--mode','base','--degraded',
        '--checkpoint',p['initial_checkpoints']['fold'],'--files-json',SNAP/'artifacts/fold_b_files.json',
        '--output',OUT/'eval/reference_degraded','--batch-size','128'],env)
    comparisons={};eligible=[]
    for arm,ckpt in checkpoints.items():
        verify(p);dest=evaluate(arm,ckpt,env)
        normal=compare(REFERENCE,dest);degraded=compare(OUT/'eval/reference_degraded',OUT/'eval'/f'{arm}_degraded',True)
        d=normal['subject']['deltas_pp'];dd=degraded['degraded_two']['deltas_pp']
        passed=d['accuracy']>=.3 and d['macro_accuracy']>0 and d['tail_accuracy']>=-.2 and dd['accuracy']>=0 and dd['tail_accuracy']>=-.2
        comparisons[arm]=dict(full_views=normal,degraded=degraded,screen_pass=bool(passed),checkpoint_sha256=sha(ckpt))
        if passed:eligible.append(arm)
    selected=max(eligible,key=lambda a:comparisons[a]['full_views']['subject']['candidate']['accuracy']) if eligible else None
    decision=dict(completed_at=now(),comparisons=comparisons,
                  robust_vs_control=compare(OUT/'eval/control',OUT/'eval/robust'),selected=selected,
                  screen_pass=bool(selected),test_data_used_in_screen=False)
    verify(p);write(OUT/'decision.json',decision);write(ROOT/f'artifacts/{OUT.name}_decision.json',decision)
    status('screen_complete',selected=selected,screen_pass=bool(selected))
    if selected is None:return
    DEADLINE=time.monotonic()+6*3600
    checkpoint=train(selected,'full',env,p)
    dest=OUT/'production'
    status('production_ten_views',arm=selected)
    views='center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,center:512:1.28,center:512:1.4,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14'
    run('production_base',['-m','aic_clip.infer_ft','--checkpoint',checkpoint,'--weights','raw',
        '--test-dir','/root/autodl-tmp/data/aic-rematch/test','--views',views,'--batch-size','192',
        '--workers','10','--expected-count','37444','--output-dir',dest,'--save-probs','--probs-dtype','float32'],env)
    status('production_subject',arm=selected)
    run('production_subject',[SNAP/'scripts/subject_view_probs.py','--checkpoint',checkpoint,
        '--test-dir','/root/autodl-tmp/data/aic-rematch/test','--weights','raw','--size','512',
        '--tau','.6','--margin','.15','--min-view-area','.02','--max-view-area','.95',
        '--batch-size','16','--output-dir',dest],env)
    status('production_export',arm=selected)
    run('production_export',[SNAP/'scripts/export_full.py','--directory',dest,'--checkpoint',checkpoint],env)
    verify(p);status('production_complete',selected=selected,output_directory=str(dest),local_scoring_pending=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
