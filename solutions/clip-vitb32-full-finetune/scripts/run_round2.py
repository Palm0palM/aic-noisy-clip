"""Round-2 dose experiment: train -> test inference -> fixed-soft50 export.

Two candidates from the loss_robust raw finals:
  r2_robust   the exact winning 3-epoch GCE+LA-CE schedule (fresh optimizer)
  d6_robust   the same schedule repeated twice (6 epochs)

Diagnostic/comparison only: each candidate is scored once against the local
truth afterwards; nothing is tuned on test, and the current best CSV hash is
pinned as the bar in the protocol.
"""
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

SNAP = Path(__file__).resolve().parents[1]
OUT = SNAP.parent
ROOT = Path('/root/autodl-tmp/projects/AIC-Robust-CLIP-复赛')
LOSS = Path('/root/autodl-tmp/aic-experiments/loss_transfer_20261009')
BATCH = Path('/root/autodl-tmp/aic-experiments/batch_test_20261009')
TESTDIR = '/root/autodl-tmp/data/aic-rematch/test'
VIEWS = ('center:512:1.0,flip:512:1.0,center:512:1.14,flip:512:1.14,'
         'center:512:1.28,center:512:1.4,tl:512:1.14,tr:512:1.14,bl:512:1.14,br:512:1.14')
DEADLINE = None

sys.path.insert(0, str(LOSS/'frozen/scripts'))
from fixed_soft50 import uniform_alignment  # noqa: E402


def now():
    return datetime.datetime.now().astimezone().isoformat()


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, d):
    t = Path(p).with_suffix('.tmp')
    t.write_text(json.dumps(d, indent=2))
    t.replace(p)


def status(phase, **kw):
    d = dict(phase=phase, updated_at=now(), pid=os.getpid(), output=str(OUT), **kw)
    write(OUT/'status.json', d)
    try:
        write(ROOT/f'artifacts/{OUT.name}_status.json', d)
    except OSError as exc:
        print('status copy failed:', exc, flush=True)
    print('STATUS', json.dumps(d), flush=True)


def verify(p):
    for path, h in p['frozen'].items():
        assert sha(path) == h, path
    for key, path in p['initial_checkpoints'].items():
        assert sha(path) == p['initial_checkpoint_hashes'][key]
    for path, h in p['reference_results'].items():
        assert sha(path) == h, path


def run(name, args, env, cwd=None):
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip(), 'GPU already in use'
    log = OUT/'logs'/f'{name}.log'
    assert not log.exists(), log
    with log.open('w') as f:
        print('RUN', name, [str(a) for a in args], flush=True)
        proc = subprocess.Popen([sys.executable]+[str(a) for a in args], cwd=cwd or SNAP, env=env,
                                stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=max(1, DEADLINE-time.monotonic()) if DEADLINE else None)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise RuntimeError('Preregistered execution time limit reached')
        if rc:
            raise RuntimeError(f'{name} failed with code {rc}; see {log}')


def train(dest, arm, env, p, smoke=False):
    name = dest.name
    status('smoke_training' if smoke else 'training', arm=arm, log=str(OUT/'logs'/f'{name}.log'))
    args = [SNAP/'scripts/train_round2.py', '--arm', arm, '--mode', 'full',
            '--checkpoint', p['initial_checkpoints']['full'], '--output', dest]
    if smoke:
        args += ['--smoke-steps', '2']
    run(name, args, env)
    hist = json.loads((dest/'history.json').read_text())
    head_epochs = p['head_epochs'][arm]
    assert [i for i, e in enumerate(hist) if e['head_only']] == head_epochs, hist
    assert len(hist) == len(p['schedules'][arm])
    return hist


def export(name, src_dir, checkpoint):
    report = json.loads((src_dir/'report.json').read_text())
    subj_report = json.loads((src_dir/'subject_view_report.json').read_text())
    h = sha(checkpoint)
    assert report['checkpoint_sha256'] == h and subj_report['checkpoint_sha256'] == h
    assert report['weights'] == 'raw' and subj_report['weights'] == 'raw'
    subject = np.load(src_dir/'subject_view_probs.npz', allow_pickle=False)
    files = [str(x) for x in subject['files']]
    fallback = subject['fallback'].astype(bool)
    p = np.load(src_dir/'mean_probs.npy').astype(np.float32)
    p = p/np.maximum(p.sum(1, keepdims=True), 1e-12)
    assert len(files) == len(set(files)) == 37444 and p.shape == (37444, 750)
    assert np.isfinite(p).all() and np.isfinite(subject['probs']).all()
    p[~fallback] = (p[~fallback]*10 + subject['probs'][~fallback])/11
    p = p/np.maximum(p.sum(1, keepdims=True), 1e-12)
    dest = OUT/'predictions'/name
    dest.mkdir(parents=True)
    aligned, info = uniform_alignment(p, strength=.5)
    assert info['converged']
    hashes = {}
    for suffix, values in [('raw', p), ('soft50', aligned)]:
        path = dest/f'{suffix}.csv'
        with path.open('w', encoding='utf-8', newline='') as fh:
            for file, label in zip(files, values.argmax(1)):
                fh.write(f'{file},{int(label):04d}\n')
        hashes[suffix] = sha(path)
    write(dest/'completion.json', dict(completed_at=now(), name=name, rows=len(files),
        prediction_hashes=hashes,
        provenance=dict(checkpoint=str(checkpoint), checkpoint_sha256=h, weights='raw',
            base_report=report, subject_report=subj_report),
        soft50=info, test_labels_used=False, parameter_search=False))
    return hashes


def main():
    global DEADLINE
    lock = (OUT/'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = json.loads((OUT/'protocol.json').read_text())
    verify(p)
    (OUT/'logs').mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(LOSS/'frozen/src'), HF_HOME='/root/autodl-tmp/huggingface',
               TORCH_HOME='/root/autodl-tmp/torch-cache', HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1', TMPDIR='/root/tmp')
    DEADLINE = time.monotonic() + 8*3600
    control_digest = [r['batch_indices_sha256']
                      for r in json.loads((BATCH/'loss_robust/history.json').read_text())]
    summary = {}
    r2_digest = None
    for name, arm in p['arm_definitions'].items():
        verify(p)
        smoke = train(OUT/f'smoke_{name}', arm, env, p, smoke=True)
        print(f'[round2] smoke {name} OK: epochs={len(smoke)} head_epochs={p["head_epochs"][arm]}', flush=True)
        dest = OUT/name
        hist = train(dest, arm, env, p, smoke=False)
        digs = [r['batch_indices_sha256'] for r in hist]
        if arm == 'robust3':
            assert digs == control_digest, 'batch order differs from recorded loss_robust'
            r2_digest = digs
        else:
            assert digs[:3] == control_digest, 'second arm first cycle differs from recorded loss_robust'
            assert digs[:3] == r2_digest, 'r2/d6 first-cycle sampling differs'
        verify(p)
        ckpt = dest/'last.pt'
        inf = OUT/'inference'/name
        status('ten_views', candidate=name)
        run(name+'_ten_views', ['-m', 'aic_clip.infer_ft', '--checkpoint', ckpt, '--weights', 'raw',
            '--test-dir', TESTDIR, '--views', VIEWS, '--batch-size', '192', '--workers', '10',
            '--expected-count', '37444', '--output-dir', inf, '--save-probs', '--probs-dtype', 'float16'],
            env, cwd=LOSS/'frozen')
        status('subject', candidate=name)
        run(name+'_subject', [LOSS/'frozen/scripts/subject_view_probs.py', '--checkpoint', ckpt,
            '--test-dir', TESTDIR, '--weights', 'raw', '--size', '512', '--tau', '.6',
            '--margin', '.15', '--min-view-area', '.02', '--max-view-area', '.95',
            '--batch-size', '16', '--output-dir', inf], env, cwd=LOSS/'frozen')
        hashes = export(name, inf, ckpt)
        summary[name] = dict(arm=arm, checkpoint_sha256=sha(ckpt),
                             prediction_hashes=hashes, history_digests=digs)
        for f in ['test_view_probs.npz', 'mean_probs.npy']:
            try:
                (inf/f).unlink()
            except FileNotFoundError:
                pass
        print(f'[round2] {name} exported: {hashes}', flush=True)
        status('candidate_done', candidate=name, prediction_hashes=hashes)
    write(OUT/'summary.json', dict(completed_at=now(), candidates=summary))
    try:
        write(ROOT/f'artifacts/{OUT.name}_summary.json', dict(completed_at=now(), candidates=summary))
    except OSError as exc:
        print('summary copy failed:', exc, flush=True)
    status('complete', candidates=list(summary))
    print('DONE', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=repr(exc), traceback=traceback.format_exc())
        raise
