"""Production-view evaluation on a frozen training-only held-out file list."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SNAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNAP / 'src'))
from aic_clip.train_ft import FTClassifier
from aic_clip.infer_ft import build_view, run_view
from compare_fold_probs import metrics
from dump_teacher_probs import degrade_image

VIEWS = ['center:512:1.0', 'flip:512:1.0', 'center:512:1.14', 'flip:512:1.14',
         'center:512:1.28', 'center:512:1.4', 'tl:512:1.14', 'tr:512:1.14',
         'bl:512:1.14', 'br:512:1.14']


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inputs(path, limit=0):
    d = json.loads(Path(path).read_text())
    n = limit or len(d['files'])
    return d['files'][:n], np.array(d['indices'][:n]), np.array(d['labels'][:n])


class DegradedView:
    def __init__(self, view):
        self.view = view

    def __call__(self, image):
        return self.view(degrade_image(image, 'jpeg45_blur1'))


def compact(probs, labels):
    assert probs.dtype == np.float32 and np.isfinite(probs).all()
    assert np.allclose(probs.sum(1), 1, atol=1e-3)
    return probs.argmax(1).astype(np.int16), probs.max(1), metrics(probs, labels)


def base(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'base_summary.npz').exists()
    files, indices, labels = inputs(args.files_json, args.limit)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = payload['config']
    assert payload['epoch'] == 5 and payload['num_classes'] == 750
    assert cfg['model']['revision'] == 'c237dc49a33fc61debc9276459120b7eac67e7ef'
    assert not cfg['model'].get('local_readout') and 'ema' in payload
    model = FTClassifier(cfg['model']['backbone'], cfg['model']['revision'], 750,
                         head=cfg['model'].get('head', 'linear'), dropout=0.,
                         feature=payload.get('feature', cfg['model'].get('feature', 'projected'))).cuda()
    model.load_state_dict(payload['ema'], strict=True)
    model.eval()
    del payload
    gc.collect()
    views = VIEWS[2:4] if args.degraded else VIEWS
    probabilities = []
    started = time.monotonic()
    for spec in views:
        transform = build_view(spec, 512)
        if args.degraded:
            transform = DegradedView(transform)
        p = run_view(model, Path(args.image_root), files, transform, args.batch_size,
                     args.workers, torch.device('cuda'), torch.bfloat16, 750)
        compact(p, labels)
        probabilities.append(p)
        print('VIEW_DONE', spec, 'minutes', round((time.monotonic()-started)/60, 2), flush=True)
    if args.degraded:
        # Historical degradation arrays are float16; quantize both arms identically.
        two = np.mean(np.stack(probabilities), axis=0).astype(np.float16).astype(np.float32)
        pred, conf, report = compact(two, labels)
        np.savez_compressed(out/'base_summary.npz', files=files, indices=indices,
                            labels=labels, two_pred=pred, two_conf=conf)
        reports = {'degraded_two': report}
    else:
        two = np.mean(np.stack(probabilities[2:4]), axis=0)
        ten = np.mean(np.stack(probabilities), axis=0)
        p2, c2, m2 = compact(two, labels)
        p10, c10, m10 = compact(ten, labels)
        np.save(out/'ten_probs.npy', ten)
        np.savez_compressed(out/'base_summary.npz', files=files, indices=indices,
                            labels=labels, two_pred=p2, two_conf=c2, ten_pred=p10, ten_conf=c10)
        reports = {'two': m2, 'ten': m10}
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
                  weights='ema', files_sha256=sha(args.files_json), n=len(files),
                  views=views, batch_size=args.batch_size, precision='BF16 forward / FP32 softmax',
                  minutes=(time.monotonic()-started)/60, metrics=reports)
    (out/'base_report.json').write_text(json.dumps(report, indent=2))
    print('BASE_DONE', json.dumps(report), flush=True)


def finish(args):
    out = Path(args.output)
    base_data = np.load(out/'base_summary.npz', allow_pickle=False)
    subject = np.load(out/'subject_view_probs.npz', allow_pickle=False)
    assert np.array_equal(base_data['files'], subject['files'])
    ten = np.load(out/'ten_probs.npy')
    sp = subject['probs']
    fallback = subject['fallback']
    assert sp.shape == ten.shape and fallback.shape == (len(ten),)
    assert np.isfinite(sp).all() and (sp >= 0).all() and np.all(sp[fallback] == 0)
    assert np.allclose(sp[~fallback].sum(1), 1, atol=1e-3)
    b = json.loads((out/'base_report.json').read_text())
    s = json.loads((out/'subject_view_report.json').read_text())
    assert b['checkpoint_sha256'] == s['checkpoint_sha256'] and b['weights'] == s['weights']
    eleven = ten.copy()
    eleven[~fallback] = (ten[~fallback]*10+sp[~fallback])/11
    assert np.array_equal(eleven[fallback], ten[fallback])
    pred, conf, m = compact(eleven, base_data['labels'])
    np.savez_compressed(out/'predictions.npz', **{k:base_data[k] for k in base_data.files},
                        subject_pred=pred, subject_conf=conf, subject_used=~fallback)
    b['metrics']['subject'] = m
    b['subject_used'] = int((~fallback).sum())
    b['subject_report'] = s
    b['predictions_sha256'] = sha(out/'predictions.npz')
    (out/'report.json').write_text(json.dumps(b, indent=2))
    # This file is created by this evaluator and is no longer needed after combination.
    (out/'ten_probs.npy').unlink()
    print('FOLD_EVAL_DONE', json.dumps(b), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['base', 'finish'], required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--files-json')
    parser.add_argument('--image-root', default='/root/autodl-tmp/data/aic-rematch/train')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--degraded', action='store_true')
    args = parser.parse_args()
    (base if args.mode == 'base' else finish)(args)


if __name__ == '__main__':
    main()
