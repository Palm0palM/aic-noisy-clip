"""Export four preregistered predictions from one new full-data checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
SNAP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SNAP/'src'))
from aic_clip.train_ft import FTClassifier
from aic_clip.infer_ft import run_view
from detail_views import whole_view
from fixed_soft50 import uniform_alignment


class Whole:
    def __call__(self,im):return whole_view(im,512)


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);ap.add_argument('--checkpoint',required=True)
    args=ap.parse_args();out=Path(args.directory)
    assert not (out/'completion.json').exists()
    sub=np.load(out/'subject_view_probs.npz',allow_pickle=False)
    files=sub['files'].tolist();fb=sub['fallback'];rows=np.flatnonzero(fb)
    assert len(files)==len(set(files))==37444
    ten=np.load(out/'mean_probs.npy').astype(np.float32);ten/=ten.sum(1,keepdims=True)
    subject=ten.copy();subject[~fb]=(ten[~fb]*10+sub['probs'][~fb])/11
    subject/=subject.sum(1,keepdims=True)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=payload['config']
    assert payload['epoch']==2 and payload['training_mode']=='full' and not payload['smoke']
    model=FTClassifier(cfg['model']['backbone'],cfg['model']['revision'],750,head='linear',dropout=0.,feature='projected').cuda()
    model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False);del payload
    p=run_view(model,Path('/root/autodl-tmp/data/aic-rematch/test'),[files[i] for i in rows],Whole(),192,10,torch.device('cuda'),torch.bfloat16,750)
    del model;torch.cuda.empty_cache()
    whole=subject.copy();whole[fb]=(ten[fb]*10+p)/11;whole/=whole.sum(1,keepdims=True)
    np.savez_compressed(out/'whole_fallback_probs.npz',files=np.array(files)[rows],rows=rows,probs=p)
    hashes={};alignments={}
    for name,probs in [('subject',subject),('subject_letterbox',whole)]:
        assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
        for suffix,values in [('raw',probs),('soft50',None)]:
            if values is None:
                values,info=uniform_alignment(probs,strength=.5)
                assert info['converged'];alignments[name]=info
            dest=out/f'{name}_{suffix}.csv'
            with dest.open('w',encoding='utf-8',newline='') as f:
                for file,pred in zip(files,values.argmax(1)):f.write(f'{file},{int(pred):04d}\n')
            hashes[dest.name]=sha(dest)
            with zipfile.ZipFile(dest.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:z.write(dest,arcname='pred_results.csv')
            if suffix=='soft50':del values
    report=dict(rows=len(files),checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
                prediction_hashes=hashes,alignments=alignments,weights='fixed final raw',
                test_labels_used=False,parameter_search=False)
    (out/'completion.json').write_text(json.dumps(report,indent=2))
    print('FULL_EXPORT_DONE',json.dumps(report),flush=True)


if __name__=='__main__':main()
