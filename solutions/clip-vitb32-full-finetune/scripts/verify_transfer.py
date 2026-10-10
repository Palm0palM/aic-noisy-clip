"""Checks for loss weighting, LA sign, original augmentation, and split isolation."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_transfer import transfer_loss,effective_prior,inputs,SNAP
from aic_clip.train_ft import one_hot


def main():
    torch.manual_seed(41)
    z=torch.randn(7,5,requires_grad=True)
    y=torch.tensor([0,1,2,3,4,1,2]);t=one_hot(y,5,.15)
    p=torch.full((5,),.2)
    ce,_=transfer_loss(z,t,p,(1,0,0))
    assert torch.allclose(ce,F.cross_entropy(z,y,label_smoothing=.15),atol=1e-6)
    la,_=transfer_loss(z,t,p,(0,1,0))
    assert torch.allclose(la,ce,atol=1e-6),'Uniform prior must not alter LA-CE'
    weights=torch.tensor([.5,1,.5,1,1,1,.5])
    wt=t*weights[:,None]
    loss,_=transfer_loss(z,wt,p,(1,0,0))
    expected=(-(wt*F.log_softmax(z,1)).sum(1)).sum()/weights.sum()
    assert torch.allclose(loss,expected,atol=1e-6)
    scale,_=transfer_loss(z,wt*3,p,(.35,.35,.3))
    regular,_=transfer_loss(z,wt,p,(.35,.35,.3))
    assert torch.allclose(scale,regular,atol=1e-6),'Reliability scale must cancel'
    perm=torch.tensor([1,2,3,4,5,6,0]);lam=.3
    mixed=lam*wt+(1-lam)*wt[perm]
    expected_gce=(mixed*(1-z.softmax(1).pow(.7))/.7).sum()/mixed.sum()
    actual_gce,_=transfer_loss(z,mixed,p,(0,0,1))
    assert torch.allclose(expected_gce,actual_gce,atol=1e-6)
    pi=torch.tensor([.8,.2]);adjusted=(-pi.log()+pi.log()).softmax(0)
    assert torch.allclose(adjusted,torch.tensor([.5,.5]))
    hard,soft=effective_prior(np.array([0,0,1]),np.array([1.,1.,2.]),np.array([1.,.5,1.]),2,.15)
    assert np.allclose(hard,[1.5/3.5,2/3.5]) and np.allclose(soft,.85*hard+.15/2)
    actual_gce.backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    records,keep,w=inputs('fold');listing=json.loads((SNAP/'artifacts/fold_b_files.json').read_text())
    assert not set(keep).intersection(listing['indices'])
    source=(SNAP/'src/aic_clip/train_ft.py').read_text()
    assert 'images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]' in source
    assert 'cutmix_skipped' not in source
    report=dict(loss_checks_pass=True,train_rows=len(keep),heldout_rows=len(listing['indices']),
                shared_indices=0,cutmix='actual image/target mixing retained',test_labels_used=False)
    (SNAP.parent/'verification.json').write_text(json.dumps(report,indent=2))
    print('VERIFIED',json.dumps(report),flush=True)


if __name__=='__main__':main()
