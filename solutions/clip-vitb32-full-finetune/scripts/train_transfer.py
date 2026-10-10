"""Three fixed continuation epochs; CE control versus GCE/effective-prior LA-CE."""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

SNAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SNAP/'src'))
from aic_clip.train_ft import (FTClassifier, ManifestDataset, read_manifest,
    build_train_transform, build_optimizer, build_scheduler, set_seed,
    one_hot, apply_mixup_cutmix)

STAGES = [(0.8,0.,0.2), (0.35,0.35,0.30), (0.15,0.70,0.15)]


def save_json(p, d):
    t = Path(p).with_suffix('.tmp')
    t.write_text(json.dumps(d, indent=2))
    t.replace(p)


def effective_prior(labels, sampling, reliability, classes, smoothing):
    mass = np.bincount(labels, weights=sampling*reliability, minlength=classes).astype(np.float64)
    hard = mass/mass.sum()
    soft = (1-smoothing)*hard + smoothing/classes
    assert np.isfinite(soft).all() and (soft>0).all() and np.isclose(soft.sum(),1)
    return hard, soft


def transfer_loss(logits, target, prior, coeff, q=.7):
    # Targets include each source image's reliability BEFORE MixUp/CutMix.
    # Every term is linear in these weighted targets, preserving pair attribution.
    z = logits.float()
    ce = -(target*F.log_softmax(z,1)).sum(1)
    la = -(target*F.log_softmax(z+prior.log()[None,:],1)).sum(1)
    gce = (target*(1-F.softmax(z,1).clamp_min(1e-12).pow(q))/q).sum(1)
    parts = torch.stack([ce,la,gce],1)
    den = target.sum().clamp_min(1.)
    loss = (parts*z.new_tensor(coeff)[None,:]).sum()/den
    return loss, (parts.sum(0)/den).detach()


def configure_epoch(model, epoch):
    for name,p in model.named_parameters():
        p.requires_grad_(epoch<2 or name.startswith('head.'))


def state_digest(model, prefix='vision.'):
    h = hashlib.sha256()
    for k,v in model.state_dict().items():
        if k.startswith(prefix):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def inputs(mode):
    records = read_manifest(SNAP/'artifacts/train_manifest.csv')
    if mode == 'fold':
        keep = np.setdiff1d(np.arange(len(records)),np.load(SNAP/'artifacts/oof_a_drop.npy'))
        folds = np.array(json.loads((SNAP/'artifacts/folds_2.json').read_text())['fold'])
        assert len(keep)==72771 and (folds[keep]==0).all()
        weights = np.ones(len(keep),np.float32)
    else:
        keep = np.setdiff1d(np.arange(len(records)),np.load(SNAP/'artifacts/dedup_drop.npy'))
        weights = np.load(SNAP/'artifacts/oof_weights_v2.npy')[keep].astype(np.float32)
        assert len(keep)==144572 and np.isin(weights,[.5,1.]).all()
    return [records[i] for i in keep], keep, weights


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=['control','robust'],required=True)
    ap.add_argument('--mode',choices=['fold','full'],default='fold')
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--smoke-steps',type=int,default=0)
    args=ap.parse_args()
    out=Path(args.output)
    assert not out.exists(), out
    out.mkdir(parents=True)
    protocol=json.loads((SNAP.parent/'protocol.json').read_text())
    cfg=json.loads((SNAP/'configs/continuation.json').read_text())
    set_seed(cfg['train']['seed'])
    torch.set_num_threads(4)
    records,keep,reliability=inputs(args.mode)
    labels=np.array([r['label'] for r in records])
    counts=np.bincount(labels,minlength=750)
    sample_weights=np.maximum(counts,1.)[labels]**(-.5)
    hard,prior=effective_prior(labels,sample_weights,reliability,750,.15)
    save_json(out/'effective_prior.json',dict(hard_prior=hard.tolist(),smoothed_prior=prior.tolist(),
        rows=len(records),smoothing=.15,sampler_power=.5,formula='sum sampling_probability * reliability, then label smoothing'))
    ds=ManifestDataset(Path(cfg['data']['train_dir']),records,
        build_train_transform(576,cfg['augment']),1152)
    sampler=WeightedRandomSampler(torch.tensor(sample_weights,dtype=torch.double),len(records),replacement=True)
    loader=DataLoader(ds,batch_size=80,sampler=sampler,drop_last=True,num_workers=10,
                      pin_memory=True,persistent_workers=True,prefetch_factor=3)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    expected=protocol['initial_checkpoint_hashes'][args.mode]
    with Path(args.checkpoint).open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==expected
    assert payload['num_classes']==750 and payload['epoch']==5
    model=FTClassifier(cfg['model']['backbone'],cfg['model']['revision'],750,
        head='linear',dropout=0.,feature='projected').cuda()
    model.load_state_dict(payload['ema'],strict=True)
    del payload
    gc.collect()
    configure_epoch(model,0)
    optimizer=build_optimizer(model,cfg['train'])
    scheduler=build_scheduler(optimizer,3,.2,len(loader),.02)
    prior_t=torch.tensor(prior,dtype=torch.float32,device='cuda')
    history=[]
    for epoch in range(3):
        configure_epoch(model,epoch)
        model.train()
        if epoch==2:
            for p in model.vision.parameters():p.grad=None
            tower_before=state_digest(model)
        coeff=STAGES[epoch] if args.arm=='robust' else (1.,0.,0.)
        started=time.monotonic();seen=0;total_loss=0.;sum_parts=np.zeros(3);batch_digest=hashlib.sha256()
        for step,(images,labels_t,idx) in enumerate(loader):
            if args.smoke_steps and step>=args.smoke_steps:break
            batch_digest.update(idx.numpy().tobytes())
            images=images.cuda(non_blocking=True);labels_t=labels_t.cuda(non_blocking=True)
            target=one_hot(labels_t,750,.15)
            target*=torch.from_numpy(reliability[idx.numpy()]).cuda(non_blocking=True)[:,None]
            images,target,_,_=apply_mixup_cutmix(images,target,.2,1.,.8,750)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images)
            loss,parts=transfer_loss(logits,target,prior_t,coeff)
            assert torch.isfinite(loss) and torch.isfinite(parts).all()
            loss.backward()
            if epoch==2:assert all(p.grad is None for p in model.vision.parameters())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm)
            optimizer.step();scheduler.step()
            seen+=len(labels_t);total_loss+=loss.item()*len(labels_t);sum_parts+=parts.cpu().numpy()*len(labels_t)
            if (step+1)%100==0 or step+1==len(loader) or (args.smoke_steps and step+1==args.smoke_steps):
                progress=dict(arm=args.arm,epoch=epoch,step=step+1,steps=len(loader),loss=total_loss/seen,
                    coeff=coeff,trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    minutes=(time.monotonic()-started)/60,pid=os.getpid(),at=datetime.datetime.now().astimezone().isoformat())
                save_json(out/'progress.json',progress)
                print('TRAIN_PROGRESS',json.dumps(progress),flush=True)
        if epoch==2:assert state_digest(model)==tower_before,'Frozen tower changed during head calibration'
        entry=dict(epoch=epoch,loss=total_loss/seen,loss_parts=(sum_parts/seen).tolist(),
                   coefficients=coeff,minutes=(time.monotonic()-started)/60,batch_indices_sha256=batch_digest.hexdigest(),
                   seen=seen,head_only=epoch==2)
        history.append(entry)
        save_json(out/'history.json',history)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        save=dict(model=state,epoch=epoch,config=cfg,image_size=576,eval_size=512,num_classes=750,
            feature='projected',backbone=cfg['model']['backbone'],history=history,
            selection='fixed last raw, no heldout-based checkpoint selection',arm=args.arm,
            initial_checkpoint_sha256=expected,training_mode=args.mode,smoke=bool(args.smoke_steps))
        tmp=out/'last.pt.tmp';torch.save(save,tmp);tmp.replace(out/'last.pt')
        del state,save
        print('EPOCH_DONE',json.dumps(entry),flush=True)
    save_json(out/'complete.json',dict(epochs=3,rows=len(records),fixed_weights='raw',arm=args.arm,
                                    smoke=bool(args.smoke_steps),heldout_used_for_training=False))


if __name__=='__main__':main()
