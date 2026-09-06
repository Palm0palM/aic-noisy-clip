import torch

l_real = torch.load('outputs/prelim_run2_352real/logits_tta.pt', map_location='cpu').float()
l_f = torch.load('outputs/prelim_run2_352/logits_tta.pt', map_location='cpu').float()
print(f'logits loaded: real={tuple(l_real.shape)} f352={tuple(l_f.shape)}')

C = l_real.shape[1]
Ts = [4, 5, 6, 7, 8, 10]
taus = [0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5]


def dist_stats(pred):
    vc = torch.bincount(pred, minlength=C)
    return int(vc.max()), int(vc.min()), int((vc > 0).sum())


def correct(logits, T, t):
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    return (x + t * adj).argmax(-1)


def sweep(logits, name, champ=None):
    rows = []
    for T in Ts:
        for t in taus:
            pred = correct(logits, T, t)
            mx, mn, nc = dist_stats(pred)
            agree = float((pred == champ).float().mean()) if champ is not None else 1.0
            rows.append((name, T, t, mx, mn, nc, agree))
    return rows


# champion reference: 352real T6 t1.6 (scored 71.2501)
champ = correct(l_real, 6, 1.6)

# ensembles
ens_logit = (l_real + l_f) / 2
p_avg = (l_real.softmax(-1) + l_f.softmax(-1)) / 2
ens_prob = torch.log(p_avg.clamp_min(1e-12))

all_rows = []
all_rows += sweep(l_real, 'real', champ)
all_rows += sweep(ens_logit, 'ens_logit', champ)
all_rows += sweep(ens_prob, 'ens_prob', champ)

print(f"\n{'name':10s} {'T':>4s} {'tau':>5s} {'max':>5s} {'min':>5s} {'cls':>4s} {'agree':>6s}")
for r in all_rows:
    print(f'{r[0]:10s} {r[1]:4d} {r[2]:5.1f} {r[3]:5d} {r[4]:5d} {r[5]:4d} {r[6]:6.3f}')

# sweet-zone filter: neighborhood of champion distribution (max 60-140, min 8-40)
sweet = [r for r in all_rows if 60 <= r[3] <= 140 and 8 <= r[4] <= 40 and r[5] == 500]
sweet.sort(key=lambda r: (r[0], abs(r[3] - 98) + abs(r[4] - 17)))
print(f'\n=== sweet-zone candidates ({len(sweet)}) ===')
for r in sweet:
    print(f'{r[0]:10s} T={r[1]:2d} tau={r[2]:4.1f}  max={r[3]:3d} min={r[4]:2d} agree={r[6]:.3f}')

torch.save({'champ': champ}, 'outputs/prelim_run2_352real/champ_pred.pt')
print('\nchamp pred saved')
