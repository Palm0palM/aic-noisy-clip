import torch
import pandas as pd
import zipfile
import os

l_real = torch.load('outputs/prelim_run2_352real/logits_tta.pt', map_location='cpu').float()
l_f = torch.load('outputs/prelim_run2_352/logits_tta.pt', map_location='cpu').float()
C = l_real.shape[1]

fams = {
    'real': l_real,
    'enslogit': (l_real + l_f) / 2,
    'ensprob': torch.log(((l_real.softmax(-1) + l_f.softmax(-1)) / 2).clamp_min(1e-12)),
}

base = pd.read_csv('outputs/prelim_run2_352real/pred_raw_352real.csv', header=None)
assert len(base) == l_real.shape[0]

def make(fam, T, t):
    logits = fams[fam]
    x = logits / T
    prior = x.softmax(-1).mean(0).clamp_min(1e-8)
    adj = -torch.log(prior * C)
    pred = (x + t * adj).argmax(-1)
    vc = torch.bincount(pred, minlength=C)
    print(f'{fam} T={T} tau={t}: max={int(vc.max())} min={int(vc.min())} classes={int((vc > 0).sum())}')
    df = base.copy()
    df[1] = [f'{v:04d}' for v in pred.tolist()]
    name = f'pred_{fam}_T{T}_t{str(t).replace(".", "")}'
    csv_path = f'outputs/submissions/{name}.csv'
    df.to_csv(csv_path, index=False, header=False)
    with zipfile.ZipFile(f'outputs/submissions/{name}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, 'pred_results.csv')
    print(f'  -> outputs/submissions/{name}.zip')

os.makedirs('outputs/submissions', exist_ok=True)
make('enslogit', 6, 1.6)
make('ensprob', 4, 2.0)
make('real', 6, 1.5)
print('done')
