import re

src = open('infer.py').read()

# Step 1: remove previously mis-inserted cap block
i = src.find('if args.cap > 0:')
if i != -1:
    line_start = src.rfind('\n', 0, i) + 1
    j = src.find('if args.balance_tau > 0:')
    assert j > i, 'balance_tau line not found after cap block'
    bal_line_start = src.rfind('\n', 0, j) + 1
    orig_indent = src[line_start:i]
    src = src[:line_start] + orig_indent + src[bal_line_start:]
    print('removed bad block, indent =', repr(orig_indent))

# Step 2: re-insert with correct indentation detected from balance_tau line
m = re.search(r'^([ \t]*)if args\.balance_tau > 0:', src, re.M)
assert m, 'balance_tau line not found'
ind = m.group(1)
block = """if args.cap > 0:
    N, C = all_logits.shape
    k = 64
    probs = all_logits.softmax(-1)
    topv, topi = probs.topk(k, dim=1)
    order = torch.argsort(topv.flatten(), descending=True)
    pred = torch.full((N,), -1, dtype=torch.long)
    counts = torch.zeros(C, dtype=torch.long)
    flat_c = topi.flatten()
    for idx in order.tolist():
        i = idx // k
        c = int(flat_c[idx])
        if pred[i] == -1 and counts[c] < args.cap:
            pred[i] = c
            counts[c] += 1
    left = (pred == -1).nonzero(as_tuple=True)[0]
    for i in left.tolist():
        avail = (counts < args.cap).nonzero(as_tuple=True)[0]
        c = int(avail[all_logits[i, avail].argmax()])
        pred[i] = c
        counts[c] += 1
    print(f"[infer] cap={args.cap}: dist max={int(counts.max())} min={int(counts.min())} classes={int((counts > 0).sum())}")
    all_logits = torch.zeros_like(all_logits)
    all_logits[torch.arange(N), pred] = 10.0
"""
cap_code = '\n'.join((ind + ln) if ln else ln for ln in block.splitlines()) + '\n\n'
src = src[:m.start()] + cap_code + src[m.start():]

# ensure --cap argument exists
if '"--cap"' not in src:
    anchor = '    parser.add_argument("--temp", type=float, default=1.0)'
    assert anchor in src, 'temp arg anchor not found'
    src = src.replace(anchor, anchor + '\n    parser.add_argument("--cap", type=int, default=0)', 1)
    print('added --cap argument')

open('infer.py', 'w').write(src)
print('re-inserted OK, indent =', repr(ind))
