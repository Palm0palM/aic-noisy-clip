src = open('infer.py').read()

if '--save_logits' not in src:
    anchor_arg = '    parser.add_argument("--cap", type=int, default=0)'
    assert anchor_arg in src, 'cap arg anchor not found'
    src = src.replace(anchor_arg, anchor_arg + '\n    parser.add_argument("--save_logits", type=str, default="")', 1)

    anchor_body = '    if args.temp != 1.0:'
    assert anchor_body in src, 'temp body anchor not found'
    save_block = (
        '    if args.save_logits:\n'
        '        torch.save(all_logits, args.save_logits)\n'
        '        print(f"[infer] saved logits -> {args.save_logits} shape={tuple(all_logits.shape)}")\n'
    )
    src = src.replace(anchor_body, save_block + anchor_body, 1)
    open('infer.py', 'w').write(src)
    print('patched OK')
else:
    print('already patched')

import ast
ast.parse(src)
print('syntax OK')
