import ast

src = open('infer.py').read()
ast.parse(src)
print('syntax OK')
for n, line in enumerate(src.splitlines(), 1):
    if 'cap' in line:
        print(f'{n}: {line}')
