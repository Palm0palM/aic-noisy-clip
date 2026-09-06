import pymupdf

doc = pymupdf.open(r'D:\AIC\comp_rules.pdf')
print('pages:', len(doc))
for i, page in enumerate(doc, 1):
    print(f"\n===== PAGE {i} =====")
    print(page.get_text())
doc.close()
