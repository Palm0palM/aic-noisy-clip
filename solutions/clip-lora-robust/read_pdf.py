import re
from pathlib import Path

from pypdf import PdfReader

pdf = next(Path(r"c:\Users\Lenovo\.trae-cn\attachments\6a91852eb1141fe142291c66").rglob("*.pdf"))
print("file:", pdf)
r = PdfReader(str(pdf))
print("pages:", len(r.pages))
for i, p in enumerate(r.pages):
    t = p.extract_text() or ""
    if re.search(r"评分|指标|准确率|噪声|测试集|提交|类别设置", t):
        print(f"===== page {i + 1} =====")
        print(t[:2500])
