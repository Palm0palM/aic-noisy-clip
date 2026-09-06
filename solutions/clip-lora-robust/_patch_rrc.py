"""为 src/data.py 与 train.py 打补丁：RRC scale 范围可配置（train.rrc_scale）。
默认行为不变（0.5,1.0）。断言保证替换生效，失败即中止。"""
p = "src/data.py"
src = open(p, encoding="utf-8").read()
old1 = '''def _make_train_tf(crop: int):
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(crop, scale=(0.5, 1.0), interpolation=v2.InterpolationMode.BICUBIC),'''
new1 = '''def _make_train_tf(crop: int, rrc_scale=(0.5, 1.0)):
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(crop, scale=rrc_scale, interpolation=v2.InterpolationMode.BICUBIC),'''
assert old1 in src, "data.py _make_train_tf block not found"
src = src.replace(old1, new1)
old2 = '''    def __init__(self, mm_path: str, n: int, labels: np.ndarray, train: bool,
                 mm_size: int = 256, img_size: int = 224):
        self.mm_path = str(mm_path)
        self.n = n
        self.mm_size = mm_size
        self.labels = labels.astype(np.int64)
        self.tf = _make_train_tf(img_size) if train else _make_eval_tf(img_size)'''
new2 = '''    def __init__(self, mm_path: str, n: int, labels: np.ndarray, train: bool,
                 mm_size: int = 256, img_size: int = 224, rrc_scale=(0.5, 1.0)):
        self.mm_path = str(mm_path)
        self.n = n
        self.mm_size = mm_size
        self.labels = labels.astype(np.int64)
        self.tf = (_make_train_tf(img_size, rrc_scale) if train
                   else _make_eval_tf(img_size))'''
assert old2 in src, "data.py MemmapTrainDataset block not found"
src = src.replace(old2, new2)
open(p, "w", encoding="utf-8").write(src)
assert "rrc_scale" in open(p, encoding="utf-8").read()
print("[patch] src/data.py rrc_scale done")

p = "train.py"
src = open(p, encoding="utf-8").read()
old1 = '''    train_ds = MemmapTrainDataset(mm_path, len(df), labels, train=True,
                                  mm_size=mm_size, img_size=img_size)'''
new1 = '''    rrc_scale = tuple(cfg["train"].get("rrc_scale", [0.5, 1.0]))
    train_ds = MemmapTrainDataset(mm_path, len(df), labels, train=True,
                                  mm_size=mm_size, img_size=img_size,
                                  rrc_scale=rrc_scale)'''
assert old1 in src, "train.py train_ds block not found"
src = src.replace(old1, new1)
old2 = '''    log(f"[data] train={n_train} val={len(val_idx)} classes={cfg['data']['num_classes']}", log_file)'''
new2 = old2 + '''
    log(f"[data] rrc_scale={rrc_scale}", log_file)'''
assert old2 in src, "train.py log line not found"
src = src.replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(src)
assert "rrc_scale" in open(p, encoding="utf-8").read()
print("[patch] train.py rrc_scale passthrough done")
