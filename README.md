# AIC Frozen CLIP Linear Baseline

面向 AIC 全球校园人工智能算法精英大赛“面向噪声标签数据的细粒度图像识别鲁棒微调”赛题的最小实验框架。

当前 Baseline 严格定义为：

```text
OpenAI CLIP ViT-B/32 官方预训练权重
→ 完全冻结的图像编码器
→ nn.Linear(512, num_classes)
→ CrossEntropyLoss
```

只有线性分类头参与训练。项目当前不包含 LoRA、Adapter、Prompt Tuning、Robust Loss、样本过滤、重加权、伪标签、表征约束、其他视觉模型或模型集成。

## 环境准备

项目要求 Python 3.12 和 NVIDIA CUDA GPU。以下命令以 CUDA 12.8 版 PyTorch 为例。

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`requirements.txt` 将官方 `openai/CLIP` 固定到提交 `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`。首次加载时，CLIP 会把官方 ViT-B/32 权重放入配置中的 `models/`；当前工作区已经有该权重。

项目源码不依赖工作区文件夹名称，根目录由脚本自身位置动态计算。若在 Windows 上移动或重命名整个项目，`.venv/Scripts` 中的启动器可能仍保存旧绝对路径；完成重命名并重新打开终端后，可先尝试刷新虚拟环境：

```powershell
py -3.12 -m venv --upgrade .venv
python -m pip install -r requirements.txt
```

如果原虚拟环境无法正常激活，最稳妥的做法是在新目录中重新创建 `.venv`。数据、manifest、源码、配置、checkpoint 和 CSV 记录本身均使用相对项目路径，不需要因根目录改名而重写。

## 数据目录

```text
data/
├── train/
│   ├── 0000/
│   │   ├── image_a.jpg
│   │   └── ...
│   ├── 0001/
│   └── ...
└── test/
    ├── test_image_a.jpg
    └── ...
```

类别数由训练目录自动扫描，不硬编码为 500。类别文件夹必须是数值编号；提交时统一格式化成四位数字。

训练入口只构建带标签的训练 Dataset，并从 `data/train/` 产生验证集。测试 Dataset 只在 `src.predict` 和 smoke test 的最终 inference 阶段读取 `data/test/`，不提供标签，不能被训练循环正常消费。

## 固定数据划分

首次运行会按类别、固定 seed 和 `val_ratio` 创建：

```text
manifests/train_val_split.json
```

manifest 保存相对路径、标签、双向类别映射和训练数据清单摘要。后续实验直接复用；如果数据内容、类别映射、seed 或验证比例发生变化，程序会报错，不会静默生成不同划分。该文件应提交到 Git，让三名开发者共享完全一致的 train/validation split。

## 配置

所有实验参数集中在 `configs/baseline.yaml`，包括 seed、数据路径、manifest、验证比例、batch size、worker 数、epoch、学习率、weight decay、AMP、CUDA 设备和实验名称。

修改配置时建议复制出新的 YAML，例如 `configs/baseline_lr_3e-4.yaml`，不要把实验参数写入 Python 源码。

## Smoke test

完整 CUDA smoke test 只从正式训练集和验证集中各取一个两样本 batch，并在 checkpoint 恢复后对两张 test 图片做纯推理：

```powershell
python scripts/smoke_test.py --config configs/baseline.yaml
```

它验证：train/validation 无重叠、test 隔离、类别映射、冻结状态、forward shape、loss、backward、AdamW、AMP、validation、checkpoint 保存/恢复、test inference 和 CSV 完整性。

只检查数据和 manifest，完全不加载 CLIP 或打开 test 图片：

```powershell
python scripts/smoke_test.py --config configs/baseline.yaml --data-only
```

运行自动化测试：

```powershell
python -m pytest tests -q
```

## 训练

```powershell
python -m src.train --config configs/baseline.yaml
```

每次训练自动创建：

```text
runs/<timestamp>_<experiment_name>/
├── config.yaml
├── metrics.json
├── train.log
├── best.pt
├── last.pt
└── class_mapping.json
```

每个 epoch 输出 train loss、train accuracy、validation loss、validation accuracy、epoch 时间和 CUDA 峰值显存。`best.pt` 按最高 validation accuracy 保存，`last.pt` 每轮更新。checkpoint 不重复保存固定的官方 CLIP 权重，因此体积较小。

训练、smoke test、独立 validation 和 predict 成功结束后，都会向开发者本地的 `runs/run_registry.csv` 追加一条基础记录。该表包含运行时间、类型、run ID、checkpoint、产物路径、loss、accuracy、样本数、预测数、耗时和显存等字段；整个 `runs/` 目录不会进入 Git。

查看最近记录：

```powershell
python scripts/show_recent_runs.py --limit 10
```

只查看最近的 predict：

```powershell
python scripts/show_recent_runs.py --type predict --limit 10
```

## 独立验证

```powershell
python -m src.validate --checkpoint runs/<run_name>/best.pt
```

验证只读取固定 manifest 中的 validation split。默认从 checkpoint 同目录读取该次运行的 `config.yaml` 和 `class_mapping.json`，结果写入 `validation_results.json`。

## Test inference 与提交文件

测试集只能用于最终 inference：

```powershell
python -m src.predict --checkpoint runs/<run_name>/best.pt
```

默认输出 `runs/<run_name>/pred_results.csv`。也可指定位置：

```powershell
python -m src.predict --checkpoint runs/<run_name>/best.pt --output pred_results.csv
```

CSV 无表头，每行为：

```text
filename.jpg,0001
```

写入前后都会检查测试图片是否全部预测、文件名是否重复、是否出现多余图片、类别索引是否合法，以及类别编号是否为有效四位字符串。CSV 按 filename 显式排序，不依赖文件系统遍历顺序。

## 协作提交系统

`runs/` 是每名开发者自己的本地工作区；`submission/` 是可以提交到 Git 的团队候选档案：

```text
submission/
├── candidates/
│   └── candidate_<timestamp>_<run_id>_<sha8>.csv
├── records.csv
└── pred_results.csv
```

`records.csv` 记录候选 ID、创建时间、开发者、来源运行、源文件、候选文件、SHA-256、行数、状态、平台分数、排名和备注。候选 CSV 使用时间戳、来源 run ID 和内容哈希命名，便于审计且不会互相覆盖。

开发者完成 predict 后，从最近的本地结果中选择一个加入团队候选：

```powershell
python scripts/promote_prediction.py
```

脚本会显示最近的有效 predict 记录，校验 CSV 格式，再把所选结果复制到 `submission/candidates/` 并更新 `submission/records.csv`。也可直接选最近一次：

```powershell
python scripts/promote_prediction.py --latest --developer alice --notes "baseline epoch 10"
```

队长查看候选、选择平台提交版本并等待录入结果：

```powershell
python scripts/manage_submission.py
```

脚本会把所选候选复制成比赛要求的 `submission/pred_results.csv`，输出 SHA-256，然后停在输入提示处。队长手动上传文件，平台显示结果后按 Enter，依次录入分数、排名和可选备注；记录会写回 `submission/records.csv`，状态变为 `scored`。

若只想先生成 `pred_results.csv`、稍后再处理平台结果：

```powershell
python scripts/manage_submission.py --index 1 --stage-only
```

候选文件始终保留，生成 `pred_results.csv` 使用复制而不是移动，因此每次平台提交都能追溯到原始候选及哈希。提交前应将 `submission/` 中新增的候选、`records.csv` 和当前 `pred_results.csv` 一并提交到团队 Git 分支。

## 目录职责

```text
configs/       YAML 实验配置
manifests/     可复用的固定数据划分
src/data/      train/test Dataset 与 split
src/models/    CLIP backbone 和线性头
src/training/  train/validation loop
src/utils/     配置、日志、checkpoint、提交校验
scripts/       端到端 smoke test
tests/         快速自动化测试
runs/          本地实验输出和 run_registry.csv，不进入 Git
submission/    团队共享的候选 CSV、提交记录和 pred_results.csv
```

## Git 协作建议

不要提交 `data/`、`models/`、`runs/`、虚拟环境或 checkpoint。建议提交源码、配置、测试、README、依赖文件以及固定 split manifest。每个新算法使用独立配置，并让算法差异保持在对应模块中，避免修改数据划分和提交格式逻辑。
