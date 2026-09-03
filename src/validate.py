"""Evaluate a trained checkpoint on the persisted validation split only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset

from src.data import LabeledImageDataset, load_or_create_split
from src.models import load_baseline_model
from src.training import evaluate, make_dataloader
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, project_path
from src.utils.experiment import load_class_mapping, save_json
from src.utils.reproducibility import seed_everything
from src.utils.run_registry import append_run_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    validation_start = time.perf_counter()
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve() if args.config else checkpoint_path.parent / "config.yaml"
    config = load_config(config_path)
    seed = int(config["seed"])
    seed_everything(seed)

    training_config = config["training"]
    device = torch.device(training_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Baseline validation requires CUDA.")

    class_to_idx, idx_to_class = load_class_mapping(
        checkpoint_path.parent / "class_mapping.json"
    )
    data_config = config["data"]
    manifest_path = project_path(data_config["split_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("Validation requires the persisted training split manifest.")
    dataset = LabeledImageDataset(project_path(data_config["train_dir"]))
    if dataset.class_to_idx != class_to_idx or dataset.idx_to_class != idx_to_class:
        raise RuntimeError("Saved class mapping does not match the training dataset.")
    manifest, _created = load_or_create_split(
        dataset,
        manifest_path,
        val_ratio=float(data_config["val_ratio"]),
        seed=seed,
    )

    model, preprocess = load_baseline_model(
        model_name=config["model"]["name"],
        download_root=project_path(config["model"]["download_root"]),
        num_classes=len(idx_to_class),
        device=device,
    )
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    dataset.transform = preprocess
    _train_indices, val_indices = manifest.indices(dataset)
    loader = make_dataloader(
        Subset(dataset, val_indices),
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        seed=seed,
        device=device,
    )
    metrics = evaluate(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device=device,
        amp_enabled=bool(training_config["amp"]),
    )
    result = {"checkpoint": str(checkpoint_path), **metrics}
    save_json(checkpoint_path.parent / "validation_results.json", result)
    append_run_record(
        event_type="validation",
        run_id=checkpoint_path.parent.name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        artifact_path=checkpoint_path.parent / "validation_results.json",
        validation_loss=metrics["loss"],
        validation_accuracy=metrics["accuracy"],
        num_samples=int(metrics["num_samples"]),
        duration_seconds=time.perf_counter() - validation_start,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
