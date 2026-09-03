"""Train the strict frozen OpenAI CLIP ViT-B/32 linear baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset

from src.data import LabeledImageDataset, load_or_create_split
from src.models import count_parameters, load_baseline_model
from src.training import evaluate, make_dataloader, train_one_epoch
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, project_path
from src.utils.experiment import (
    create_logger,
    create_run_directory,
    save_class_mapping,
    save_json,
    save_yaml,
)
from src.utils.reproducibility import seed_everything
from src.utils.run_registry import append_run_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed = int(config["seed"])
    seed_everything(seed)

    training_config = config["training"]
    device = torch.device(training_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Baseline training requires CUDA.")

    run_dir = create_run_directory(config["experiment"]["name"])
    logger = create_logger(run_dir)
    save_yaml(run_dir / "config.yaml", config)

    data_config = config["data"]
    dataset = LabeledImageDataset(project_path(data_config["train_dir"]))
    manifest, created = load_or_create_split(
        dataset,
        project_path(data_config["split_manifest"]),
        val_ratio=float(data_config["val_ratio"]),
        seed=seed,
    )
    save_class_mapping(
        run_dir / "class_mapping.json", dataset.class_to_idx, dataset.idx_to_class
    )

    model, preprocess = load_baseline_model(
        model_name=config["model"]["name"],
        download_root=project_path(config["model"]["download_root"]),
        num_classes=len(dataset.classes),
        device=device,
    )
    dataset.transform = preprocess
    train_indices, val_indices = manifest.indices(dataset)
    train_loader = make_dataloader(
        Subset(dataset, train_indices),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        seed=seed,
        device=device,
    )
    val_loader = make_dataloader(
        Subset(dataset, val_indices),
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        seed=seed,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    amp_enabled = bool(training_config["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    counts = count_parameters(model)

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", torch.cuda.get_device_name(device))
    logger.info("Split manifest: %s (%s)", data_config["split_manifest"], "created" if created else "reused")
    logger.info("Samples: train=%d validation=%d classes=%d", len(train_indices), len(val_indices), len(dataset.classes))
    logger.info("Parameters: total=%d trainable=%d", counts.total, counts.trainable)

    epochs = int(training_config["epochs"])
    if epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    best_accuracy = -1.0
    history: list[dict[str, float | int]] = []
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device=device,
            amp_enabled=amp_enabled,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device=device,
            amp_enabled=amp_enabled,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "epoch_seconds": epoch_seconds,
            "gpu_peak_allocated_gib": allocated_gib,
            "gpu_peak_reserved_gib": reserved_gib,
        }
        history.append(record)

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_validation_accuracy=best_accuracy,
                model_name=config["model"]["name"],
            )
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_validation_accuracy=best_accuracy,
            model_name=config["model"]["name"],
        )
        metrics_document = {
            "best_validation_accuracy": best_accuracy,
            "total_training_seconds": time.perf_counter() - training_start,
            "epochs": history,
        }
        save_json(run_dir / "metrics.json", metrics_document)
        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%% | %.1fs | GPU peak %.2f/%.2f GiB allocated/reserved",
            epoch,
            epochs,
            train_metrics["loss"],
            100.0 * train_metrics["accuracy"],
            val_metrics["loss"],
            100.0 * val_metrics["accuracy"],
            epoch_seconds,
            allocated_gib,
            reserved_gib,
        )

    total_training_seconds = time.perf_counter() - training_start
    logger.info("Training complete in %.1fs; best validation accuracy %.2f%%", total_training_seconds, 100.0 * best_accuracy)
    final_record = history[-1]
    append_run_record(
        event_type="train",
        run_id=run_dir.name,
        config_path=run_dir / "config.yaml",
        checkpoint_path=run_dir / "best.pt",
        train_loss=float(final_record["train_loss"]),
        train_accuracy=float(final_record["train_accuracy"]),
        validation_loss=float(final_record["validation_loss"]),
        validation_accuracy=best_accuracy,
        num_samples=len(train_indices) + len(val_indices),
        duration_seconds=total_training_seconds,
        gpu_peak_allocated_gib=float(final_record["gpu_peak_allocated_gib"]),
        notes=f"epochs={epochs}",
    )


if __name__ == "__main__":
    main()
