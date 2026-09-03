"""Incremental smoke test; later stages extend this through CUDA inference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CompetitionTestDataset, LabeledImageDataset, load_or_create_split
from src.models import count_parameters, load_baseline_model
from src.training import evaluate, make_dataloader, train_one_epoch
from src.utils.checkpoint import load_checkpoint, save_checkpoint
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
from src.utils.submission import predict_unlabeled, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIC baseline smoke test")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Run only data integrity checks without loading CLIP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config["seed"]))
    data_config = config["data"]

    train_dataset = LabeledImageDataset(project_path(data_config["train_dir"]))
    manifest, created = load_or_create_split(
        train_dataset,
        project_path(data_config["split_manifest"]),
        val_ratio=float(data_config["val_ratio"]),
        seed=int(config["seed"]),
    )
    train_paths = {record.path for record in manifest.train}
    val_paths = {record.path for record in manifest.validation}
    assert train_paths.isdisjoint(val_paths)
    assert set(train_dataset.class_to_idx.values()) == set(range(len(train_dataset.classes)))
    assert all("test" not in Path(path).parts for path in train_paths | val_paths)

    print("Data smoke test passed")
    print(f"  manifest: {project_path(data_config['split_manifest'])} ({'created' if created else 'reused'})")
    print(f"  classes: {len(train_dataset.classes)}")
    print(f"  train samples: {len(manifest.train)}")
    print(f"  validation samples: {len(manifest.validation)}")

    if args.data_only:
        return

    training_config = config["training"]
    device = torch.device(training_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The baseline smoke test requires a real CUDA device.")

    model, preprocess = load_baseline_model(
        model_name=config["model"]["name"],
        download_root=project_path(config["model"]["download_root"]),
        num_classes=len(train_dataset.classes),
        device=device,
    )
    train_dataset.transform = preprocess
    train_indices, val_indices = manifest.indices(train_dataset)
    batch_size = 2
    train_loader = make_dataloader(
        Subset(train_dataset, train_indices[:batch_size]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        seed=int(config["seed"]),
        device=device,
    )
    val_loader = make_dataloader(
        Subset(train_dataset, val_indices[:batch_size]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        seed=int(config["seed"]),
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
    smoke_start = time.perf_counter()
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

    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad is not None for parameter in model.classifier.parameters())
    counts = count_parameters(model)
    expected_trainable = model.feature_dim * model.num_classes + model.num_classes
    assert counts.trainable == expected_trainable

    run_dir = create_run_directory("smoke_test")
    logger = create_logger(run_dir)
    save_yaml(run_dir / "config.yaml", config)
    save_class_mapping(
        run_dir / "class_mapping.json",
        train_dataset.class_to_idx,
        train_dataset.idx_to_class,
    )
    for checkpoint_name in ("best.pt", "last.pt"):
        save_checkpoint(
            run_dir / checkpoint_name,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=1,
            best_validation_accuracy=val_metrics["accuracy"],
            model_name=config["model"]["name"],
        )

    saved_weight = model.classifier.weight.detach().clone()
    with torch.no_grad():
        model.classifier.weight.zero_()
    load_checkpoint(
        run_dir / "best.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        map_location=device,
    )
    assert torch.equal(model.classifier.weight, saved_weight)

    # Test data is instantiated only now, for the final inference stage.
    test_dataset = CompetitionTestDataset(
        project_path(data_config["test_dir"]), transform=preprocess
    )
    test_subset = Subset(test_dataset, list(range(min(2, len(test_dataset)))))
    test_loader = make_dataloader(
        test_subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        seed=int(config["seed"]),
        device=device,
    )
    filenames, predictions = predict_unlabeled(
        model,
        test_loader,
        device=device,
        amp_enabled=amp_enabled,
    )
    expected_filenames = test_dataset.filenames[: len(test_subset)]
    write_submission(
        run_dir / "pred_results.csv",
        filenames=filenames,
        predictions=predictions,
        idx_to_class=train_dataset.idx_to_class,
        expected_filenames=expected_filenames,
    )
    metrics = {
        "best_validation_accuracy": val_metrics["accuracy"],
        "total_training_seconds": time.perf_counter() - smoke_start,
        "epochs": [
            {
                "epoch": 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
            }
        ],
    }
    save_json(run_dir / "metrics.json", metrics)
    logger.info("Smoke checkpoint save/load passed")
    append_run_record(
        event_type="smoke_test",
        run_id=run_dir.name,
        config_path=run_dir / "config.yaml",
        checkpoint_path=run_dir / "best.pt",
        artifact_path=run_dir / "pred_results.csv",
        train_loss=train_metrics["loss"],
        train_accuracy=train_metrics["accuracy"],
        validation_loss=val_metrics["loss"],
        validation_accuracy=val_metrics["accuracy"],
        num_samples=int(train_metrics["num_samples"] + val_metrics["num_samples"]),
        num_predictions=len(predictions),
        duration_seconds=metrics["total_training_seconds"],
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / (1024**3),
        notes="2 train + 2 validation + 2 inference images",
    )

    print("CUDA training smoke test passed")
    print(f"  device: {torch.cuda.get_device_name(device)}")
    print(f"  feature dimension: {model.feature_dim}")
    print(f"  train loss: {train_metrics['loss']:.4f}")
    print(f"  validation loss: {val_metrics['loss']:.4f}")
    print(f"  total parameters: {counts.total:,}")
    print(f"  trainable parameters: {counts.trainable:,}")
    print(f"  inference images: {len(predictions)}")
    print(f"  run directory: {run_dir}")


if __name__ == "__main__":
    main()
