"""Run test-only inference and create a validated pred_results.csv."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.data import CompetitionTestDataset
from src.models import load_baseline_model
from src.training import make_dataloader
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, project_path
from src.utils.experiment import load_class_mapping
from src.utils.reproducibility import seed_everything
from src.utils.run_registry import append_run_record
from src.utils.submission import predict_unlabeled, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    prediction_start = time.perf_counter()
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve() if args.config else checkpoint_path.parent / "config.yaml"
    output_path = args.output.resolve() if args.output else checkpoint_path.parent / "pred_results.csv"
    config = load_config(config_path)
    seed = int(config["seed"])
    seed_everything(seed)

    training_config = config["training"]
    device = torch.device(training_config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Baseline prediction requires CUDA.")
    _class_to_idx, idx_to_class = load_class_mapping(
        checkpoint_path.parent / "class_mapping.json"
    )

    model, preprocess = load_baseline_model(
        model_name=config["model"]["name"],
        download_root=project_path(config["model"]["download_root"]),
        num_classes=len(idx_to_class),
        device=device,
    )
    load_checkpoint(checkpoint_path, model=model, map_location=device)

    # The competition test directory is first touched here, after all training and
    # validation decisions have already been completed and persisted.
    dataset = CompetitionTestDataset(
        project_path(config["data"]["test_dir"]), transform=preprocess
    )
    loader = make_dataloader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        seed=seed,
        device=device,
    )
    filenames, predictions = predict_unlabeled(
        model,
        loader,
        device=device,
        amp_enabled=bool(training_config["amp"]),
    )
    write_submission(
        output_path,
        filenames=filenames,
        predictions=predictions,
        idx_to_class=idx_to_class,
        expected_filenames=dataset.filenames,
    )
    append_run_record(
        event_type="predict",
        run_id=checkpoint_path.parent.name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        artifact_path=output_path,
        num_predictions=len(predictions),
        duration_seconds=time.perf_counter() - prediction_start,
    )
    print(f"Wrote and validated {len(predictions)} predictions: {output_path}")


if __name__ == "__main__":
    main()
