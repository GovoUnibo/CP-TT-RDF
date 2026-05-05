from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


RNDF_DEFAULT_REPO = Path("/home/andrea/SDF_comparison/RNDF")
RNDF_DEFAULT_DATASET = RNDF_DEFAULT_REPO / "dataset"


@dataclass
class DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    files: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_rndf_model_class(repo_dir: Path):
    models_dir = repo_dir / "models"
    models_dir_str = str(models_dir)
    if models_dir_str not in sys.path:
        sys.path.insert(0, models_dir_str)
    from networks import RobotNDF  # type: ignore

    return RobotNDF


def collect_dataset_files(
    dataset_dir: Path,
    include_inside: bool = True,
    include_outside: bool = True,
) -> list[Path]:
    files: list[Path] = []
    if include_inside:
        files.extend(sorted((dataset_dir / "inside").glob("*.npy")))
    if include_outside:
        files.extend(sorted((dataset_dir / "outside").glob("*.npy")))
    if not files:
        raise FileNotFoundError(f"No RNDF dataset files found under {dataset_dir}")
    return files


def load_rndf_dataset(
    dataset_dir: Path,
    include_inside: bool = True,
    include_outside: bool = True,
    max_samples: int | None = None,
    seed: int = 42,
) -> DatasetBundle:
    files = collect_dataset_files(
        dataset_dir=dataset_dir,
        include_inside=include_inside,
        include_outside=include_outside,
    )
    arrays = [np.load(path).astype(np.float32) for path in files]
    data = np.concatenate(arrays, axis=0)

    if max_samples is not None and max_samples < data.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(data.shape[0], size=max_samples, replace=False)
        data = data[idx]

    x = data[:, :10].astype(np.float32)
    # Trimesh returns positive inside, negative outside. RNDF inference flips the sign.
    y = (-1.0 * data[:, 10:]).astype(np.float32)
    return DatasetBundle(x=x, y=y, files=[str(path) for path in files])


def split_indices(
    num_samples: int,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    idx = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(round(num_samples * val_ratio))
    if val_ratio > 0 and val_size == 0:
        val_size = 1
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    if train_idx.size == 0:
        raise ValueError("Validation split consumed the full dataset.")
    return train_idx, val_idx


def infer_feature_size_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    weight = state_dict["backbone.fc_1.weight"]
    return int(weight.shape[0])


def create_model(
    repo_dir: Path,
    feature_size: int,
    dropout_rate: float,
    device: torch.device,
) -> torch.nn.Module:
    RobotNDF = import_rndf_model_class(repo_dir)
    model = RobotNDF(N=feature_size, dropout_rate=dropout_rate)
    return model.to(device)


def load_checkpoint(
    checkpoint_path: Path,
    repo_dir: Path,
    device: torch.device,
    feature_size: int | None = None,
    dropout_rate: float = 0.2,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint
        if feature_size is None:
            feature_size = int(checkpoint.get("config", {}).get("feature_size", 0)) or None
        dropout_rate = float(checkpoint.get("config", {}).get("dropout_rate", dropout_rate))
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")

    if feature_size is None:
        feature_size = infer_feature_size_from_state_dict(state_dict)

    model = create_model(
        repo_dir=repo_dir,
        feature_size=feature_size,
        dropout_rate=dropout_rate,
        device=device,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, metadata


def summarize_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    abs_error = (pred - target).abs()
    sq_error = (pred - target).pow(2)
    min_pred = pred.min(dim=1).values
    min_target = target.min(dim=1).values
    min_abs_error = (min_pred - min_target).abs()

    return {
        "num_samples": int(pred.shape[0]),
        "num_outputs": int(pred.shape[1]),
        "mae": float(abs_error.mean().item()),
        "rmse": float(torch.sqrt(sq_error.mean()).item()),
        "max_abs_error": float(abs_error.max().item()),
        "robot_min_distance_mae": float(min_abs_error.mean().item()),
        "robot_min_distance_rmse": float(torch.sqrt((min_pred - min_target).pow(2).mean()).item()),
        "mean_prediction": float(pred.mean().item()),
        "mean_target": float(target.mean().item()),
        "per_link_mae": [float(v) for v in abs_error.mean(dim=0).tolist()],
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
