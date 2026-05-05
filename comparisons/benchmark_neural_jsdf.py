#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


DEFAULT_BATCH_SIZES = (1_000, 10_000, 100_000)


@dataclass
class BenchmarkRow:
    device: str
    batch_size: int
    repetitions: int
    warmup: int
    avg_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float
    samples_per_second: float
    mean_prediction_cm: float
    mean_target_cm: float
    mae_cm: float
    max_abs_error_cm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the pretrained Neural-JSDF model on included dataset queries."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("/home/andrea/SDF_comparison/Neural-JSDF"),
        help="Path to the Neural-JSDF repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/andrea/SDF_comparison/results/neural_jsdf_benchmark"),
        help="Directory where benchmark artifacts will be written.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Timed repetitions per device and batch size.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup repetitions per device and batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when resampling dataset rows for large batches.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda"],
        help="Devices to benchmark. Unsupported devices are skipped.",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_devices(device_names: list[str]) -> list[torch.device]:
    resolved: list[torch.device] = []
    for device_name in device_names:
        if device_name == "cuda":
            if torch.cuda.is_available():
                resolved.append(torch.device("cuda"))
            else:
                print("[skip] cuda requested but not available")
        elif device_name == "cpu":
            resolved.append(torch.device("cpu"))
        else:
            raise ValueError(f"Unsupported device {device_name!r}")
    if not resolved:
        raise RuntimeError("No usable devices selected for the benchmark.")
    return resolved


def load_repo_model(repo_dir: Path, device: torch.device):
    nn_learning_dir = repo_dir / "learning" / "nn-learning"
    sys.path.insert(0, str(nn_learning_dir))
    from sdf.robot_sdf import RobotSdfCollisionNet  # pylint: disable=import-error

    tensor_args = {"device": device, "dtype": torch.float32}
    model = RobotSdfCollisionNet(
        in_channels=10,
        out_channels=9,
        layers=[256] * 4,
        skips=[],
    )
    model.load_weights(str(nn_learning_dir / "franka_collision_model.pt"), tensor_args)
    model.model.eval()
    return model


def load_dataset(repo_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = repo_dir / "learning" / "data-sampling" / "datasets" / "data_mesh_test.mat"
    data = loadmat(str(dataset_path))["dataset"]
    x = data[:, :10].astype(np.float32)
    y_cm = (100.0 * data[:, 10:]).astype(np.float32)
    return x, y_cm


def build_batch(
    x_all: np.ndarray,
    y_all: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= x_all.shape[0]:
        return x_all[:batch_size], y_all[:batch_size]
    rng = np.random.default_rng(seed + batch_size)
    idx = rng.choice(x_all.shape[0], size=batch_size, replace=True)
    return x_all[idx], y_all[idx]


def benchmark_one(
    model,
    device: torch.device,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    repetitions: int,
    warmup: int,
) -> BenchmarkRow:
    x_tensor = torch.from_numpy(x_batch).to(device=device, dtype=torch.float32)
    y_tensor = torch.from_numpy(y_batch).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.compute_signed_distance(x_tensor)
            synchronize(device)

        durations: list[float] = []
        pred = None
        for _ in range(repetitions):
            synchronize(device)
            start = time.perf_counter()
            pred = model.compute_signed_distance(x_tensor)
            synchronize(device)
            durations.append(time.perf_counter() - start)

    if pred is None:
        raise RuntimeError("No predictions were produced during the benchmark.")

    pred_detached = pred.detach()
    abs_error = (pred_detached - y_tensor).abs()
    avg_seconds = float(np.mean(durations))
    return BenchmarkRow(
        device=str(device),
        batch_size=int(x_tensor.shape[0]),
        repetitions=repetitions,
        warmup=warmup,
        avg_seconds=avg_seconds,
        std_seconds=float(np.std(durations)),
        min_seconds=float(np.min(durations)),
        max_seconds=float(np.max(durations)),
        samples_per_second=float(x_tensor.shape[0] / avg_seconds),
        mean_prediction_cm=float(pred_detached.mean().cpu()),
        mean_target_cm=float(y_tensor.mean().cpu()),
        mae_cm=float(abs_error.mean().cpu()),
        max_abs_error_cm=float(abs_error.max().cpu()),
    )


def write_csv(rows: list[BenchmarkRow], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    devices = resolve_devices(args.devices)
    x_all, y_all = load_dataset(args.repo_dir)
    rows: list[BenchmarkRow] = []

    summary: dict[str, object] = {
        "script": str(Path(__file__).resolve()),
        "repo_dir": str(args.repo_dir),
        "dataset_rows": int(x_all.shape[0]),
        "batch_sizes": list(args.batch_sizes),
        "devices": [str(device) for device in devices],
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "rows": [],
    }

    for device in devices:
        print(f"[device] {device}")
        model = load_repo_model(args.repo_dir, device)
        for batch_size in args.batch_sizes:
            x_batch, y_batch = build_batch(x_all, y_all, batch_size, args.seed)
            row = benchmark_one(
                model=model,
                device=device,
                x_batch=x_batch,
                y_batch=y_batch,
                repetitions=args.repetitions,
                warmup=args.warmup,
            )
            rows.append(row)
            summary["rows"].append(asdict(row))
            print(
                f"  - {batch_size:>6} | avg={row.avg_seconds:.6f}s "
                f"| throughput={row.samples_per_second:,.0f} samples/s "
                f"| mae={row.mae_cm:.4f} cm"
            )

    csv_path = args.output_dir / "benchmark.csv"
    json_path = args.output_dir / "benchmark.json"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Wrote CSV report to {csv_path}")
    print(f"Wrote JSON report to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
