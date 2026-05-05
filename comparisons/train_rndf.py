#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rndf_common import (
    RNDF_DEFAULT_DATASET,
    RNDF_DEFAULT_REPO,
    DatasetBundle,
    ensure_dir,
    load_rndf_dataset,
    save_history_csv,
    save_json,
    set_seed,
    split_indices,
    summarize_metrics,
    create_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNDF on local .npy datasets.")
    parser.add_argument("--repo-dir", type=Path, default=RNDF_DEFAULT_REPO)
    parser.add_argument("--dataset-dir", type=Path, default=RNDF_DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/andrea/SDF_comparison/results/rndf_train"))
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--feature-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loss", choices=["smooth_l1", "l1", "mse"], default="smooth_l1")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-inside", action="store_true", default=True)
    parser.add_argument("--include-outside", action="store_true", default=True)
    parser.add_argument("--no-inside", dest="include_inside", action="store_false")
    parser.add_argument("--no-outside", dest="include_outside", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def loss_fn(name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if name == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if name == "l1":
        return F.l1_loss(pred, target)
    if name == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported loss {name}")


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            batch_loss = F.l1_loss(pred, yb, reduction="sum")
            total_loss += float(batch_loss.item())
            total_count += int(xb.shape[0])
            preds.append(pred.cpu())
            targets.append(yb.cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    metrics = summarize_metrics(pred_all, target_all)
    metrics["l1_sum"] = total_loss
    metrics["l1_mean"] = total_loss / max(total_count, 1)
    return metrics


def make_loaders(bundle: DatasetBundle, args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None, dict]:
    train_idx, val_idx = split_indices(bundle.x.shape[0], args.val_ratio, args.seed)

    x_train = torch.from_numpy(bundle.x[train_idx])
    y_train = torch.from_numpy(bundle.y[train_idx])
    train_dataset = TensorDataset(x_train, y_train)

    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    split_info = {
        "num_total": int(bundle.x.shape[0]),
        "num_train": int(train_idx.shape[0]),
        "num_val": int(val_idx.shape[0]),
    }
    if val_idx.size > 0:
        x_val = torch.from_numpy(bundle.x[val_idx])
        y_val = torch.from_numpy(bundle.y[val_idx])
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    return train_loader, val_loader, split_info


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    run_dir = ensure_dir(args.output_dir / args.run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")

    bundle = load_rndf_dataset(
        dataset_dir=args.dataset_dir,
        include_inside=args.include_inside,
        include_outside=args.include_outside,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    train_loader, val_loader, split_info = make_loaders(bundle, args)

    model = create_model(
        repo_dir=args.repo_dir,
        feature_size=args.feature_size,
        dropout_rate=args.dropout_rate,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history: list[dict] = []
    best_metric = float("inf")
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"

    print(f"[train] device={device} samples={split_info['num_total']} train={split_info['num_train']} val={split_info['num_val']}")
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(args.loss, pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * int(xb.shape[0])
            sample_count += int(xb.shape[0])

        scheduler.step()

        train_metrics = evaluate_model(model, train_loader, device)
        val_metrics = evaluate_model(model, val_loader, device) if val_loader is not None else None
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": epoch_loss / max(sample_count, 1),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_robot_min_distance_mae": train_metrics["robot_min_distance_mae"],
            "val_mae": None if val_metrics is None else val_metrics["mae"],
            "val_rmse": None if val_metrics is None else val_metrics["rmse"],
            "val_robot_min_distance_mae": None if val_metrics is None else val_metrics["robot_min_distance_mae"],
        }
        history.append(row)
        val_mae_text = "n/a" if val_metrics is None else f"{val_metrics['mae']:.6f}"
        print(
            f"[epoch {epoch:03d}] "
            f"train_mae={train_metrics['mae']:.6f} "
            f"val_mae={val_mae_text}"
        )

        monitor_metric = train_metrics["mae"] if val_metrics is None else val_metrics["mae"]
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "repo_dir": str(args.repo_dir),
                "dataset_dir": str(args.dataset_dir),
                "feature_size": args.feature_size,
                "dropout_rate": args.dropout_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "loss": args.loss,
                "include_inside": args.include_inside,
                "include_outside": args.include_outside,
                "dataset_files": bundle.files,
            },
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
            },
        }
        torch.save(checkpoint, last_path)
        if monitor_metric < best_metric:
            best_metric = monitor_metric
            torch.save(checkpoint, best_path)

    elapsed = time.perf_counter() - start_time
    save_history_csv(run_dir / "history.csv", history)
    save_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "elapsed_seconds": elapsed,
            "best_mae": best_metric,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "split": split_info,
            "history": history,
        },
    )

    print()
    print(f"Wrote checkpoints to {checkpoints_dir}")
    print(f"Wrote training summary to {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
