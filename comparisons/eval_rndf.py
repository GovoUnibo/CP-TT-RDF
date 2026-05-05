#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from rndf_common import (
    RNDF_DEFAULT_DATASET,
    RNDF_DEFAULT_REPO,
    ensure_dir,
    load_checkpoint,
    load_rndf_dataset,
    save_json,
    set_seed,
    split_indices,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RNDF checkpoints on local .npy datasets.")
    parser.add_argument("--repo-dir", type=Path, default=RNDF_DEFAULT_REPO)
    parser.add_argument("--dataset-dir", type=Path, default=RNDF_DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/andrea/SDF_comparison/RNDF/models/weight/128_params.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/andrea/SDF_comparison/results/rndf_eval"))
    parser.add_argument("--split", choices=["full", "train", "val"], default="full")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-size", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-inside", action="store_true", default=True)
    parser.add_argument("--include-outside", action="store_true", default=True)
    parser.add_argument("--no-inside", dest="include_inside", action="store_false")
    parser.add_argument("--no-outside", dest="include_outside", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)

    bundle = load_rndf_dataset(
        dataset_dir=args.dataset_dir,
        include_inside=args.include_inside,
        include_outside=args.include_outside,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if args.split == "full":
        x_eval = bundle.x
        y_eval = bundle.y
    else:
        train_idx, val_idx = split_indices(bundle.x.shape[0], args.val_ratio, args.seed)
        idx = train_idx if args.split == "train" else val_idx
        if idx.size == 0:
            raise ValueError(f"Requested split {args.split!r} is empty.")
        x_eval = bundle.x[idx]
        y_eval = bundle.y[idx]

    dataset = TensorDataset(torch.from_numpy(x_eval), torch.from_numpy(y_eval))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model, checkpoint_meta = load_checkpoint(
        checkpoint_path=args.checkpoint,
        repo_dir=args.repo_dir,
        device=device,
        feature_size=args.feature_size,
        dropout_rate=args.dropout_rate,
    )

    preds = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu())
            targets.append(yb.cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    metrics = summarize_metrics(pred_all, target_all)

    payload = {
        "repo_dir": str(args.repo_dir),
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "split": args.split,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_dataset_files": len(bundle.files),
        "dataset_files": bundle.files,
        "checkpoint_meta_keys": sorted(checkpoint_meta.keys()) if checkpoint_meta else [],
        "metrics": metrics,
    }

    stem = args.checkpoint.stem
    output_path = output_dir / f"{stem}_{args.split}_metrics.json"
    save_json(output_path, payload)

    print(
        f"[eval] split={args.split} samples={metrics['num_samples']} "
        f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} "
        f"robot_min_mae={metrics['robot_min_distance_mae']:.6f}"
    )
    print(f"Wrote evaluation report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
