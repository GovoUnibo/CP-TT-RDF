#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_BATCH_SIZES = (1_000, 10_000, 100_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a public RMMI/iSDF architecture proxy and summarize "
            "public accuracy traces from vox_res.json files."
        )
    )
    parser.add_argument(
        "--rmmi-dir",
        type=Path,
        default=Path("/home/andrea/SDF_comparison/rmmi"),
    )
    parser.add_argument(
        "--isdf-dir",
        type=Path,
        default=Path("/home/andrea/SDF_comparison/iSDF_fb"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/andrea/SDF_comparison/iSDF_fb/results/iSDF/exp0"),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="apt_2_nav_*",
        help="Glob for public vox_res/config directories used in the accuracy summary.",
    )
    parser.add_argument(
        "--config-seq",
        type=str,
        default="apt_2_nav_1",
        help="Directory under results-root whose config.json is used for the proxy architecture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/andrea/SDF_comparison/results/rmmi_public_proxy"),
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ensure_imports(rmmi_dir: Path) -> None:
    import sys

    rmmi_str = str(rmmi_dir)
    if rmmi_str not in sys.path:
        sys.path.insert(0, rmmi_str)


def translate_public_config(config: dict) -> dict:
    model = config["model"]
    if "embedding" in model:
        return config

    translated = {
        "model": {
            "hidden_feature_size": model["hidden_feature_size"],
            "hidden_layers_block": model["hidden_layers_block"],
            "scale_output": model["scale_output"],
            "embedding": {
                "gauss_embed": model["gauss_embed"],
                "n_embed_funcs": model["n_embed_funcs"],
                "scale_input": model["scale_input"],
            },
        }
    }
    return translated


def build_proxy_folder(config_file: Path, output_dir: Path) -> Path:
    from isdf.modules.embedding import PostionalEncoding
    from isdf.modules.fc_map import SDFMap

    config = json.loads(config_file.read_text())
    translated = translate_public_config(config)

    model_dir = output_dir / "proxy_model"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "config.json").write_text(
        json.dumps(translated, indent=2), encoding="utf-8"
    )

    model_params = translated["model"]
    enc_params = model_params["embedding"]
    encoding = PostionalEncoding(
        enc_params["gauss_embed"],
        max_deg=enc_params["n_embed_funcs"],
        scale=enc_params["scale_input"],
    ).type(torch.float16)
    encoding = encoding.half()

    sdf_map = SDFMap(
        encoding,
        hidden_size=model_params["hidden_feature_size"],
        hidden_layers_block=model_params["hidden_layers_block"],
        scale_output=model_params["scale_output"],
    )

    checkpoint_path = checkpoint_dir / "last_step.pth"
    torch.save({"model_state_dict": sdf_map.state_dict()}, checkpoint_path)
    return model_dir


def benchmark_proxy(
    rmmi_dir: Path,
    model_dir: Path,
    device: torch.device,
    batch_sizes: list[int],
    repetitions: int,
    warmup: int,
) -> list[dict]:
    ensure_imports(rmmi_dir)
    from mm_neo.sdf import SDF

    sdf_model = SDF(str(model_dir), np.eye(4), device=str(device))
    rows: list[dict] = []

    for batch_size in batch_sizes:
        batch = torch.rand(batch_size, 3, device=device, dtype=torch.float32)
        with torch.inference_mode():
            for _ in range(warmup):
                _ = sdf_model.sdf_forward(batch)
                synchronize(device)

            durations: list[float] = []
            for _ in range(repetitions):
                synchronize(device)
                t0 = time.perf_counter()
                _ = sdf_model.sdf_forward(batch)
                synchronize(device)
                durations.append(time.perf_counter() - t0)

        avg_seconds = float(np.mean(durations))
        rows.append(
            {
                "batch_size": batch_size,
                "avg_seconds": avg_seconds,
                "avg_ms": avg_seconds * 1000.0,
                "queries_per_ms": batch_size / (avg_seconds * 1000.0),
                "min_ms": min(durations) * 1000.0,
                "max_ms": max(durations) * 1000.0,
            }
        )

    return rows


def summarize_public_accuracy(results_root: Path, pattern: str) -> dict:
    per_run = []
    for vox_file in sorted(results_root.glob(f"{pattern}/vox_res.json")):
        payload = json.loads(vox_file.read_text())
        best_visible_surf_vis = None
        best_rays_vis = None
        best_time = None
        for t_str, step_payload in payload.items():
            surf_val = step_payload["visible_surf"]["vis"]["av_l1"]
            rays_val = step_payload["rays"]["vis"]["av_l1"]
            if best_visible_surf_vis is None or surf_val < best_visible_surf_vis:
                best_visible_surf_vis = surf_val
                best_rays_vis = rays_val
                best_time = float(t_str)
        if best_visible_surf_vis is None:
            continue
        per_run.append(
            {
                "run": vox_file.parent.name,
                "best_time_s": best_time,
                "best_visible_surf_vis_l1_m": best_visible_surf_vis,
                "best_visible_surf_vis_l1_mm": best_visible_surf_vis * 1000.0,
                "rays_vis_l1_m_at_best_surface": best_rays_vis,
                "rays_vis_l1_mm_at_best_surface": best_rays_vis * 1000.0,
            }
        )

    if not per_run:
        raise FileNotFoundError(
            f"No vox_res.json files matched pattern {pattern!r} under {results_root}"
        )

    avg_surface_mm = sum(x["best_visible_surf_vis_l1_mm"] for x in per_run) / len(per_run)
    avg_rays_mm = sum(x["rays_vis_l1_mm_at_best_surface"] for x in per_run) / len(per_run)

    return {
        "pattern": pattern,
        "num_runs": len(per_run),
        "per_run": per_run,
        "mean_best_visible_surf_vis_l1_mm": avg_surface_mm,
        "mean_rays_vis_l1_mm_at_best_surface": avg_rays_mm,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config_file = args.results_root / args.config_seq / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(config_file)

    model_dir = build_proxy_folder(config_file, args.output_dir)
    timing_rows = benchmark_proxy(
        rmmi_dir=args.rmmi_dir,
        model_dir=model_dir,
        device=device,
        batch_sizes=args.batch_sizes,
        repetitions=args.repetitions,
        warmup=args.warmup,
    )
    accuracy_summary = summarize_public_accuracy(args.results_root, args.pattern)

    payload = {
        "device": str(device),
        "config_seq": args.config_seq,
        "proxy_model_dir": str(model_dir),
        "timing_rows": timing_rows,
        "accuracy_summary": accuracy_summary,
        "note": (
            "Accuracy comes from public vox_res.json traces (best visible-surface "
            "L1 over matching runs). Timing is a local architecture proxy measured "
            "with a randomly initialized compatible RMMI/iSDF checkpoint because "
            "public model weights were not bundled with the repo."
        ),
    }

    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "timing.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(timing_rows, csv_path)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
