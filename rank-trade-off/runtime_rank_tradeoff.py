#!/usr/bin/env python3
"""Independent GPU-runtime sweep for dense, CP and TT Bernstein RDF models."""

from __future__ import annotations

import csv
import gc
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def discover_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "panda_test").is_dir():
            return candidate
    raise RuntimeError(f"Repository root not found from: {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = discover_repo_root(SCRIPT_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rdf_3Dcp import RDF_3D_CP  # noqa: E402
from src.rdf_3Dtt import RDF_TT  # noqa: E402
from src.rdf_weights import RDF_Weights  # noqa: E402


LINK_NAME = "panda_link0"
POLYNOMIAL_ORDERS = (8, 16, 24)
CP_RANKS = (4, 8, 16, 24, 32, 48)
TT_RANKS = (2, 4, 8, 12, 16, 24)
DENSE_TRAIN_ITERS = 800
CP_TRAIN_ITERS = 800
TT_TRAIN_ITERS = 800
RUNTIME_BATCH_SIZE = 10_000
WARMUP_PASSES = 10
TIMING_PASSES = 50
RUNTIME_OUTLIER_MAD_Z = 3.5
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float32

WS_PATH = ROOT / "panda_test"
RESULT_DIR = SCRIPT_DIR / "result"
CSV_PATH = RESULT_DIR / f"{LINK_NAME}_runtime.csv"


def backup_models() -> list[tuple[Path, Path, bool]]:
    saved = []
    for suffix in ("_w.pt", "_cp.pt", "_tt.pt"):
        model = WS_PATH / "Models" / f"{LINK_NAME}{suffix}"
        backup = model.with_name(model.name + ".runtime_backup")
        if backup.exists():
            shutil.copy2(backup, model)
            backup.unlink()
        existed = model.exists()
        if existed:
            shutil.copy2(model, backup)
        saved.append((model, backup, existed))
    return saved


def restore_models(saved: list[tuple[Path, Path, bool]]) -> None:
    for model, backup, existed in saved:
        if existed and backup.exists():
            shutil.copy2(backup, model)
            backup.unlink()
        elif not existed and model.exists():
            model.unlink()


def trim_outliers(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad <= np.finfo(values.dtype).eps:
        return values
    z = 0.6745 * (values - median) / mad
    kept = values[np.abs(z) <= RUNTIME_OUTLIER_MAD_Z]
    return kept if kept.size >= max(3, values.size // 2) else values


def parameter_bytes(rdf, representation: str) -> int:
    model = getattr(rdf, LINK_NAME + rdf.model_extension)
    tensors = (
        (model.weights,) if representation == "dense" else
        (model.A, model.B, model.C, model.lamd) if representation == "cp" else
        (model.G1, model.G2, model.G3)
    )
    return sum(int(t.numel() * t.element_size()) for t in tensors)


def benchmark(rdf, representation: str) -> tuple[float, int]:
    model = getattr(rdf, LINK_NAME + rdf.model_extension)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    normalized = torch.rand((RUNTIME_BATCH_SIZE, 3), generator=generator, dtype=DTYPE) * 2.0 - 1.0
    centroid = torch.as_tensor(model.centroid_offset, device=DEVICE, dtype=DTYPE).reshape(1, 3)
    scale = torch.as_tensor(model.scale_factor, device=DEVICE, dtype=DTYPE).reshape(1, 1)
    points = (normalized.to(DEVICE) * scale + centroid).contiguous()
    for _ in range(WARMUP_PASSES):
        rdf.sdf_kernel_for_tests(LINK_NAME, points, get_grad=False, get_min=False)
    torch.cuda.synchronize()
    samples = []
    for _ in range(TIMING_PASSES):
        torch.cuda.synchronize()
        start = time.perf_counter()
        rdf.sdf_kernel_for_tests(LINK_NAME, points, get_grad=False, get_min=False)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return float(np.median(trim_outliers(np.asarray(samples, dtype=np.float64)))), parameter_bytes(rdf, representation)


def train_dense(n: int):
    rdf = RDF_Weights(device=DEVICE, dtype=DTYPE)
    rdf.init_robot_folder(str(WS_PATH), robot_name="panda_robot")
    rdf.train_links([LINK_NAME], n_func=n, iters=DENSE_TRAIN_ITERS, robot_name="panda_robot")
    rdf.add_models([LINK_NAME], robot_name="panda_robot")
    return rdf


def train_cp(n: int, rank: int):
    rdf = RDF_3D_CP(device=DEVICE, dtype=DTYPE)
    rdf.init_robot_folder(str(WS_PATH), robot_name="panda_robot")
    rdf.train_links([LINK_NAME], n_func=n, ranks=rank, iters=CP_TRAIN_ITERS, robot_name="panda_robot")
    rdf.add_models([LINK_NAME], robot_name="panda_robot")
    return rdf


def train_tt(n: int, rank: int):
    rdf = RDF_TT(device=DEVICE, dtype=DTYPE)
    rdf.init_robot_folder(str(WS_PATH), robot_name="panda_robot")
    rdf.train_links([LINK_NAME], n_func=n, ranks=(rank, rank), iters=TT_TRAIN_ITERS, robot_name="panda_robot")
    rdf.add_models([LINK_NAME], robot_name="panda_robot")
    return rdf


def plot(rows: list[dict[str, object]]) -> None:
    for representation, xlabel in (("cp", "CP rank $R$"), ("tt", "TT rank $r_1=r_2$")):
        fig, ax = plt.subplots(figsize=(14.5, 8.6))
        for color, n in zip(("tab:blue", "tab:orange", "tab:green"), POLYNOMIAL_ORDERS):
            curve = sorted([r for r in rows if r["representation"] == representation and r["N"] == n], key=lambda r: r["rank"])
            ax.plot([r["rank"] for r in curve], [r["runtime_ms"] for r in curve], color=color, linewidth=3.2, label=f"N={n}")
            dense = next((r for r in rows if r["representation"] == "dense" and r["N"] == n), None)
            if dense:
                ax.axhline(dense["runtime_ms"], color=color, linestyle="--", linewidth=3.2, alpha=0.8)
        ax.set_title(f"{representation.upper()} GPU inference runtime", fontsize=13)
        ax.set_xlabel(xlabel, fontsize=23)
        ax.set_ylabel("Inference time [ms]", fontsize=23)
        ax.tick_params(axis="both", which="both", labelsize=22)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=18)
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.88)
        fig.savefig(RESULT_DIR / f"{LINK_NAME}_{representation}_runtime_benchmark.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU runtime benchmark.")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    saved = backup_models()
    rows = []
    try:
        for n in POLYNOMIAL_ORDERS:
            for representation, trainer, ranks in (("dense", train_dense, (0,)), ("cp", train_cp, CP_RANKS), ("tt", train_tt, TT_RANKS)):
                for rank in ranks:
                    if representation == "tt" and rank > n:
                        continue
                    rdf = trainer(n) if representation == "dense" else trainer(n, rank)
                    runtime_ms, storage = benchmark(rdf, representation)
                    rows.append({"representation": representation, "N": n, "rank": rank, "runtime_ms": runtime_ms, "parameter_bytes": storage})
                    print(f"{representation:>5s} N={n:2d} rank={rank:2d}: {runtime_ms:.4f} ms")
                    del rdf
                    gc.collect()
                    torch.cuda.empty_cache()
        with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("representation", "N", "rank", "runtime_ms", "parameter_bytes"))
            writer.writeheader()
            writer.writerows(rows)
        plot(rows)
    finally:
        restore_models(saved)


if __name__ == "__main__":
    main()
