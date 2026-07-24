#!/usr/bin/env python3
"""Independent parameter-storage sweep for dense, CP and TT RDF models."""

from __future__ import annotations

import csv
import gc
import shutil
import sys
from pathlib import Path

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
DENSE_TRAIN_ITERS = 100
CP_TRAIN_ITERS = 100
TT_TRAIN_ITERS = 100
PEAK_BATCH_SIZE = 100_000
DEVICE = "cuda"
DTYPE = torch.float32

WS_PATH = ROOT / "panda_test"
RESULT_DIR = SCRIPT_DIR / "result"
CSV_PATH = RESULT_DIR / f"{LINK_NAME}_memory.csv"


def backup_models() -> list[tuple[Path, Path, bool]]:
    backups = []
    for suffix in ("_w.pt", "_cp.pt", "_tt.pt"):
        model = WS_PATH / "Models" / f"{LINK_NAME}{suffix}"
        backup = model.with_name(model.name + ".memory_backup")
        if backup.exists():
            shutil.copy2(backup, model)
            backup.unlink()
        existed = model.exists()
        if existed:
            shutil.copy2(model, backup)
        backups.append((model, backup, existed))
    return backups


def restore_models(backups: list[tuple[Path, Path, bool]]) -> None:
    for model, backup, existed in backups:
        if existed and backup.exists():
            shutil.copy2(backup, model)
            backup.unlink()
        elif not existed and model.exists():
            model.unlink()


def stored_parameter_bytes(rdf, representation: str) -> int:
    model = getattr(rdf, LINK_NAME + rdf.model_extension)
    tensors = (
        (model.weights,) if representation == "dense" else
        (model.A, model.B, model.C, model.lamd) if representation == "cp" else
        (model.G1, model.G2, model.G3)
    )
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


@torch.inference_mode()
def peak_gpu_bytes(rdf) -> int:
    """Peak allocated GPU memory for one fixed inference batch."""
    model = getattr(rdf, LINK_NAME + rdf.model_extension)
    generator = torch.Generator(device="cpu").manual_seed(0)
    normalized = torch.rand((PEAK_BATCH_SIZE, 3), generator=generator, dtype=DTYPE) * 2.0 - 1.0
    centroid = torch.as_tensor(model.centroid_offset, device=DEVICE, dtype=DTYPE).reshape(1, 3)
    scale = torch.as_tensor(model.scale_factor, device=DEVICE, dtype=DTYPE).reshape(1, 1)
    points = (normalized.to(DEVICE) * scale + centroid).contiguous()

    rdf.sdf_kernel_for_tests(LINK_NAME, points, get_grad=False, get_min=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rdf.sdf_kernel_for_tests(LINK_NAME, points, get_grad=False, get_min=False)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the existing RDF training path.")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    backups = backup_models()
    rows = []
    try:
        for n in POLYNOMIAL_ORDERS:
            for representation, trainer, ranks in (("dense", train_dense, (0,)), ("cp", train_cp, CP_RANKS), ("tt", train_tt, TT_RANKS)):
                for rank in ranks:
                    if representation == "tt" and rank > n:
                        continue
                    rdf = trainer(n) if representation == "dense" else trainer(n, rank)
                    byte_count = stored_parameter_bytes(rdf, representation)
                    peak_bytes = peak_gpu_bytes(rdf)
                    rows.append({"representation": representation, "N": n, "rank": rank, "parameter_bytes": byte_count, "parameter_kib": byte_count / 1024.0, "peak_gpu_bytes": peak_bytes, "peak_gpu_gb": peak_bytes / 1e9})
                    print(f"{representation:>5s} N={n:2d} rank={rank:2d}: peak={peak_bytes / 1e9:.3f} GB")
                    del rdf
                    gc.collect()
                    torch.cuda.empty_cache()
        with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("representation", "N", "rank", "parameter_bytes", "parameter_kib", "peak_gpu_bytes", "peak_gpu_gb"))
            writer.writeheader()
            writer.writerows(rows)
    finally:
        restore_models(backups)
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
