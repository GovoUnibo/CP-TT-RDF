#!/usr/bin/env python3
"""Measure CP/TT compression error against dense Bernstein tensors by order."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

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

from src.rdf_weights import RDF_Weights  # noqa: E402


LINK_NAME = "panda_link0"
POLYNOMIAL_ORDERS = (8, 16, 24)
CP_RANKS = (4, 8, 16, 24, 32, 48)
TT_RANKS = (2, 4, 8, 12, 16, 24)
DENSE_TRAIN_ITERS = 200
DEVICE = "cuda"
DTYPE = torch.float32

WS_PATH = ROOT / "panda_test"
RESULT_DIR = SCRIPT_DIR / "result"
CSV_PATH = RESULT_DIR / f"{LINK_NAME}_weight_decomposition.csv"


def load_or_train_dense(n_func: int) -> RDF_Weights:
    rdf = RDF_Weights(device=DEVICE, dtype=DTYPE)
    rdf.init_robot_folder(str(WS_PATH), robot_name="panda_robot")
    try:
        rdf.add_models([LINK_NAME], robot_name="panda_robot")
        model = getattr(rdf, LINK_NAME + rdf.model_extension)
        if int(model.n_func) == n_func:
            return rdf
    except (FileNotFoundError, AttributeError, TypeError, ValueError):
        pass

    rdf.train_links(
        link_names=[LINK_NAME],
        n_func=n_func,
        iters=DENSE_TRAIN_ITERS,
        robot_name="panda_robot",
        debug=False,
    )
    rdf.add_models([LINK_NAME], robot_name="panda_robot")
    return rdf


def backup_dense_model() -> tuple[Path, Path, bool]:
    model_path = WS_PATH / "Models" / f"{LINK_NAME}_w.pt"
    backup_path = model_path.with_name(model_path.name + ".weight_rank_backup")
    if backup_path.exists():
        shutil.copy2(backup_path, model_path)
        backup_path.unlink()
    existed = model_path.exists()
    if existed:
        shutil.copy2(model_path, backup_path)
    return model_path, backup_path, existed


def restore_dense_model(model_path: Path, backup_path: Path, existed: bool) -> None:
    if existed and backup_path.exists():
        shutil.copy2(backup_path, model_path)
        backup_path.unlink()
    elif not existed and model_path.exists():
        model_path.unlink()


def write_csv(rows: list[dict[str, object]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ("representation", "N", "rank", "rmse_weights", "parameter_count")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the existing dense RDF training path.")
    model_path, backup_path, existed = backup_dense_model()
    try:
        rows: list[dict[str, object]] = []
        for n_func in POLYNOMIAL_ORDERS:
            rdf = load_or_train_dense(n_func)
            for rank in CP_RANKS:
                metrics = rdf.decompose_weights_cp(LINK_NAME, rank=rank)
                rows.append({"representation": "cp", "N": n_func, "rank": rank, "rmse_weights": metrics["rmse"], "parameter_count": metrics["params"]})
            for rank in TT_RANKS:
                if rank > n_func:
                    continue
                metrics = rdf.decompose_weights_tt(LINK_NAME, ranks=(rank, rank))
                rows.append({"representation": "tt", "N": n_func, "rank": rank, "rmse_weights": metrics["rmse"], "parameter_count": metrics["params"]})
            del rdf
        write_csv(rows)
    finally:
        restore_dense_model(model_path, backup_path, existed)

    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
