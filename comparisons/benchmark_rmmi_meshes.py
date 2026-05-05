#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mm_neo  # noqa: F401  # Smoke test that the editable install is importable.
import numpy as np
import open3d as o3d
import trimesh


DEFAULT_POINT_COUNTS = (1_000, 10_000, 100_000)


@dataclass
class BenchmarkRow:
    mesh_name: str
    points: int
    vertices: int
    faces: int
    watertight: bool
    load_seconds: float
    scene_build_seconds: float
    sample_seconds: float
    distance_seconds: float
    signed_distance_seconds: float | None
    mean_unsigned_distance: float
    max_unsigned_distance: float
    mean_signed_distance: float | None
    min_signed_distance: float | None
    max_signed_distance: float | None
    bbox_diag: float
    noise_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark mesh sampling and distance queries in the local RMMI "
            "Python environment."
        )
    )
    parser.add_argument(
        "--meshes-dir",
        type=Path,
        default=Path("/home/andrea/SDF_comparison/Meshes"),
        help="Directory containing STL meshes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/andrea/SDF_comparison/results/rmmi_mesh_benchmark"),
        help="Directory where CSV and JSON reports will be written.",
    )
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=list(DEFAULT_POINT_COUNTS),
        help="Point counts to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used for deterministic sampling and perturbation.",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.01,
        help=(
            "Standard deviation for the normal perturbation as a ratio of the "
            "mesh bounding-box diagonal."
        ),
    )
    return parser.parse_args()


def load_mesh(mesh_path: Path) -> tuple[trimesh.Trimesh, float]:
    start = time.perf_counter()
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Scene {mesh_path} does not contain any geometry.")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{mesh_path} did not load as a trimesh.Trimesh.")
    return mesh, time.perf_counter() - start


def build_scene(mesh: trimesh.Trimesh) -> tuple[o3d.t.geometry.RaycastingScene, float]:
    start = time.perf_counter()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.faces, dtype=np.int32)
    triangle_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.Int32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(triangle_mesh)
    return scene, time.perf_counter() - start


def sample_query_points(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
    noise_std: float,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    surface_points, face_index = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals[face_index], dtype=np.float32)
    offsets = np.random.default_rng(seed).normal(
        loc=0.0,
        scale=noise_std,
        size=(count, 1),
    ).astype(np.float32)
    query_points = np.asarray(surface_points, dtype=np.float32) + normals * offsets
    return query_points, time.perf_counter() - start


def run_queries(
    scene: o3d.t.geometry.RaycastingScene,
    query_points: np.ndarray,
    compute_signed_distance: bool,
) -> tuple[np.ndarray, float, np.ndarray | None, float | None]:
    tensor_points = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)

    distance_start = time.perf_counter()
    unsigned = scene.compute_distance(tensor_points).numpy()
    distance_seconds = time.perf_counter() - distance_start

    signed = None
    signed_seconds = None
    if compute_signed_distance:
        signed_start = time.perf_counter()
        signed = scene.compute_signed_distance(tensor_points).numpy()
        signed_seconds = time.perf_counter() - signed_start

    return unsigned, distance_seconds, signed, signed_seconds


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

    mesh_paths = sorted(args.meshes_dir.glob("*.stl"))
    if not mesh_paths:
        raise FileNotFoundError(f"No STL meshes found in {args.meshes_dir}")

    rows: list[BenchmarkRow] = []
    summary: dict[str, object] = {
        "mm_neo_version": metadata.version("mm-neo"),
        "python": sys.version,
        "meshes_dir": str(args.meshes_dir),
        "point_counts": list(args.points),
        "seed": args.seed,
        "noise_ratio": args.noise_ratio,
        "rows": [],
    }
    summary["script"] = str(Path(__file__).resolve())

    for mesh_path in mesh_paths:
        mesh, load_seconds = load_mesh(mesh_path)
        bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        noise_std = max(bbox_diag * args.noise_ratio, 1e-6)
        scene, scene_build_seconds = build_scene(mesh)

        print(
            f"[mesh] {mesh_path.name}: "
            f"faces={len(mesh.faces)} vertices={len(mesh.vertices)} "
            f"watertight={mesh.is_watertight}"
        )

        for index, point_count in enumerate(args.points):
            query_points, sample_seconds = sample_query_points(
                mesh=mesh,
                count=point_count,
                seed=args.seed + index,
                noise_std=noise_std,
            )
            unsigned, distance_seconds, signed, signed_seconds = run_queries(
                scene=scene,
                query_points=query_points,
                compute_signed_distance=bool(mesh.is_watertight),
            )

            row = BenchmarkRow(
                mesh_name=mesh_path.name,
                points=point_count,
                vertices=int(len(mesh.vertices)),
                faces=int(len(mesh.faces)),
                watertight=bool(mesh.is_watertight),
                load_seconds=load_seconds,
                scene_build_seconds=scene_build_seconds,
                sample_seconds=sample_seconds,
                distance_seconds=distance_seconds,
                signed_distance_seconds=signed_seconds,
                mean_unsigned_distance=float(np.mean(unsigned)),
                max_unsigned_distance=float(np.max(unsigned)),
                mean_signed_distance=None if signed is None else float(np.mean(signed)),
                min_signed_distance=None if signed is None else float(np.min(signed)),
                max_signed_distance=None if signed is None else float(np.max(signed)),
                bbox_diag=bbox_diag,
                noise_std=noise_std,
            )
            rows.append(row)
            summary["rows"].append(asdict(row))
            signed_text = (
                "n/a"
                if signed_seconds is None
                else f"{signed_seconds:.4f}s"
            )
            print(
                f"  - {point_count:>6} pts | sample={sample_seconds:.4f}s "
                f"| distance={distance_seconds:.4f}s | signed={signed_text}"
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
