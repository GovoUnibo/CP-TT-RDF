#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark 3D RDF (Weights vs CP vs TT) — SOLO TEMPI
- Panda links
- FK included inside the timed closure
- NO correctness
- NO plot
- NO allocazioni OOM
- Warmup escluso
- Tempo medio = (tempo totale / iters)
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import mesh_to_sdf  # noqa: F401
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panda_test.panda_fk import fk_panda
from src.rdf_3Dcp import RDF_3D_CP
from src.rdf_3Dtt import RDF_TT
from src.rdf_weights import RDF_Weights

PREFERRED_LINKS = ("panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6")
PANDA_Q = np.array([0.4, -0.7, 0.0, -2.1, 0.1, 1.9, 0.8], dtype=np.float32)
PANDA_FINGER_Q = 0.0
RUN_GRAD = False  # metti True se vuoi misurare anche il ramo SDF + GRAD


def build_forward_tensor(link_names, device, dtype, q=PANDA_Q, finger_q=PANDA_FINGER_Q):
    fk_list = fk_panda(q=np.asarray(q, dtype=np.float32), include_gripper=False, finger_q=float(finger_q))
    link_idx = [int(name.replace("panda_link", "")) for name in link_names]
    stacked = np.stack([fk_list[idx] for idx in link_idx], axis=0)
    return torch.from_numpy(stacked).to(device=device, dtype=dtype)


# ---------------- utils ----------------

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def time_mean_ms(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()

    sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def make_points(L, P, device, dtype):
    pts = (torch.rand(P, 3, device=device, dtype=dtype) * 2.0 - 1.0) * 0.5
    return pts.unsqueeze(0).repeat(L, 1, 1).contiguous()


def as_point_list(value):
    if isinstance(value, int):
        return [int(value)]
    return [int(v) for v in value]


def common_links(ws_path: Path) -> list[str]:
    models = ws_path / "Models"
    w = {p.stem[:-2] for p in models.glob("*_w.pt")}
    cp = {p.stem[:-3] for p in models.glob("*_cp.pt")}
    tt = {p.stem[:-3] for p in models.glob("*_tt.pt")}
    return sorted(w & cp & tt)


def load_n_func(ws_path: Path, link_name: str) -> int:
    model_path = ws_path / "Models" / f"{link_name}_cp.pt"
    model_dict = torch.load(model_path, map_location="cpu")
    return int(model_dict["n_func"])


def choose_link_names(ws_path: Path) -> tuple[list[str], int]:
    links = common_links(ws_path)
    if not links:
        raise ValueError(f"Nessun link con modelli w/cp/tt trovato in {ws_path / 'Models'}")

    groups: dict[int, list[str]] = {}
    preferred_links = [ln for ln in PREFERRED_LINKS if ln in links]
    if not preferred_links:
        preferred_links = links

    for link_name in preferred_links:
        n_func = load_n_func(ws_path, link_name)
        groups.setdefault(n_func, []).append(link_name)

    best_n_func = max(groups, key=lambda n: (len(groups[n]), -n))
    selected = [ln for ln in PREFERRED_LINKS if ln in groups[best_n_func]]
    if not selected:
        selected = sorted(groups[best_n_func])
    return selected, int(best_n_func)


# ---------------- main ----------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ws_path = ROOT / "panda_test"
    robot_name = "panda_robot"
    link_names, n_func = choose_link_names(ws_path)
    L = len(link_names)

    # -------- CONFIG --------
    P_SDF = [3000, 5000]
    P_GRAD = 20000
    

    GET_MIN = False

    WARMUP_SDF, ITERS_SDF = 100, 500
    WARMUP_GRAD, ITERS_GRAD = 50, 200
    # ------------------------

    print(f"\nDevice={device}  dtype={dtype}")
    print(f"Links={link_names}  n_func={n_func}")
    p_sdf_list = as_point_list(P_SDF)
    p_grad_list = as_point_list(P_GRAD)
    if RUN_GRAD:
        print(f"L={L}  P_SDF={p_sdf_list}  P_GRAD={p_grad_list}\n")
        print(f"Points used: SDF={p_sdf_list}, SDF+GRAD={p_grad_list}")
    else:
        print(f"L={L}  P_SDF={p_sdf_list}  P_GRAD=disabled\n")
        print(f"Points used: SDF={p_sdf_list}, SDF+GRAD=disabled")
    print(f"FK q={PANDA_Q.tolist()}  finger_q={PANDA_FINGER_Q}")

    # ---- init CP ----
    rdf_cp = RDF_3D_CP(device=device.type, dtype=dtype)
    rdf_cp.init_robot_folder(str(ws_path), robot_name=robot_name)
    rdf_cp.add_models(link_names=link_names, robot_name=robot_name)
    rdf_cp.set_ordered_batch_params(link_names)

    # ---- init Weights ----
    rdf_w = RDF_Weights(device=device.type, dtype=dtype)
    rdf_w.init_robot_folder(str(ws_path), robot_name=robot_name)
    rdf_w.add_models(link_names=link_names, robot_name=robot_name)
    rdf_w.set_ordered_batch_params(link_names)

    # ---- init TT ----
    rdf_tt = RDF_TT(device=device.type, dtype=dtype)
    rdf_tt.init_robot_folder(str(ws_path), robot_name=robot_name)
    rdf_tt.add_models(link_names=link_names, robot_name=robot_name)
    rdf_tt.set_ordered_batch_params(link_names)

    # =====================================================
    # SDF ONLY
    # =====================================================
    for p_sdf in p_sdf_list:
        pts_sdf = make_points(L, p_sdf, device, dtype)

        def cp_sdf():
            forward_tensor = build_forward_tensor(link_names, device=device, dtype=dtype)
            rdf_cp.inference_link_batch(points=pts_sdf, get_grad=False, get_min=GET_MIN, forward_tensor=forward_tensor)

        def w_sdf():
            forward_tensor = build_forward_tensor(link_names, device=device, dtype=dtype)
            rdf_w.inference_link_batch(points=pts_sdf, get_grad=False, get_min=GET_MIN, forward_tensor=forward_tensor)

        def tt_sdf():
            forward_tensor = build_forward_tensor(link_names, device=device, dtype=dtype)
            rdf_tt.inference_link_batch(points=pts_sdf, get_grad=False, get_min=GET_MIN, forward_tensor=forward_tensor)

        cp_ms = time_mean_ms(cp_sdf, device, ITERS_SDF, WARMUP_SDF)
        w_ms = time_mean_ms(w_sdf, device, ITERS_SDF, WARMUP_SDF)
        tt_ms = time_mean_ms(tt_sdf, device, ITERS_SDF, WARMUP_SDF)

        print(f"[SDF ONLY | P={p_sdf}]")
        print(f"CP : {cp_ms:.3f} ms")
        print(f"W  : {w_ms:.3f} ms")
        print(f"TT : {tt_ms:.3f} ms")
        print(f"Speedup (W/CP): {w_ms / cp_ms:.2f}x")
        print(f"Speedup (W/TT): {w_ms / tt_ms:.2f}x")
        print(f"Speedup (TT/CP): {tt_ms / cp_ms:.2f}x\n")

    # =====================================================
    # SDF + GRAD
    # =====================================================
    if RUN_GRAD:
        for p_grad in p_grad_list:
            pts_grad = make_points(L, p_grad, device, dtype)

            def cp_grad():
                forward_tensor = build_forward_tensor(link_names, device=device, dtype=dtype)
                rdf_cp.inference_link_batch(
                    points=pts_grad,
                    get_grad=True,
                    get_min=GET_MIN,
                    forward_tensor=forward_tensor,
                )

            def w_grad():
                forward_tensor = build_forward_tensor(link_names, device=device, dtype=dtype)
                rdf_w.inference_link_batch(
                    points=pts_grad,
                    get_grad=True,
                    get_min=GET_MIN,
                    forward_tensor=forward_tensor,
                )

            def tt_grad():
                forward_tensor = build_forward_tensor(link_names, device=device, dtype=dtype)
                rdf_tt.inference_link_batch(
                    points=pts_grad,
                    get_grad=True,
                    get_min=GET_MIN,
                    forward_tensor=forward_tensor,
                )

            cp_ms = time_mean_ms(cp_grad, device, ITERS_GRAD, WARMUP_GRAD)
            w_ms = time_mean_ms(w_grad, device, ITERS_GRAD, WARMUP_GRAD)
            tt_ms = time_mean_ms(tt_grad, device, ITERS_GRAD, WARMUP_GRAD)

            print(f"[SDF + GRAD | P={p_grad}]")
            print(f"CP : {cp_ms:.3f} ms")
            print(f"W  : {w_ms:.3f} ms")
            print(f"TT : {tt_ms:.3f} ms")
            print(f"Speedup (W/CP): {w_ms / cp_ms:.2f}x")
            print(f"Speedup (W/TT): {w_ms / tt_ms:.2f}x")
            print(f"Speedup (TT/CP): {tt_ms / cp_ms:.2f}x\n")
    else:
        print("[SDF + GRAD] skipped (RUN_GRAD=False)\n")

    print("DONE.\n")


if __name__ == "__main__":
    main()
