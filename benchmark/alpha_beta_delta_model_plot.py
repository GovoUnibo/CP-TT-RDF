#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.rdf_3Dcp import RDF_3D_CP
from src.rdf_3Dtt import RDF_TT
from src.rdf_weights import RDF_Weights

WS_PATH = ROOT / "panda_test"
LINK_NAME = None
PREFERRED_LINKS = ("panda_link3", "panda_link2", "panda_link1")
DEVICE = "auto"
DTYPE = torch.float32
BATCH_SIZES = tuple(2**k for k in range(2, 21))
SEARCH_MAX_BATCH = 1 << 25
BASE_PLOT_POINTS = 8
TAIL_OOM_FRACS = ( 0.97, 0.995)
CURVE_POINTS = 160
WARMUP = 2
ITERS = 3
SEED = 1234
FLOP_CONST = 2.0
CPEAK_TFLOPS = None
BW_GB_S = None
OUT_PATH = ROOT / "benchmark" / "alpha_beta_delta_model_plot.png"
KNOWN_GPU_SPECS = {
    "nvidia geforce rtx 3080": (29.8, 760.0),
    "nvidia geforce rtx 4060 ti": (22.0, 288.0),
}
METHODS = {"weights": RDF_Weights, "cp": RDF_3D_CP, "tt": RDF_TT}
LABELS = {"weights": "Weights", "cp": "CP", "tt": "TT"}
STYLE = {"weights": ("tab:red", "o"), "cp": ("tab:orange", "s"), "tt": ("tab:blue", "^")}
TICK_FONT_SIZE = 22
LEGEND_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 23
TITLE_FONT_SIZE = 13
OOM_LABEL_FONT_SIZE = 23
OOM_LABEL_Y = 1.02
MEASURED_LINE_WIDTH = 3.2
MODEL_LINE_WIDTH = 3.0
OOM_LINE_WIDTH = 2.0
MARKER_SIZE = 6.8


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync(device)


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def choose_link(ws_path: Path) -> str:
    models = ws_path / "Models"
    common = {p.stem[:-2] for p in models.glob("*_w.pt")}
    common &= {p.stem[:-3] for p in models.glob("*_cp.pt")}
    common &= {p.stem[:-3] for p in models.glob("*_tt.pt")}
    common = sorted(common)
    if LINK_NAME:
        if LINK_NAME not in common:
            raise ValueError(f"Link '{LINK_NAME}' non disponibile. Disponibili: {common}")
        return LINK_NAME
    for name in PREFERRED_LINKS:
        if name in common:
            return name
    if not common:
        raise ValueError(f"Nessun link con modelli w/cp/tt in {models}")
    return common[0]


def resolve_device() -> torch.device:
    if DEVICE == "cpu":
        return torch.device("cpu")
    if DEVICE == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_engine(engine_cls, ws_path: Path, link: str, device: torch.device):
    engine = engine_cls(device=str(device), dtype=DTYPE)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.init_robot_folder(str(ws_path), robot_name="")
        engine.add_models([link], robot_name="")
    return engine, getattr(engine, link + engine.model_extension)


def machine_constants(device: torch.device) -> tuple[float, float, str]:
    if CPEAK_TFLOPS is not None and BW_GB_S is not None:
        return CPEAK_TFLOPS * 1e12, BW_GB_S * 1e9, "manual"
    if device.type != "cuda":
        return float("nan"), float("nan"), "missing"
    name = str(torch.cuda.get_device_properties(device).name).lower()
    spec = next((v for k, v in KNOWN_GPU_SPECS.items() if k in name), None)
    if spec is None:
        return float("nan"), float("nan"), "missing"
    return (CPEAK_TFLOPS or spec[0]) * 1e12, (BW_GB_S or spec[1]) * 1e9, "lookup"


def gamma_eta(method: str, model) -> tuple[float, float, float]:
    if method == "weights":
        n = float(model.n_func)
        return n**3, 1.0 / (n * n), 1.0 / (n**3)
    if method == "cp":
        n, r = float(model.A.shape[0]), float(model.A.shape[1])
        d = 3.0 * n * r + r
        return d, n / d, 1.0 / d
    n, r1, r2 = float(model.G1.shape[1]), float(model.G1.shape[2]), float(model.G3.shape[0])
    d = n * (r1 * r2 + r1 + r2)
    return d, 1.0 / (r1 * r2 + r1 + r2), 1.0 / d


def num_functions(method: str, model) -> int:
    if method == "weights":
        return int(model.n_func)
    if method == "cp":
        return int(model.A.shape[0])
    return int(model.G1.shape[1])


def build_points(model, count: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    low, high = float(model.domain_min), float(model.domain_max)
    pts = torch.rand((count, 3), generator=g, dtype=DTYPE) * (high - low) + low
    centroid = torch.as_tensor(model.centroid_offset, device=device, dtype=DTYPE).reshape(1, 3)
    scale = torch.as_tensor(model.scale_factor, device=device, dtype=DTYPE).reshape(1, 1)
    return (centroid + scale * pts.to(device=device)).contiguous()


def get_test_kernel(engine):
    kernel = getattr(engine, "sdf_kernel_for_tests", None)
    if kernel is not None:
        return kernel
    kernel = getattr(engine, "inner_kernel_for_tests", None)
    if kernel is not None:
        return kernel
    return engine.inference_link


def time_kernel(engine, link: str, points: torch.Tensor, device: torch.device) -> float:
    kernel = get_test_kernel(engine)
    with torch.no_grad():
        for _ in range(WARMUP):
            kernel(link, points, get_grad=False, get_min=False)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            kernel(link, points, get_grad=False, get_min=False)
        sync(device)
    return (time.perf_counter() - t0) / ITERS


def get_points(cache: dict[int, torch.Tensor], model, p: int, device: torch.device) -> torch.Tensor:
    if p not in cache:
        cache[p] = build_points(model, p, device)
    return cache[p]


def try_batch(engine, link: str, model, p: int, device: torch.device, cache: dict[int, torch.Tensor]) -> float:
    try:
        return time_kernel(engine, link, get_points(cache, model, p, device), device)
    except RuntimeError as exc:
        if not is_oom(exc):
            raise
        clear_cuda(device)
        return float("nan")


def find_method_oom(engine, link: str, model, device: torch.device, cache: dict[int, torch.Tensor]) -> tuple[dict[int, float], int | None]:
    measured, last_ok, first_oom = {}, None, None
    for p in sorted(BATCH_SIZES):
        t = try_batch(engine, link, model, p, device, cache)
        measured[p] = t
        if np.isfinite(t):
            last_ok = p
            continue
        first_oom = p
        break
    grow = max(BATCH_SIZES)
    while first_oom is None and grow < SEARCH_MAX_BATCH:
        grow *= 2
        t = try_batch(engine, link, model, grow, device, cache)
        measured[grow] = t
        if np.isfinite(t):
            last_ok = grow
        else:
            first_oom = grow
    if last_ok is None or first_oom is None:
        return measured, first_oom
    low, high = last_ok + 1, first_oom
    while low < high:
        mid = (low + high) // 2
        t = try_batch(engine, link, model, mid, device, cache)
        measured[mid] = t
        if np.isfinite(t):
            low = mid + 1
        else:
            high = mid
    return measured, low


def spaced_batches(start: int, stop: int, count: int) -> list[int]:
    if stop <= start:
        return [int(stop)]
    vals = np.geomspace(float(start), float(stop), num=count)
    pts = sorted({int(round(v)) for v in vals})
    pts[0] = int(start)
    pts[-1] = int(stop)
    return pts


def plot_batches(last_ok: int | None, first_oom: int | None) -> list[int]:
    if last_ok is None:
        return []
    start = int(min(BATCH_SIZES))
    if last_ok <= start:
        return [int(last_ok)]
    if first_oom is None:
        return spaced_batches(start, last_ok, BASE_PLOT_POINTS + len(TAIL_OOM_FRACS))
    tail = [min(last_ok, max(start, int(first_oom * frac))) for frac in TAIL_OOM_FRACS]
    base_stop = min(last_ok, tail[0])
    batches = spaced_batches(start, base_stop, BASE_PLOT_POINTS)
    batches.extend(tail)
    batches.append(last_ok)
    return sorted(set(batches))


def measure_batches(engine, link: str, model, batches: list[int], device: torch.device, cache: dict[int, torch.Tensor], existing: dict[int, float] | None = None) -> dict[int, float]:
    measured = {} if existing is None else dict(existing)
    for p in batches:
        if p not in measured or not np.isfinite(measured[p]):
            measured[p] = try_batch(engine, link, model, p, device, cache)
    return {p: measured[p] for p in batches}


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), r2


def fit_plateau(line_vals: list[float], ys: list[float]) -> tuple[float, float]:
    if not ys:
        return float("nan"), float("nan")
    candidates = sorted({0.0, *[float(v) for v in line_vals], *[float(v) for v in ys]})
    best_t0, best_sse = float("nan"), float("inf")
    line_arr = np.asarray(line_vals, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    for t0 in candidates:
        pred = np.maximum(t0, line_arr)
        sse = float(np.sum((y_arr - pred) ** 2))
        if sse < best_sse:
            best_sse, best_t0 = sse, float(t0)
    return best_t0, best_sse


def fit_r2(ys: list[float], yhat: list[float]) -> float:
    y = np.asarray(ys, dtype=float)
    yp = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot


def fmt(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.3e}"


def main() -> None:
    device, link = resolve_device(), choose_link(WS_PATH)
    cpeak, bw, source = machine_constants(device)
    engines, models = zip(*[load_engine(cls, WS_PATH, link, device) for cls in METHODS.values()])
    engines, models = dict(zip(METHODS, engines)), dict(zip(METHODS, models))
    point_cache = {}
    scalar_bytes = float(torch.tensor([], dtype=DTYPE).element_size())
    stats, method_oom = {}, {}

    print(f"Workspace: {WS_PATH}\nLink:      {link}\nDevice:    {device}\nMode:      inner kernel only (no sphere projection)\nBatches:   {list(BATCH_SIZES)}")
    print(f"Machine:   Cpeak={cpeak / 1e12:.3f} TFLOP/s | Bw={bw / 1e9:.3f} GB/s | source={source}" if np.isfinite(cpeak) and np.isfinite(bw) else "Machine:   Cpeak/Bw non disponibili")

    for method in METHODS:
        n_func = num_functions(method, models[method])
        gamma, eta1, eta2 = gamma_eta(method, models[method])
        alpha = float("nan") if not (np.isfinite(cpeak) and np.isfinite(bw)) else FLOP_CONST / cpeak + (3.0 * scalar_bytes / bw) * eta1 + (scalar_bytes / bw) * eta2
        beta = float("nan") if not np.isfinite(bw) else scalar_bytes / bw
        search_measured, method_oom[method] = find_method_oom(engines[method], link, models[method], device, point_cache)
        finite_search = sorted(p for p, v in search_measured.items() if np.isfinite(v))
        last_ok = finite_search[-1] if finite_search else None
        measured = measure_batches(
            engines[method],
            link,
            models[method],
            plot_batches(last_ok, method_oom[method]),
            device,
            point_cache,
            existing=search_measured,
        )
        print(f"{method}: n_func={n_func} | plotted batches: {list(measured)}")
        curve_x = spaced_batches(min(measured), max(measured), CURVE_POINTS) if measured else []
        theory_x = sorted(set(curve_x) | set(measured))
        theory = {p: alpha * gamma * p + beta * gamma if np.isfinite(alpha) and np.isfinite(beta) else float("nan") for p in theory_x}
        xs = sorted(p for p, v in measured.items() if np.isfinite(v) and np.isfinite(theory[p]))
        residuals = {p: measured[p] - theory[p] for p in xs}
        delta_q_m = delta_0_m = r2_m = float("nan")
        if len(xs) >= 2:
            delta_q_m, delta_0_m, r2_m = fit_line([float(p) for p in xs], [residuals[p] for p in xs])
        corrected = {
            p: theory[p] + delta_q_m * p + delta_0_m if np.isfinite(delta_q_m) and np.isfinite(delta_0_m) else float("nan")
            for p in theory
        }
        t0_m = piece_r2 = float("nan")
        if xs:
            t0_m, _ = fit_plateau([corrected[p] for p in xs], [measured[p] for p in xs])
        piecewise = {p: max(t0_m, corrected[p]) if np.isfinite(t0_m) and np.isfinite(corrected[p]) else float("nan") for p in corrected}
        piece_residuals = {p: measured[p] - piecewise[p] for p in xs if np.isfinite(piecewise[p])}
        if xs and np.isfinite(t0_m):
            piece_r2 = fit_r2([measured[p] for p in xs], [piecewise[p] for p in xs])
        alpha_eff = alpha + delta_q_m / gamma if np.isfinite(alpha) and np.isfinite(delta_q_m) else float("nan")
        beta_eff = beta + delta_0_m / gamma if np.isfinite(beta) and np.isfinite(delta_0_m) else float("nan")
        stats[method] = {
            "gamma": gamma,
            "alpha": alpha,
            "beta": beta,
            "alpha_eff": alpha_eff,
            "beta_eff": beta_eff,
            "t0_m": t0_m,
            "piece_r2": piece_r2,
            "measured": measured,
            "theory": theory,
            "corrected": corrected,
            "piecewise": piecewise,
            "curve_x": curve_x,
            "residuals": residuals,
            "piece_residuals": piece_residuals,
            "delta_q_m": delta_q_m,
            "delta_0_m": delta_0_m,
            "r2_m": r2_m,
        }
    print("\n[Per-method residual fits]")
    print("method | delta_q_m | delta_0_m | T0_m | alpha_eff | beta_eff | piece_R^2")
    print("--------------------------------------------------------------------------")
    for method in ("weights", "cp", "tt"):
        print(
            f"{method:>7s} | {fmt(stats[method]['delta_q_m']):>9s} | {fmt(stats[method]['delta_0_m']):>9s} | "
            f"{fmt(stats[method]['t0_m']):>8s} | {fmt(stats[method]['alpha_eff']):>9s} | {fmt(stats[method]['beta_eff']):>8s} | {fmt(stats[method]['piece_r2']):>8s}"
        )

    fig, ax0 = plt.subplots(1, 1, figsize=(14.5, 8.6))
    for method in ("weights", "cp", "tt"):
        color, marker = STYLE[method]
        label = LABELS[method]
        xm = sorted(p for p, v in stats[method]["measured"].items() if np.isfinite(v))
        ym = [stats[method]["measured"][p] * 1e3 for p in xm]
        xc = [p for p in stats[method]["curve_x"] if np.isfinite(stats[method]["piecewise"].get(p, float("nan")))]
        yc = [stats[method]["piecewise"][p] * 1e3 for p in xc]
        if xm:
            ax0.plot(xm, ym, color=color, marker=marker, linewidth=MEASURED_LINE_WIDTH, markersize=MARKER_SIZE, label=f"{label} measured")
        if xc:
            ax0.plot(xc, yc, color=color, linestyle="-.", linewidth=MODEL_LINE_WIDTH, alpha=0.95, label=f"{label} model")

    for method in ("weights", "cp", "tt"):
        oom_p = method_oom.get(method)
        if oom_p is None:
            continue
        color, _ = STYLE[method]
        ax0.axvline(oom_p, color=color, linestyle=":", linewidth=OOM_LINE_WIDTH, alpha=0.9)

    ax0.set_xscale("log", base=2)
    ax0.grid(True, which="both", alpha=0.25)
    ax0.set_yscale("log")
    for method in ("weights", "cp", "tt"):
        oom_p = method_oom.get(method)
        if oom_p is None:
            continue
        color, _ = STYLE[method]
        ax0.text(
            oom_p,
            OOM_LABEL_Y,
            "OOM",
            transform=ax0.get_xaxis_transform(),
            color=color,
            fontsize=OOM_LABEL_FONT_SIZE,
            fontweight="bold",
            ha="center",
            va="bottom",
            clip_on=False,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    ax0.set_ylabel("Runtime [ms]", fontsize=AXIS_LABEL_FONT_SIZE)
    ax0.set_xlabel("Batch Size P", fontsize=AXIS_LABEL_FONT_SIZE)
    ax0.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)
    handles, labels = ax0.get_legend_handles_labels()
    pairwise_handles = handles[::2] + handles[1::2]
    pairwise_labels = labels[::2] + labels[1::2]
    ax0.legend(pairwise_handles, pairwise_labels, ncol=2, fontsize=LEGEND_FONT_SIZE, columnspacing=1.4, handletextpad=0.6)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.88)
    fig.savefig(OUT_PATH, dpi=180)
    for method in ("weights", "cp", "tt"):
        print(f"{method} first OOM: {method_oom[method] if method_oom[method] is not None else f'not observed up to {SEARCH_MAX_BATCH}'}")
    print(f"saved plot: {OUT_PATH}")


if __name__ == "__main__":
    main()
