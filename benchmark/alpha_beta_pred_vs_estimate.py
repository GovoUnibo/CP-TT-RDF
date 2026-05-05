#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

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
PREFERRED_LINKS = ("panda_link3", "panda_link2", "panda_link1", "panda_link0")
DEVICE = "auto"
DTYPE = torch.float32
BATCH_SIZES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
WARMUP = 2
ITERS = 5
SEED = 1234
FLOP_CONST = 2.0
CPEAK_TFLOPS = None
BW_GB_S = None

KNOWN_GPU_SPECS = {
    "nvidia geforce rtx 3080": (29.8, 760.0),
    "nvidia geforce rtx 4060 ti": (22.0, 288.0),
}
METHODS = {"w": RDF_Weights, "cp": RDF_3D_CP, "tt": RDF_TT}
LABELS = {"w": "weights", "cp": "cp", "tt": "tt"}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync(device)


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def common_links(ws_path: Path) -> list[str]:
    models = ws_path / "Models"
    w = {p.stem[:-2] for p in models.glob("*_w.pt")}
    cp = {p.stem[:-3] for p in models.glob("*_cp.pt")}
    tt = {p.stem[:-3] for p in models.glob("*_tt.pt")}
    return sorted(w & cp & tt)


def choose_link(ws_path: Path) -> str:
    links = common_links(ws_path)
    if LINK_NAME:
        if LINK_NAME not in links:
            raise ValueError(f"Link '{LINK_NAME}' non disponibile. Disponibili: {links}")
        return LINK_NAME
    for name in PREFERRED_LINKS:
        if name in links:
            return name
    if not links:
        raise ValueError(f"Nessun link con modelli w/cp/tt trovato in {ws_path / 'Models'}")
    return links[0]


def resolve_device() -> torch.device:
    if DEVICE == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if DEVICE == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def silent_load(engine_cls, ws_path: Path, link: str, device: torch.device):
    engine = engine_cls(device=str(device), dtype=DTYPE)
    with contextlib.redirect_stdout(io.StringIO()):
        engine.init_robot_folder(str(ws_path), robot_name="")
        engine.add_models([link], robot_name="")
    return engine, getattr(engine, link + engine.model_extension)


def machine_constants(device: torch.device) -> tuple[float, float, str]:
    if CPEAK_TFLOPS is not None and BW_GB_S is not None:
        return CPEAK_TFLOPS * 1e12, BW_GB_S * 1e9, "manual"
    if device.type != "cuda":
        cpeak = float("nan") if CPEAK_TFLOPS is None else CPEAK_TFLOPS * 1e12
        bw = float("nan") if BW_GB_S is None else BW_GB_S * 1e9
        return cpeak, bw, "missing"
    props = torch.cuda.get_device_properties(device)
    name = str(props.name).lower()
    spec = next((v for k, v in KNOWN_GPU_SPECS.items() if k in name), None)
    cpeak = (CPEAK_TFLOPS or (spec[0] if spec else float("nan"))) * 1e12
    bw = (BW_GB_S or (spec[1] if spec else float("nan"))) * 1e9
    source = "manual" if CPEAK_TFLOPS or BW_GB_S else ("lookup" if spec else "missing")
    return cpeak, bw, source


def gamma_eta(method: str, model) -> tuple[float, float, float]:
    if method == "w":
        n = float(model.n_func)
        return n**3, 1.0 / (n * n), 1.0 / (n**3)
    if method == "cp":
        n, r = float(model.A.shape[0]), float(model.A.shape[1])
        d = 3.0 * n * r + r
        return d, n / d, 1.0 / d
    n, r1, r2 = float(model.G1.shape[1]), float(model.G1.shape[2]), float(model.G3.shape[0])
    d = n * (r1 * r2 + r1 + r2)
    return d, 1.0 / (r1 * r2 + r1 + r2), 1.0 / d


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


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    yhat = slope * np.asarray(xs) + intercept
    ss_res = float(np.sum((np.asarray(ys) - yhat) ** 2))
    ss_tot = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
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


def fmt_ms(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x * 1e3:.4f}"


def fmt_pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.2f}"


def safe_rel_err_pct(actual: float, pred: float) -> float:
    if not (np.isfinite(actual) and np.isfinite(pred)) or abs(actual) < 1e-30:
        return float("nan")
    return abs(pred - actual) / abs(actual) * 100.0


def mean_rel_err_pct(actuals: list[float], preds: list[float]) -> float:
    vals = [safe_rel_err_pct(a, p) for a, p in zip(actuals, preds)]
    vals = [v for v in vals if np.isfinite(v)]
    return float("nan") if not vals else float(np.mean(vals))


def mean_abs_err(actuals: list[float], preds: list[float]) -> float:
    vals = [abs(a - p) for a, p in zip(actuals, preds) if np.isfinite(a) and np.isfinite(p)]
    return float("nan") if not vals else float(np.mean(vals))


def ratio(num: float, den: float) -> str:
    return "nan" if (not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-30) else f"{num / den:.3f}"


def print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(val))
    print(f"\n[{title}]")
    print(" | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(val.rjust(widths[idx]) if idx else val.ljust(widths[idx]) for idx, val in enumerate(row)))


def print_model_equations() -> None:
    print("\n[Model equations]")
    print("T_ideal_m(P)  = alpha_th_m * gamma_m * P + beta_th_m * gamma_m")
    print("Delta_impl_m(P) = delta_q_m * P + delta_0_m")
    print("T_affine_m(P) = T_ideal_m(P) + Delta_impl_m(P)")
    print("T_wall_m(P)   = max(T0_m, T_affine_m(P))")
    print("Derived only if needed: alpha_impl_m = delta_q_m / gamma_m, beta_impl_m = delta_0_m / gamma_m")
    print("                        alpha_eff_m  = alpha_th_m + alpha_impl_m")
    print("                        beta_eff_m   = beta_th_m + beta_impl_m")


def print_column_map() -> None:
    print("\n[Column map]")
    print("gamma      -> gamma_m")
    print("alpha_th   -> alpha_th_m, theoretical slope coefficient from the roofline surrogate")
    print("beta_th    -> beta_th_m, theoretical offset coefficient from the roofline surrogate")
    print("delta_q    -> delta_q_m, empirical residual slope correction in time/query")
    print("delta_0    -> delta_0_m, empirical residual constant correction in time")
    print("T0_ms      -> T0_m in milliseconds, small-batch latency floor")
    print("affine_MAE_ms -> mean absolute error between T_affine and measured time, in milliseconds")


def print_expanded_models(rows: list[dict[str, str]]) -> None:
    print("\n[Expanded models in ms]")
    for row in rows:
        print(f"{row['method']}:")
        print(f"  T_ideal_ms(P)    = {row['ideal_slope_ms']} * P + {row['ideal_offset_ms']}")
        print(f"  Delta_impl_ms(P) = {row['delta_q_ms']} * P + {row['delta_0_ms']}")
        print(f"  T_wall_ms(P)     = max({row['t0_ms']}, T_ideal_ms(P) + Delta_impl_ms(P))")


def main() -> None:
    device = resolve_device()
    link = choose_link(WS_PATH)
    cpeak, bw, source = machine_constants(device)
    engines, models = {}, {}
    for method, cls in METHODS.items():
        engines[method], models[method] = silent_load(cls, WS_PATH, link, device)
    points_all = build_points(models["w"], max(BATCH_SIZES), device)
    s = float(torch.tensor([], dtype=DTYPE).element_size())

    print(f"Workspace: {WS_PATH}\nLink:      {link}\nDevice:    {device}\nDtype:     {DTYPE}\nMode:      inner kernel only (no sphere projection)\nBatches:   {list(BATCH_SIZES)}")
    if np.isfinite(cpeak) and np.isfinite(bw):
        print(f"Machine:   Cpeak={cpeak / 1e12:.3f} TFLOP/s | Bw={bw / 1e9:.3f} GB/s | source={source}")
    else:
        print("Machine:   Cpeak/Bw non disponibili, alpha_pred/beta_pred = nan")

    results = {}
    for method in ("w", "cp", "tt"):
        gamma, eta1, eta2 = gamma_eta(method, models[method])
        alpha_pred = float("nan") if not (np.isfinite(cpeak) and np.isfinite(bw)) else FLOP_CONST / cpeak + (3.0 * s / bw) * eta1 + (s / bw) * eta2
        beta_pred = float("nan") if not np.isfinite(bw) else s / bw
        xs, ys = [], []
        measured = {}
        batch_rows = []
        for batch in BATCH_SIZES:
            status, t = "ok", float("nan")
            try:
                t = time_kernel(engines[method], link, points_all[:batch].contiguous(), device)
                xs.append(float(batch))
                ys.append(t)
                measured[batch] = t
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                status = "oom"
                clear_cuda(device)
        theory = {
            int(p): alpha_pred * gamma * float(p) + beta_pred * gamma if np.isfinite(alpha_pred) and np.isfinite(beta_pred) else float("nan")
            for p in BATCH_SIZES
        }
        delta_q = delta_0 = residual_r2 = float("nan")
        if len(xs) >= 2:
            residuals = [y - theory[int(p)] for p, y in zip(xs, ys)]
            delta_q, delta_0, residual_r2 = fit_line(xs, residuals)
        corrected = {
            int(p): theory[int(p)] + delta_q * float(p) + delta_0 if np.isfinite(delta_q) and np.isfinite(delta_0) else float("nan")
            for p in BATCH_SIZES
        }
        t0 = float("nan")
        if ys:
            t0, _ = fit_plateau([corrected[int(p)] for p in xs], ys)
        wall = {
            int(p): max(t0, corrected[int(p)]) if np.isfinite(t0) and np.isfinite(corrected[int(p)]) else float("nan")
            for p in BATCH_SIZES
        }
        ideal_preds = [theory[int(p)] for p in xs]
        corrected_preds = [corrected[int(p)] for p in xs]
        wall_preds = [wall[int(p)] for p in xs]
        ideal_r2 = fit_r2(ys, ideal_preds) if len(xs) >= 2 else float("nan")
        corrected_r2 = fit_r2(ys, corrected_preds) if len(xs) >= 2 else float("nan")
        wall_r2 = fit_r2(ys, wall_preds) if len(xs) >= 2 else float("nan")
        alpha_eff = alpha_pred + delta_q / gamma if np.isfinite(alpha_pred) and np.isfinite(delta_q) else float("nan")
        beta_eff = beta_pred + delta_0 / gamma if np.isfinite(beta_pred) and np.isfinite(delta_0) else float("nan")
        ideal_slope = alpha_pred * gamma if np.isfinite(alpha_pred) else float("nan")
        ideal_offset = beta_pred * gamma if np.isfinite(beta_pred) else float("nan")
        ideal_mre = mean_rel_err_pct(ys, ideal_preds)
        corrected_mre = mean_rel_err_pct(ys, corrected_preds)
        wall_mre = mean_rel_err_pct(ys, wall_preds)
        corrected_mae = mean_abs_err(ys, corrected_preds)
        for batch in BATCH_SIZES:
            measured_time = measured.get(batch, float("nan"))
            batch_rows.append(
                [
                    str(batch),
                    fmt_ms(measured_time),
                    fmt_ms(theory[batch]),
                    fmt_ms(wall[batch]),
                    fmt_pct(safe_rel_err_pct(measured_time, theory[batch])),
                    fmt_pct(safe_rel_err_pct(measured_time, wall[batch])),
                ]
            )
        results[method] = {
            "label": LABELS[method],
            "gamma": gamma,
            "alpha_pred": alpha_pred,
            "beta_pred": beta_pred,
            "delta_q": delta_q,
            "delta_0": delta_0,
            "t0": t0,
            "corrected_mae": corrected_mae,
            "alpha_eff": alpha_eff,
            "beta_eff": beta_eff,
            "residual_r2": residual_r2,
            "ideal_r2": ideal_r2,
            "corrected_r2": corrected_r2,
            "wall_r2": wall_r2,
            "ideal_mre": ideal_mre,
            "corrected_mre": corrected_mre,
            "wall_mre": wall_mre,
            "ideal_slope": ideal_slope,
            "ideal_offset": ideal_offset,
            "points": len(xs),
            "batch_rows": batch_rows,
        }

    for method in ("w", "cp", "tt"):
        print_table(
            f"Per-batch {LABELS[method]}",
            ["P", "meas_ms", "ideal_ms", "wall_ms", "ideal_err_%", "wall_err_%"],
            results[method]["batch_rows"],
        )

    paper_rows = []
    expanded_rows = []
    for method in ("w", "cp", "tt"):
        item = results[method]
        paper_rows.append(
            [
                item["label"],
                f"{item['gamma']:.0f}",
                fmt(item["alpha_pred"]),
                fmt(item["beta_pred"]),
                fmt(item["delta_q"]),
                fmt(item["delta_0"]),
                fmt_ms(item["t0"]),
                fmt_ms(item["corrected_mae"]),
            ]
        )
        expanded_rows.append(
            {
                "method": item["label"],
                "ideal_slope_ms": fmt(item["ideal_slope"] * 1e3),
                "ideal_offset_ms": fmt(item["ideal_offset"] * 1e3),
                "delta_q_ms": fmt(item["delta_q"] * 1e3),
                "delta_0_ms": fmt(item["delta_0"] * 1e3),
                "t0_ms": fmt_ms(item["t0"]),
            }
        )

    fit_rows = []
    for method in ("w", "cp", "tt"):
        item = results[method]
        fit_rows.append(
            [
                item["label"],
                fmt(item["ideal_r2"]),
                fmt(item["corrected_r2"]),
                fmt(item["wall_r2"]),
                fmt(item["residual_r2"]),
                fmt_pct(item["ideal_mre"]),
                fmt_pct(item["corrected_mre"]),
                fmt_pct(item["wall_mre"]),
                str(item["points"]),
            ]
        )

    print_model_equations()
    print_table(
        "Paper parameters + empirical corrections",
        ["method", "gamma", "alpha_th", "beta_th", "delta_q", "delta_0", "T0_ms", "affine_MAE_ms"],
        paper_rows,
    )
    print_column_map()
    print_expanded_models(expanded_rows)
    print_table(
        "Fit quality",
        ["method", "ideal_R2", "affine_R2", "wall_R2", "resid_R2", "ideal_MRE_%", "affine_MRE_%", "wall_MRE_%", "points"],
        fit_rows,
    )


if __name__ == "__main__":
    main()
