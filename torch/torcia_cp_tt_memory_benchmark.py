#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

MPL_CACHE = Path("/tmp/matplotlib-codex")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import psutil
import torch

def _discover_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "torch").is_dir():
            return candidate
    return start


ROOT = _discover_repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.assets.entities.models import CpLinkModel, TtLinkModel
from src.core.common import CommonSdfMethods
from src.core.math.projection_point import correct_sphere_gradient, spherical_projection
from src.rdf_3Dcp import RDF_3D_CP
from src.rdf_3Dtt import RDF_TT

# ==================== USER CONFIG ====================
WS_PATH = str(ROOT / "torch")
ROBOT_NAME = "panda_robot"
LINK_NAME = "torcia"
FIGURE_REF_LABEL = "fig:four_torch_repr"

DEVICE = "auto"          # "auto", "cpu", "cuda", "cuda:0"
DTYPE = torch.float32
WITH_GRAD = False

BENCHMARK_POINTS = 100_000
WARMUP = 3
ITERS = 10

MEMORY_BUDGET_MB = 11141.12  # 10.88 GiB
BUDGET_SAFETY = 0.70     # usato solo se MEMORY_BUDGET_MB is None
SEARCH_START = 4_096
SEARCH_CAP = 20_000_000

PROBE_SAMPLE_INTERVAL_MS = 1.0
SEED = 1234
PRINT_LATEX_TABLE = True
LATEX_TABLE_OUTPUT_PATH = None  # lasciare None per stampare soltanto a schermo
# =====================================================


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def human_points(points: int) -> str:
    return f"{points:,}".replace(",", "_")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def current_rss_bytes() -> int:
    return int(psutil.Process(os.getpid()).memory_info().rss)


def clear_runtime_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync_if_needed(device)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_name == "cuda":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


@dataclass
class PreparedBatch:
    points: torch.Tensor
    pts_scaled: torch.Tensor
    outside: torch.Tensor
    d_sphere: torch.Tensor
    Bx: torch.Tensor
    By: torch.Tensor
    Bz: torch.Tensor
    dBx: torch.Tensor | None
    dBy: torch.Tensor | None
    dBz: torch.Tensor | None
    link_length: int
    point_length: int
    r1: float


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def cp_param_bytes(model: CpLinkModel) -> int:
    return sum(
        tensor_nbytes(t)
        for t in (model.A, model.B, model.C, model.lamd, model.scale_factor, model.centroid_offset)
    )


def tt_param_bytes(model: TtLinkModel) -> int:
    return sum(
        tensor_nbytes(t)
        for t in (model.G1, model.G2, model.G3, model.scale_factor, model.centroid_offset)
    )


def cp_batch_bytes(rdf) -> int:
    return sum(
        tensor_nbytes(t)
        for t in (
            getattr(rdf, "A_batch", None),
            getattr(rdf, "B_batch", None),
            getattr(rdf, "C_batch", None),
            getattr(rdf, "lam_batch", None),
            getattr(rdf, "centroids_batch", None),
            getattr(rdf, "scale_factors_batch", None),
        )
    )


def tt_batch_bytes(rdf) -> int:
    return sum(
        tensor_nbytes(t)
        for t in (
            getattr(rdf, "G1_batch", None),
            getattr(rdf, "G2_batch", None),
            getattr(rdf, "G3_batch", None),
            getattr(rdf, "G2_batch_flat", None),
            getattr(rdf, "G3_batch_t", None),
            getattr(rdf, "centroids_batch", None),
            getattr(rdf, "scale_factors_batch", None),
        )
    )


def cp_runtime_bytes(model: CpLinkModel, rdf) -> int:
    return cp_param_bytes(model) + cp_batch_bytes(rdf)


def tt_runtime_bytes(model: TtLinkModel, rdf) -> int:
    return tt_param_bytes(model) + tt_batch_bytes(rdf)


def runtime_model_bytes(method: str, rdf, model) -> int:
    if method == "cp":
        return cp_runtime_bytes(model, rdf)
    if method == "tt":
        return tt_runtime_bytes(model, rdf)
    raise ValueError(f"Metodo non supportato: {method}")


def cp_param_numel(model: CpLinkModel) -> int:
    return sum(
        int(t.numel())
        for t in (model.A, model.B, model.C, model.lamd, model.scale_factor, model.centroid_offset)
        if t is not None
    )


def tt_param_numel(model: TtLinkModel) -> int:
    return sum(
        int(t.numel())
        for t in (model.G1, model.G2, model.G3, model.scale_factor, model.centroid_offset)
        if t is not None
    )


def load_raw_model_dict(method: str, ws_path: str, link_name: str) -> tuple[Path, dict]:
    suffix = "_cp.pt" if method == "cp" else "_tt.pt"
    path = Path(ws_path) / "Models" / f"{link_name}{suffix}"
    return path, torch.load(path, map_location="cpu")


def raw_model_storage_bytes(model_dict: dict) -> int:
    total = 0
    for value in model_dict.values():
        if hasattr(value, "nbytes"):
            total += int(value.nbytes)
        elif hasattr(value, "dtype") and hasattr(value.dtype, "itemsize"):
            total += int(value.dtype.itemsize)
    return total


def estimate_cp_forward_extra_bytes(
    model: CpLinkModel,
    num_points: int,
    with_grad: bool,
    dtype: torch.dtype,
) -> int:
    n = int(model.n_func)
    r = int(model.rank)
    fsize = torch.tensor([], dtype=dtype).element_size()
    bool_size = 1

    # Input + sphere projection + normalized points + returned zero-grad tensor.
    float_scalars_per_point = 21
    # Bernstein bases Bx/By/Bz.
    float_scalars_per_point += 3 * n
    # Sx/Sy/Sz + one extra temporary for the elementwise CP product.
    float_scalars_per_point += 5 * r

    if with_grad:
        float_scalars_per_point += 3 * n      # dBx/dBy/dBz
        float_scalars_per_point += 4 * r      # dSx/dSy/dSz + one extra contraction temp
        float_scalars_per_point += 9          # gx/gy/gz + g_in + corrected grad

    total = num_points * float_scalars_per_point * fsize
    total += num_points * bool_size
    return int(total)


def estimate_tt_forward_extra_bytes(
    model: TtLinkModel,
    num_points: int,
    with_grad: bool,
    dtype: torch.dtype,
) -> int:
    n = int(model.n_func)
    r1, r2 = (int(model.ranks[0]), int(model.ranks[1]))
    fsize = torch.tensor([], dtype=dtype).element_size()
    bool_size = 1

    # Input + sphere projection + normalized points + returned zero-grad tensor.
    float_scalars_per_point = 21
    # Bernstein bases Bx/By/Bz.
    float_scalars_per_point += 3 * n
    # U1, U2 and one extra temporary in the final contraction.
    float_scalars_per_point += r1 + 2 * r2

    if with_grad:
        float_scalars_per_point += 3 * n      # dBx/dBy/dBz
        float_scalars_per_point += r1 + 2 * r2
        float_scalars_per_point += 9          # dfdx/dfdy/dfdz + g_in + corrected grad

    total = num_points * float_scalars_per_point * fsize
    total += num_points * bool_size
    return int(total)


def estimate_forward_extra_bytes(method: str, model, num_points: int, with_grad: bool, dtype: torch.dtype) -> int:
    if method == "cp":
        return estimate_cp_forward_extra_bytes(model=model, num_points=num_points, with_grad=with_grad, dtype=dtype)
    if method == "tt":
        return estimate_tt_forward_extra_bytes(model=model, num_points=num_points, with_grad=with_grad, dtype=dtype)
    raise ValueError(f"Metodo non supportato: {method}")


def prepare_common_batch(
    rdf,
    points: torch.Tensor,
    with_grad: bool,
    forward_tensor: torch.Tensor | None = None,
) -> PreparedBatch:
    points_link = CommonSdfMethods.to_link_frame(points_w=points, H_wl=forward_tensor) if forward_tensor is not None else points
    link_length = int(points_link.shape[0])
    point_length = int(points_link.shape[1])

    pts_scaled = ((points_link - rdf.centroids_batch) / rdf.scale_factors_batch).reshape(-1, 3)
    r1 = (float(rdf.domain_max) - float(rdf.domain_min)) / 2.0
    x_in, d_sphere, outside = spherical_projection(pts_scaled, r1=r1)

    p = rdf.normalize_points(x_in)
    Bx, dBx = rdf.build_bernstein_t(p[:, 0], use_derivative=with_grad)
    By, dBy = rdf.build_bernstein_t(p[:, 1], use_derivative=with_grad)
    Bz, dBz = rdf.build_bernstein_t(p[:, 2], use_derivative=with_grad)

    n_func = int(Bx.shape[1])
    Bx = Bx.reshape(link_length, point_length, n_func)
    By = By.reshape(link_length, point_length, n_func)
    Bz = Bz.reshape(link_length, point_length, n_func)

    if with_grad:
        dBx = dBx.reshape(link_length, point_length, n_func)
        dBy = dBy.reshape(link_length, point_length, n_func)
        dBz = dBz.reshape(link_length, point_length, n_func)
    else:
        dBx = None
        dBy = None
        dBz = None

    return PreparedBatch(
        points=points_link,
        pts_scaled=pts_scaled,
        outside=outside,
        d_sphere=d_sphere,
        Bx=Bx,
        By=By,
        Bz=Bz,
        dBx=dBx,
        dBy=dBy,
        dBz=dBz,
        link_length=link_length,
        point_length=point_length,
        r1=r1,
    )


def contract_cp_batch(rdf, batch: PreparedBatch, with_grad: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    Sx = torch.matmul(batch.Bx, rdf.A_batch)
    Sy = torch.matmul(batch.By, rdf.B_batch)
    Sz = torch.matmul(batch.Bz, rdf.C_batch)

    lam = rdf.lam_batch.unsqueeze(1)
    sdf_inner = (Sx * Sy * Sz * lam).sum(dim=-1)

    if not with_grad:
        return sdf_inner, None

    dSx = torch.matmul(batch.dBx, rdf.A_batch)
    dSy = torch.matmul(batch.dBy, rdf.B_batch)
    dSz = torch.matmul(batch.dBz, rdf.C_batch)

    gx = (dSx * Sy * Sz * lam).sum(dim=-1)
    gy = (Sx * dSy * Sz * lam).sum(dim=-1)
    gz = (Sx * Sy * dSz * lam).sum(dim=-1)
    g_in = torch.stack([gx, gy, gz], dim=-1)
    return sdf_inner, g_in


def contract_tt_batch(rdf, batch: PreparedBatch, with_grad: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    U1 = torch.einsum("lpn,lnr->lpr", batch.Bx, rdf.G1_batch)
    U2 = torch.einsum("lpr,lpn,lrnq->lpq", U1, batch.By, rdf.G2_batch)
    sdf_inner = torch.einsum("lpq,lpn,lqn->lp", U2, batch.Bz, rdf.G3_batch)

    if not with_grad:
        return sdf_inner, None

    dU1_dx = torch.einsum("lpn,lnr->lpr", batch.dBx, rdf.G1_batch)
    dU2_dx = torch.einsum("lpr,lpn,lrnq->lpq", dU1_dx, batch.By, rdf.G2_batch)
    dU2_dy = torch.einsum("lpr,lpn,lrnq->lpq", U1, batch.dBy, rdf.G2_batch)

    dfdx = torch.einsum("lpq,lpn,lqn->lp", dU2_dx, batch.Bz, rdf.G3_batch)
    dfdy = torch.einsum("lpq,lpn,lqn->lp", dU2_dy, batch.Bz, rdf.G3_batch)
    dfdz = torch.einsum("lpq,lpn,lqn->lp", U2, batch.dBz, rdf.G3_batch)
    g_in = torch.stack([dfdx, dfdy, dfdz], dim=-1)
    return sdf_inner, g_in


def contract_tt_batch_matmul(
    batch: PreparedBatch,
    g2_flat: torch.Tensor,
    g3_t: torch.Tensor,
    g1: torch.Tensor,
    with_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    TT contraction reordered around batched matmul:
    - contract x first
    - fold (y, G2) into a single batched matmul
    - fold z with the same pattern
    """
    u1 = torch.matmul(batch.Bx, g1)
    y_flat = torch.matmul(batch.By, g2_flat)
    r1 = int(g1.shape[-1])
    r2 = int(g3_t.shape[-1])
    y_tensor = y_flat.reshape(batch.link_length, batch.point_length, r1, r2)
    u2 = (u1.unsqueeze(-1) * y_tensor).sum(dim=2)

    z = torch.matmul(batch.Bz, g3_t)
    sdf_inner = (u2 * z).sum(dim=-1)

    if not with_grad:
        return sdf_inner, None

    du1_dx = torch.matmul(batch.dBx, g1)
    du2_dx = (du1_dx.unsqueeze(-1) * y_tensor).sum(dim=2)

    y_flat_dy = torch.matmul(batch.dBy, g2_flat)
    y_tensor_dy = y_flat_dy.reshape(batch.link_length, batch.point_length, r1, r2)
    du2_dy = (u1.unsqueeze(-1) * y_tensor_dy).sum(dim=2)

    dz = torch.matmul(batch.dBz, g3_t)
    dfdx = (du2_dx * z).sum(dim=-1)
    dfdy = (du2_dy * z).sum(dim=-1)
    dfdz = (u2 * dz).sum(dim=-1)
    g_in = torch.stack([dfdx, dfdy, dfdz], dim=-1)
    return sdf_inner, g_in


def finalize_batch(
    rdf,
    batch: PreparedBatch,
    sdf_inner: torch.Tensor,
    g_in: torch.Tensor | None,
    with_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    sdf = sdf_inner + batch.d_sphere.reshape(batch.link_length, batch.point_length)
    sdf = sdf * rdf.scale_factors_batch.reshape(batch.link_length, 1)

    if not with_grad:
        d_sdf = torch.zeros((batch.link_length, batch.point_length, 3), device=sdf.device, dtype=sdf.dtype)
        return sdf, d_sdf

    den = torch.as_tensor(float(rdf.domain_max) - float(rdf.domain_min), device=sdf.device, dtype=sdf.dtype)
    g_in = g_in / den
    g_corr = correct_sphere_gradient(
        points_scaled=batch.pts_scaled,
        g_in=g_in.reshape(-1, 3),
        outside=batch.outside,
        r1=batch.r1,
    )
    d_sdf = g_corr.reshape(batch.link_length, batch.point_length, 3)
    return sdf, d_sdf


def benchmark_method_forward(
    method: str,
    rdf,
    points: torch.Tensor,
    with_grad: bool,
    forward_tensor: torch.Tensor | None = None,
):
    batch = prepare_common_batch(rdf=rdf, points=points, with_grad=with_grad, forward_tensor=forward_tensor)
    if method == "cp":
        sdf_inner, g_in = contract_cp_batch(rdf=rdf, batch=batch, with_grad=with_grad)
    elif method == "tt":
        sdf_inner, g_in = contract_tt_batch_matmul(
            batch=batch,
            g2_flat=getattr(rdf, "G2_batch_flat"),
            g3_t=getattr(rdf, "G3_batch_t"),
            g1=rdf.G1_batch,
            with_grad=with_grad,
        )
    else:
        raise ValueError(f"Metodo non supportato: {method}")
    sdf, d_sdf = finalize_batch(rdf=rdf, batch=batch, sdf_inner=sdf_inner, g_in=g_in, with_grad=with_grad)
    return sdf, d_sdf, batch.points


def benchmark_tt_matmul_variant(
    rdf,
    points: torch.Tensor,
    with_grad: bool,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict:
    batch = prepare_common_batch(rdf=rdf, points=points, with_grad=with_grad)
    g2_flat = getattr(rdf, "G2_batch_flat", None)
    if g2_flat is None:
        g2_flat = rdf.G2_batch.permute(0, 2, 1, 3).reshape(rdf.G2_batch.shape[0], rdf.G2_batch.shape[2], -1).contiguous()
    g3_t = getattr(rdf, "G3_batch_t", None)
    if g3_t is None:
        g3_t = rdf.G3_batch.transpose(1, 2).contiguous()
    g1 = rdf.G1_batch

    sdf_einsum, grad_einsum = contract_tt_batch(rdf=rdf, batch=batch, with_grad=with_grad)
    sdf_matmul, grad_matmul = contract_tt_batch_matmul(
        batch=batch,
        g2_flat=g2_flat,
        g3_t=g3_t,
        g1=g1,
        with_grad=with_grad,
    )

    sdf_diff = float((sdf_einsum - sdf_matmul).abs().max().item())
    grad_diff = 0.0
    if with_grad and grad_einsum is not None and grad_matmul is not None:
        grad_diff = float((grad_einsum - grad_matmul).abs().max().item())

    einsum_ms = time_block_ms(
        lambda: contract_tt_batch(rdf=rdf, batch=batch, with_grad=with_grad),
        device=device,
        warmup=warmup,
        iters=iters,
    )
    matmul_ms = time_block_ms(
        lambda: contract_tt_batch_matmul(batch=batch, g2_flat=g2_flat, g3_t=g3_t, g1=g1, with_grad=with_grad),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    return {
        "einsum_ms": float(einsum_ms),
        "matmul_ms": float(matmul_ms),
        "sdf_diff": sdf_diff,
        "grad_diff": grad_diff,
    }


def build_points(num_points: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    pts = (torch.rand((num_points, 3), generator=g, dtype=dtype) * 2.0 - 1.0) * 0.5
    return pts.to(device=device, dtype=dtype).unsqueeze(0).contiguous()


def load_engine(
    method: str,
    ws_path: str,
    robot_name: str,
    link_name: str,
    device: torch.device,
    dtype: torch.dtype,
):
    if method == "cp":
        rdf = RDF_3D_CP(device=device.type, dtype=dtype)
        rdf.init_robot_folder(ws_path, robot_name=robot_name)
        rdf.add_models([link_name], robot_name=robot_name)
        rdf.set_ordered_batch_params([link_name])
        model = getattr(rdf, link_name + rdf.model_extension)
        rdf.set_points_domain(domain_min=float(model.domain_min), domain_max=float(model.domain_max))
        return rdf, model, cp_param_bytes(model)

    if method == "tt":
        rdf = RDF_TT(device=device.type, dtype=dtype)
        rdf.init_robot_folder(ws_path, robot_name=robot_name)
        rdf.add_models([link_name], robot_name=robot_name)
        rdf.set_ordered_batch_params([link_name])
        model = getattr(rdf, link_name + rdf.model_extension)
        rdf.set_points_domain(domain_min=float(model.domain_min), domain_max=float(model.domain_max))
        return rdf, model, tt_param_bytes(model)

    raise ValueError(f"Metodo non supportato: {method}")


def infer_once(rdf, points: torch.Tensor, with_grad: bool):
    if isinstance(rdf, RDF_3D_CP):
        return benchmark_method_forward("cp", rdf, points, with_grad, forward_tensor=None)
    if isinstance(rdf, RDF_TT):
        return benchmark_method_forward("tt", rdf, points, with_grad, forward_tensor=None)
    return rdf.inference_link_batch(points=points, get_grad=with_grad, get_min=False, forward_tensor=None)


def warmup_forward(
    rdf,
    num_points: int,
    with_grad: bool,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> None:
    points = None
    out = None
    try:
        points = build_points(num_points=num_points, device=device, dtype=dtype, seed=seed)
        out = infer_once(rdf, points=points, with_grad=with_grad)
        sync_if_needed(device)
    finally:
        del out
        del points
        clear_runtime_state(device)


def time_forward_ms(
    rdf,
    points: torch.Tensor,
    with_grad: bool,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(max(0, int(warmup))):
        out = infer_once(rdf, points, with_grad=with_grad)
        del out
    sync_if_needed(device)

    t0 = time.perf_counter()
    for _ in range(max(1, int(iters))):
        out = infer_once(rdf, points, with_grad=with_grad)
        del out
    sync_if_needed(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / max(1, int(iters))


def time_block_ms(fn, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(max(0, int(warmup))):
        out = fn()
        del out
    sync_if_needed(device)

    t0 = time.perf_counter()
    for _ in range(max(1, int(iters))):
        out = fn()
        del out
    sync_if_needed(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / max(1, int(iters))


def time_common_ms(
    rdf,
    points: torch.Tensor,
    with_grad: bool,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    return time_block_ms(
        lambda: prepare_common_batch(rdf=rdf, points=points, with_grad=with_grad),
        device=device,
        warmup=warmup,
        iters=iters,
    )


def time_contract_ms(
    method: str,
    rdf,
    batch: PreparedBatch,
    with_grad: bool,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    if method == "cp":
        fn = lambda: contract_cp_batch(rdf=rdf, batch=batch, with_grad=with_grad)
    elif method == "tt":
        fn = lambda: contract_tt_batch_matmul(
            batch=batch,
            g2_flat=getattr(rdf, "G2_batch_flat"),
            g3_t=getattr(rdf, "G3_batch_t"),
            g1=rdf.G1_batch,
            with_grad=with_grad,
        )
    else:
        raise ValueError(f"Metodo non supportato: {method}")
    return time_block_ms(fn, device=device, warmup=warmup, iters=iters)


def time_tail_ms(
    rdf,
    batch: PreparedBatch,
    sdf_inner: torch.Tensor,
    g_in: torch.Tensor | None,
    with_grad: bool,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    return time_block_ms(
        lambda: finalize_batch(rdf=rdf, batch=batch, sdf_inner=sdf_inner, g_in=g_in, with_grad=with_grad),
        device=device,
        warmup=warmup,
        iters=iters,
    )


def measure_timing_breakdown(
    method: str,
    rdf,
    points: torch.Tensor,
    with_grad: bool,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict:
    clear_runtime_state(device)
    common_ms = time_common_ms(
        rdf=rdf,
        points=points,
        with_grad=with_grad,
        device=device,
        warmup=warmup,
        iters=iters,
    )

    batch = prepare_common_batch(rdf=rdf, points=points, with_grad=with_grad)
    if method == "cp":
        sdf_inner, g_in = contract_cp_batch(rdf=rdf, batch=batch, with_grad=with_grad)
    elif method == "tt":
        sdf_inner, g_in = contract_tt_batch_matmul(
            batch=batch,
            g2_flat=getattr(rdf, "G2_batch_flat"),
            g3_t=getattr(rdf, "G3_batch_t"),
            g1=rdf.G1_batch,
            with_grad=with_grad,
        )
    else:
        raise ValueError(f"Metodo non supportato: {method}")

    contract_ms = time_contract_ms(
        method=method,
        rdf=rdf,
        batch=batch,
        with_grad=with_grad,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    tail_ms = time_tail_ms(
        rdf=rdf,
        batch=batch,
        sdf_inner=sdf_inner,
        g_in=g_in,
        with_grad=with_grad,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    clear_runtime_state(device)
    return {
        "method": method,
        "common_ms": float(common_ms),
        "contract_ms": float(contract_ms),
        "tail_ms": float(tail_ms),
    }


def cpu_peak_extra_bytes(
    rdf,
    num_points: int,
    with_grad: bool,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    sample_interval_s: float,
) -> int:
    warmup_forward(
        rdf=rdf,
        num_points=min(int(num_points), 4_096),
        with_grad=with_grad,
        device=device,
        dtype=dtype,
        seed=seed - 1,
    )

    baseline_rss = current_rss_bytes()
    peak_rss = baseline_rss
    stop_event = threading.Event()
    proc = psutil.Process(os.getpid())

    def sampler():
        nonlocal peak_rss
        while not stop_event.is_set():
            rss = int(proc.memory_info().rss)
            if rss > peak_rss:
                peak_rss = rss
            time.sleep(sample_interval_s)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    points = None
    out = None
    try:
        points = build_points(num_points, device=device, dtype=dtype, seed=seed)
        out = infer_once(rdf, points=points, with_grad=with_grad)
        sync_if_needed(device)
    finally:
        peak_rss = max(peak_rss, current_rss_bytes())
        stop_event.set()
        thread.join()
        del out
        del points
        clear_runtime_state(device)

    return max(0, peak_rss - baseline_rss)


def cuda_peak_extra_bytes(
    rdf,
    num_points: int,
    with_grad: bool,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> int:
    warmup_forward(
        rdf=rdf,
        num_points=min(int(num_points), 4_096),
        with_grad=with_grad,
        device=device,
        dtype=dtype,
        seed=seed - 1,
    )

    clear_runtime_state(device)
    baseline_alloc = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)

    points = None
    out = None
    try:
        points = build_points(num_points, device=device, dtype=dtype, seed=seed)
        out = infer_once(rdf, points=points, with_grad=with_grad)
        sync_if_needed(device)
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
    finally:
        del out
        del points
        clear_runtime_state(device)

    return max(0, peak_alloc - baseline_alloc)


def measure_peak_extra_bytes(
    method: str,
    model,
    rdf,
    num_points: int,
    with_grad: bool,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    sample_interval_s: float,
) -> int:
    if device.type == "cpu":
        return estimate_forward_extra_bytes(
            method=method,
            model=model,
            num_points=num_points,
            with_grad=with_grad,
            dtype=dtype,
        )

    if device.type == "cuda":
        return cuda_peak_extra_bytes(
            rdf=rdf,
            num_points=num_points,
            with_grad=with_grad,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    return cpu_peak_extra_bytes(
        rdf=rdf,
        num_points=num_points,
        with_grad=with_grad,
        device=device,
        dtype=dtype,
        seed=seed,
        sample_interval_s=sample_interval_s,
    )


def measure_total_footprint(
    method: str,
    model,
    rdf,
    num_points: int,
    with_grad: bool,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    sample_interval_s: float,
) -> dict:
    weight_bytes = cp_param_bytes(model) if method == "cp" else tt_param_bytes(model)
    batch_bytes = cp_batch_bytes(rdf) if method == "cp" else tt_batch_bytes(rdf)
    model_bytes = int(runtime_model_bytes(method, rdf, model))
    try:
        peak_extra = measure_peak_extra_bytes(
            method=method,
            model=model,
            rdf=rdf,
            num_points=int(num_points),
            with_grad=with_grad,
            device=device,
            dtype=dtype,
            seed=seed,
            sample_interval_s=sample_interval_s,
        )
        return {
            "success": True,
            "method": method,
            "points": int(num_points),
            "weight_bytes": int(weight_bytes),
            "batch_bytes": int(batch_bytes),
            "model_bytes": int(model_bytes),
            "peak_extra_bytes": int(peak_extra),
            "total_footprint_bytes": int(model_bytes + peak_extra),
        }
    except (RuntimeError, MemoryError) as exc:
        clear_runtime_state(device)
        return {
            "success": False,
            "method": method,
            "points": int(num_points),
            "weight_bytes": int(weight_bytes),
            "batch_bytes": int(batch_bytes),
            "model_bytes": int(model_bytes),
            "error": str(exc),
        }


def estimate_max_points_cpu(method: str, model, rdf, budget_bytes: int, dtype: torch.dtype, with_grad: bool, search_cap: int) -> dict:
    weight_bytes = cp_param_bytes(model) if method == "cp" else tt_param_bytes(model)
    batch_bytes = cp_batch_bytes(rdf) if method == "cp" else tt_batch_bytes(rdf)
    model_bytes = int(runtime_model_bytes(method, rdf, model))
    bytes_per_point = estimate_forward_extra_bytes(
        method=method,
        model=model,
        num_points=1,
        with_grad=with_grad,
        dtype=dtype,
    )
    if budget_bytes <= model_bytes or bytes_per_point <= 0:
        return {
            "success": False,
            "method": method,
            "points": 0,
            "weight_bytes": int(weight_bytes),
            "batch_bytes": int(batch_bytes),
            "model_bytes": int(model_bytes),
            "peak_extra_bytes": 0,
            "total_footprint_bytes": int(model_bytes),
            "error": "budget troppo piccolo per il modello",
            "within_budget": False,
        }

    max_points = min(int(search_cap), max(0, (int(budget_bytes) - int(model_bytes)) // int(bytes_per_point)))
    peak_extra = estimate_forward_extra_bytes(
        method=method,
        model=model,
        num_points=max_points,
        with_grad=with_grad,
        dtype=dtype,
    )
    return {
        "success": True,
        "method": method,
        "points": int(max_points),
        "weight_bytes": int(weight_bytes),
        "batch_bytes": int(batch_bytes),
        "model_bytes": int(model_bytes),
        "peak_extra_bytes": int(peak_extra),
        "total_footprint_bytes": int(model_bytes + peak_extra),
        "within_budget": True,
        "budget_bytes": int(budget_bytes),
    }


def find_max_points_cuda(
    method: str,
    model,
    rdf,
    budget_bytes: int,
    device: torch.device,
    dtype: torch.dtype,
    with_grad: bool,
    search_start: int,
    search_cap: int,
    seed: int,
    sample_interval_s: float,
) -> dict:
    low = 0
    high = max(1, int(search_start))
    best = None

    while high <= int(search_cap):
        result = measure_total_footprint(
            method=method,
            model=model,
            rdf=rdf,
            num_points=high,
            with_grad=with_grad,
            device=device,
            dtype=dtype,
            seed=seed,
            sample_interval_s=sample_interval_s,
        )
        if result.get("success") and int(result["total_footprint_bytes"]) <= budget_bytes:
            low = high
            best = result
            high *= 2
            continue
        break

    if best is None:
        failed = measure_total_footprint(
            method=method,
            model=model,
            rdf=rdf,
            num_points=low or high,
            with_grad=with_grad,
            device=device,
            dtype=dtype,
            seed=seed,
            sample_interval_s=sample_interval_s,
        )
        failed["within_budget"] = False
        return failed

    upper = min(high, int(search_cap) + 1)
    while low + 1 < upper:
        mid = (low + upper) // 2
        result = measure_total_footprint(
            method=method,
            model=model,
            rdf=rdf,
            num_points=mid,
            with_grad=with_grad,
            device=device,
            dtype=dtype,
            seed=seed,
            sample_interval_s=sample_interval_s,
        )
        if result.get("success") and int(result["total_footprint_bytes"]) <= budget_bytes:
            low = mid
            best = result
        else:
            upper = mid

    best["within_budget"] = True
    best["budget_bytes"] = int(budget_bytes)
    return best


def default_budget_bytes(device: torch.device, safety: float) -> int:
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(float(free_bytes) * float(safety))
    return int(psutil.virtual_memory().available * float(safety))


def print_summary_row(tag: str, stats: dict) -> None:
    pts = int(stats["benchmark_points"])
    mean_ms = float(stats["mean_ms"])
    throughput = pts / max(mean_ms / 1000.0, 1e-12)
    print(
        f"{tag:>2s} | weights={human_bytes(int(stats['weight_bytes'])):>10s} | "
        f"batch={human_bytes(int(stats['batch_bytes'])):>10s} | "
        f"model={human_bytes(int(stats['model_bytes'])):>10s} | "
        f"peak@{human_points(pts):>9s}={human_bytes(int(stats['total_footprint_bytes'])):>10s} | "
        f"time={mean_ms:8.3f} ms | throughput={throughput:12.1f} pts/s"
    )


def print_timing_breakdown_row(tag: str, stats: dict) -> None:
    print(
        f"{tag:>2s} | common={float(stats['common_ms']):8.3f} ms | "
        f"contract={float(stats['contract_ms']):8.3f} ms | "
        f"tail={float(stats['tail_ms']):8.3f} ms | "
        f"total={float(stats['mean_ms']):8.3f} ms"
    )


def print_tt_variant_row(stats: dict) -> None:
    print(
        f"TT | einsum={float(stats['einsum_ms']):8.3f} ms | "
        f"matmul={float(stats['matmul_ms']):8.3f} ms | "
        f"speedup={float(stats['einsum_ms']) / max(float(stats['matmul_ms']), 1e-12):6.2f}x | "
        f"sdf_diff={float(stats['sdf_diff']):.3e} | grad_diff={float(stats['grad_diff']):.3e}"
    )


def print_capacity_row(tag: str, result: dict) -> None:
    if not result.get("success"):
        err = result.get("error", "probe failed")
        print(f"{tag:>2s} | max_points=FAILED | {err}")
        return
    print(
        f"{tag:>2s} | max_points={human_points(int(result['points'])):>12s} | "
        f"footprint={human_bytes(int(result['total_footprint_bytes'])):>10s}"
    )


def print_model_memory_table(runtime_stats: dict, raw_stats: dict) -> None:
    print("\n[Model memory]")
    print("     runtime tensors = weights + batch tensors usati in inferenza")
    print("     raw arrays       = bytes numerici realmente serializzati nel file .pt")
    print("     file size        = dimensione del file su disco")
    print(
        f"CP | params={human_points(int(runtime_stats['cp']['param_numel'])):>10s} | "
        f"weights={human_bytes(int(runtime_stats['cp']['weight_bytes'])):>10s} | "
        f"batch={human_bytes(int(runtime_stats['cp']['batch_bytes'])):>10s} | "
        f"runtime={human_bytes(int(runtime_stats['cp']['model_bytes'])):>10s} | "
        f"raw arrays={human_bytes(int(raw_stats['cp']['raw_array_bytes'])):>10s} | "
        f"file size={human_bytes(int(raw_stats['cp']['file_size_bytes'])):>10s}"
    )
    print(
        f"TT | params={human_points(int(runtime_stats['tt']['param_numel'])):>10s} | "
        f"weights={human_bytes(int(runtime_stats['tt']['weight_bytes'])):>10s} | "
        f"batch={human_bytes(int(runtime_stats['tt']['batch_bytes'])):>10s} | "
        f"runtime={human_bytes(int(runtime_stats['tt']['model_bytes'])):>10s} | "
        f"raw arrays={human_bytes(int(raw_stats['tt']['raw_array_bytes'])):>10s} | "
        f"file size={human_bytes(int(raw_stats['tt']['file_size_bytes'])):>10s}"
    )

    delta_runtime = int(runtime_stats["cp"]["model_bytes"]) - int(runtime_stats["tt"]["model_bytes"])
    delta_file = int(raw_stats["cp"]["file_size_bytes"]) - int(raw_stats["tt"]["file_size_bytes"])
    ratio = float(runtime_stats["cp"]["model_bytes"]) / max(float(runtime_stats["tt"]["model_bytes"]), 1.0)
    print(
        f"TT saves {human_bytes(delta_runtime)} of runtime tensor memory and "
        f"{human_bytes(delta_file)} on disk vs CP ({ratio:.2f}x smaller runtime model)."
    )


def latex_ratio(value: float) -> str:
    return f"{value:.2f}\\times"


def compact_points_label(points: int) -> str:
    points = int(points)
    if points >= 1_000_000:
        if points % 1_000_000 == 0:
            return f"{points // 1_000_000}M"
        return f"{points / 1_000_000:.2f}M"
    if points >= 1_000:
        if points % 1_000 == 0:
            return f"{points // 1_000}k"
        return f"{points / 1_000:.2f}k"
    return str(points)


def compact_tt_ranks_label(ranks: tuple[int, ...]) -> str:
    return "(" + ",".join(str(int(rank)) for rank in ranks) + ")"


def shared_n_func_label(cp_stats: dict, tt_stats: dict) -> str:
    cp_n = int(cp_stats["n_func"])
    tt_n = int(tt_stats["n_func"])
    if cp_n == tt_n:
        return str(cp_n)
    return f"{cp_n}/{tt_n}"


def format_best_tag(winner_name: str, winner_value: float, loser_value: float, adjective: str) -> str:
    return f"{winner_name} ({latex_ratio(loser_value / max(winner_value, 1e-12))} {adjective})"


def format_more_tag(winner_name: str, winner_value: float, loser_value: float) -> str:
    return f"{winner_name} ({latex_ratio(winner_value / max(loser_value, 1e-12))} more)"


def build_compact_latex_table(
    runtime_stats: dict,
    cp_best: dict,
    tt_best: dict,
    method_models: dict,
    device: torch.device,
    budget_bytes: int,
) -> str:
    cp = runtime_stats["cp"]
    tt = runtime_stats["tt"]
    benchmark_points = int(cp["benchmark_points"])
    benchmark_label = compact_points_label(benchmark_points)
    budget_gib = budget_bytes / (1024.0 ** 3)
    mode_label = "SDF+GRAD" if WITH_GRAD else "SDF only"

    cp_runtime_mib = cp["model_bytes"] / 1024.0 ** 2
    tt_runtime_mib = tt["model_bytes"] / 1024.0 ** 2
    cp_peak_mib = cp["total_footprint_bytes"] / 1024.0 ** 2
    tt_peak_mib = tt["total_footprint_bytes"] / 1024.0 ** 2

    cp_time = float(cp["mean_ms"])
    tt_time = float(tt["mean_ms"])

    cp_points_success = bool(cp_best.get("success"))
    tt_points_success = bool(tt_best.get("success"))
    cp_points = int(cp_best.get("points", 0)) if cp_points_success else 0
    tt_points = int(tt_best.get("points", 0)) if tt_points_success else 0

    cp_rank = int(getattr(method_models["cp"], "rank"))
    tt_ranks = tuple(int(r) for r in getattr(method_models["tt"], "ranks"))
    tt_ranks_latex = compact_tt_ranks_label(tt_ranks)
    n_func_label = shared_n_func_label(cp, tt)

    if cp_runtime_mib <= tt_runtime_mib:
        runtime_best = format_best_tag("CP", cp_runtime_mib, tt_runtime_mib, "smaller")
    else:
        runtime_best = format_best_tag("TT", tt_runtime_mib, cp_runtime_mib, "smaller")

    if cp_peak_mib <= tt_peak_mib:
        peak_best = format_best_tag("CP", cp_peak_mib, tt_peak_mib, "lower")
    else:
        peak_best = format_best_tag("TT", tt_peak_mib, cp_peak_mib, "lower")

    if cp_time <= tt_time:
        time_best = format_best_tag("CP", cp_time, tt_time, "faster")
    else:
        time_best = format_best_tag("TT", tt_time, cp_time, "faster")

    if cp_points_success and tt_points_success:
        if cp_points >= tt_points:
            points_best = format_more_tag("CP", float(cp_points), float(tt_points))
        else:
            points_best = format_more_tag("TT", float(tt_points), float(cp_points))
    elif cp_points_success:
        points_best = "CP"
    elif tt_points_success:
        points_best = "TT"
    else:
        points_best = "FAILED"

    caption = (
        rf"\caption{{Compact CP--TT comparison for Fig.\ref{{{FIGURE_REF_LABEL}}} at "
        f"$N={n_func_label}$, with CP rank $R={cp_rank}$ and TT ranks ${tt_ranks_latex}$. "
        f"Measured on {device.type} in {mode_label} mode under a {budget_gib:.2f}\\,GiB memory budget."
        "}"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        caption,
        rf"\label{{tab:{LINK_NAME}-compact}}",
        r"\begin{tabular}{|l|c|c|}",
        r"\hline",
        r"\textbf{Metric} & \textbf{CP / TT} & \textbf{Best} \\",
        r"\hline",
        fr"Runtime memory        & {cp_runtime_mib * 1024:.1f} / {tt_runtime_mib * 1024:.1f} KiB   & {runtime_best} \\",
        fr"Peak memory @ {benchmark_label}    & {cp_peak_mib:.1f} / {tt_peak_mib:.1f} MiB   & {peak_best} \\",
        fr"Eval.\ time @ {benchmark_label} [ms] & {cp_time:.2f} / {tt_time:.2f}     & {time_best} \\",
        fr"Max query points      & {compact_points_label(cp_points) if cp_points_success else 'FAILED'} / "
        fr"{compact_points_label(tt_points) if tt_points_success else 'FAILED'}     & {points_best} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def emit_compact_latex_table(
    runtime_stats: dict,
    cp_best: dict,
    tt_best: dict,
    method_models: dict,
    device: torch.device,
    budget_bytes: int,
    output_path: Path | None = None,
) -> str:
    table = build_compact_latex_table(
        runtime_stats=runtime_stats,
        cp_best=cp_best,
        tt_best=tt_best,
        method_models=method_models,
        device=device,
        budget_bytes=budget_bytes,
    )
    print("\n[LaTeX table]")
    print(table)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")
        print(f"[LaTeX table saved] {output_path}")
    return table


def run_main() -> int:
    device = resolve_device(DEVICE)
    dtype = DTYPE
    sample_interval_s = max(0.0005, float(PROBE_SAMPLE_INTERVAL_MS) / 1000.0)

    if MEMORY_BUDGET_MB is not None:
        budget_bytes = int(float(MEMORY_BUDGET_MB) * 1024**2)
        budget_label = f"user budget ({float(MEMORY_BUDGET_MB):.1f} MiB)"
    else:
        budget_bytes = default_budget_bytes(device=device, safety=float(BUDGET_SAFETY))
        budget_label = f"detected budget ({float(BUDGET_SAFETY) * 100.0:.0f}% free {device.type.upper()} memory)"

    print(f"Workspace: {WS_PATH}")
    print(f"Robot: {ROBOT_NAME}")
    print(f"Link: {LINK_NAME}")
    print(f"Device requested: {DEVICE}")
    print(f"Device used: {device.type}")
    print(f"Mode: {'SDF+GRAD' if WITH_GRAD else 'SDF only'}")
    print(f"Benchmark points: {human_points(int(BENCHMARK_POINTS))}")
    print(f"Budget for max-points search: {human_bytes(budget_bytes)} [{budget_label}]")
    if device.type == "cpu":
        print("Memory metric: analytical activation estimate on CPU; real peak allocation on CUDA.")

    method_stats = {}
    method_models = {}
    method_engines = {}
    raw_stats = {}
    tt_variant_stats = None
    benchmark_seed = int(SEED)
    breakdown_warmup = max(1, min(2, int(WARMUP)))
    breakdown_iters = max(1, min(5, int(ITERS)))
    for method in ("cp", "tt"):
        rdf, model, _weight_bytes = load_engine(
            method=method,
            ws_path=WS_PATH,
            robot_name=ROBOT_NAME,
            link_name=LINK_NAME,
            device=device,
            dtype=dtype,
        )
        raw_path, raw_dict = load_raw_model_dict(method=method, ws_path=WS_PATH, link_name=LINK_NAME)

        points = build_points(
            num_points=int(BENCHMARK_POINTS),
            device=device,
            dtype=dtype,
            seed=benchmark_seed,
        )
        mean_ms = time_forward_ms(
            rdf=rdf,
            points=points,
            with_grad=bool(WITH_GRAD),
            device=device,
            warmup=int(WARMUP),
            iters=int(ITERS),
        )
        clear_runtime_state(device)

        peak_extra = measure_peak_extra_bytes(
            method=method,
            model=model,
            rdf=rdf,
            num_points=int(BENCHMARK_POINTS),
            with_grad=bool(WITH_GRAD),
            device=device,
            dtype=dtype,
            seed=benchmark_seed,
            sample_interval_s=sample_interval_s,
        )

        clear_runtime_state(device)
        timing_breakdown = measure_timing_breakdown(
            method=method,
            rdf=rdf,
            points=points,
            with_grad=bool(WITH_GRAD),
            device=device,
            warmup=breakdown_warmup,
            iters=breakdown_iters,
        )
        if method == "tt":
            tt_variant_stats = benchmark_tt_matmul_variant(
                rdf=rdf,
                points=points,
                with_grad=bool(WITH_GRAD),
                device=device,
                warmup=breakdown_warmup,
                iters=breakdown_iters,
            )

        if method == "cp":
            rank_desc = f"rank={int(model.rank)}"
            param_numel = cp_param_numel(model)
            weight_bytes = cp_param_bytes(model)
            batch_bytes = cp_batch_bytes(rdf)
        else:
            rank_desc = f"ranks={tuple(int(r) for r in model.ranks)}"
            param_numel = tt_param_numel(model)
            weight_bytes = tt_param_bytes(model)
            batch_bytes = tt_batch_bytes(rdf)
        model_bytes = int(runtime_model_bytes(method, rdf, model))

        method_models[method] = model
        method_engines[method] = rdf
        method_stats[method] = {
            "method": method,
            "rank_desc": rank_desc,
            "n_func": int(model.n_func),
            "param_numel": int(param_numel),
            "weight_bytes": int(weight_bytes),
            "batch_bytes": int(batch_bytes),
            "model_bytes": int(model_bytes),
            "peak_extra_bytes": int(peak_extra),
            "total_footprint_bytes": int(model_bytes + peak_extra),
            "mean_ms": float(mean_ms),
            "common_ms": float(timing_breakdown["common_ms"]),
            "contract_ms": float(timing_breakdown["contract_ms"]),
            "tail_ms": float(timing_breakdown["tail_ms"]),
            "benchmark_points": int(BENCHMARK_POINTS),
        }
        raw_stats[method] = {
            "path": str(raw_path),
            "raw_array_bytes": int(raw_model_storage_bytes(raw_dict)),
            "file_size_bytes": int(raw_path.stat().st_size),
        }

        del points
        clear_runtime_state(device)

    print_model_memory_table(runtime_stats=method_stats, raw_stats=raw_stats)

    print("\n[Sample batch]")
    print_summary_row("CP", method_stats["cp"])
    print_summary_row("TT", method_stats["tt"])
    print(
        f"TT uses {human_bytes(method_stats['cp']['weight_bytes'] - method_stats['tt']['weight_bytes'])} less weight memory "
        f"than CP on this torcia setup."
    )

    print(f"\n[Timing breakdown] (warmup={breakdown_warmup}, iters={breakdown_iters})")
    print_timing_breakdown_row("CP", method_stats["cp"])
    print_timing_breakdown_row("TT", method_stats["tt"])
    cp_contract_ms = float(method_stats["cp"]["contract_ms"])
    tt_contract_ms = float(method_stats["tt"]["contract_ms"])
    if cp_contract_ms > 0 and tt_contract_ms > 0:
        print(f"Contract speedup (TT/CP): {cp_contract_ms / tt_contract_ms:.2f}x")

    if tt_variant_stats is not None:
        print("\n[TT contraction variants]")
        print_tt_variant_row(tt_variant_stats)
        matmul_ms = float(tt_variant_stats["matmul_ms"])
        if matmul_ms > 0:
            print(f"Matmul speedup over einsum path: {float(tt_variant_stats['einsum_ms']) / matmul_ms:.2f}x")

    if device.type == "cpu":
        cp_best = estimate_max_points_cpu(
            method="cp",
            model=method_models["cp"],
            rdf=method_engines["cp"],
            budget_bytes=budget_bytes,
            dtype=dtype,
            with_grad=bool(WITH_GRAD),
            search_cap=int(SEARCH_CAP),
        )
        tt_best = estimate_max_points_cpu(
            method="tt",
            model=method_models["tt"],
            rdf=method_engines["tt"],
            budget_bytes=budget_bytes,
            dtype=dtype,
            with_grad=bool(WITH_GRAD),
            search_cap=int(SEARCH_CAP),
        )
    else:
        cp_best = find_max_points_cuda(
            method="cp",
            model=method_models["cp"],
            rdf=method_engines["cp"],
            budget_bytes=budget_bytes,
            device=device,
            dtype=dtype,
            with_grad=bool(WITH_GRAD),
            search_start=int(SEARCH_START),
            search_cap=int(SEARCH_CAP),
            seed=int(SEED),
            sample_interval_s=sample_interval_s,
        )
        tt_best = find_max_points_cuda(
            method="tt",
            model=method_models["tt"],
            rdf=method_engines["tt"],
            budget_bytes=budget_bytes,
            device=device,
            dtype=dtype,
            with_grad=bool(WITH_GRAD),
            search_start=int(SEARCH_START),
            search_cap=int(SEARCH_CAP),
            seed=int(SEED),
            sample_interval_s=sample_interval_s,
        )

    print("\n[Max points at same budget]")
    print_capacity_row("CP", cp_best)
    print_capacity_row("TT", tt_best)

    if cp_best.get("success") and tt_best.get("success"):
        cp_points = int(cp_best["points"])
        tt_points = int(tt_best["points"])
        delta = tt_points - cp_points
        ratio = tt_points / max(cp_points, 1)
        winner = "TT" if delta >= 0 else "CP"
        print(
            f"{winner} can process {human_points(abs(delta))} more points than "
            f"{'CP' if winner == 'TT' else 'TT'} at the same memory budget "
            f"({ratio:.2f}x points for TT/CP)."
        )

    print("\n[Model config]")
    print(f"CP -> N={method_stats['cp']['n_func']} {method_stats['cp']['rank_desc']}")
    print(f"TT -> N={method_stats['tt']['n_func']} {method_stats['tt']['rank_desc']}")
    if PRINT_LATEX_TABLE:
        emit_compact_latex_table(
            runtime_stats=method_stats,
            cp_best=cp_best,
            tt_best=tt_best,
            method_models=method_models,
            device=device,
            budget_bytes=budget_bytes,
            output_path=LATEX_TABLE_OUTPUT_PATH,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main())
