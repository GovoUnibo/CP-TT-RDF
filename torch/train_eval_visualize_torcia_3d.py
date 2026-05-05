#!/usr/bin/env python3
import time
import sys
from pathlib import Path

import numpy as np
import torch

def _discover_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "torch").is_dir():
            return candidate
    return start


ROOT = _discover_repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.sdf_creator import SDFTrain
from src.core.assets.MeshHandler import MeshHandler
from src.core.train.train_config import CPLinkCfg, TTLinkCfg, TrainConfig, Train_W
from src.rdf_3Dcp import RDF_3D_CP
from src.rdf_3Dtt import RDF_TT
from src.rdf_weights import RDF_Weights

# ========================= USER CONFIG =========================
WS_PATH = str(ROOT / "torch")
ROBOT_NAME = "torch"

# Può essere:
# - nome mesh già in WS_PATH/Meshes (es: "torcia")
# - path completo a file .stl (es: "/path/to/torcia.stl")
MESH_INPUT = "torcia"

N_FUNC = 64            # N basi Bernstein per tutti i metodi
CP_RANK = 128          # rank CP
TT_RANKS = (12, 12)    # ranks TT 3D (più alti = migliore resa geometrica, più costo)

ITERS_W = 250
ITERS_CP = 10
ITERS_TT = 10
CP_METHOD = "adam"      # {"adam", "als"}
CP_LR = 5e-4           # usato solo da Adam
CP_BATCH_SIZE = 16_384
CP_RIDGE = 2e-4

TT_METHOD = "adam"     # {"adam", "mals"} (consigliato: mals)
TT_LR = 1e-3
TT_BATCH_SIZE = 8_192
TT_RIDGE = 5e-5

DATASET_NUM_POINTS = 800_000  # usato solo se il dataset non esiste e va creato
EVAL_MAX_POINTS = 120_000
TORCIA_EVAL_POINTS = 20_000
EVAL_CHUNK = 10_000
EVAL_SEED = 0
EVAL_USE_DATASET = False   # False: usa punti campionati uniformemente sulla superficie mesh
GET_GRAD = False

RUN_TRAIN_WEIGHTS = False
RUN_TRAIN_CP = True
RUN_TRAIN_TT = True
RUN_VISUALIZATION = True
SHOW_EVAL_POINTS_IN_VISUALIZATION = False
SHOW_CARTESIAN_FRAME = False  # False: nasconde il frame cartesiano nelle viste
EVAL_POINTS_COLOR = "#9E9E9E"
EVAL_POINTS_SIZE = 8.0
EVAL_POINTS_OPACITY = 0.3

MESH_COLOR = "#C7C7C7"
MESH_OPACITY = 1.0
RDF_OPACITY = 1
RDF_COLORS = {"weights": "#1f77b4", "cp": "#788291eb", "tt": "#2ca02c"}
# =============================================================


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_mesh_input(mesh_input):
    p = Path(mesh_input).expanduser()
    if p.suffix.lower() == ".stl" or p.exists():
        if not p.exists():
            raise FileNotFoundError(f"Mesh path non trovato: {p}")
        return p.stem, str(p)
    return mesh_input, None


def _search_mesh_file(mesh_name):
    roots = [Path(WS_PATH), ROOT / "torch", ROOT / "panda_test"]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.stl"):
            stem = p.stem.lower()
            name = mesh_name.lower()
            if stem == name:
                return str(p)
            if name in stem:
                candidates.append(str(p))
    return candidates[0] if len(candidates) == 1 else None


def _mesh_aliases(mesh_name):
    aliases = {"torcia": ["torch"], "torch": ["torcia"]}
    return [mesh_name] + aliases.get(mesh_name.lower(), [])


def ensure_mesh_in_workspace(trainer, ws_path, robot_name, mesh_name, mesh_src_path):
    mesh_target = Path(ws_path) / "Meshes" / f"{mesh_name}.stl"
    if mesh_target.exists():
        return
    src = mesh_src_path
    if src is None:
        for candidate_name in _mesh_aliases(mesh_name):
            src = _search_mesh_file(candidate_name)
            if src is not None:
                print(f"[INFO] Mesh trovata automaticamente: {src}")
                break
    if src is None:
        available = sorted([p.stem for p in (Path(ws_path) / "Meshes").glob("*.stl")])
        preview = ", ".join(available[:15]) + (" ..." if len(available) > 15 else "")
        raise FileNotFoundError(
            f"Mesh '{mesh_name}.stl' non trovata in {mesh_target.parent}.\n"
            "Imposta MESH_INPUT con path completo al file .stl.\n"
            f"Mesh disponibili in workspace: {preview}"
        )
    trainer.copy_file(src, file_rename=mesh_name, robot_name=robot_name)


def train_models(mesh_name, mesh_src_path, device, dtype, run_w, run_cp, run_tt):
    if not (run_w or run_cp or run_tt):
        print("\n[TRAIN] Nessun training richiesto (tutti i flag train sono False).")
        return

    trainer = SDFTrain(device=device.type, dtype=dtype)
    trainer.init_robot_folder(WS_PATH, robot_name=ROBOT_NAME)
    ensure_mesh_in_workspace(trainer, WS_PATH, ROBOT_NAME, mesh_name, mesh_src_path)

    cfg = TrainConfig(
        debug=False,
        links_to_train=[mesh_name],
        classic=Train_W(run=run_w, n_func=N_FUNC, iters=ITERS_W, batch_near=1024, batch_rand=64),
        cp_link=CPLinkCfg(run=run_cp, method=CP_METHOD, n_func=N_FUNC, rank=CP_RANK, iters=ITERS_CP, batch_size=CP_BATCH_SIZE, lr=CP_LR, ridge=CP_RIDGE),
        tt_link=TTLinkCfg(run=run_tt, method=TT_METHOD, n_func=N_FUNC, ranks=TT_RANKS, iters=ITERS_TT, batch_size=TT_BATCH_SIZE, ridge=TT_RIDGE, lr=TT_LR),
    )
    trainer.create_model(cfg, robot_name=ROBOT_NAME, number_of_points=DATASET_NUM_POINTS)


def load_eval_dataset(ws_path, mesh_name, max_points, seed):
    ds_path = Path(ws_path) / "Dataset" / f"{mesh_name}.npy"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {ds_path}")
    ds = np.load(ds_path, allow_pickle=True).item()
    scale_factor = float(ds.get("mesh_scale_factor", 1.0))
    pts = np.concatenate([np.asarray(ds["near_points"]), np.asarray(ds["query_points"])], axis=0)
    sdf = np.concatenate([np.asarray(ds["near_sdf"]), np.asarray(ds["query_sdf"])], axis=0)
    if max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        sdf = sdf[idx]
    # I modelli RDF ritornano SDF in unita' mesh originali (dopo scale_factor).
    # Convertiamo il ground-truth dataset dalla scala unitaria alla scala originale.
    sdf_world = sdf * scale_factor
    return pts.astype(np.float32, copy=False), sdf_world.astype(np.float32, copy=False), scale_factor


def load_eval_surface_points(ws_path, mesh_name, n_points, seed):
    """
    Campiona punti uniformi sulla superficie mesh (no dataset).
    Per punti di superficie, il target SDF e' 0.
    """
    import trimesh

    mesh = MeshHandler(ws_path).load(mesh_name).mesh
    n_points = int(n_points)
    rng = np.random.default_rng(int(seed))

    # Sampling uniforme superficie senza KMeans (molto piu' leggero su 20k punti).
    pts, _ = trimesh.sample.sample_surface_even(mesh, n_points, seed=int(seed))
    pts = np.asarray(pts)
    if pts.shape[0] < n_points:
        extra, _ = trimesh.sample.sample_surface(mesh, n_points - pts.shape[0])
        pts = np.vstack([pts, extra])
    elif pts.shape[0] > n_points:
        idx = rng.choice(pts.shape[0], size=n_points, replace=False)
        pts = pts[idx]

    pts = pts.astype(np.float32, copy=False)
    sdf = np.zeros((pts.shape[0],), dtype=np.float32)
    return pts, sdf


def build_rdf_engines(mesh_name, device, dtype):
    engines = {
        "weights": RDF_Weights(device=device.type, dtype=dtype),
        "cp": RDF_3D_CP(device=device.type, dtype=dtype),
        "tt": RDF_TT(device=device.type, dtype=dtype),
    }
    for rdf in engines.values():
        rdf.init_robot_folder(WS_PATH, robot_name=ROBOT_NAME)
        rdf.show_cartesian_frame = SHOW_CARTESIAN_FRAME
        rdf.add_models(link_names=[mesh_name], robot_name=ROBOT_NAME)
        rdf.add_mesh(link_names=[mesh_name], robot_name=ROBOT_NAME)
        rdf.set_ordered_batch_params([mesh_name])
    return engines


def evaluate_torcia_20k(mesh_name, device, dtype, seed=EVAL_SEED, chunk_size=EVAL_CHUNK, get_grad=GET_GRAD):
    """
    Valuta torcia su 20k punti campionati uniformemente sulla superficie mesh (no dataset).
    Target SDF: zero sui punti di superficie.
    """
    points_np, sdf_np = load_eval_surface_points(WS_PATH, mesh_name, TORCIA_EVAL_POINTS, seed)
    print(f"\n[EVAL-20K SURFACE] punti usati: {points_np.shape[0]} | grad={get_grad}")

    engines = build_rdf_engines(mesh_name, device, dtype)
    results = {}
    for name, rdf in engines.items():
        dt_ms, mae, rmse, maxe = evaluate_engine(
            rdf=rdf,
            points_np=points_np,
            sdf_np=sdf_np,
            device=device,
            dtype=dtype,
            get_grad=get_grad,
            chunk_size=chunk_size,
        )
        results[name] = (dt_ms, mae, rmse, maxe)
        print(
            f"[EVAL-20K SURFACE] {name:>8s} | total_ms={dt_ms:9.3f} | "
            f"mae={mae:.6e} ({mae * 1000.0:.3f} mm) | "
            f"rmse={rmse:.6e} ({rmse * 1000.0:.3f} mm) | "
            f"max={maxe:.6e} ({maxe * 1000.0:.3f} mm)"
        )
    return results


@torch.no_grad()
def evaluate_engine(rdf, points_np, sdf_np, device, dtype, get_grad, chunk_size):
    n = points_np.shape[0]
    abs_sum = 0.0
    sq_sum = 0.0
    max_err = 0.0

    sync(device)
    t0 = time.perf_counter()

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        pts = torch.as_tensor(points_np[start:end], device=device, dtype=dtype).unsqueeze(0)
        sdf_true = torch.as_tensor(sdf_np[start:end], device=device, dtype=dtype)
        sdf_pred, _, _ = rdf.inference_link_batch(pts, get_grad=get_grad, get_min=False, forward_tensor=None)
        err = (sdf_pred[0] - sdf_true).abs()
        abs_sum += float(err.sum().item())
        sq_sum += float((err * err).sum().item())
        max_err = max(max_err, float(err.max().item()))

    sync(device)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    mae = abs_sum / max(1, n)
    rmse = (sq_sum / max(1, n)) ** 0.5
    return dt_ms, mae, rmse, max_err


def visualize_engine(name, rdf, mesh_name, device, dtype, mesh_only=False, eval_points_np=None):
    fw_dict = {mesh_name: torch.eye(4, device=device, dtype=dtype)}
    title = f"{name.upper()} - SOLO MESH" if mesh_only else f"{name.upper()} - SOLO RDF"
    print(f"\n[VISUALIZE] {title} (chiudi la finestra per continuare)")
    if eval_points_np is None:
        eval_points = torch.empty((0, 3), device=device, dtype=dtype)
    else:
        eval_points = torch.as_tensor(eval_points_np, device=device, dtype=dtype).reshape(-1, 3)
    rdf.visualize_scene(
        forward_as_dict=fw_dict,
        links_as_mesh=mesh_only,
        mesh_link_names=[mesh_name],
        links_as_rdf=not mesh_only,
        rdf_link_names=[mesh_name],
        additional_point_cloud=eval_points,
        point_cloud_color=EVAL_POINTS_COLOR,
        point_cloud_size=EVAL_POINTS_SIZE,
        point_cloud_opacity=EVAL_POINTS_OPACITY,
        mesh_color=MESH_COLOR,
        mesh_opacity=MESH_OPACITY,
        rdf_color=RDF_COLORS[name],
        rdf_opacity=RDF_OPACITY,
    )


def _weight_tensor_from_model(model, device, dtype):
    n = int(model.n_func)
    w = torch.as_tensor(model.weights, device=device, dtype=dtype).reshape(-1)
    exp = n ** 3
    if w.numel() < exp:
        w = torch.nn.functional.pad(w, (0, exp - w.numel()))
    elif w.numel() > exp:
        w = w[:exp]
    return w.reshape(n, n, n)


def _cp_tensor_from_model(model, device, dtype):
    a = torch.as_tensor(model.A, device=device, dtype=dtype)
    b = torch.as_tensor(model.B, device=device, dtype=dtype)
    c = torch.as_tensor(model.C, device=device, dtype=dtype)
    lam = torch.as_tensor(model.lamd, device=device, dtype=dtype).reshape(-1)
    return torch.einsum("ir,jr,kr,r->ijk", a, b, c, lam)


def _tt_tensor_from_model(model, device, dtype):
    g1 = torch.as_tensor(model.G1, device=device, dtype=dtype)
    g2 = torch.as_tensor(model.G2, device=device, dtype=dtype)
    g3 = torch.as_tensor(model.G3, device=device, dtype=dtype)
    return torch.einsum("ia,ajb,bk->ijk", g1[0], g2, g3[..., 0])


def print_weight_decomp_debug(mesh_name, engines, device, dtype):
    w = _weight_tensor_from_model(
        getattr(engines["weights"], mesh_name + engines["weights"].model_extension), device, dtype
    )
    cp = _cp_tensor_from_model(
        getattr(engines["cp"], mesh_name + engines["cp"].model_extension), device, dtype
    )
    tt = _tt_tensor_from_model(
        getattr(engines["tt"], mesh_name + engines["tt"].model_extension), device, dtype
    )

    def _metrics(ref, hat):
        err = hat - ref
        abs_err = err.abs()
        rel_fro = float(err.norm().item() / max(ref.norm().item(), 1e-12))
        mae = float(abs_err.mean().item())
        rmse = float((err.square().mean().sqrt()).item())
        maxe = float(abs_err.max().item())
        return rel_fro, mae, rmse, maxe

    print("[debug] tensor approximation vs weights")
    if tuple(cp.shape) == tuple(w.shape):
        cp_m = _metrics(w, cp)
        print(
            f"    w->cp | rel_fro={cp_m[0]:.6e} | mae={cp_m[1]:.6e} | "
            f"rmse={cp_m[2]:.6e} | max={cp_m[3]:.6e}"
        )
    else:
        print(f"    w->cp | skipped (shape mismatch: W={tuple(w.shape)} CP={tuple(cp.shape)})")

    if tuple(tt.shape) == tuple(w.shape):
        tt_m = _metrics(w, tt)
        print(
            f"    w->tt | rel_fro={tt_m[0]:.6e} | mae={tt_m[1]:.6e} | "
            f"rmse={tt_m[2]:.6e} | max={tt_m[3]:.6e}"
        )
    else:
        print(f"    w->tt | skipped (shape mismatch: W={tuple(w.shape)} TT={tuple(tt.shape)})")


def main():
    mesh_name, mesh_src_path = resolve_mesh_input(MESH_INPUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Device: {device.type}")
    print(f"Mesh input: {MESH_INPUT} -> mesh_name: {mesh_name}")
    print(f"N={N_FUNC} | CP rank={CP_RANK} | TT ranks={TT_RANKS}")
    print(
        f"CP method={CP_METHOD} lr={CP_LR} bs={CP_BATCH_SIZE} ridge={CP_RIDGE} | "
        f"TT method={TT_METHOD} lr={TT_LR} bs={TT_BATCH_SIZE} ridge={TT_RIDGE}"
    )

    print(
        f"\n[TRAIN FLAGS] weights={RUN_TRAIN_WEIGHTS} | cp={RUN_TRAIN_CP} | tt={RUN_TRAIN_TT}"
    )
    train_models(
        mesh_name=mesh_name,
        mesh_src_path=mesh_src_path,
        device=device,
        dtype=dtype,
        run_w=RUN_TRAIN_WEIGHTS,
        run_cp=RUN_TRAIN_CP,
        run_tt=RUN_TRAIN_TT,
    )

    if EVAL_USE_DATASET:
        points_np, sdf_np, scale_factor = load_eval_dataset(WS_PATH, mesh_name, EVAL_MAX_POINTS, EVAL_SEED)
        print(
            f"\n[EVAL DATASET] punti usati: {points_np.shape[0]} | grad={GET_GRAD} | "
            f"scale_factor={scale_factor:.6e}"
        )
    else:
        points_np, sdf_np = load_eval_surface_points(WS_PATH, mesh_name, TORCIA_EVAL_POINTS, EVAL_SEED)
        print(
            f"\n[EVAL SURFACE] punti usati: {points_np.shape[0]} | grad={GET_GRAD} | "
            "target_sdf=0 (punti su superficie mesh)"
        )

    engines = build_rdf_engines(mesh_name, device, dtype)
    results = {}
    for name, rdf in engines.items():
        dt_ms, mae, rmse, maxe = evaluate_engine(
            rdf=rdf,
            points_np=points_np,
            sdf_np=sdf_np,
            device=device,
            dtype=dtype,
            get_grad=GET_GRAD,
            chunk_size=EVAL_CHUNK,
        )
        results[name] = (dt_ms, mae, rmse, maxe)
        print(
            f"{name:>8s} | total_ms={dt_ms:9.3f} | "
            f"mae={mae:.6e} ({mae * 1000.0:.3f} mm) | "
            f"rmse={rmse:.6e} ({rmse * 1000.0:.3f} mm) | "
            f"max={maxe:.6e} ({maxe * 1000.0:.3f} mm)"
        )
    print_weight_decomp_debug(mesh_name, engines, device, dtype)
    # Valutazione dedicata su 20k punti uniformi di superficie (no dataset).
    evaluate_torcia_20k(mesh_name=mesh_name, device=device, dtype=dtype)

    if RUN_VISUALIZATION:
        vis_points = points_np if SHOW_EVAL_POINTS_IN_VISUALIZATION else None
        # 1) solo torcia mesh
        visualize_engine("weights", engines["weights"], mesh_name, device, dtype, mesh_only=True, eval_points_np=vis_points)
        # 2) solo torcia weights
        visualize_engine("weights", engines["weights"], mesh_name, device, dtype, mesh_only=False, eval_points_np=vis_points)
        # 3) solo torcia CP
        visualize_engine("cp", engines["cp"], mesh_name, device, dtype, mesh_only=False, eval_points_np=vis_points)
        # 4) solo torcia TT
        visualize_engine("tt", engines["tt"], mesh_name, device, dtype, mesh_only=False, eval_points_np=vis_points)
    else:
        print("\n[VISUALIZE] Saltato (RUN_VISUALIZATION=False)")


if __name__ == "__main__":
    main()
