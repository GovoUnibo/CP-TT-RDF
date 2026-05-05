import torch
import skimage.measure
import trimesh
import numpy as np
from src.core.math.transformations import matrix_to_pos_quat
from src.utils.MeshUtils import denormalize_mesh
from src.core.assets.entities.MeshModel import Mesh
from typing import Union
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from trimesh.util import concatenate as trimesh_concatenate



def sdf_to_mesh(
    weights: torch.Tensor,
    nbData: int,
    domain_min: float,
    domain_max: float,
    scaling_factor: torch.Tensor,
    centroid_offset: torch.Tensor,
    basis_function_from_3Dpoints=None,
):
    """
    Genera la mesh isosuperficie (livello 0) da un campo SDF espresso tramite pesi e funzioni base.
    Tutto rimane sullo stesso device di `weights`.
    """
    # --- device coerente ---
    device = weights.device
    dtype = weights.dtype

    # --- griglia regolare nello stesso device/dtype ---
    domain = torch.linspace(domain_min, domain_max, nbData, device=device, dtype=dtype)
    grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain, indexing="ij")
    p = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    # --- calcolo SDF in batch per evitare OOM ---
    d_list = []
    for p_s in torch.split(p, 10_000, dim=0):
        phi_p, _ = basis_function_from_3Dpoints(p_s, use_derivative=False)
        d_s = torch.matmul(phi_p, weights)
        d_list.append(d_s)

    d = torch.cat(d_list, dim=0)
    d_np = d.view(nbData, nbData, nbData).detach().cpu().numpy()

    # --- marching cubes su CPU ---
    d_smooth = gaussian_filter(d_np, sigma=1.0)
    spacing = (domain_max - domain_min) / max(nbData - 1, 1)
    verts, faces, normals, values = skimage.measure.marching_cubes(
        d_smooth,
        level=0.0,
        spacing=(spacing, spacing, spacing),
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh = mesh.subdivide_to_size(max_edge=1)
    mesh.apply_translation([domain_min, domain_min, domain_min])

    # --- conversione sicura CPU per i parametri ---
    scaling_factor_np = scaling_factor.detach().cpu().numpy() if torch.is_tensor(scaling_factor) else np.asarray(scaling_factor)
    centroid_offset_np = centroid_offset.detach().cpu().numpy() if torch.is_tensor(centroid_offset) else np.asarray(centroid_offset)

    # --- denormalizzazione finale nel frame originale ---
    mesh = denormalize_mesh(mesh.copy(), scaling_factor_np, centroid_offset_np)

    return mesh



def cp4d_sdf_to_mesh_robot(
    A: torch.Tensor,          # (N, R)
    B: torch.Tensor,          # (N, R)
    C: torch.Tensor,          # (N, R)
    lam: torch.Tensor,        # (R,)
    V: torch.Tensor,          # (L, R)
    list_ds: list,            # uno per link, per scala/centro
    T_list: list,             # lista di 4x4 (np.array) FK per ogni link
    nbData: int,
    domain_min: float,
    domain_max: float,
    bernstein_matrix_1d: callable,
    sigma_smooth: float = 1.0,
    batch_points: int = 50_000,
):
    """
    Costruisce la mesh del robot intero (tutti i link) in una posa fissata,
    a partire dalla decomposizione CP 4D (A,B,C,lam,V) e dalle trasformazioni FK.

    Parametri
    ---------
    A,B,C,lam,V : fattori CP 4D
    list_ds     : lista di dataset, uno per link, usata per 'mesh_scale_factor' e 'mesh_centroid_offset'
    T_list      : lista di trasformazioni 4x4 (numpy) base->link
    nbData      : risoluzione griglia SDF per marching cubes
    domain_min/max : dominio SDF
    bernstein_matrix_1d : funzione build_bernstein_t(t, use_derivative=False)

    Ritorna
    -------
    mesh_robot : trimesh.Trimesh oppure None se nessun link produce livello 0
    """
    device = A.device
    dtype  = A.dtype
    N, R   = A.shape
    assert B.shape == (N, R) and C.shape == (N, R), "A,B,C shape mismatch"
    lam = lam.view(R).to(device=device, dtype=dtype)

    L = V.shape[0]
    assert V.shape[1] == R, "Rank mismatch tra V e lam/A/B/C"
    assert len(list_ds) >= L,  "list_ds deve contenere almeno L dataset"
    assert len(T_list)  >= L,  "T_list deve contenere almeno L trasformazioni"

    # griglia 3D condivisa (stesso dominio per tutti i link)
    domain = torch.linspace(domain_min, domain_max, nbData, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(domain, domain, domain, indexing="ij")
    P = nbData ** 3

    denom = (domain_max - domain_min)
    tx = (gx - domain_min) / denom
    ty = (gy - domain_min) / denom
    tz = (gz - domain_min) / denom

    x = tx.reshape(-1)
    y = ty.reshape(-1)
    z = tz.reshape(-1)

    meshes = []

    # loop su tutti i link
    for ell in range(L):
        ds = list_ds[ell]

        # coeff per link: lam_eff = lam * V[ell,:]
        v_ell   = V[ell].to(device=device, dtype=dtype).view(-1)  # (R,)
        lam_eff = lam * v_ell

        sdf_vals = torch.empty(P, device=device, dtype=dtype)

        # valutazione SDF CP4D batched per questo link
        for start in range(0, P, batch_points):
            end = min(start + batch_points, P)
            xb, yb, zb = x[start:end], y[start:end], z[start:end]

            Phi_x, _ = bernstein_matrix_1d(xb, use_derivative=False)  # (m,N)
            Phi_y, _ = bernstein_matrix_1d(yb, use_derivative=False)
            Phi_z, _ = bernstein_matrix_1d(zb, use_derivative=False)

            Sx = Phi_x @ A      # (m,R)
            Sy = Phi_y @ B
            Sz = Phi_z @ C

            f_batch = (Sx * Sy * Sz) @ lam_eff   # (m,)
            sdf_vals[start:end] = f_batch

        sdf_grid = sdf_vals.view(nbData, nbData, nbData).detach().cpu().numpy()

        mn, mx = float(np.min(sdf_grid)), float(np.max(sdf_grid))
        if not (mn <= 0.0 <= mx):
            # nessun livello 0 per questo link, lo saltiamo
            continue

        if sigma_smooth and sigma_smooth > 0:
            sdf_grid = gaussian_filter(sdf_grid, sigma=sigma_smooth)

        spacing = float(domain_max - domain_min) / (nbData - 1)
        verts, faces, normals, values = marching_cubes(
            sdf_grid, level=0.0, spacing=(spacing, spacing, spacing)
        )

        mesh_link = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # riallinea al dominio SDF
        mesh_link.apply_translation([domain_min, domain_min, domain_min])

        # denormalizzazione (scale + offset nel frame link)
        sf = ds['mesh_scale_factor']
        co = ds['mesh_centroid_offset']

        sf_np = sf.detach().cpu().numpy() if torch.is_tensor(sf) else np.asarray(sf)
        co_np = co.detach().cpu().numpy() if torch.is_tensor(co) else np.asarray(co)
        sf_np = np.atleast_1d(sf_np).astype(float)
        co_np = np.atleast_1d(co_np).astype(float)

        if sf_np.size == 1:
            mesh_link.apply_scale(float(sf_np[0]))
        elif sf_np.size >= 3:
            S = np.eye(4)
            S[0, 0], S[1, 1], S[2, 2] = float(sf_np[0]), float(sf_np[1]), float(sf_np[2])
            mesh_link.apply_transform(S)
        else:
            mesh_link.apply_scale(float(sf_np.ravel()[0]))

        if co_np.size >= 3:
            mesh_link.apply_translation(co_np[:3])
        elif co_np.size == 1:
            mesh_link.apply_translation([float(co_np[0])] * 3)

        # applica FK T0^ell (4x4 numpy) per portare il link nel mondo
        T = T_list[ell]
        mesh_link.apply_transform(T)

        meshes.append(mesh_link)

    if not meshes:
        return None

    # una sola mesh del robot intero
    mesh_robot = trimesh_concatenate(meshes)
    return mesh_robot

def tt4d_sdf_to_mesh_robot(
    G0: torch.Tensor,         # (1, L, r0)
    G1: torch.Tensor,         # (r0, N, r1)
    G2: torch.Tensor,         # (r1, N, r2)
    G3: torch.Tensor,         # (r2, N, 1)
    list_ds: list,            # uno per link, per scala/centro
    T_list: list,             # lista di 4x4 (np.array) FK per ogni link
    nbData: int,
    domain_min: float,
    domain_max: float,
    bernstein_matrix_1d: callable,
    sigma_smooth: float = 1.0,
    batch_points: int = 50_000,
):
    """
    Mesh del robot intero a partire da TT 4D: W[l,i,j,k] in TT(G0,G1,G2,G3).
    """
    device = G0.device
    dtype  = G0.dtype

    # --- check shape ---
    assert G0.dim() == 3 and G1.dim() == 3 and G2.dim() == 3 and G3.dim() == 3
    _, L, r0 = G0.shape
    r0_g1, N, r1 = G1.shape
    r1_g2, N2, r2 = G2.shape
    r2_g3, N3, one = G3.shape

    assert r0_g1 == r0
    assert r1_g2 == r1
    assert r2_g3 == r2
    assert one == 1
    assert N == N2 == N3
    assert len(list_ds) >= L
    assert len(T_list)  >= L

    # griglia 3D
    domain = torch.linspace(domain_min, domain_max, nbData, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(domain, domain, domain, indexing="ij")
    P = nbData ** 3

    denom = (domain_max - domain_min)
    tx = (gx - domain_min) / denom
    ty = (gy - domain_min) / denom
    tz = (gz - domain_min) / denom

    x = tx.reshape(-1)
    y = ty.reshape(-1)
    z = tz.reshape(-1)

    meshes = []

    # G3_red: (r2, N)
    G3_red = G3[..., 0]

    for ell in range(L):
        ds = list_ds[ell]

        # vettore TT del link ell: g0_ell ∈ (r0,)
        g0_ell = G0[0, ell, :].to(device=device, dtype=dtype)  # (r0,)

        sdf_vals = torch.empty(P, device=device, dtype=dtype)

        for start in range(0, P, batch_points):
            end = min(start + batch_points, P)
            xb, yb, zb = x[start:end], y[start:end], z[start:end]
            B = xb.numel()

            Phi_x, _ = bernstein_matrix_1d(xb, use_derivative=False)  # (B,N)
            Phi_y, _ = bernstein_matrix_1d(yb, use_derivative=False)  # (B,N)
            Phi_z, _ = bernstein_matrix_1d(zb, use_derivative=False)  # (B,N)

            # ----- forward TT 4D (stesso schema di train_tt_robot) -----
            # G1: (r0, N, r1) → indici a0, n, a1
            # Bx: (B, r0, r1)
            Bx = torch.einsum('bn,anr->bar', Phi_x, G1)       # (B,r0,r1)

            # G2: (r1, N, r2)
            # By: (B, r1, r2)
            By = torch.einsum('bn,anr->bar', Phi_y, G2)       # (B,r1,r2)

            # G3_red: (r2, N)
            # Bz: (B, r2) = Σ_n Phi_z[b,n] * G3_red[a2,n]
            Bz = torch.einsum('bn,an->ba', Phi_z, G3_red)     # (B,r2)

            # T1: (B, r1) = Σ_{a0} g0_ell[a0] * Bx[b,a0,a1]
            T1 = torch.einsum('a,bar->br', g0_ell, Bx)        # (B,r1)

            # T2: (B, r2) = Σ_{a1} T1[b,a1] * By[b,a1,a2]
            T2 = torch.einsum('ba,bar->br', T1, By)           # (B,r2)

            # y_hat: (B,) = Σ_{a2} T2[b,a2] * Bz[b,a2]
            f_batch = (T2 * Bz).sum(dim=1)                    # (B,)

            sdf_vals[start:end] = f_batch

        sdf_grid = sdf_vals.view(nbData, nbData, nbData).detach().cpu().numpy()

        mn, mx = float(np.min(sdf_grid)), float(np.max(sdf_grid))
        if not (mn <= 0.0 <= mx):
            # nessun livello 0 → salta link
            continue

        if sigma_smooth and sigma_smooth > 0:
            sdf_grid = gaussian_filter(sdf_grid, sigma=sigma_smooth)

        spacing = float(domain_max - domain_min) / (nbData - 1)
        verts, faces, normals, values = marching_cubes(
            sdf_grid, level=0.0, spacing=(spacing, spacing, spacing)
        )

        mesh_link = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh_link.apply_translation([domain_min, domain_min, domain_min])

        # ---- denormalizzazione nel frame del link ----
        sf = ds['mesh_scale_factor']
        co = ds['mesh_centroid_offset']

        sf_np = sf.detach().cpu().numpy() if torch.is_tensor(sf) else np.asarray(sf)
        co_np = co.detach().cpu().numpy() if torch.is_tensor(co) else np.asarray(co)
        sf_np = np.atleast_1d(sf_np).astype(float)
        co_np = np.atleast_1d(co_np).astype(float)

        if sf_np.size == 1:
            mesh_link.apply_scale(float(sf_np[0]))
        elif sf_np.size >= 3:
            S = np.eye(4)
            S[0, 0], S[1, 1], S[2, 2] = float(sf_np[0]), float(sf_np[1]), float(sf_np[2])
            mesh_link.apply_transform(S)
        else:
            mesh_link.apply_scale(float(sf_np.ravel()[0]))

        if co_np.size >= 3:
            mesh_link.apply_translation(co_np[:3])
        elif co_np.size == 1:
            mesh_link.apply_translation([float(co_np[0])] * 3)

        # FK base->link
        T = T_list[ell]
        mesh_link.apply_transform(T)

        meshes.append(mesh_link)

    if not meshes:
        return None

    mesh_robot = trimesh_concatenate(meshes)
    return mesh_robot



def cp_sdf_to_mesh(
    A: torch.Tensor,   # (N, R)
    B: torch.Tensor,   # (N, R)
    C: torch.Tensor,   # (N, R)
    lam: torch.Tensor, # (R,)
    nbData: int,
    domain_min: float,
    domain_max: float,
    scaling_factor: torch.Tensor,   # shape (3,) o scalare
    centroid_offset: torch.Tensor,  # shape (3,)
    sigma_smooth: float = 1.0,
    batch_points: int = 50_000,
    bernstein_matrix_1d: callable = None,
):
    """
    Ricostruisce l'isuperficie livello 0 dell'SDF CP:
        f(p) = sum_r lam_r * (Phi_x^T a_r) * (Phi_y^T b_r) * (Phi_z^T c_r)
    Ritorna: trimesh.Trimesh | None (se il livello 0 non esiste).
    """
    # --- coerenza device/dtype ---
    device = A.device
    dtype  = A.dtype
    N, R   = A.shape
    assert B.shape == (N, R) and C.shape == (N, R)
    lam = lam.view(R).to(device=device, dtype=dtype)

    # --- griglia regolare ---
    domain = torch.linspace(domain_min, domain_max, nbData, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(domain, domain, domain, indexing="ij")
    P = nbData ** 3

    # normalizzazione a [0,1] per le basi di Bernstein
    denom = (domain_max - domain_min)
    tx = (gx - domain_min) / denom
    ty = (gy - domain_min) / denom
    tz = (gz - domain_min) / denom

    # Flatten per batch
    x = tx.reshape(-1)
    y = ty.reshape(-1)
    z = tz.reshape(-1)

    # --- valutazione SDF CP in batch ---
    sdf_vals = torch.empty(P, device=device, dtype=dtype)

    for start in range(0, P, batch_points):
        end = min(start + batch_points, P)
        xb, yb, zb = x[start:end], y[start:end], z[start:end]

        # la tua build_bernstein_t(t, use_derivative=False) -> (phi, dphi)
        Phi_x, _ = bernstein_matrix_1d(xb, use_derivative=False)  # (m,N)
        Phi_y, _ = bernstein_matrix_1d(yb, use_derivative=False)  # (m,N)
        Phi_z, _ = bernstein_matrix_1d(zb, use_derivative=False)  # (m,N)

        # proiezioni su fattori: (m,N) @ (N,R) -> (m,R)
        Sx = Phi_x @ A
        Sy = Phi_y @ B
        Sz = Phi_z @ C

        # f = (Sx * Sy * Sz) @ lam
        f_batch = (Sx * Sy * Sz) @ lam
        sdf_vals[start:end] = f_batch

    # reshape su griglia e passaggio a CPU per marching cubes
    sdf_grid = sdf_vals.view(nbData, nbData, nbData).detach().cpu().numpy()

    # se il livello 0 non è presente, abort pulito
    mn, mx = float(np.min(sdf_grid)), float(np.max(sdf_grid))
    if not (mn <= 0.0 <= mx):
        return None

    # --- marching cubes su CPU ---
    if sigma_smooth and sigma_smooth > 0:
        sdf_grid = gaussian_filter(sdf_grid, sigma=sigma_smooth)

    spacing = float(domain_max - domain_min) / (nbData - 1)
    verts, faces, normals, values = marching_cubes(
        sdf_grid, level=0.0, spacing=(spacing, spacing, spacing)
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # riallinea dal voxel-space all'origine del dominio
    mesh.apply_translation([domain_min, domain_min, domain_min])

    # post-process (IN-PLACE: non riassegnare!)
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
    except Exception:
        pass

    # --- denormalizzazione come nel "caso classico" ---
    # robusto a scalare, 0-D, 3-D
    sf = scaling_factor.detach().cpu().numpy() if torch.is_tensor(scaling_factor) else np.asarray(scaling_factor)
    co = centroid_offset.detach().cpu().numpy() if torch.is_tensor(centroid_offset) else np.asarray(centroid_offset)
    sf = np.atleast_1d(sf).astype(float)
    co = np.atleast_1d(co).astype(float)

    if sf.size == 1:
        mesh.apply_scale(float(sf[0]))
    elif sf.size >= 3:
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = float(sf[0]), float(sf[1]), float(sf[2])
        mesh.apply_transform(S)
    else:
        mesh.apply_scale(float(sf.ravel()[0]))

    if co.size >= 3:
        mesh.apply_translation(co[:3])
    elif co.size == 1:
        mesh.apply_translation([float(co[0])]*3)

    return mesh

def tt_sdf_to_mesh(
    G1: torch.Tensor,   # (1, N, r1)
    G2: torch.Tensor,   # (r1, N, r2)
    G3: torch.Tensor,   # (r2, N, 1)
    nbData: int,
    domain_min: float,
    domain_max: float,
    scaling_factor: torch.Tensor,   # shape (3,) o scalare
    centroid_offset: torch.Tensor,  # shape (3,)
    sigma_smooth: float = 1.0,
    batch_points: int = 50_000,
    bernstein_matrix_1d: callable = None,
):
    """
    Ricostruisce l'isuperficie livello 0 dell'SDF TT:
        W(i,j,k) ≈ TT(G1,G2,G3)
        f(p) = Σ_{b,i} U2[p,b] * Phi_z[p,i] * G3[b,i,0]
    dove:
        U1[p,a] = Σ_i Phi_x[p,i] * G1[0,i,a]
        U2[p,b] = Σ_{a,i} U1[p,a] * Phi_y[p,i] * G2[a,i,b]

    Ritorna: trimesh.Trimesh | None (se il livello 0 non esiste).
    """
    # --- coerenza device/dtype ---
    device = G1.device
    dtype  = G1.dtype

    # controlli shape basilari
    assert G1.ndim == 3 and G2.ndim == 3 and G3.ndim == 3, "G1,G2,G3 devono essere 3D"
    _, N1, r1 = G1.shape       # (1, N, r1)
    r1b, N2, r2 = G2.shape     # (r1, N, r2)
    r2b, N3, one = G3.shape    # (r2, N, 1)
    assert one == 1, "G3 deve avere ultima dimensione = 1"
    assert N1 == N2 == N3, "Dimensione N incoerente tra G1,G2,G3"
    N = N1
    assert r1 == r1b and r2 == r2b, "TT ranks incoerenti tra G1,G2,G3"

    # --- griglia regolare ---
    domain = torch.linspace(domain_min, domain_max, nbData, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(domain, domain, domain, indexing="ij")
    P = nbData ** 3

    # normalizzazione a [0,1] per le basi di Bernstein
    denom = (domain_max - domain_min)
    tx = (gx - domain_min) / denom
    ty = (gy - domain_min) / denom
    tz = (gz - domain_min) / denom

    # Flatten per batch
    x = tx.reshape(-1)
    y = ty.reshape(-1)
    z = tz.reshape(-1)

    # --- valutazione SDF TT in batch ---
    sdf_vals = torch.empty(P, device=device, dtype=dtype)

    for start in range(0, P, batch_points):
        end = min(start + batch_points, P)
        xb, yb, zb = x[start:end], y[start:end], z[start:end]

        # la tua build_bernstein_t(t, use_derivative=False) -> (phi, dphi)
        Phi_x, _ = bernstein_matrix_1d(xb, use_derivative=False)  # (m,N)
        Phi_y, _ = bernstein_matrix_1d(yb, use_derivative=False)  # (m,N)
        Phi_z, _ = bernstein_matrix_1d(zb, use_derivative=False)  # (m,N)
        m = Phi_x.shape[0]

        # U1: (m, r1) = Σ_i Phi_x[m,i] * G1[0,i,a]
        # G1[0] ha shape (N, r1)
        U1 = torch.einsum('mi,ir->mr', Phi_x, G1[0])

        # U2: (m, r2) = Σ_{a,i} U1[m,a] * Phi_y[m,i] * G2[a,i,b]
        U2 = torch.einsum('ma,mi,aib->mb', U1, Phi_y, G2)

        # yhat: (m,) = Σ_{b,i} U2[m,b] * Phi_z[m,i] * G3[b,i,0]
        G3_red = G3[..., 0]  # (r2, N)
        f_batch = torch.einsum('mb,mi,bi->m', U2, Phi_z, G3_red)

        sdf_vals[start:end] = f_batch

    # reshape su griglia e passaggio a CPU per marching cubes
    sdf_grid = sdf_vals.view(nbData, nbData, nbData).detach().cpu().numpy()

    # se il livello 0 non è presente, abort pulito
    mn, mx = float(np.min(sdf_grid)), float(np.max(sdf_grid))
    if not (mn <= 0.0 <= mx):
        return None

    # --- marching cubes su CPU ---
    if sigma_smooth and sigma_smooth > 0:
        sdf_grid = gaussian_filter(sdf_grid, sigma=sigma_smooth)

    spacing = float(domain_max - domain_min) / (nbData - 1)
    verts, faces, normals, values = marching_cubes(
        sdf_grid, level=0.0, spacing=(spacing, spacing, spacing)
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # riallinea dal voxel-space all'origine del dominio
    mesh.apply_translation([domain_min, domain_min, domain_min])

    # post-process (IN-PLACE: non riassegnare!)
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
    except Exception:
        pass

    # --- denormalizzazione come nel "caso classico" ---
    # robusto a scalare, 0-D, 3-D
    sf = scaling_factor.detach().cpu().numpy() if torch.is_tensor(scaling_factor) else np.asarray(scaling_factor)
    co = centroid_offset.detach().cpu().numpy() if torch.is_tensor(centroid_offset) else np.asarray(centroid_offset)
    sf = np.atleast_1d(sf).astype(float)
    co = np.atleast_1d(co).astype(float)

    if sf.size == 1:
        mesh.apply_scale(float(sf[0]))
    elif sf.size >= 3:
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = float(sf[0]), float(sf[1]), float(sf[2])
        mesh.apply_transform(S)
    else:
        mesh.apply_scale(float(sf.ravel()[0]))

    if co.size >= 3:
        mesh.apply_translation(co[:3])
    elif co.size == 1:
        mesh.apply_translation([float(co[0])] * 3)

    return mesh


def rdf_as_mesh(sdf_mesh, tf):
    mesh = mesh_transfrom(sdf_mesh, tf)
    return mesh

def mesh_transfrom(mesh : Union[trimesh.Trimesh, Mesh], tf) -> trimesh.Trimesh:
    if isinstance(mesh, Mesh):
        mesh = mesh.mesh

    translation, quaternion = matrix_to_pos_quat(tf)
    mesh.apply_transform(trimesh.transformations.quaternion_matrix(quaternion.cpu().numpy()))
    mesh.apply_translation(translation)
    return mesh

def point_cloud_transform(points:torch.Tensor, tf:np.ndarray) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        if points.is_cuda:
            points = points.cpu()
        points = points.numpy()
    
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = points_homogeneous @ tf.T
    return transformed_points[:, :3]


def sample_nearby_points(points, sdf, central_point, radius, n_samples, device):
    def get_nearby_points(points, query_point, radius):
        distances = np.linalg.norm(points - query_point, axis=1)
        nearby_idx = np.where(distances < radius)[0]
        return nearby_idx

    nearby_indices = get_nearby_points(points, central_point, radius)
    
    # Assicurati di avere sempre n_samples punti campionati
    
    choice_indices = np.random.choice(nearby_indices, n_samples, replace=True)  # Campiona con replacement


    # Estrai i punti vicini e i loro sdf
    sampled_points = torch.from_numpy(points[choice_indices]).float().to(device)
    sampled_sdf = torch.from_numpy(sdf[choice_indices]).float().to(device)
    
    return sampled_points, sampled_sdf

def check_ellipsoid_fit(points, center, axes, scales):
    print("Verifica ortogonalità matrice assi (axes):")
    ortonorm = np.allclose(axes.T @ axes, np.eye(3), atol=1e-7)
    print(f"axes.T @ axes ≈ I? {ortonorm}")

    # Calcolo punti locali (coordinate nel sistema degli assi)
    points_centered = points - center
    points_local = points_centered @ axes
    print(f"Shape points_local: {points_local.shape}")

    # Controlla max valori assoluti nei punti locali e semiassi
    max_local = np.max(np.abs(points_local), axis=0)
    print(f"Max valori assoluti nei punti locali: {max_local}")
    print(f"Semiassi forniti (scales): {scales}")

    # Check che i semiassi siano almeno i max valori assoluti (dovrebbero essere uguali o più grandi)
    for i, (m, s) in enumerate(zip(max_local, scales)):
        if s < m:
            print(f"Attenzione: semi-asse {i} ({s}) è minore del massimo valore locale {m}!")

    # Verifica formula ellissoide
    val = (points_local[:, 0] / scales[0])**2 + (points_local[:, 1] / scales[1])**2 + (points_local[:, 2] / scales[2])**2
    all_inside = np.all(val <= 1 + 1e-12)
    n_outside = np.sum(val > 1 + 1e-12)

    print(f"Tutti i punti sono dentro l'ellissoide? {all_inside}")
    if not all_inside:
        print(f"Punti fuori: {n_outside} su {len(points)}")

    # Mostra alcuni valori anomali
    if n_outside > 0:
        idx_outside = np.where(val > 1 + 1e-12)[0]
        print("Valori ellissoide dei primi 5 punti fuori:", val[idx_outside[:5]])
        print("Coordinate locali dei primi 5 punti fuori:", points_local[idx_outside[:5]])

    return all_inside, val
