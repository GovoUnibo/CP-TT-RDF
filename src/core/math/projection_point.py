import torch
from typing import Union





def spherical_projection(points: torch.Tensor, r1: float = 1.0, center: torch.Tensor  = None):
    """
    Versione senza boolean indexing: evita nonzero/index nel profiler.
    points: (N,3)
    """
    if center is None:
        center = torch.zeros((1,3), device=points.device, dtype=points.dtype)
    else:
        center = center.reshape(1,3).to(device=points.device, dtype=points.dtype)

    rel = points - center                               # (N,3)
    r = rel.norm(dim=1).clamp_min(1e-12)               # (N,)
    outside = r > r1                                    # (N,) bool

    # scale = 1 (inside), r1/r (outside)
    scale = torch.where(outside, (r1 / r), torch.ones_like(r))   # (N,)
    proj = center + rel * scale.unsqueeze(1)                     # (N,3)

    dist = torch.where(outside, r - r1, torch.zeros_like(r))     # (N,)

    return proj, dist, outside

def jacobian_spherical_projection(z: torch.Tensor, r1: float):
    """
    z: (N,3)
    Ritorna J_out: (N,3,3), r: (N,1), u: (N,3)
    """
    r = z.norm(dim=1, keepdim=True).clamp_min(1e-12)   # (N,1)
    u = z / r                                          # (N,3)
    # I - u u^T
    I = torch.eye(3, device=z.device, dtype=z.dtype).unsqueeze(0)  # (1,3,3)
    uuT = u.unsqueeze(2) @ u.unsqueeze(1)                           # (N,3,3)
    J = (r1 / r).unsqueeze(-1) * (I - uuT)                          # (N,3,3)
    return J, r, u


def correct_sphere_gradient(points_scaled: torch.Tensor,
                            g_in: torch.Tensor,
                            outside: torch.Tensor,
                            r1: float):
    r = points_scaled.norm(dim=1, keepdim=True).clamp_min(1e-12)  # (N,1)
    u = points_scaled / r                                         # (N,3)

    ug = (u * g_in).sum(dim=1, keepdim=True)                      # (N,1)
    Jtg = (r1 / r) * (g_in - u * ug)                              # (N,3)

    mask = outside.reshape(-1, 1)
    g_bernstein_z = torch.where(mask, Jtg, g_in)
    g_sphere_z = torch.where(mask, u, 0.0)
    return g_bernstein_z + g_sphere_z



# def spherical_projection(points: torch.Tensor, r1: float = 1.0, center=None):
#     """
#     Proietta i punti fuori da una sfera di raggio r1 (opz. centrata in 'center').
#     - I punti dentro restano invariati
#     - I punti fuori vengono proiettati sulla superficie
#     - Ritorna:
#         proj   : (N,3) punti proiettati
#         dist   : (N,) distanza radiale (0 dentro, r-r1 fuori)
#         outside: (N,) bool, True per i punti proiettati
#     """
#     if center is None:
#         center = torch.zeros(3, device=points.device, dtype=points.dtype)

#     rel = points - center
#     r = torch.norm(rel, dim=1, keepdim=True).clamp_min(1e-12)  # (N,1)

#     outside = (r.squeeze(1) > r1)

#     # fattore di scala = r1/r solo per fuori, 1 altrimenti
#     scale = torch.ones_like(r)
#     scale[outside] = r1 / r[outside]

#     proj = center + rel * scale

#     dist = torch.zeros_like(r.squeeze(1))
#     dist[outside] = r.squeeze(1)[outside] - r1

#     return proj, dist, outside

# def jacobian_spherical_projection(z, r1):
#     """
#     x_hat = r1 * z / ||z||
#     J = d x_hat / d z = (r1/||z||) * (I - u u^T), u = z/||z||
#     z: (N,3), r1: float
#     """
#     r = torch.norm(z, dim=1, keepdim=True).clamp_min(1e-12)           # (N,1)
#     u = z / r                                                         # (N,3)
#     I = torch.eye(3, device=z.device, dtype=z.dtype).unsqueeze(0).expand(z.size(0), 3, 3)
#     J = (r1 / r).unsqueeze(-1) * (I - u.unsqueeze(2) @ u.unsqueeze(1)) # (N,3,3)
#     return J, r, u


# def correct_sphere_gradient(points_scaled:torch.Tensor, g_in:torch.Tensor, outside:torch.Tensor, r1:torch.Tensor):
#     """
#     g = P_tan (g_in) + grad_distanza_proiezione (solo fuori)
#     con P_tan = I - (points_scaled points_scaled^T) / ||points_scaled||^2
#     """
#     device, dtype = points_scaled.device, points_scaled.dtype
#     N = points_scaled.shape[0]

#     # Jacobiani
#     I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N,3,3)
#     J_out, _, _ = jacobian_spherical_projection(points_scaled, r1)  # definita come prima

#     # seleziona J = I per inside, J = J_out per outside
#     J = torch.where(outside.view(-1,1,1), J_out, I)

#     g_bernstein_z = torch.bmm(J.transpose(1,2), g_in.unsqueeze(-1)).squeeze(-1)

#     # grad della distanza sferica solo per fuori
#     g_sphere_z = torch.zeros_like(points_scaled)
#     g_sphere_z[outside] = (points_scaled / torch.norm(points_scaled, dim=1, keepdim=True))[outside]

#     return g_bernstein_z + g_sphere_z

def eberly_batch_newton(points:torch.Tensor, center:torch.Tensor, axes:torch.Tensor, D:torch.Tensor, max_iter=20, tol=1e-8, device='cpu'):

    q = (points - center) @ axes
    D2 = D ** 2
    D2_expand = D2[None, :]

    def F(lmbd):
        lmbd = lmbd[:, None]
        denom = D2_expand + lmbd
        return torch.sum((D * q / denom) ** 2, dim=1) - 1

    def dF(lmbd):
        lmbd = lmbd[:, None]
        denom = D2_expand + lmbd
        term = (D * q / denom) ** 2
        return -2 * torch.sum(term / denom, dim=1)

    lmbd = torch.full((points.shape[0],), 1e-6, dtype=points.dtype, device=device)

    for _ in range(max_iter):
        f_val = F(lmbd)
        df_val = dF(lmbd)
        delta = f_val / df_val
        lmbd = lmbd - delta
        if torch.max(torch.abs(f_val)) < tol:
            # print(f"\033[92mConverged after {_+1} iterations\033[0m")
            break
        # print(f"\033[93mIteration {_+1}: max |f| = {torch.max(torch.abs(f_val)):.6f}\033[0m")
    

    lmbd_final = lmbd[:, None]
    denom = D2_expand + lmbd_final
    proj_local = (D2_expand * q) / denom
    proj_global = proj_local @ axes.T + center
    return proj_global


def eberly_batch_hybrid(points:torch.Tensor, center:torch.Tensor, axes:torch.Tensor, D:torch.Tensor,
                        max_iter_bisect=2, max_iter_newton=100, tol=1e-8, device='cpu'):
    eps = 1e-12
    q = (points - center) @ axes
    D2 = D ** 2
    D2_expand = D2[None, :]

    def F(lmbd):
        lmbd = lmbd[:, None]
        denom = D2_expand + lmbd
        return torch.sum((D * q / denom) ** 2, dim=1) - 1

    def dF(lmbd):
        lmbd = lmbd[:, None]
        denom = D2_expand + lmbd
        term = (D * q / denom) ** 2
        return -2 * torch.sum(term / denom, dim=1)

    # --- Step 1: Bisezione (max_iter_bisect passi) ---
    l_low = torch.zeros((points.shape[0],), dtype=points.dtype, device=device)
    l_high = torch.full_like(l_low, 1e6)
    lmbd = torch.zeros_like(l_low)
    # lmbd = torch.full((points.shape[0],), 1e-6, dtype=points.dtype, device=device)
    # lmbd = initial_lambda_estimate(q, D2)

    for _ in range(max_iter_bisect):
        l_mid = (l_low + l_high) / 2
        f_mid = F(l_mid)
        mask = f_mid > 0
        l_low = torch.where(mask, l_mid, l_low)
        l_high = torch.where(~mask, l_mid, l_high)
        lmbd = l_mid  # aggiorna lambda alla media

    # --- Step 2: Newton-Raphson ---
    for k in range(max_iter_newton):
        f_val = F(lmbd)
        df_val = dF(lmbd)

        # Prevenzione divisione per zero
        df_val = torch.where(torch.abs(df_val) < eps, torch.full_like(df_val, eps), df_val)

        delta = f_val / df_val

        # Update e clamp
        lmbd = lmbd - delta
        lmbd = torch.clamp(lmbd, min=1e-8, max=1e4)  # evita lambda troppo grandi

        if torch.max(torch.abs(f_val)) < tol:
            break

    lmbd_final = lmbd[:, None]
    denom = D2_expand + lmbd_final
    proj_local = (D2_expand * q) / denom
    proj_global = proj_local @ axes.T + center

    return proj_global

def ellipsoid_projection(points: torch.Tensor,
                                        center: torch.Tensor,
                                        axes: torch.Tensor,
                                        D: torch.Tensor,
                                        r: float = 1.0,
                                        device='cpu'):
    """
    Proietta solo i punti con ||p|| > 1 su un ellissoide, gli altri restano invariati.
    Args:
        - points: tensore di punti 3D (N, 3)
        - center: centro dell'ellissoide (3,)
        - axes: eigen vettori degli assi dell'ellissoide (3, 3)
        - D: semiasse dell'ellissoide (3,)
    Restituisce:
      - new_points: tensore con alcuni punti modificati
      - distances: distanza euclidea tra punto e proiezione (0 per i punti interni)
    """
    # Norme euclidee dei punti
    norms = torch.linalg.norm(points, dim=1)

    # Maschera dei punti fuori dalla sfera di raggio 1
    mask_outside = norms > r
    # print(f"\033[96m[DEBUG] Punti fuori dalla sfera di raggio 1: {mask_outside.sum().item()} / {points.shape[0]}\033[0m")

    # Inizializza new_points come copia
    new_points = points.clone()
    distances = torch.zeros_like(norms)

    if mask_outside.any():
        print(f"\033[96m[DEBUG] Proiezione dei punti fuori dalla sfera su ellissoide...\033[0m")
        points_out = points[mask_outside]
        proj = eberly_batch_newton(points_out, center, axes, D, device=device, max_iter=100, tol=1e-5)
        new_points[mask_outside] = proj
        distances[mask_outside] = torch.norm(points_out - proj, dim=1)


        mask_outside = mask_outside.to(torch.bool)  # Assicurati che sia un tensore booleano per la visualizzazione
    
        # plot_projections(points.cpu(), new_points.cpu(), mask_outside.cpu(),
        #                  ellipsoid_center=center.cpu(),
        #                  ellipsoid_axes=D.cpu(),
        #                  ellipsoid_eigenvectors=axes.cpu())

    return new_points, distances, mask_outside



@torch.no_grad()
def correct_ellipsoid_gradient(
    points_scaled: torch.Tensor,          # (N,3) punti prima della proiezione (scaled)
    g_in: torch.Tensor,                   # (N,3) grad wrt x_in
    outside: torch.Tensor,                # (N,) bool
    ellipsoid_center: torch.Tensor,       # (3,)
    ellipsoid_axes: torch.Tensor,         # (3,3) SAME convention as projection: q = (p-c) @ axes
    r1: Union[float, torch.Tensor],       # scalar (normalized-space)
    *,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    g = J^T g_in + grad(d_extra) (solo outside)

    Approx:
      pe_local = (p-c) @ axes
      r_eq = ||pe_local||
      P_tan = I - (pe_local pe_local^T)/r_eq^2
      J_e ≈ (r1 / r_eq) * P_tan
      J_world = axes * J_e * axes^T
    """
    assert points_scaled.ndim == 2 and points_scaled.shape[1] == 3
    assert g_in.shape == points_scaled.shape
    assert outside.ndim == 1 and outside.shape[0] == points_scaled.shape[0]
    assert ellipsoid_center.shape == (3,)
    assert ellipsoid_axes.shape == (3, 3)

    device, dtype = points_scaled.device, points_scaled.dtype
    c = ellipsoid_center.to(device=device, dtype=dtype)
    A = ellipsoid_axes.to(device=device, dtype=dtype)  # "axes" matrix

    # r1 as tensor
    r1_t = r1 if isinstance(r1, torch.Tensor) else torch.tensor(r1, device=device, dtype=dtype)

    N = points_scaled.shape[0]

    # pe in world
    pe = points_scaled - c  # (N,3)

    # local coords (coherent with Eberly q = (p-c) @ axes)
    pe_local = pe @ A  # (N,3)

    # radial norm in local frame
    r_eq = pe_local.norm(dim=1, keepdim=True).clamp_min(eps)  # (N,1)

    # tangent projection in local frame
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N, 3, 3)
    outer = torch.einsum("ni,nj->nij", pe_local, pe_local)  # (N,3,3)
    P_tan = I - outer / (r_eq ** 2).unsqueeze(-1)           # (N,3,3)

    # approximate Jacobian in local frame
    J_e = (r1_t / r_eq).unsqueeze(-1) * P_tan               # (N,3,3)

    # back to world
    Ab = A.unsqueeze(0).expand(N, 3, 3)
    J_world = torch.bmm(torch.bmm(Ab, J_e), Ab.transpose(1, 2))  # (N,3,3)

    # transport gradient
    g_bern = torch.bmm(J_world.transpose(1, 2), g_in.unsqueeze(-1)).squeeze(-1)  # (N,3)

    # grad of extra distance (only outside)
    unit = pe / pe.norm(dim=1, keepdim=True).clamp_min(eps)
    g_dist = torch.zeros_like(points_scaled)
    g_dist[outside] = unit[outside]

    return g_bern + g_dist









