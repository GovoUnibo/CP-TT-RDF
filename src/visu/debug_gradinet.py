# ===== Stand-alone plotting utilities for SDF slices & gradient correction =====
import os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal

from src.core.math.projection_point import spherical_projection, _jacobian_spherical_projection


# --- piccolo helper per estrarre il "modello" da SDF e tipi/device coerenti
def _extract_model_params(sdf_obj, link_name: str):
    device, dtype = sdf_obj.device, sdf_obj.dtype
    model = getattr(sdf_obj, link_name + sdf_obj.model_extension)

    w = model.weights.to(device=device, dtype=dtype)                  # (M,)
    s = torch.as_tensor(model.scale_factor,   device=device, dtype=dtype)  # scala (float/tensor)
    c = torch.as_tensor(model.centroid_offset,device=device, dtype=dtype)  # offset (3,)
    dom_min = float(model.domain_min)
    dom_max = float(model.domain_max)

    return model, w, s, c, dom_min, dom_max, device, dtype


def _grads_at_points_z(
    sdf_obj,
    pts_z: torch.Tensor,
    r1: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcola gradiente Bernstein in z-space (dopo la mappa), gradiente sferico e totale,
    e restituisce anche i punti proiettati Xin (nello spazio interno).
    """
    device, dtype = sdf_obj.device, sdf_obj.dtype
    Xin, _, outside_p = spherical_projection(pts_z, r1=r1)

    Phi, Dphi = sdf_obj.basis_function_from_3Dpoints(Xin, use_derivative=True)
    # pesi sono già impostati dentro sdf_obj? per robustezza leggiamoli dal modello via caller.
    # Qui assumiamo che chi chiama abbia già impostato i pesi nel sdf_obj (true nel nostro uso).
    # In alternativa, potremmo passarli come argomento.
    # Per minimizzare side-effects, leggiamo i pesi dal modello solo dove serve.
    # Ma qui calcoliamo ∑ Dphi_m * w_m: chiediamo a chi invoca di aver fatto set_*? Non necessario,
    # usiamo direttamente torch.einsum con i pesi forniti nel contesto chiamante.
    # => Lasciamo la responsabilità al wrapper che prepara 'w' e fa la einsum qui sotto.

    # NOTA: eseguiamo qui la einsum con i pesi attuali dentro sdf_obj
    # ma per rimanere puri, torniamo anche Xin e outside_p per costruire J.
    return Xin, Dphi, outside_p


def plot_sdf_slice_2d_external(
    sdf_obj: "SDF",
    link_name: str,
    x_range = (-0.2, 1.2),
    y_range = (-0.2, 1.2),
    z0: float = 0.0,
    res: int = 300,
    cmap: str = "coolwarm",
    show_zero_level: bool = True,
    show_training_sphere: bool = True,
    points_world: Optional[torch.Tensor] = None,   # (N,3)
    arrow_scale_pts: float = 0.22,
    view_scaled: bool = True,
):
    """
    Versione stand-alone del tuo plot slice 2D:
    - legge i parametri dal modello interno dell'istanza SDF
    - visualizza la SDF su una sezione piana a z=z0 (nel dominio scelto)
    - opzionale: mostra frecce ∇B (arancione), ∇S (verde), ∇T (viola) su punti forniti
    """
    model, w, s, c, dom_min, dom_max, device, dtype = _extract_model_params(sdf_obj, link_name)
    # raggio unitario nello spazio normalizzato; altrimenti metà del dominio
    r1 = 1.0 if view_scaled else (dom_max - dom_min) / 2.0

    # --- griglia nello z-space (spazio scalato/traslato)
    if view_scaled:
        xz_min, xz_max = (x_range[0] - float(c[0]))/float(s), (x_range[1] - float(c[0]))/float(s)
        yz_min, yz_max = (y_range[0] - float(c[1]))/float(s), (y_range[1] - float(c[1]))/float(s)
        z0_hat = (z0 - float(c[2]))/float(s)

        xs = np.linspace(xz_min, xz_max, res)
        ys = np.linspace(yz_min, yz_max, res)
        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, z0_hat)
        Pz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        pts_z = torch.tensor(Pz, device=device, dtype=dtype)

        # proiezione e valutazione SDF
        Xin, d_sphere, _ = spherical_projection(pts_z, r1=r1)
        phi, _ = sdf_obj.basis_function_from_3Dpoints(Xin, use_derivative=False)
        sdf_vals = torch.einsum('nm,m->n', phi, w) + d_sphere
        SDF = sdf_vals.detach().cpu().numpy().reshape(res, res)

        # --- plot mappa
        fig, ax = plt.subplots(figsize=(7.8, 7.0))
        im = ax.contourf(X, Y, SDF, levels=50, cmap=cmap, alpha=0.95)
        cb = plt.colorbar(im, ax=ax); cb.set_label("Signed distance (z-space)")

        if show_zero_level:
            ax.contour(X, Y, SDF, levels=[0.0], colors="k", linewidths=1.6)

        if show_training_sphere:
            r_circ_sq = 1.0 - (z0_hat)**2
            if r_circ_sq > 0:
                r_circ = math.sqrt(r_circ_sq)
                theta = np.linspace(0, 2*np.pi, 400)
                ax.plot(r_circ*np.cos(theta), r_circ*np.sin(theta),
                        "b--", linewidth=1.2, label="unit sphere slice")

        # --- frecce/grad
        def grads_at(points_z_batch: torch.Tensor):
            Xin, _, outside_p = spherical_projection(points_z_batch, r1=r1)
            Phi, Dphi = sdf_obj.basis_function_from_3Dpoints(Xin, use_derivative=True)
            Gin = torch.einsum('nmk,m->nk', Dphi, w)   # grad nello spazio interno

            I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(points_z_batch.size(0),3,3)
            Jout, _, _ = _jacobian_spherical_projection(points_z_batch, r1)
            Jp = torch.where(outside_p.view(-1,1,1), Jout, I)
            gB = torch.bmm(Jp.transpose(1,2), Gin.unsqueeze(-1)).squeeze(-1)  # ∇B in z-space

            gS = torch.zeros_like(points_z_batch)
            mask = outside_p.bool()
            if mask.any():
                Zm = points_z_batch[mask]
                gS[mask] = Zm / (torch.norm(Zm, dim=1, keepdim=True) + 1e-12)    # ∇S
            gT = gB + gS
            return gB, gS, gT, Xin

        if points_world is not None:
            pts = torch.as_tensor(points_world, device=device, dtype=dtype).clone()
            if pts.ndim == 1:
                pts = pts.reshape(1,3)
            pts_z = (pts - c) / s
            pts_z[:, 2] = (z0 - float(c[2]))/float(s)

            gB, gS, gT, Xproj = grads_at(pts_z)
            gB2, gS2, gT2, _  = grads_at(Xproj)

            pts_np, xproj_np = pts_z.detach().cpu().numpy(), Xproj.detach().cpu().numpy()

            def arrow2d(p, v, color, label=None, alpha=1.0):
                v2 = np.array([float(v[0]), float(v[1])])
                n  = np.linalg.norm(v2)
                if n < 1e-12: return None
                v2 /= n
                return ax.quiver(p[0], p[1], v2[0], v2[1],
                                 angles='xy', scale_units='xy', scale=1/arrow_scale_pts,
                                 color=color, alpha=alpha, label=label, zorder=7)

            for i in range(len(pts_np)):
                p, pp = pts_np[i], xproj_np[i]
                ax.scatter(p[0],  p[1],  s=55, c='red',  edgecolors='k', zorder=8)
                ax.scatter(pp[0], pp[1], s=45, c='blue', edgecolors='k', zorder=8)
                ax.plot([p[0], pp[0]], [p[1], pp[1]], 'b--', alpha=0.8, zorder=6)

                arrow2d(p,  gB[i], "tab:orange", r"$\nabla_B$")
                arrow2d(p,  gS[i], "tab:green",  r"$\nabla_S$")
                arrow2d(p,  gT[i], "tab:purple", r"$\nabla_T$")
                arrow2d(pp, gB2[i], "tab:orange")
                arrow2d(pp, gS2[i], "tab:green")
                arrow2d(pp, gT2[i], "tab:purple")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xz_min, xz_max); ax.set_ylim(yz_min, yz_max)
        ax.set_xlabel("x̂ (z-space)"); ax.set_ylabel("ŷ (z-space)")
        ax.set_title(f"SDF slice 2D (z-space) @ ẑ={z0_hat:.3f} — {link_name}")
        # legenda compatta (etichette uniche)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")
        plt.tight_layout(); plt.show()

def plot_transport_and_correction_gif(
    sdf_obj: "SDF",
    link_name: str,
    points_world: torch.Tensor,
    z0: float = 0.0,
    x_range = (-0.2, 0.7),
    y_range = (-0.2, 0.7),
    res_bg: int = 200,
    n_frames: int = 40,                 # metà trasporto, metà correzione
    out_dir: str = "transport_and_correction_frames",
    make_gif: bool = True,
    gif_name: str = "transport_and_correction.gif",
    view_scaled: bool = True,
):
    """
    GIF didattica in due fasi:
      Fase A (0 -> 0.5): trasporto del gradiente arancione da \\hat p a p
                         G_in(\\hat p) --> \\nabla_B(p) = J^T(z) G_in
      Fase B (0.5 -> 1): correzione al punto p
                         mostra grad verde \\nabla_S(p) che cresce (alpha), e la viola \\nabla_T(p) = \\nabla_B + \\nabla_S(alpha)

    Nota: per compatibilità con mathtext, uso \\mathrm invece di \\text.
    """
    import os, numpy as np, torch, matplotlib.pyplot as plt
    from src.core.math.projection_point import spherical_projection, _jacobian_spherical_projection
    os.makedirs(out_dir, exist_ok=True)

    # --- parametri modello
    model, w, s, c, dom_min, dom_max, device, dtype = _extract_model_params(sdf_obj, link_name)
    r1 = 1.0 if view_scaled else (dom_max - dom_min) / 2.0

    # --- slice di sfondo
    if not view_scaled:
        raise NotImplementedError("Usare view_scaled=True per questa GIF.")
    xz_min, xz_max = (x_range[0] - float(c[0]))/float(s), (x_range[1] - float(c[0]))/float(s)
    yz_min, yz_max = (y_range[0] - float(c[1]))/float(s), (y_range[1] - float(c[1]))/float(s)
    z0_hat = (z0 - float(c[2]))/float(s)

    xs = np.linspace(xz_min, xz_max, res_bg)
    ys = np.linspace(yz_min, yz_max, res_bg)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, z0_hat)
    Pz_bg = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    pts_bg = torch.tensor(Pz_bg, device=device, dtype=dtype)
    Xin_bg, d_sphere_bg, _ = spherical_projection(pts_bg, r1=r1)
    phi_bg, _ = sdf_obj.basis_function_from_3Dpoints(Xin_bg, use_derivative=False)
    sdf_bg = torch.einsum('nm,m->n', phi_bg, w) + d_sphere_bg
    SDF_BG = sdf_bg.detach().cpu().numpy().reshape(res_bg, res_bg)

    # --- punti
    pts = torch.as_tensor(points_world, device=device, dtype=dtype).clone()
    if pts.ndim == 1: pts = pts.reshape(1,3)
    pts_z = (pts - c) / s
    pts_z[:, 2] = z0_hat

    # --- gradienti a \hat p e a p
    def grads_proj_and_map(points_z_batch: torch.Tensor):
        Xin, _, outside = spherical_projection(points_z_batch, r1=r1)              # \hat p
        Phi_in, Dphi_in = sdf_obj.basis_function_from_3Dpoints(Xin, use_derivative=True)
        Gin = torch.einsum('nmk,m->nk', Dphi_in, w)                                 # G_in(\hat p)
        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(points_z_batch.size(0),3,3)
        Jout, _, _ = _jacobian_spherical_projection(points_z_batch, r1)
        Jp = torch.where(outside.view(-1,1,1), Jout, I)
        gB_at_p = torch.bmm(Jp.transpose(1,2), Gin.unsqueeze(-1)).squeeze(-1)      # ∇B(p)
        # componente sferica al punto p
        gS_at_p = torch.zeros_like(points_z_batch)
        mask = outside.bool()
        if mask.any():
            Zm = points_z_batch[mask]
            gS_at_p[mask] = Zm / (torch.norm(Zm, dim=1, keepdim=True) + 1e-12)     # ∇S(p)
        gT_at_p = gB_at_p + gS_at_p                                                  # ∇T(p)
        return Xin, Gin, gB_at_p, gS_at_p, gT_at_p

    Xin_p, Gin_hat, gB_p, gS_p, gT_p = grads_proj_and_map(pts_z)

    pts_np   = pts_z.detach().cpu().numpy()       # p (rosso)
    xproj_np = Xin_p.detach().cpu().numpy()       # \hat p (blu)
    Gin_np   = Gin_hat.detach().cpu().numpy()     # direzione in \hat p
    gB_np    = gB_p.detach().cpu().numpy()        # arancione a p
    gS_np    = gS_p.detach().cpu().numpy()        # verde a p
    gT_np    = gT_p.detach().cpu().numpy()        # viola a p

    # --- helper: freccia con lunghezza fissa per visibilità
    def draw_arrow(ax, p, v, color, label=None, arrow_len=0.45, zorder=12):
        vx, vy = float(v[0]), float(v[1])
        n = (vx*vx + vy*vy) ** 0.5
        if n < 1e-12: return
        ux, uy = vx/n, vy/n
        dx, dy = ux*arrow_len, uy*arrow_len
        ax.quiver(
            p[0], p[1], dx, dy,
            angles='xy', scale_units='xy', scale=1.0,
            color=color, label=label, zorder=zorder,
            width=0.006, headwidth=5.0, headlength=6.0, headaxislength=4.5, pivot='tail'
        )

    # timeline: metà trasporto (fase A), metà correzione (fase B)
    frames_A = n_frames // 2
    frames_B = n_frames - frames_A
    alphas_A = np.linspace(0.0, 1.0, frames_A)        # posizione/direzione: \hat p -> p, Gin -> ∇B
    alphas_B = np.linspace(0.0, 1.0, frames_B)        # correzione: aggiungo ∇S e mostro ∇T

    for k in range(n_frames):
        fig, ax = plt.subplots(figsize=(7.4, 6.6))
        im = ax.contourf(X, Y, SDF_BG, levels=40, cmap="coolwarm", alpha=0.90)
        cb = plt.colorbar(im, ax=ax); cb.set_label("Signed distance (z-space)")
        ax.contour(X, Y, SDF_BG, levels=[0.0], colors="k", linewidths=1.2)

        phase_A = (k < frames_A)
        if phase_A:
            alpha = alphas_A[k]
        else:
            beta  = alphas_B[k - frames_A]   # 0 -> 1

        for i in range(len(pts_np)):
            p   = pts_np[i]
            ph  = xproj_np[i]
            v0  = Gin_np[i]   # direzione a \hat p
            v1  = gB_np[i]    # arancione a p (mappato)

            # collegamento p -- \hat p
            ax.plot([p[0], ph[0]], [p[1], ph[1]], 'b--', alpha=0.8, zorder=7)

            # marker punti
            ax.scatter(p[0],  p[1],  s=60, c='red',  edgecolors='k', zorder=10, label='p' if k==0 and i==0 else None)
            ax.scatter(ph[0], ph[1], s=50, c='blue', edgecolors='k', zorder=10, label=r'$\hat p$' if k==0 and i==0 else None)

            if phase_A:
                # FASE A: trasporto arancione dal punto proiettato al punto esterno
                pos = (1-alpha) * ph + alpha * p
                v_interp = (1-alpha) * v0 + alpha * v1
                lbl = r"$G_{\mathrm{in}}$" if (k==0 and i==0) else None
                if k == frames_A-1 and i==0:
                    lbl = r"$\nabla_B$"
                draw_arrow(ax, pos, v_interp, "tab:orange", label=lbl, arrow_len=0.45)
            else:
                # FASE B: mostra correzione al punto p
                # arancione: fisso a p
                draw_arrow(ax, p, v1, "tab:orange", label=r"$\nabla_B$" if (k==frames_A and i==0) else None, arrow_len=0.45)
                # verde: cresce con beta (solo direzione, lunghezza fissa ma alpha di disegno aumenta)
                draw_arrow(ax, p, gS_np[i], "tab:green",  label=r"$\nabla_S$" if (k==frames_A and i==0) else None, arrow_len=0.45)
                # per rendere la crescita percettiva, moduliamo l'alpha del disegno verde
                # (quiver non supporta alpha per singola freccia in modo facile dopo il draw;
                #  quindi la "crescita" concettuale la mostriamo tramite sovrapposizione del risultante)
                # risultante viola: vT(beta) = vB + beta * vS
                vT_beta = v1 + beta * gS_np[i]
                lblT = r"$\nabla_T$" if (k==frames_A and i==0) else None
                draw_arrow(ax, p, vT_beta, "tab:purple", label=lblT, arrow_len=0.48)

        # titoli per le due fasi
        if phase_A:
            ax.set_title(f"Trasporto: $G_{{\\mathrm{{in}}}}@\\hat p \\to \\nabla_B@p$   (α={alpha:.2f}, ẑ={z0_hat:.3f})")
        else:
            ax.set_title(f"Correzione a p: $\\nabla_T=\\nabla_B+\\beta\\,\\nabla_S$   (β={beta:.2f}, ẑ={z0_hat:.3f})")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xz_min, xz_max); ax.set_ylim(yz_min, yz_max)
        ax.set_xlabel("x̂ (z-space)"); ax.set_ylabel("ŷ (z-space)")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        plt.tight_layout()
        frame_path = os.path.join(out_dir, f"frame_{k:03d}.png")
        plt.savefig(frame_path, dpi=140)
        plt.close(fig)

    # --- GIF
    gif_path = None
    if make_gif:
        try:
            import imageio.v2 as imageio
            frames = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(".png")]
            if frames:
                gif_path = os.path.join(out_dir, gif_name)
                imgs = [imageio.imread(fp) for fp in frames]
                imageio.mimsave(gif_path, imgs, fps=10)
        except Exception as e:
            print("[WARN] Impossibile creare la GIF:", e)

    return out_dir, gif_path
