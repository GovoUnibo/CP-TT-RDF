from src.core.math.Bernstain_P import BersteinPoly
import torch
import numpy as np
from typing import Optional, Tuple, Sequence
import tensorly as tl
import torch.nn as nn

class BernsteinTTrain(BersteinPoly):
    def __init__(self, n_func: int = 16, domain_min: float = -1, domain_max: float = 1,
                 device: str = 'cpu', dtype: torch.dtype = torch.float32):
        super().__init__(n_func, domain_min, domain_max, device, dtype)

    @staticmethod
    def _solve_ridge_system(ata: torch.Tensor, atb: torch.Tensor, ridge: float):
        n = ata.shape[0]
        ata_reg = ata + ridge * torch.eye(n, device=ata.device, dtype=ata.dtype)
        try:
            return torch.linalg.solve(ata_reg, atb)
        except RuntimeError:
            sol = torch.linalg.lstsq(ata_reg, atb.unsqueeze(-1)).solution.squeeze(-1)
            return sol

    def _solve_from_chunked_design(self, y: torch.Tensor, w: torch.Tensor, out_dim: int, ridge: float, build_design, chunk: int = 1024):
        ata = torch.zeros((out_dim, out_dim), device=y.device, dtype=y.dtype)
        atb = torch.zeros((out_dim,), device=y.device, dtype=y.dtype)
        n = y.numel()
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            d = build_design(start, end)                        # (B, out_dim)
            wc = w[start:end]
            dw = d * wc.unsqueeze(1)
            ata += dw.T @ dw
            atb += dw.T @ (wc * y[start:end])
        return self._solve_ridge_system(ata, atb, ridge=ridge)

    def _vector_ls_update(
        self,
        phi: torch.Tensor,
        coeff: torch.Tensor,
        target: torch.Tensor,
        ridge: float,
        rhs_weight: Optional[torch.Tensor] = None,
    ):
        if coeff.abs().max() < 1e-12:
            return None
        d = phi * coeff.unsqueeze(1)
        rhs = target if rhs_weight is None else (rhs_weight * target)
        ata = d.T @ d
        atb = d.T @ rhs
        return self._solve_ridge_system(ata, atb, ridge=ridge)

    @staticmethod
    def _choose_svd_rank(s: torch.Tensor, rank_cap: int, svd_rtol: float = 1e-4) -> int:
        if s.numel() == 0:
            return 1
        s0 = s[0].abs().clamp_min(1e-12)
        if svd_rtol > 0.0:
            keep = int((s / s0 >= svd_rtol).sum().item())
            keep = max(1, keep)
        else:
            keep = int(s.numel())
        keep = min(keep, int(s.numel()))
        keep = min(keep, int(rank_cap))
        return max(1, keep)

    def _split_two_site(
        self,
        theta: torch.Tensor,
        r_left: int,
        n_left: int,
        n_right: int,
        r_right: int,
        rank_cap: int,
        svd_rtol: float = 1e-4,
    ):
        mat = theta.reshape(r_left * n_left, n_right * r_right)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        r_new = self._choose_svd_rank(s, rank_cap=rank_cap, svd_rtol=svd_rtol)
        u = u[:, :r_new]
        s = s[:r_new]
        vh = vh[:r_new, :]
        left = u.reshape(r_left, n_left, r_new)
        right = (s.unsqueeze(1) * vh).reshape(r_new, n_right, r_right)
        return left, right

    def _fit_theta_local(
        self,
        theta_init: torch.Tensor,
        predict_fn,
        y_b: torch.Tensor,
        sw_b: torch.Tensor,
        ridge: float,
        lr: float,
        local_steps: int = 2,
    ):
        theta = nn.Parameter(theta_init.clone())
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(max(1, int(local_steps))):
            y_hat = predict_fn(theta)
            resid = sw_b * (y_hat - y_b)
            loss = (resid @ resid) / float(y_b.numel()) + ridge * theta.square().mean()
            if not torch.isfinite(loss):
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], max_norm=10.0)
            opt.step()
            with torch.no_grad():
                theta.data.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
        return theta.detach()

    def train_tt(self,
                 points_near: torch.Tensor, sdf_near: torch.Tensor,
                 points_rand: torch.Tensor, sdf_rand: torch.Tensor,
                 tt_ranks: Tuple[int, int] = (8, 8),
                 iters: int = 200, lr: float = 5e-3,
                 ridge: float = 1e-6, batch_size: int = 65_536,
                 weights: Optional[torch.Tensor] = None):
        """
        Training TT su punti (NO tensore denso).
        Approssima W(i,j,k) ≈ TT(G1,G2,G3) con:
          G1 ∈ R^{1  x N x r1}
          G2 ∈ R^{r1 x N x r2}
          G3 ∈ R^{r2 x N x 1}
        """
        device, dtype = self.device, self.dtype
        r1, r2 = tt_ranks

        # --- prepara dataset come nel CP ---
        points_near = torch.as_tensor(points_near, device=device, dtype=dtype).reshape(-1, 3)
        sdf_near    = torch.as_tensor(sdf_near,    device=device, dtype=dtype).reshape(-1)
        points_rand = torch.as_tensor(points_rand, device=device, dtype=dtype).reshape(-1, 3)
        sdf_rand    = torch.as_tensor(sdf_rand,    device=device, dtype=dtype).reshape(-1)

        pts = torch.cat([points_near, points_rand], dim=0)
        y   = torch.cat([sdf_near, sdf_rand], dim=0)
        assert pts.shape[0] == y.numel(), "points e sdf devono avere lo stesso P."

        # --- normalizza in [0,1] ---
        t = super().normalize_points(pts).clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        y = y.reshape(-1).to(device=device, dtype=dtype)

        P = y.numel()
        Nx = Ny = Nz = self.n_func

        if weights is None:
            sw = torch.ones(P, device=device, dtype=dtype)
        else:
            sw = torch.as_tensor(weights, device=device, dtype=dtype).reshape(-1)
            sw = torch.sqrt(sw.clamp_min(0) + torch.finfo(dtype).eps)

        # ================== PARAMETRI TT ==================
        # G1: (1, N, r1)
        # G2: (r1, N, r2)
        # G3: (r2, N, 1)
        G1 = nn.Parameter(1e-2 * torch.randn(1,  Nx, r1, device=device, dtype=dtype))
        G2 = nn.Parameter(1e-2 * torch.randn(r1, Nx, r2, device=device, dtype=dtype))
        G3 = nn.Parameter(1e-2 * torch.randn(r2, Nx, 1, device=device, dtype=dtype))

        opt = torch.optim.Adam([G1, G2, G3], lr=lr)

        bs = int(batch_size) if (batch_size is not None and batch_size > 0) else P
        log_every = max(1, iters // 20)
        for it in range(iters):
            perm = torch.randperm(P, device=device)
            epoch_loss = 0.0
            nbatches = 0

            for start in range(0, P, bs):
                end = min(start + bs, P)
                idx = perm[start:end]

                Phi_x, _ = self.build_bernstein_t(tx[idx], use_derivative=False)  # (B,N)
                Phi_y, _ = self.build_bernstein_t(ty[idx], use_derivative=False)
                Phi_z, _ = self.build_bernstein_t(tz[idx], use_derivative=False)

                # U1: (B, r1) = Σ_i Phi_x[b,i] * G1[0,i,a]
                U1 = torch.einsum('bn,nr->br', Phi_x, G1[0])
                # U2: (B, r2) = Σ_{a,i} U1[b,a] * Phi_y[b,i] * G2[a,i,r]
                U2 = torch.einsum('ba,bn,anr->br', U1, Phi_y, G2)

                # yhat: (B,) = Σ_{r,i} U2[b,r] * Phi_z[b,i] * G3[r,i,0]
                G3_red = G3[..., 0]
                yhat = torch.einsum('br,bn,rn->b', U2, Phi_z, G3_red)

                y_b = y[idx]
                sw_b = sw[idx]
                resid = sw_b * (yhat - y_b)
                data_loss = (resid @ resid) / float(y_b.numel())
                reg_loss  = G1.square().mean() + G2.square().mean() + G3.square().mean()
                loss = data_loss + ridge * reg_loss

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"TT loss non-finite (iter={it+1}). "
                        f"Prova con lr più basso, ranks/N più piccoli o ridge più alto."
                    )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([G1, G2, G3], max_norm=10.0)
                opt.step()

                with torch.no_grad():
                    G1.data.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                    G2.data.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                    G3.data.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)

                epoch_loss += float(loss.item())
                nbatches += 1

            epoch_loss /= max(nbatches, 1)
            if (it % log_every == 0) or (it == iters - 1):
                print(f"\033[96m TT iter {it+1}/{iters} | loss={epoch_loss:.6e} | bs={bs}\033[0m")

        # (opzionale) piccola rinormalizzazione per evitare drift di scala
        # with torch.no_grad():
        #     # normalizza Frobenius complessivo a 1 (o altro) e lascia tutto nei core
        #     norm = torch.sqrt(
        #         G1.square().sum() + G2.square().sum() + G3.square().sum()
        #     ).clamp_min(1e-12)
        #     G1.div_(norm); G2.div_(norm); G3.div_(norm)

        return G1.detach(), G2.detach(), G3.detach()

    def train_tt_als(self,
                     points_near: torch.Tensor, sdf_near: torch.Tensor,
                     points_rand: torch.Tensor, sdf_rand: torch.Tensor,
                     tt_ranks: Tuple[int, int] = (8, 8),
                     iters: int = 200, ridge: float = 1e-6,
                     batch_size: int = 65_536, weights: Optional[torch.Tensor] = None):
        """
        Alternating Least Squares per TT 3D (core-wise LS).
        Alternativa robusta ad Adam.
        """
        device, dtype = self.device, self.dtype
        r1, r2 = tt_ranks

        points_near = torch.as_tensor(points_near, device=device, dtype=dtype).reshape(-1, 3)
        sdf_near    = torch.as_tensor(sdf_near,    device=device, dtype=dtype).reshape(-1)
        points_rand = torch.as_tensor(points_rand, device=device, dtype=dtype).reshape(-1, 3)
        sdf_rand    = torch.as_tensor(sdf_rand,    device=device, dtype=dtype).reshape(-1)

        pts = torch.cat([points_near, points_rand], dim=0)
        y_all = torch.cat([sdf_near, sdf_rand], dim=0).reshape(-1)
        p_total = y_all.numel()

        t = super().normalize_points(pts).clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        if weights is None:
            sw_all = torch.ones(p_total, device=device, dtype=dtype)
        else:
            sw_all = torch.as_tensor(weights, device=device, dtype=dtype).reshape(-1)
            sw_all = torch.sqrt(sw_all.clamp_min(0) + torch.finfo(dtype).eps)

        n = self.n_func
        g1 = 1e-2 * torch.randn(1, n, r1, device=device, dtype=dtype)
        g2 = 1e-2 * torch.randn(r1, n, r2, device=device, dtype=dtype)
        g3 = 1e-2 * torch.randn(r2, n, 1, device=device, dtype=dtype)

        bs = int(batch_size) if (batch_size is not None and batch_size > 0) else p_total
        ls_chunk = min(1024, bs)
        log_every = max(1, iters // 20)

        with torch.no_grad():
            for it in range(iters):
                if bs < p_total:
                    idx = torch.randperm(p_total, device=device)[:bs]
                else:
                    idx = torch.arange(p_total, device=device)

                phi_x, _ = self.build_bernstein_t(tx[idx], use_derivative=False)  # (B,N)
                phi_y, _ = self.build_bernstein_t(ty[idx], use_derivative=False)
                phi_z, _ = self.build_bernstein_t(tz[idx], use_derivative=False)
                y_b = y_all[idx]
                sw_b = sw_all[idx]
                bsz = y_b.numel()

                # -------- update G1 --------
                g3_red = g3[..., 0]                                              # (r2,N)
                bz = torch.einsum('bn,rn->br', phi_z, g3_red)                    # (B,r2)
                t1 = torch.einsum('bn,anr,br->ba', phi_y, g2, bz)                # (B,r1)
                out1 = n * r1
                def build_d1(s, e):
                    return torch.einsum('bi,ba->bia', phi_x[s:e], t1[s:e]).reshape(e - s, out1)
                v1 = self._solve_from_chunked_design(y_b, sw_b, out1, ridge, build_d1, chunk=ls_chunk)
                g1 = v1.reshape(n, r1).unsqueeze(0)

                # -------- update G2 --------
                lx = torch.einsum('bn,nr->br', phi_x, g1[0])                     # (B,r1)
                g3_red = g3[..., 0]
                rz = torch.einsum('bn,rn->br', phi_z, g3_red)                    # (B,r2)
                out2 = r1 * n * r2
                def build_d2(s, e):
                    return torch.einsum('ba,bn,br->banr', lx[s:e], phi_y[s:e], rz[s:e]).reshape(e - s, out2)
                v2 = self._solve_from_chunked_design(y_b, sw_b, out2, ridge, build_d2, chunk=ls_chunk)
                g2 = v2.reshape(r1, n, r2)

                # -------- update G3 --------
                lx = torch.einsum('bn,nr->br', phi_x, g1[0])                     # (B,r1)
                t3 = torch.einsum('ba,bn,anr->br', lx, phi_y, g2)                # (B,r2)
                out3 = r2 * n
                def build_d3(s, e):
                    return torch.einsum('br,bn->brn', t3[s:e], phi_z[s:e]).reshape(e - s, out3)
                v3 = self._solve_from_chunked_design(y_b, sw_b, out3, ridge, build_d3, chunk=ls_chunk)
                g3 = v3.reshape(r2, n, 1)

                # -------- monitor --------
                u1 = torch.einsum('bn,nr->br', phi_x, g1[0])
                u2 = torch.einsum('ba,bn,anr->br', u1, phi_y, g2)
                y_hat = torch.einsum('br,bn,rn->b', u2, phi_z, g3[..., 0])
                resid = sw_b * (y_hat - y_b)
                loss = (resid @ resid) / float(bsz)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"TT-ALS loss non-finite (iter={it+1}). "
                        "Prova con batch_size più piccolo o ranks/N ridotti."
                    )
                if (it % log_every == 0) or (it == iters - 1):
                    print(f"\033[96m TT-ALS iter {it+1}/{iters} | loss={loss.item():.6e} | bs={bs}\033[0m")

        return g1.detach(), g2.detach(), g3.detach()

    def train_tt_mals(
        self,
        points_near: torch.Tensor,
        sdf_near: torch.Tensor,
        points_rand: torch.Tensor,
        sdf_rand: torch.Tensor,
        tt_ranks: Tuple[int, int] = (8, 8),
        iters: int = 200,
        lr: float = 5e-3,
        ridge: float = 1e-6,
        batch_size: int = 65_536,
        weights: Optional[torch.Tensor] = None,
        local_steps: int = 2,
        svd_rtol: float = 1e-4,
    ):
        """
        TT 3D two-site MALS/DMRG-like:
        - update locale su coppie di core (bond 0 e bond 1),
        - split via SVD con truncation (rank-adaptive entro i cap di tt_ranks).
        """
        work_dtype = torch.float64 if (self.n_func >= 128 and self.dtype == torch.float32) else self.dtype
        if work_dtype != self.dtype:
            print("\033[93m[TT-MALS] auto-switch a float64 per stabilità numerica (N>=128)\033[0m")

        device, dtype = self.device, work_dtype
        r1_cap, r2_cap = int(tt_ranks[0]), int(tt_ranks[1])

        points_near = torch.as_tensor(points_near, device=device, dtype=dtype).reshape(-1, 3)
        sdf_near = torch.as_tensor(sdf_near, device=device, dtype=dtype).reshape(-1)
        points_rand = torch.as_tensor(points_rand, device=device, dtype=dtype).reshape(-1, 3)
        sdf_rand = torch.as_tensor(sdf_rand, device=device, dtype=dtype).reshape(-1)

        pts = torch.cat([points_near, points_rand], dim=0)
        y_all = torch.cat([sdf_near, sdf_rand], dim=0).reshape(-1)
        p_total = y_all.numel()

        t = super().normalize_points(pts).clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        if weights is None:
            sw_all = torch.ones(p_total, device=device, dtype=dtype)
        else:
            sw_all = torch.as_tensor(weights, device=device, dtype=dtype).reshape(-1)
            sw_all = torch.sqrt(sw_all.clamp_min(0) + torch.finfo(dtype).eps)

        n = int(self.n_func)
        g1 = 1e-2 * torch.randn(1, n, r1_cap, device=device, dtype=dtype)
        g2 = 1e-2 * torch.randn(r1_cap, n, r2_cap, device=device, dtype=dtype)
        g3 = 1e-2 * torch.randn(r2_cap, n, 1, device=device, dtype=dtype)

        bs = int(batch_size) if (batch_size is not None and batch_size > 0) else p_total
        log_every = max(1, int(iters) // 20)

        for it in range(int(iters)):
            if bs < p_total:
                idx = torch.randperm(p_total, device=device)[:bs]
            else:
                idx = torch.arange(p_total, device=device)

            phi_x, _ = self.build_bernstein_t(tx[idx], use_derivative=False)
            phi_y, _ = self.build_bernstein_t(ty[idx], use_derivative=False)
            phi_z, _ = self.build_bernstein_t(tz[idx], use_derivative=False)
            y_b = y_all[idx]
            sw_b = sw_all[idx]

            # sweep: left->right e ritorno rapido
            for bond in (0, 1, 0):
                if bond == 0:
                    theta = torch.einsum('anr,rmq->anmq', g1, g2)  # (1,N,N,r2)
                    z_proj = torch.einsum('bn,rn->br', phi_z, g3[..., 0]).detach()

                    def predict_theta01(theta_param):
                        t01 = torch.einsum('bn,bm,anms->bs', phi_x, phi_y, theta_param)
                        return (t01 * z_proj).sum(dim=1)

                    theta_fit = self._fit_theta_local(
                        theta_init=theta,
                        predict_fn=predict_theta01,
                        y_b=y_b,
                        sw_b=sw_b,
                        ridge=ridge,
                        lr=lr,
                        local_steps=local_steps,
                    )
                    g1, g2 = self._split_two_site(
                        theta=theta_fit,
                        r_left=1,
                        n_left=n,
                        n_right=n,
                        r_right=int(g2.shape[2]),
                        rank_cap=r1_cap,
                        svd_rtol=svd_rtol,
                    )
                else:
                    theta = torch.einsum('anr,rmq->anmq', g2, g3)  # (r1,N,N,1)
                    u1 = torch.einsum('bn,nr->br', phi_x, g1[0]).detach()

                    def predict_theta12(theta_param):
                        return torch.einsum('br,bn,bm,rnm->b', u1, phi_y, phi_z, theta_param[..., 0])

                    theta_fit = self._fit_theta_local(
                        theta_init=theta,
                        predict_fn=predict_theta12,
                        y_b=y_b,
                        sw_b=sw_b,
                        ridge=ridge,
                        lr=lr,
                        local_steps=local_steps,
                    )
                    g2, g3 = self._split_two_site(
                        theta=theta_fit,
                        r_left=int(g2.shape[0]),
                        n_left=n,
                        n_right=n,
                        r_right=1,
                        rank_cap=r2_cap,
                        svd_rtol=svd_rtol,
                    )

                g1.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                g2.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                g3.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)

            u1 = torch.einsum('bn,nr->br', phi_x, g1[0])
            u2 = torch.einsum('ba,bn,anr->br', u1, phi_y, g2)
            y_hat = torch.einsum('br,bn,rn->b', u2, phi_z, g3[..., 0])
            resid = sw_b * (y_hat - y_b)
            loss = (resid @ resid) / float(y_b.numel())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"TT-MALS loss non-finite (iter={it+1}). "
                    "Prova con ranks più piccoli, lr più basso o batch_size più basso."
                )
            if (it % log_every == 0) or (it == int(iters) - 1):
                print(
                    f"\033[96m TT-MALS iter {it+1}/{iters} | loss={loss.item():.6e} | "
                    f"r=({g1.shape[2]},{g2.shape[2]}) | bs={bs}\033[0m"
                )

        return g1.detach(), g2.detach(), g3.detach()

    def train_tt_robot_als(
        self,
        points_near_list:  Sequence[torch.Tensor],
        sdf_near_list:     Sequence[torch.Tensor],
        points_rand_list:  Sequence[torch.Tensor],
        sdf_rand_list:     Sequence[torch.Tensor],
        tt_ranks: Tuple[int, ...] = (4, 8, 4),
        iters: int = 10,
        ridge: float = 1e-6,
        batch_size: int = 65_536,
        weights_near_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        weights_rand_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ):
        """
        TT 4D ALS (mini-batch): aggiornamento alternato core-wise per G0,G1,G2,G3.
        Alternativa robusta ad Adam per il robot-wide TT.
        """
        device, dtype = self.device, self.dtype
        l_num = len(points_near_list)
        assert len(sdf_near_list) == l_num
        assert len(points_rand_list) == l_num
        assert len(sdf_rand_list) == l_num
        if weights_near_list is not None:
            assert len(weights_near_list) == l_num
        if weights_rand_list is not None:
            assert len(weights_rand_list) == l_num

        pts_list, sdf_list, lid_list, w_list = [], [], [], []
        for ell in range(l_num):
            p_near = torch.as_tensor(points_near_list[ell], device=device, dtype=dtype).reshape(-1, 3)
            s_near = torch.as_tensor(sdf_near_list[ell],    device=device, dtype=dtype).reshape(-1)
            p_rand = torch.as_tensor(points_rand_list[ell], device=device, dtype=dtype).reshape(-1, 3)
            s_rand = torch.as_tensor(sdf_rand_list[ell],    device=device, dtype=dtype).reshape(-1)

            pts_link = torch.cat([p_near, p_rand], dim=0)
            sdf_link = torch.cat([s_near, s_rand], dim=0)
            n_link = pts_link.shape[0]

            pts_list.append(pts_link)
            sdf_list.append(sdf_link)
            lid_list.append(torch.full((n_link,), ell, device=device, dtype=torch.long))

            if (weights_near_list is not None and weights_near_list[ell] is not None) or \
               (weights_rand_list is not None and weights_rand_list[ell] is not None):
                w_near = weights_near_list[ell] if weights_near_list is not None else None
                w_rand = weights_rand_list[ell] if weights_rand_list is not None else None
                w_near_t = torch.ones(p_near.shape[0], device=device, dtype=dtype) if w_near is None else torch.as_tensor(w_near, device=device, dtype=dtype).reshape(-1)
                w_rand_t = torch.ones(p_rand.shape[0], device=device, dtype=dtype) if w_rand is None else torch.as_tensor(w_rand, device=device, dtype=dtype).reshape(-1)
                w_list.append(torch.cat([w_near_t, w_rand_t], dim=0))
            else:
                w_list.append(torch.ones(n_link, device=device, dtype=dtype))

        pts_all = torch.cat(pts_list, dim=0)
        sdf_all = torch.cat(sdf_list, dim=0)
        lid_all = torch.cat(lid_list, dim=0)
        w_all = torch.cat(w_list, dim=0)
        p_total = pts_all.shape[0]

        print(f"[TRAIN 4D TT ALS] L={l_num}, P_tot={p_total}, N={self.n_func}, tt_ranks={tt_ranks}")

        t = self.normalize_points(pts_all).clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        sw_all = torch.sqrt(w_all.clamp_min(0) + torch.finfo(dtype).eps)

        n = self.n_func
        if len(tt_ranks) == 3:
            r0, r1, r2 = tt_ranks
        elif len(tt_ranks) == 2:
            r0 = 1
            r1, r2 = tt_ranks
        else:
            raise ValueError(f"tt_ranks deve avere lunghezza 2 o 3, trovato {len(tt_ranks)}")

        g0 = 1e-2 * torch.randn(l_num, r0, device=device, dtype=dtype)
        g1 = 1e-2 * torch.randn(r0, n, r1, device=device, dtype=dtype)
        g2 = 1e-2 * torch.randn(r1, n, r2, device=device, dtype=dtype)
        g3 = 1e-2 * torch.randn(r2, n, device=device, dtype=dtype)

        bs = int(batch_size) if (batch_size is not None and batch_size > 0) else p_total
        log_every = max(1, iters // 20)

        with torch.no_grad():
            for it in range(iters):
                if bs < p_total:
                    idx = torch.randperm(p_total, device=device)[:bs]
                else:
                    idx = torch.arange(p_total, device=device)

                phi_x, _ = self.build_bernstein_t(tx[idx], use_derivative=False)
                phi_y, _ = self.build_bernstein_t(ty[idx], use_derivative=False)
                phi_z, _ = self.build_bernstein_t(tz[idx], use_derivative=False)
                y_b = sdf_all[idx]
                lid_b = lid_all[idx]
                sw_b = sw_all[idx]

                def _forward_current():
                    bx_ = torch.einsum('bn,anr->bar', phi_x, g1)   # (B,r0,r1)
                    by_ = torch.einsum('bn,anr->bar', phi_y, g2)   # (B,r1,r2)
                    bz_ = torch.einsum('bn,an->ba', phi_z, g3)     # (B,r2)
                    g0_b_ = g0[lid_b, :]                            # (B,r0)
                    t1_ = torch.einsum('ba,bar->br', g0_b_, bx_)    # (B,r1)
                    t2_ = torch.einsum('ba,bar->br', t1_, by_)      # (B,r2)
                    y_hat_ = (t2_ * bz_).sum(dim=1)
                    return y_hat_, bx_, by_, bz_, g0_b_, t1_, t2_

                y_hat, bx, by, bz, g0_b, t1, t2 = _forward_current()

                # ---- update G0 (per-link, vettore r0) ----
                for l_id in lid_b.unique():
                    mask = (lid_b == l_id)
                    bx_m = bx[mask]                                  # (M,r0,r1)
                    by_m = by[mask]                                  # (M,r1,r2)
                    bz_m = bz[mask]                                  # (M,r2)
                    tmp = torch.einsum('bar,brs->bas', bx_m, by_m)   # (M,r0,r2)
                    feat = torch.einsum('bas,bs->ba', tmp, bz_m)     # (M,r0)
                    old = (g0_b[mask] * feat).sum(dim=1)
                    residual = y_b[mask] - y_hat[mask] + old
                    w_m = sw_b[mask]

                    d = feat * w_m.unsqueeze(1)
                    ata = d.T @ d + ridge * torch.eye(r0, device=device, dtype=dtype)
                    atb = d.T @ (w_m * residual)
                    try:
                        g_new = torch.linalg.solve(ata, atb)
                    except RuntimeError:
                        g_new = torch.linalg.lstsq(ata, atb.unsqueeze(-1)).solution.squeeze(-1)

                    if torch.isfinite(g_new).all():
                        y_hat[mask] = y_hat[mask] - old + (feat @ g_new)
                        g0[l_id, :] = g_new
                        g0_b[mask, :] = g_new

                # ---- update G1 (r0 x r1 vettori da lunghezza N) ----
                y_hat, bx, by, bz, g0_b, t1, t2 = _forward_current()
                q = torch.einsum('bij,bj->bi', by, bz)              # (B,r1)
                for a in range(r0):
                    for i in range(r1):
                        pred = phi_x @ g1[a, :, i]
                        coef0 = g0_b[:, a] * q[:, i]
                        old = coef0 * pred
                        residual = y_b - y_hat + old
                        g_new = self._vector_ls_update(phi_x, coef0 * sw_b, residual, ridge, rhs_weight=sw_b)
                        if g_new is not None and torch.isfinite(g_new).all():
                            pred_new = phi_x @ g_new
                            y_hat = y_hat - old + coef0 * pred_new
                            g1[a, :, i] = g_new

                # ---- update G2 (r1 x r2 vettori da lunghezza N) ----
                y_hat, bx, by, bz, g0_b, t1, t2 = _forward_current()
                for i in range(r1):
                    for j in range(r2):
                        pred = phi_y @ g2[i, :, j]
                        coef0 = t1[:, i] * bz[:, j]
                        old = coef0 * pred
                        residual = y_b - y_hat + old
                        g_new = self._vector_ls_update(phi_y, coef0 * sw_b, residual, ridge, rhs_weight=sw_b)
                        if g_new is not None and torch.isfinite(g_new).all():
                            pred_new = phi_y @ g_new
                            y_hat = y_hat - old + coef0 * pred_new
                            g2[i, :, j] = g_new

                # ---- update G3 (r2 vettori da lunghezza N) ----
                y_hat, bx, by, bz, g0_b, t1, t2 = _forward_current()
                for j in range(r2):
                    pred = phi_z @ g3[j, :]
                    coef0 = t2[:, j]
                    old = coef0 * pred
                    residual = y_b - y_hat + old
                    g_new = self._vector_ls_update(phi_z, coef0 * sw_b, residual, ridge, rhs_weight=sw_b)
                    if g_new is not None and torch.isfinite(g_new).all():
                        pred_new = phi_z @ g_new
                        y_hat = y_hat - old + coef0 * pred_new
                        g3[j, :] = g_new

                g0.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                g1.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                g2.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                g3.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)

                y_hat, _, _, _, _, _, _ = _forward_current()
                resid = sw_b * (y_hat - y_b)
                loss = (resid @ resid) / float(y_b.numel())
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"TT-4D ALS loss non-finite (iter={it+1}). "
                        "Prova con ranks più piccoli, batch_size più basso o ridge più alto."
                    )
                if (it % log_every == 0) or (it == iters - 1):
                    print(f"\033[95m[TRAIN 4D TT ALS] iter {it+1}/{iters}, loss={loss.item():.6e} | bs={bs}\033[0m")

        return g0.unsqueeze(0).detach(), g1.detach(), g2.detach(), g3.unsqueeze(-1).detach()

    def train_tt_robot_mals(
        self,
        points_near_list: Sequence[torch.Tensor],
        sdf_near_list: Sequence[torch.Tensor],
        points_rand_list: Sequence[torch.Tensor],
        sdf_rand_list: Sequence[torch.Tensor],
        tt_ranks: Tuple[int, ...] = (4, 8, 4),
        iters: int = 10,
        lr: float = 5e-3,
        ridge: float = 1e-6,
        batch_size: int = 65_536,
        weights_near_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        weights_rand_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        local_steps: int = 2,
        svd_rtol: float = 1e-4,
    ):
        """
        TT 4D two-site MALS/DMRG-like:
        - update locale su bond (G0,G1), (G1,G2), (G2,G3),
        - split con SVD truncation (rank-adaptive entro i cap di tt_ranks).
        """
        work_dtype = torch.float64 if (self.n_func >= 128 and self.dtype == torch.float32) else self.dtype
        if work_dtype != self.dtype:
            print("\033[93m[TT-4D MALS] auto-switch a float64 per stabilità numerica (N>=128)\033[0m")

        device, dtype = self.device, work_dtype
        l_num = len(points_near_list)
        assert len(sdf_near_list) == l_num
        assert len(points_rand_list) == l_num
        assert len(sdf_rand_list) == l_num
        if weights_near_list is not None:
            assert len(weights_near_list) == l_num
        if weights_rand_list is not None:
            assert len(weights_rand_list) == l_num

        if len(tt_ranks) == 3:
            r0_cap, r1_cap, r2_cap = int(tt_ranks[0]), int(tt_ranks[1]), int(tt_ranks[2])
        elif len(tt_ranks) == 2:
            r0_cap = 1
            r1_cap, r2_cap = int(tt_ranks[0]), int(tt_ranks[1])
        else:
            raise ValueError(f"tt_ranks deve avere lunghezza 2 o 3, trovato {len(tt_ranks)}")

        pts_list, sdf_list, lid_list, w_list = [], [], [], []
        for ell in range(l_num):
            p_near = torch.as_tensor(points_near_list[ell], device=device, dtype=dtype).reshape(-1, 3)
            s_near = torch.as_tensor(sdf_near_list[ell], device=device, dtype=dtype).reshape(-1)
            p_rand = torch.as_tensor(points_rand_list[ell], device=device, dtype=dtype).reshape(-1, 3)
            s_rand = torch.as_tensor(sdf_rand_list[ell], device=device, dtype=dtype).reshape(-1)

            pts_link = torch.cat([p_near, p_rand], dim=0)
            sdf_link = torch.cat([s_near, s_rand], dim=0)
            n_link = pts_link.shape[0]

            pts_list.append(pts_link)
            sdf_list.append(sdf_link)
            lid_list.append(torch.full((n_link,), ell, device=device, dtype=torch.long))

            if (weights_near_list is not None and weights_near_list[ell] is not None) or \
               (weights_rand_list is not None and weights_rand_list[ell] is not None):
                w_near = weights_near_list[ell] if weights_near_list is not None else None
                w_rand = weights_rand_list[ell] if weights_rand_list is not None else None
                w_near_t = torch.ones(p_near.shape[0], device=device, dtype=dtype) if w_near is None else torch.as_tensor(w_near, device=device, dtype=dtype).reshape(-1)
                w_rand_t = torch.ones(p_rand.shape[0], device=device, dtype=dtype) if w_rand is None else torch.as_tensor(w_rand, device=device, dtype=dtype).reshape(-1)
                w_list.append(torch.cat([w_near_t, w_rand_t], dim=0))
            else:
                w_list.append(torch.ones(n_link, device=device, dtype=dtype))

        pts_all = torch.cat(pts_list, dim=0)
        sdf_all = torch.cat(sdf_list, dim=0)
        lid_all = torch.cat(lid_list, dim=0)
        w_all = torch.cat(w_list, dim=0)
        p_total = pts_all.shape[0]

        print(f"[TRAIN 4D TT MALS] L={l_num}, P_tot={p_total}, N={self.n_func}, tt_ranks={tt_ranks}")

        t = self.normalize_points(pts_all).clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        sw_all = torch.sqrt(w_all.clamp_min(0) + torch.finfo(dtype).eps)

        n = int(self.n_func)
        g0 = 1e-2 * torch.randn(1, l_num, r0_cap, device=device, dtype=dtype)
        g1 = 1e-2 * torch.randn(r0_cap, n, r1_cap, device=device, dtype=dtype)
        g2 = 1e-2 * torch.randn(r1_cap, n, r2_cap, device=device, dtype=dtype)
        g3 = 1e-2 * torch.randn(r2_cap, n, 1, device=device, dtype=dtype)

        bs = int(batch_size) if (batch_size is not None and batch_size > 0) else p_total
        log_every = max(1, int(iters) // 20)

        for it in range(int(iters)):
            if bs < p_total:
                idx = torch.randperm(p_total, device=device)[:bs]
            else:
                idx = torch.arange(p_total, device=device)

            phi_x, _ = self.build_bernstein_t(tx[idx], use_derivative=False)
            phi_y, _ = self.build_bernstein_t(ty[idx], use_derivative=False)
            phi_z, _ = self.build_bernstein_t(tz[idx], use_derivative=False)
            y_b = sdf_all[idx]
            lid_b = lid_all[idx]
            sw_b = sw_all[idx]

            # sweep two-site bidirezionale
            for bond in (0, 1, 2, 1, 0):
                if bond == 0:
                    theta = torch.einsum('ilr,rnq->ilnq', g0, g1)  # (1,L,N,r1)
                    by_proj = torch.einsum('bn,anr->bar', phi_y, g2).detach()  # (B,r1,r2)
                    bz_proj = torch.einsum('bn,rn->br', phi_z, g3[..., 0]).detach()  # (B,r2)

                    def predict_theta01(theta_param):
                        theta_l = theta_param[0, lid_b]  # (B,N,r1)
                        t1 = (theta_l * phi_x.unsqueeze(-1)).sum(dim=1)  # (B,r1)
                        t2 = torch.einsum('br,brs->bs', t1, by_proj)
                        return (t2 * bz_proj).sum(dim=1)

                    theta_fit = self._fit_theta_local(
                        theta_init=theta,
                        predict_fn=predict_theta01,
                        y_b=y_b,
                        sw_b=sw_b,
                        ridge=ridge,
                        lr=lr,
                        local_steps=local_steps,
                    )
                    g0, g1 = self._split_two_site(
                        theta=theta_fit,
                        r_left=1,
                        n_left=l_num,
                        n_right=n,
                        r_right=int(g1.shape[2]),
                        rank_cap=r0_cap,
                        svd_rtol=svd_rtol,
                    )

                elif bond == 1:
                    theta = torch.einsum('anr,rmq->anmq', g1, g2)  # (r0,N,N,r2)
                    g0_b = g0[0, lid_b, :].detach()  # (B,r0)
                    bz_proj = torch.einsum('bn,rn->br', phi_z, g3[..., 0]).detach()  # (B,r2)

                    def predict_theta12(theta_param):
                        t2 = torch.einsum('ba,bn,bm,anms->bs', g0_b, phi_x, phi_y, theta_param)
                        return (t2 * bz_proj).sum(dim=1)

                    theta_fit = self._fit_theta_local(
                        theta_init=theta,
                        predict_fn=predict_theta12,
                        y_b=y_b,
                        sw_b=sw_b,
                        ridge=ridge,
                        lr=lr,
                        local_steps=local_steps,
                    )
                    g1, g2 = self._split_two_site(
                        theta=theta_fit,
                        r_left=int(g1.shape[0]),
                        n_left=n,
                        n_right=n,
                        r_right=int(g2.shape[2]),
                        rank_cap=r1_cap,
                        svd_rtol=svd_rtol,
                    )

                else:
                    theta = torch.einsum('anr,rmq->anmq', g2, g3)  # (r1,N,N,1)
                    g0_b = g0[0, lid_b, :].detach()
                    u1 = torch.einsum('ba,bn,anr->br', g0_b, phi_x, g1).detach()  # (B,r1)

                    def predict_theta23(theta_param):
                        return torch.einsum('br,bn,bm,rnm->b', u1, phi_y, phi_z, theta_param[..., 0])

                    theta_fit = self._fit_theta_local(
                        theta_init=theta,
                        predict_fn=predict_theta23,
                        y_b=y_b,
                        sw_b=sw_b,
                        ridge=ridge,
                        lr=lr,
                        local_steps=local_steps,
                    )
                    g2, g3 = self._split_two_site(
                        theta=theta_fit,
                        r_left=int(g2.shape[0]),
                        n_left=n,
                        n_right=n,
                        r_right=1,
                        rank_cap=r2_cap,
                        svd_rtol=svd_rtol,
                    )

                g0.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                g1.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                g2.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                g3.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)

            bx = torch.einsum('bn,anr->bar', phi_x, g1)
            by = torch.einsum('bn,anr->bar', phi_y, g2)
            bz = torch.einsum('bn,rn->br', phi_z, g3[..., 0])
            g0_b = g0[0, lid_b, :]
            t1 = torch.einsum('ba,bar->br', g0_b, bx)
            t2 = torch.einsum('ba,bar->br', t1, by)
            y_hat = (t2 * bz).sum(dim=1)
            resid = sw_b * (y_hat - y_b)
            loss = (resid @ resid) / float(y_b.numel())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"TT-4D MALS loss non-finite (iter={it+1}). "
                    "Prova con ranks più piccoli, lr più basso o batch_size più basso."
                )
            if (it % log_every == 0) or (it == int(iters) - 1):
                print(
                    f"\033[95m[TRAIN 4D TT MALS] iter {it+1}/{iters}, loss={loss.item():.6e} | "
                    f"r=({g0.shape[2]},{g1.shape[2]},{g2.shape[2]}) | bs={bs}\033[0m"
                )

        return g0.detach(), g1.detach(), g2.detach(), g3.detach()

    def train_tt_robot(
        self,
        points_near_list:  Sequence[torch.Tensor],
        sdf_near_list:     Sequence[torch.Tensor],
        points_rand_list:  Sequence[torch.Tensor],
        sdf_rand_list:     Sequence[torch.Tensor],
        tt_ranks: Tuple[int, ...] = (4, 8, 4),
        iters: int = 10,
        lr: float = 5e-3,
        ridge: float = 1e-6,
        batch_size: int = 65_536,
        weights_near_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        weights_rand_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ):
        """
        TT 4D per un robot con L link (W ∈ R^{L×N×N×N}).

        Core TT:
          G0 ∈ R^{1 × L × r0}
          G1 ∈ R^{r0 × N × r1}
          G2 ∈ R^{r1 × N × r2}
          G3 ∈ R^{r2 × N × 1}
        """
        device, dtype = self.device, self.dtype
        L = len(points_near_list)
        assert len(sdf_near_list)    == L
        assert len(points_rand_list) == L
        assert len(sdf_rand_list)    == L

        if weights_near_list is not None:
            assert len(weights_near_list) == L
        if weights_rand_list is not None:
            assert len(weights_rand_list) == L

        # ----- pack globale -----
        pts_list, sdf_list, lid_list, w_list = [], [], [], []

        for ell in range(L):
            p_near = torch.as_tensor(points_near_list[ell], device=device, dtype=dtype).reshape(-1, 3)
            s_near = torch.as_tensor(sdf_near_list[ell],    device=device, dtype=dtype).reshape(-1)
            p_rand = torch.as_tensor(points_rand_list[ell], device=device, dtype=dtype).reshape(-1, 3)
            s_rand = torch.as_tensor(sdf_rand_list[ell],    device=device, dtype=dtype).reshape(-1)

            pts_link = torch.cat([p_near, p_rand], dim=0)   # (P_l,3)
            sdf_link = torch.cat([s_near, s_rand], dim=0)   # (P_l,)
            n_link   = pts_link.shape[0]

            lid = torch.full((n_link,), ell, device=device, dtype=torch.long)

            pts_list.append(pts_link)
            sdf_list.append(sdf_link)
            lid_list.append(lid)

            # pesi
            if (weights_near_list is not None and weights_near_list[ell] is not None) or \
               (weights_rand_list is not None and weights_rand_list[ell] is not None):

                w_near = weights_near_list[ell] if (weights_near_list is not None) else None
                w_rand = weights_rand_list[ell] if (weights_rand_list is not None) else None

                if w_near is None:
                    w_near_t = torch.ones(p_near.shape[0], device=device, dtype=dtype)
                else:
                    w_near_t = torch.as_tensor(w_near, device=device, dtype=dtype).reshape(-1)

                if w_rand is None:
                    w_rand_t = torch.ones(p_rand.shape[0], device=device, dtype=dtype)
                else:
                    w_rand_t = torch.as_tensor(w_rand, device=device, dtype=dtype).reshape(-1)

                w_link = torch.cat([w_near_t, w_rand_t], dim=0)
            else:
                w_link = torch.ones(n_link, device=device, dtype=dtype)

            w_list.append(w_link)

        pts_all  = torch.cat(pts_list, dim=0)   # (P_tot,3)
        sdf_all  = torch.cat(sdf_list, dim=0)   # (P_tot,)
        link_ids = torch.cat(lid_list, dim=0)   # (P_tot,)
        w_all    = torch.cat(w_list, dim=0)     # (P_tot,)
        P_tot    = pts_all.shape[0]

        print(f"[TRAIN 4D TT WHOLE ROBOT] L={L}, P_tot={P_tot}, N={self.n_func}, tt_ranks={tt_ranks}")

        # ----- normalizzazione + basis -----
        t = self.normalize_points(pts_all).clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        N = self.n_func

        # gestisci tt_ranks di lunghezza 2 o 3
        if len(tt_ranks) == 3:
            r0, r1, r2 = tt_ranks
        elif len(tt_ranks) == 2:
            r0 = 1
            r1, r2 = tt_ranks
        else:
            raise ValueError(f"tt_ranks deve avere lunghezza 2 o 3, trovato {len(tt_ranks)}")

        # core TT
        G0 = nn.Parameter(1e-2 * torch.randn(1,  L,  r0, device=device, dtype=dtype))
        G1 = nn.Parameter(1e-2 * torch.randn(r0, N,  r1, device=device, dtype=dtype))
        G2 = nn.Parameter(1e-2 * torch.randn(r1, N,  r2, device=device, dtype=dtype))
        G3 = nn.Parameter(1e-2 * torch.randn(r2, N,  1, device=device, dtype=dtype))

        opt = torch.optim.Adam([G0, G1, G2, G3], lr=lr)
        sw_all = torch.sqrt(w_all.clamp_min(0) + torch.finfo(dtype).eps)

        for it in range(iters):
            perm = torch.randperm(P_tot, device=device)
            epoch_loss = 0.0
            nbatches = 0

            for start in range(0, P_tot, batch_size):
                end = min(start + batch_size, P_tot)
                idx = perm[start:end]

                tx_b = tx[idx]
                ty_b = ty[idx]
                tz_b = tz[idx]
                y_b   = sdf_all[idx]
                lid_b = link_ids[idx]
                sw_b  = sw_all[idx]
                B     = y_b.numel()

                Phi_x, _ = self.build_bernstein_t(tx_b, use_derivative=False)  # (B,N)
                Phi_y, _ = self.build_bernstein_t(ty_b, use_derivative=False)  # (B,N)
                Phi_z, _ = self.build_bernstein_t(tz_b, use_derivative=False)  # (B,N)

                # ---- forward TT 4D ----
                # G1: (r0, N, r1)  → indici a,n,r
                # Bx: (B, r0, r1) = Σ_n Phi_x[b,n] * G1[a,n,r]
                Bx = torch.einsum('bn,anr->bar', Phi_x, G1)      # (B,r0,r1)

                # G2: (r1, N, r2)
                # By: (B, r1, r2)
                By = torch.einsum('bn,anr->bar', Phi_y, G2)      # (B,r1,r2)

                # G3: (r2, N, 1)  → G3_red: (r2, N)
                G3_red = G3[..., 0]                              # (r2,N)
                # Bz: (B, r2) = Σ_n Phi_z[b,n] * G3_red[a2,n]
                Bz = torch.einsum('bn,an->ba', Phi_z, G3_red)    # (B,r2)

                # G0 per i link del batch: (B, r0)
                G0_b = G0[0, lid_b, :]                           # (B,r0)

                # T1: (B, r1) = Σ_{α0} G0_b[b,α0] * Bx[b,α0,α1]
                T1 = torch.einsum('ba,bar->br', G0_b, Bx)        # (B,r1)

                # T2: (B, r2) = Σ_{α1} T1[b,α1] * By[b,α1,α2]
                T2 = torch.einsum('ba,bar->br', T1, By)          # (B,r2)

                # y_hat: (B,) = Σ_{α2} T2[b,α2] * Bz[b,α2]
                y_hat = (T2 * Bz).sum(dim=1)                     # (B,)

                # ---- loss ----
                resid     = sw_b * (y_hat - y_b)
                data_loss = (resid @ resid) / float(B)
                reg_loss  = (
                    G0.square().mean() +
                    G1.square().mean() +
                    G2.square().mean() +
                    G3.square().mean()
                )
                loss = data_loss + ridge * reg_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                nbatches   += 1
            data_loss = (resid @ resid) / float(B)
            reg_loss  = (
                G0.square().mean() +
                G1.square().mean() +
                G2.square().mean() +
                G3.square().mean()
            )
            loss = data_loss + ridge * reg_loss
            print(f"[TT] it={it} batch_loss={loss.item():.3e} data={data_loss.item():.3e} reg={reg_loss.item():.3e}")

            epoch_loss /= max(nbatches, 1)
            print(f"\033[95m[TRAIN 4D TT WHOLE ROBOT] iter {it+1}/{iters}, loss={epoch_loss:.6e}\033[0m")

        # # rinormalizzazione globale (opzionale)
        # with torch.no_grad():
        #     norm = torch.sqrt(
        #         G0.square().sum() +
        #         G1.square().sum() +
        #         G2.square().sum() +
        #         G3.square().sum()
        #     ).clamp_min(1e-12)
        #     G0.div_(norm); G1.div_(norm); G2.div_(norm); G3.div_(norm)

        return G0.detach(), G1.detach(), G2.detach(), G3.detach()
