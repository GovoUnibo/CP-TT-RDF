from src.core.math.Bernstain_P import BersteinPoly
import torch
import numpy as np
from typing import Optional, Tuple, Sequence
import tensorly as tl
import torch.nn as nn


class BernsteinCPTrain(BersteinPoly):
    def __init__(self, n_func: int = 16, domain_min: float = -1, domain_max: float = 1,
                 device: str = 'cpu', dtype: torch.dtype = torch.float32):
        super().__init__(n_func, domain_min, domain_max, device, dtype)

    @staticmethod
    def _renorm_triplet(A, B, C, lam, r: int):
        na = A[:, r].norm().clamp_min(1e-12)
        nb = B[:, r].norm().clamp_min(1e-12)
        nc = C[:, r].norm().clamp_min(1e-12)
        A[:, r] /= na; B[:, r] /= nb; C[:, r] /= nc
        lam[r]  *= (na * nb * nc)

    @staticmethod
    def _solve_ridge_system(ata: torch.Tensor, atb: torch.Tensor, ridge: float):
        n = ata.shape[0]
        ata_reg = ata + ridge * torch.eye(n, device=ata.device, dtype=ata.dtype)
        try:
            return torch.linalg.solve(ata_reg, atb)
        except RuntimeError:
            sol = torch.linalg.lstsq(ata_reg, atb.unsqueeze(-1)).solution.squeeze(-1)
            return sol

    def _rankwise_ls_update(
        self,
        phi: torch.Tensor,
        coeff: torch.Tensor,
        target: torch.Tensor,
        ridge: float,
        rhs_weight: Optional[torch.Tensor] = None,
    ):
        # Solve min || diag(coeff) * (phi @ x) - rhs_weight*target ||^2 + ridge ||x||^2
        # Typical weighted rank-wise CP linearization:
        # coeff = sw * g, rhs_weight = sw, target = residual.
        if coeff.abs().max() < 1e-12:
            return None
        d = phi * coeff.unsqueeze(1)
        rhs = target if rhs_weight is None else (rhs_weight * target)
        ata = d.T @ d
        atb = d.T @ rhs
        return self._solve_ridge_system(ata, atb, ridge=ridge)


    def train_cp(self,
                 points_near: torch.Tensor, sdf_near: torch.Tensor,
                 points_rand: torch.Tensor, sdf_rand: torch.Tensor,
                 rank: int = 8, iters: int = 200, lr: float = 5e-3,
                 ridge: float = 1e-6, batch_size: int = 65_536,
                 weights: Optional[torch.Tensor] = None):
        """
        Wrapper 'su punti': costruisce Φx,Φy,Φz e chiama train_cp_on_design.
        """
        # Per N alti (es. 256) il Bernstein può diventare numericamente instabile in float32.
        # Usiamo float64 nel solo training CP quando necessario.
        work_dtype = torch.float64 if (self.n_func >= 128 and self.dtype == torch.float32) else self.dtype
        if work_dtype != self.dtype:
            print("\033[93m[CP] auto-switch a float64 per stabilità numerica (N>=128)\033[0m")

        points_near = torch.as_tensor(points_near, device=self.device, dtype=work_dtype).reshape(-1, 3)
        sdf_near    = torch.as_tensor(sdf_near,    device=self.device, dtype=work_dtype).reshape(-1)
        points_rand = torch.as_tensor(points_rand, device=self.device, dtype=work_dtype).reshape(-1, 3)
        sdf_rand    = torch.as_tensor(sdf_rand,    device=self.device, dtype=work_dtype).reshape(-1)


        device, dtype = self.device, work_dtype
        pts = torch.cat([points_near, points_rand], dim=0).to(device=device, dtype=dtype)
        y   = torch.cat([sdf_near, sdf_rand], dim=0).to(device=device, dtype=dtype)
        # pts = torch.as_tensor(points, device=device, dtype=dtype).reshape(-1, 3)
        # y   = torch.as_tensor(sdf,     device=device, dtype=dtype).reshape(-1)
        assert pts.shape[0] == y.numel(), "points e sdf devono avere lo stesso P."

        # Normalizza una sola volta i punti in [0,1] (leggero in memoria).
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

        # Parametri CP (leaf tensors)
        A   = nn.Parameter(1e-2 * torch.randn(Nx, rank, device=device, dtype=dtype))
        B   = nn.Parameter(1e-2 * torch.randn(Ny, rank, device=device, dtype=dtype))
        C   = nn.Parameter(1e-2 * torch.randn(Nz, rank, device=device, dtype=dtype))
        lam = nn.Parameter(torch.ones(rank, device=device, dtype=dtype))

        opt = torch.optim.Adam([A, B, C, lam], lr=lr)
        log_every = max(1, iters // 20)   # ~20 log totali

        bs = int(batch_size) if (batch_size is not None and batch_size > 0) else P
        for i in range(iters):
            perm = torch.randperm(P, device=device)
            epoch_loss = 0.0
            nbatches = 0

            for start in range(0, P, bs):
                end = min(start + bs, P)
                idx = perm[start:end]

                Phi_x, _ = self.build_bernstein_t(tx[idx], use_derivative=False)  # (B,N)
                Phi_y, _ = self.build_bernstein_t(ty[idx], use_derivative=False)
                Phi_z, _ = self.build_bernstein_t(tz[idx], use_derivative=False)

                X = Phi_x @ A
                Y = Phi_y @ B
                Z = Phi_z @ C
                K = X * Y * Z
                yhat = K @ lam

                y_b = y[idx]
                sw_b = sw[idx]
                resid = sw_b * (yhat - y_b)
                loss = (resid @ resid) / float(y_b.numel()) \
                     + ridge * (A.square().mean() + B.square().mean() + C.square().mean())

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"CP loss non-finite (iter={i+1}). "
                        f"Prova con lr più basso, rank/N più piccoli o ridge più alto."
                    )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([A, B, C, lam], max_norm=10.0)
                opt.step()

                with torch.no_grad():
                    A.data.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                    B.data.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                    C.data.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                    lam.data.nan_to_num_(nan=1.0, posinf=1e2, neginf=-1e2)
                    lam.data.clamp_(-1e3, 1e3)

                epoch_loss += float(loss.item())
                nbatches += 1

            epoch_loss /= max(nbatches, 1)
            if (i % log_every == 0) or (i == iters - 1):
                print(f"\033[95m CP iter {i+1}/{iters} | loss={epoch_loss:.6e} | bs={bs}\033[0m")

        with torch.no_grad():
            for r in range(rank):
                self._renorm_triplet(A, B, C, lam, r)
        
        return A.detach(), B.detach(), C.detach(), lam.detach()

    def train_cp_als(self,
                     points_near: torch.Tensor, sdf_near: torch.Tensor,
                     points_rand: torch.Tensor, sdf_rand: torch.Tensor,
                     rank: int = 8, iters: int = 200,
                     ridge: float = 1e-6, batch_size: int = 65_536,
                     weights: Optional[torch.Tensor] = None):
        """
        Alternating Least Squares (rank-wise) per CP 3D.
        Più robusto di Adam quando N/rank sono grandi.
        """
        work_dtype = torch.float64 if (self.n_func >= 128 and self.dtype == torch.float32) else self.dtype
        if work_dtype != self.dtype:
            print("\033[93m[CP-ALS] auto-switch a float64 per stabilità numerica (N>=128)\033[0m")

        device, dtype = self.device, work_dtype
        points_near = torch.as_tensor(points_near, device=device, dtype=dtype).reshape(-1, 3)
        sdf_near    = torch.as_tensor(sdf_near,    device=device, dtype=dtype).reshape(-1)
        points_rand = torch.as_tensor(points_rand, device=device, dtype=dtype).reshape(-1, 3)
        sdf_rand    = torch.as_tensor(sdf_rand,    device=device, dtype=dtype).reshape(-1)

        pts = torch.cat([points_near, points_rand], dim=0)
        y   = torch.cat([sdf_near, sdf_rand], dim=0).reshape(-1)
        p_total = y.numel()
        t = super().normalize_points(pts).clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        if weights is None:
            sw = torch.ones(p_total, device=device, dtype=dtype)
        else:
            sw = torch.as_tensor(weights, device=device, dtype=dtype).reshape(-1)
            sw = torch.sqrt(sw.clamp_min(0) + torch.finfo(dtype).eps)

        n = self.n_func
        a = 1e-2 * torch.randn(n, rank, device=device, dtype=dtype)
        b = 1e-2 * torch.randn(n, rank, device=device, dtype=dtype)
        c = 1e-2 * torch.randn(n, rank, device=device, dtype=dtype)
        lam = torch.ones(rank, device=device, dtype=dtype)

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
                y_b = y[idx]
                sw_b = sw[idx]

                x = phi_x @ a
                yv = phi_y @ b
                z = phi_z @ c
                y_hat = ((x * yv * z) * lam.unsqueeze(0)).sum(dim=1)

                for r in range(rank):
                    old = lam[r] * x[:, r] * yv[:, r] * z[:, r]
                    residual = y_b - y_hat + old
                    coeff = lam[r] * yv[:, r] * z[:, r] * sw_b
                    a_new = self._rankwise_ls_update(phi_x, coeff, residual, ridge, rhs_weight=sw_b)
                    if a_new is not None and torch.isfinite(a_new).all():
                        x_new = phi_x @ a_new
                        new = lam[r] * x_new * yv[:, r] * z[:, r]
                        y_hat = y_hat - old + new
                        x[:, r] = x_new
                        a[:, r] = a_new

                    old = lam[r] * x[:, r] * yv[:, r] * z[:, r]
                    residual = y_b - y_hat + old
                    coeff = lam[r] * x[:, r] * z[:, r] * sw_b
                    b_new = self._rankwise_ls_update(phi_y, coeff, residual, ridge, rhs_weight=sw_b)
                    if b_new is not None and torch.isfinite(b_new).all():
                        y_new = phi_y @ b_new
                        new = lam[r] * x[:, r] * y_new * z[:, r]
                        y_hat = y_hat - old + new
                        yv[:, r] = y_new
                        b[:, r] = b_new

                    old = lam[r] * x[:, r] * yv[:, r] * z[:, r]
                    residual = y_b - y_hat + old
                    coeff = lam[r] * x[:, r] * yv[:, r] * sw_b
                    c_new = self._rankwise_ls_update(phi_z, coeff, residual, ridge, rhs_weight=sw_b)
                    if c_new is not None and torch.isfinite(c_new).all():
                        z_new = phi_z @ c_new
                        new = lam[r] * x[:, r] * yv[:, r] * z_new
                        y_hat = y_hat - old + new
                        z[:, r] = z_new
                        c[:, r] = c_new

                    old = lam[r] * x[:, r] * yv[:, r] * z[:, r]
                    residual = y_b - y_hat + old
                    g = x[:, r] * yv[:, r] * z[:, r]
                    wg = sw_b * g
                    den = (wg * wg).sum() + ridge
                    if den > 1e-12:
                        lam_new = ((sw_b * residual) * wg).sum() / den
                        lam_new = lam_new.clamp(-1e3, 1e3)
                        new = lam_new * g
                        y_hat = y_hat - old + new
                        lam[r] = lam_new

                a.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                b.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                c.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                lam.nan_to_num_(nan=1.0, posinf=1e2, neginf=-1e2)

                if (it % log_every == 0) or (it == iters - 1):
                    resid = sw_b * (y_hat - y_b)
                    loss = (resid @ resid) / float(y_b.numel())
                    print(f"\033[95m CP-ALS iter {it+1}/{iters} | loss={loss.item():.6e} | bs={bs}\033[0m")

            for r in range(rank):
                self._renorm_triplet(a, b, c, lam, r)

        return a.detach(), b.detach(), c.detach(), lam.detach()

    def train_cp_robot_als(
        self,
        points_near_list:  Sequence[torch.Tensor],
        sdf_near_list:     Sequence[torch.Tensor],
        points_rand_list:  Sequence[torch.Tensor],
        sdf_rand_list:     Sequence[torch.Tensor],
        rank: int = 8,
        iters: int = 10,
        ridge: float = 1e-6,
        batch_size: int = 65_536,
        weights_near_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        weights_rand_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ):
        """
        CP 4D ALS (mini-batch): aggiorna alternatamente A,B,C,V,lambda.
        Più robusto di Adam in scenari difficili (N/rank alti).
        """
        work_dtype = torch.float64 if (self.n_func >= 128 and self.dtype == torch.float32) else self.dtype
        if work_dtype != self.dtype:
            print("\033[93m[CP-4D ALS] auto-switch a float64 per stabilità numerica (N>=128)\033[0m")

        device, dtype = self.device, work_dtype
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
        link_ids = torch.cat(lid_list, dim=0)
        w_all = torch.cat(w_list, dim=0)
        p_total = pts_all.shape[0]

        print(f"[TRAIN 4D CP ALS] L={l_num}, P_tot={p_total}, rank={rank}, N={self.n_func}")

        t = self.normalize_points(pts_all).clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        sw_all = torch.sqrt(w_all.clamp_min(0) + torch.finfo(dtype).eps)

        n = self.n_func
        r = rank
        a = 1e-2 * torch.randn(n, r, device=device, dtype=dtype)
        b = 1e-2 * torch.randn(n, r, device=device, dtype=dtype)
        c = 1e-2 * torch.randn(n, r, device=device, dtype=dtype)
        v = torch.ones(l_num, r, device=device, dtype=dtype)
        lam = torch.ones(r, device=device, dtype=dtype)

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
                lid_b = link_ids[idx]
                sw_b = sw_all[idx]

                x = phi_x @ a
                yv = phi_y @ b
                z = phi_z @ c
                v_b = v[lid_b, :]
                y_hat = ((x * yv * z * v_b) * lam.unsqueeze(0)).sum(dim=1)

                for rr in range(r):
                    old = lam[rr] * x[:, rr] * yv[:, rr] * z[:, rr] * v_b[:, rr]
                    residual = y_b - y_hat + old
                    coeff = lam[rr] * yv[:, rr] * z[:, rr] * v_b[:, rr] * sw_b
                    a_new = self._rankwise_ls_update(phi_x, coeff, residual, ridge, rhs_weight=sw_b)
                    if a_new is not None and torch.isfinite(a_new).all():
                        x_new = phi_x @ a_new
                        new = lam[rr] * x_new * yv[:, rr] * z[:, rr] * v_b[:, rr]
                        y_hat = y_hat - old + new
                        x[:, rr] = x_new
                        a[:, rr] = a_new

                    old = lam[rr] * x[:, rr] * yv[:, rr] * z[:, rr] * v_b[:, rr]
                    residual = y_b - y_hat + old
                    coeff = lam[rr] * x[:, rr] * z[:, rr] * v_b[:, rr] * sw_b
                    b_new = self._rankwise_ls_update(phi_y, coeff, residual, ridge, rhs_weight=sw_b)
                    if b_new is not None and torch.isfinite(b_new).all():
                        y_new = phi_y @ b_new
                        new = lam[rr] * x[:, rr] * y_new * z[:, rr] * v_b[:, rr]
                        y_hat = y_hat - old + new
                        yv[:, rr] = y_new
                        b[:, rr] = b_new

                    old = lam[rr] * x[:, rr] * yv[:, rr] * z[:, rr] * v_b[:, rr]
                    residual = y_b - y_hat + old
                    coeff = lam[rr] * x[:, rr] * yv[:, rr] * v_b[:, rr] * sw_b
                    c_new = self._rankwise_ls_update(phi_z, coeff, residual, ridge, rhs_weight=sw_b)
                    if c_new is not None and torch.isfinite(c_new).all():
                        z_new = phi_z @ c_new
                        new = lam[rr] * x[:, rr] * yv[:, rr] * z_new * v_b[:, rr]
                        y_hat = y_hat - old + new
                        z[:, rr] = z_new
                        c[:, rr] = c_new

                    old = lam[rr] * x[:, rr] * yv[:, rr] * z[:, rr] * v_b[:, rr]
                    residual = y_b - y_hat + old
                    g_no_v = lam[rr] * x[:, rr] * yv[:, rr] * z[:, rr]
                    v_col = v[:, rr].clone()
                    for l_id in lid_b.unique():
                        mask = (lid_b == l_id)
                        wg = sw_b[mask] * g_no_v[mask]
                        den = (wg * wg).sum() + ridge
                        if den > 1e-12:
                            v_col[l_id] = ((sw_b[mask] * residual[mask]) * wg).sum() / den
                    v_col = v_col.clamp(-1e3, 1e3)
                    new = g_no_v * v_col[lid_b]
                    y_hat = y_hat - old + new
                    v[:, rr] = v_col
                    v_b[:, rr] = v_col[lid_b]

                    old = lam[rr] * x[:, rr] * yv[:, rr] * z[:, rr] * v_b[:, rr]
                    residual = y_b - y_hat + old
                    g = x[:, rr] * yv[:, rr] * z[:, rr] * v_b[:, rr]
                    wg = sw_b * g
                    den = (wg * wg).sum() + ridge
                    if den > 1e-12:
                        lam_new = ((sw_b * residual) * wg).sum() / den
                        lam_new = lam_new.clamp(-1e3, 1e3)
                        new = lam_new * g
                        y_hat = y_hat - old + new
                        lam[rr] = lam_new

                a.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                b.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                c.nan_to_num_(nan=0.0, posinf=1e2, neginf=-1e2)
                v.nan_to_num_(nan=1.0, posinf=1e2, neginf=-1e2)
                lam.nan_to_num_(nan=1.0, posinf=1e2, neginf=-1e2)

                resid = sw_b * (y_hat - y_b)
                loss = (resid @ resid) / float(y_b.numel())
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"CP-4D ALS loss non-finite (iter={it+1}). "
                        "Prova con rank più basso, batch_size più piccolo o ridge più alto."
                    )
                if (it % log_every == 0) or (it == iters - 1):
                    print(f"\033[96m[TRAIN 4D CP ALS] iter {it+1}/{iters}, loss={loss.item():.6e} | bs={bs}\033[0m")

            for rr in range(r):
                self._renorm_triplet(a, b, c, lam, rr)

        return v.detach(), a.detach(), b.detach(), c.detach(), lam.detach()

    def train_cp_robot(
        self,
        points_near_list:  Sequence[torch.Tensor],
        sdf_near_list:     Sequence[torch.Tensor],
        points_rand_list:  Sequence[torch.Tensor],
        sdf_rand_list:     Sequence[torch.Tensor],
        rank: int = 8,
        iters: int = 10,
        lr: float = 5e-3,
        ridge: float = 1e-6,
        batch_size: int = 65_536,
        weights_near_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
        weights_rand_list: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ):
        """
        CP 4D per un robot con L link, usando liste per-link:
          - points_near_list[l] : (P_near_l, 3)
          - sdf_near_list[l]    : (P_near_l,)
          - points_rand_list[l] : (P_rand_l, 3)
          - sdf_rand_list[l]    : (P_rand_l,)

        Tutti i link condividono lo stesso dominio e stesso numero di funzioni:
          self.domain_min, self.domain_max, self.n_func

        Parametri
        ---------
        rank       : rank R del CP 4D
        iters      : numero di epoche di training
        lr         : learning rate Adam
        ridge      : regolarizzazione L2 sui fattori
        batch_size : dimensione dei mini-batch
        weights_*_list : opzionali pesi per campione (near/rand), stessa lunghezza delle liste punti

        Ritorna
        -------
        V   : (L, R)
        A,B,C : (N, R)
        lam : (R,)
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

        pts_list  = []
        sdf_list  = []
        lid_list  = []
        w_list    = []

        for ell in range(L):
            # --- near ---
            p_near = torch.as_tensor(points_near_list[ell], device=device, dtype=dtype).reshape(-1, 3)
            s_near = torch.as_tensor(sdf_near_list[ell],    device=device, dtype=dtype).reshape(-1)

            # --- rand ---
            p_rand = torch.as_tensor(points_rand_list[ell], device=device, dtype=dtype).reshape(-1, 3)
            s_rand = torch.as_tensor(sdf_rand_list[ell],    device=device, dtype=dtype).reshape(-1)

            pts_link = torch.cat([p_near, p_rand], dim=0)   # (P_l, 3)
            sdf_link = torch.cat([s_near, s_rand], dim=0)   # (P_l,)
            n_link   = pts_link.shape[0]

            # link id
            lid = torch.full((n_link,), ell, device=device, dtype=torch.long)

            pts_list.append(pts_link)
            sdf_list.append(sdf_link)
            lid_list.append(lid)

            # pesi (se forniti)
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

                w_link = torch.cat([w_near_t, w_rand_t], dim=0)  # (P_l,)
            else:
                w_link = torch.ones(n_link, device=device, dtype=dtype)

            w_list.append(w_link)

        # --- stack globale ---
        pts_all  = torch.cat(pts_list, dim=0)   # (P_tot, 3)
        sdf_all  = torch.cat(sdf_list, dim=0)   # (P_tot,)
        link_ids = torch.cat(lid_list, dim=0)   # (P_tot,)
        w_all    = torch.cat(w_list, dim=0)     # (P_tot,)
        P_tot    = pts_all.shape[0]

        print(f"[TRAIN 4D CP WHOLE ROBOT] L={L}, P_tot={P_tot}, rank={rank}, N={self.n_func}")

        # --- normalizzazione punti in [0,1] ---
        t = self.normalize_points(pts_all)
        t = t.clamp_(0.0, 1.0)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        N = self.n_func
        R = rank

        # fattori condivisi
        A = nn.Parameter(1e-2 * torch.randn(N, R, device=device, dtype=dtype))
        B = nn.Parameter(1e-2 * torch.randn(N, R, device=device, dtype=dtype))
        C = nn.Parameter(1e-2 * torch.randn(N, R, device=device, dtype=dtype))
        # fattore per-link e lambda globale
        V   = nn.Parameter(torch.ones(L, R, device=device, dtype=dtype))
        lam = nn.Parameter(torch.ones(R,     device=device, dtype=dtype))

        opt = torch.optim.Adam([A, B, C, V, lam], lr=lr)
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

                # basi 1D per batch
                Phi_x, _ = self.build_bernstein_t(tx_b, use_derivative=False)  # (B,N)
                Phi_y, _ = self.build_bernstein_t(ty_b, use_derivative=False)
                Phi_z, _ = self.build_bernstein_t(tz_b, use_derivative=False)

                # proiezioni CP spaziali
                Sx = Phi_x @ A   # (B,R)
                Sy = Phi_y @ B   # (B,R)
                Sz = Phi_z @ C   # (B,R)
                S  = Sx * Sy * Sz  # (B,R)

                # fattore per link: gamma_{i,r} = lam_r * V[link_i, r]
                V_b   = V[lid_b, :]                     # (B,R)
                gamma = lam.unsqueeze(0) * V_b          # (B,R)

                y_hat = (S * gamma).sum(dim=1)          # (B,)

                resid = sw_b * (y_hat - y_b)
                loss = (resid @ resid) / float(y_b.numel()) \
                        + ridge * (
                            A.square().mean() +
                            B.square().mean() +
                            C.square().mean() +
                            V.square().mean()
                        )

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                nbatches += 1

            epoch_loss /= max(nbatches, 1)
            print(f"\033[96m[TRAIN 4D CP WHOLE ROBOT] iter {it+1}/{iters}, loss={epoch_loss:.6e}\033[0m")

        # rinormalizza i triplette spaziali
        with torch.no_grad():
            for r in range(R):
                self._renorm_triplet(A, B, C, lam, r)

        return V.detach(), A.detach(), B.detach(), C.detach(), lam.detach()
