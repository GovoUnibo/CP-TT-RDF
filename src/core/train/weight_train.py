from src.core.math.Bernstain_P import BersteinPoly
import torch
import numpy as np
from typing import Optional, Tuple, Sequence
import tensorly as tl
import torch.nn as nn


class BernsteinWeightsTrain(BersteinPoly):
    def __init__(self, n_func=8, domain_min = -1, domain_max=1, device='cpu', dtype=torch.float32):
        super().__init__(n_func, domain_min, domain_max, device, dtype)
    
    def set_weights(self, wb:torch.Tensor):
        self.weights = wb.to(self.device).to(self.dtype)
        # print(f'\033[95mWeights set with shape {self.wb.shape}\033[0m')

    def _to_device_dtype(self, x):
        if isinstance(x, torch.Tensor):
            # detach to avoid gradients leaking in; clone if you need your own storage
            return x.detach().to(device=self.device, dtype=self.dtype)
        else:
            # numpy array, list, scalar...
            return torch.tensor(x, device=self.device, dtype=self.dtype)

    def set_scale_and_offset(self, scale, translate):
        self.scale_factor   = self._to_device_dtype(scale)
        self.centroid_offset = self._to_device_dtype(translate)

        # print(f'\033[95mScale and translate set to scale {self.scale} and translate {self.translate}\033[0m')
    
    def train(self, point_near:np.ndarray, near_sdf:np.ndarray, query_points:np.ndarray, query_sdf:np.ndarray, epoches=400,
              sample_near=1024, sample_rand=64):
        # print(f'\033[95m Training Bernstain function using domain: [{self.domain_min},{self.domain_max}] for {epoches} epoches\033[0m')
        # print(f'Point near shape: {point_near.shape}, Near sdf shape: {near_sdf.shape}')
        # print(f'Query points shape: {query_points.shape}, Query sdf shape: {query_sdf.shape}')
        # query_sdf[query_sdf <-1] = -query_sdf[query_sdf <-1]

        wb = torch.zeros(self.n_func**3).float().to(self.device)
        B = (torch.eye(self.n_func**3)/1e-2).float().to(self.device)

        for iter in range(epoches):
            choice_near = np.random.choice(len(point_near), sample_near,replace=False) 
            p_near,sdf_near = torch.from_numpy(point_near[choice_near]).float().to(self.device),torch.from_numpy(near_sdf[choice_near]).float().to(self.device)
            choice_random = np.random.choice(len(query_points), sample_rand,replace=False)
            p_random,sdf_random = torch.from_numpy(query_points[choice_random]).float().to(self.device),torch.from_numpy(query_sdf[choice_random]).float().to(self.device)
            p = torch.cat([p_near,p_random],dim=0)
            sdf = torch.cat([sdf_near,sdf_random],dim=0)
            
            phi_xyz, _ = self.basis_function_from_3Dpoints(p.float().to(self.device),use_derivative=False)


            K = torch.matmul(B,phi_xyz.T).matmul(torch.linalg.inv((torch.eye(len(p)).float().to(self.device)+torch.matmul(torch.matmul(phi_xyz,B),phi_xyz.T))))
            B -= torch.matmul(K,phi_xyz).matmul(B)
            delta_wb = torch.matmul(K,(sdf - torch.matmul(phi_xyz,wb)).squeeze())
            loss = torch.nn.functional.mse_loss(torch.matmul(phi_xyz,wb).squeeze(), sdf, reduction='mean').item()

            print(f'\033[95m Iteration {iter} loss {loss}\033[0m')
            # loss_list.append(loss)
            wb += delta_wb   
            # print(f'iter {iter} wb {wb.shape}')

        return wb



    #     return self.train_cp_on_design(Phi_x, Phi_y, Phi_z, y,
    #                                    rank=rank, iters=iters, lr=lr,
    #                                    ridge=ridge, weights=weights)



    # def train_cp_on_design(self,
    #                        Phi_x: torch.Tensor, Phi_y: torch.Tensor, Phi_z: torch.Tensor,
    #                        y: torch.Tensor,
    #                        rank: int = 8, iters: int = 200, lr: float = 5e-3,
    #                        ridge: float = 1e-6, weights: Optional[torch.Tensor] = None):
        

    #     device, dtype = self.device, self.dtype
    #     Phi_x = Phi_x.to(device=device, dtype=dtype)
    #     Phi_y = Phi_y.to(device=device, dtype=dtype)
    #     Phi_z = Phi_z.to(device=device, dtype=dtype)
    #     y     = y.reshape(-1).to(device=device, dtype=dtype)

    #     P, Nx = Phi_x.shape
    #     Ny = Phi_y.shape[1]; Nz = Phi_z.shape[1]
    #     assert Nx == Ny == Nz == self.n_func, "Phi_* devono avere N colonne ciascuna == self.n_func"
    #     assert y.numel() == P, "y deve avere P elementi"

    #     if weights is None:
    #         sw = torch.ones(P, device=device, dtype=dtype)
    #     else:
    #         sw = torch.as_tensor(weights, device=device, dtype=dtype).reshape(-1)
    #         sw = torch.sqrt(sw.clamp_min(0) + torch.finfo(dtype).eps)

    #     # Parametri CP (leaf tensors)
    #     A   = nn.Parameter(1e-2 * torch.randn(Nx, rank, device=device, dtype=dtype))
    #     B   = nn.Parameter(1e-2 * torch.randn(Ny, rank, device=device, dtype=dtype))
    #     C   = nn.Parameter(1e-2 * torch.randn(Nz, rank, device=device, dtype=dtype))
    #     lam = nn.Parameter(torch.ones(rank, device=device, dtype=dtype))

    #     opt = torch.optim.Adam([A, B, C, lam], lr=lr)
    #     for i in range(iters):
    #         X = Phi_x @ A
    #         Y = Phi_y @ B
    #         Z = Phi_z @ C
    #         K = X * Y * Z               # (P,R)
    #         yhat = K @ lam              # (P,)
    #         resid = sw * (yhat - y)
    #         loss = (resid @ resid) / P \
    #              + ridge * (A.square().mean() + B.square().mean() + C.square().mean())
    #         print(f'\033[95m CP Training Iteration {i+1}/{iters}; loss={loss.item():.6f} \033[0m')
    #         opt.zero_grad(); loss.backward(); opt.step()

    #     with torch.no_grad():
    #         for r in range(rank):
    #             self._renorm_triplet(A, B, C, lam, r)
        
    #     return A.detach(), B.detach(), C.detach(), lam.detach()

    # def train_eikonal(self, point_near: np.ndarray, near_sdf: np.ndarray,
    #         query_points: np.ndarray, query_sdf: np.ndarray,
    #         epoches=400, lr=1e-2, lambda_eikonal=0.05):
        
    #     print(f'\033[95m Training Bernstein SDF with Eikonal loss [{self.domain_min}, {self.domain_max}] for {epoches} epochs\033[0m')

    #     # Inizializza i pesi come parametro ottimizzabile
    #     wb = torch.zeros(self.n_func**3, requires_grad=True, device=self.device, dtype=torch.float32)
    #     optimizer = torch.optim.Adam([wb], lr=lr)

    #     for iter in range(epoches):
    #         optimizer.zero_grad()

    #         # Campionamento
    #         idx_near = np.random.choice(len(point_near), 1024, replace=False)
    #         idx_rand = np.random.choice(len(query_points), 64, replace=False)

    #         p_near = torch.from_numpy(point_near[idx_near]).float().to(self.device)
    #         sdf_near = torch.from_numpy(near_sdf[idx_near]).float().to(self.device)

    #         p_rand = torch.from_numpy(query_points[idx_rand]).float().to(self.device)
    #         sdf_rand = torch.from_numpy(query_sdf[idx_rand]).float().to(self.device)

    #         # Input completo
    #         p = torch.cat([p_near, p_rand], dim=0)
    #         sdf_gt = torch.cat([sdf_near, sdf_rand], dim=0)

    #         # Base di Bernstein + derivate
    #         phi_xyz, d_phi_xyz = self.basis_function_from_3Dpoints(p, use_derivative=True)

    #         sdf_pred = torch.matmul(phi_xyz, wb)  # Predizione SDF

    #         # Gradiente analitico ∇SDF
    #         grad = torch.einsum('ijk,j->ik', d_phi_xyz, wb)
    #         grad_norm = grad.norm(dim=1)

    #         # Loss componenti
    #         loss_sdf = torch.nn.functional.mse_loss(sdf_pred.squeeze(), sdf_gt)
    #         loss_eikonal = ((grad_norm - 1) ** 2).mean()
    #         loss_total = loss_sdf + lambda_eikonal * loss_eikonal

    #         # Backprop e aggiornamento
    #         loss_total.backward()
    #         optimizer.step()

    #         print(f"\033[95mEpoch {iter:03d} | SDF Loss: {loss_sdf.item():.5f} | Eikonal: {loss_eikonal.item():.5f} | Total: {loss_total.item():.5f}\033[0m")

    #     return wb.detach()

    # def train_recursive_with_eikonal(self, point_near: np.ndarray, near_sdf: np.ndarray,
    #                                 query_points: np.ndarray, query_sdf: np.ndarray,
    #                                 epoches=400, lambda_eikonal=0.01):

    #     print(f"\033[95mRecursive Non-Linear SDF Training with Eikonal | Epochs: {epoches} | λ={lambda_eikonal}\033[0m")

    #     wb = torch.zeros(self.n_func**3).float().to(self.device)
    #     B = (torch.eye(self.n_func**3) / 1e-4).float().to(self.device)  # inizializza B come matrice regolarizzata

    #     for epoch in range(epoches):
    #         idx_near = np.random.choice(len(point_near), 1024, replace=False)
    #         idx_rand = np.random.choice(len(query_points), 64, replace=False)

    #         p_near = torch.from_numpy(point_near[idx_near]).float().to(self.device)
    #         sdf_near = torch.from_numpy(near_sdf[idx_near]).float().to(self.device)

    #         p_rand = torch.from_numpy(query_points[idx_rand]).float().to(self.device)
    #         sdf_rand = torch.from_numpy(query_sdf[idx_rand]).float().to(self.device)

    #         p = torch.cat([p_near, p_rand], dim=0)
    #         sdf_gt = torch.cat([sdf_near, sdf_rand], dim=0)

    #         phi_xyz, d_phi_xyz = self.basis_function_from_3Dpoints(p, use_derivative=True)

    #         # Parte SDF classica
    #         residual_sdf = sdf_gt - torch.matmul(phi_xyz, wb)
    #         K_sdf = B @ phi_xyz.T @ torch.linalg.inv(torch.eye(len(p)).to(self.device) + phi_xyz @ B @ phi_xyz.T)
    #         B = B - K_sdf @ phi_xyz @ B
    #         delta_wb_sdf = K_sdf @ residual_sdf

    #         # Parte Eikonal: ||∇SDF|| ≈ 1
    #         grad = torch.einsum('ijk,j->ik', d_phi_xyz, wb)  # (N, 3)
    #         grad_norm = grad.norm(dim=1)
    #         residual_grad = (1.0 - grad_norm).detach()  # solo direzione di correzione, no grad

    #         # Costruiamo "pseudo-osservazioni" sulla norma del gradiente
    #         # Vettori direzionali: ∇SDF / ∥∇SDF∥
    #         unit_grad = grad / (grad_norm.unsqueeze(1) + 1e-8)  # (N, 3)
    #         phi_grad = torch.einsum('ijk,ik->ij', d_phi_xyz, unit_grad)

    #         K_eik = B @ phi_grad.T @ torch.linalg.inv(torch.eye(len(p)).to(self.device) + phi_grad @ B @ phi_grad.T)
    #         B = B - lambda_eikonal * K_eik @ phi_grad @ B
    #         delta_wb_eik = lambda_eikonal * K_eik @ residual_grad

    #         # Aggiorna pesi
    #         wb += delta_wb_sdf + delta_wb_eik

    #         # Log
    #         loss_sdf = residual_sdf.pow(2).mean().item()
    #         loss_eikonal = residual_grad.pow(2).mean().item()
    #         print(f"\033[95mEpoch {epoch:03d} | SDF Loss: {loss_sdf:.5f} | Eikonal: {loss_eikonal:.5f}\033[0m")

    #     return wb
