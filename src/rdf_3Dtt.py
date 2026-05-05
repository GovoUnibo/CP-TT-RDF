import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import List, Union, Literal

from src.core.common import CommonSdfMethods
from src.core.assets.entities.models import TtLinkModel
from src.core.train.train_config import TrainConfig, TTLinkCfg
from src.core.sdf_creator import SDFTrain
from src.core.math.projection_point import spherical_projection, correct_sphere_gradient


class RDF_TT(CommonSdfMethods):
    def __init__(self, device='cuda', dtype=torch.float32):
        CommonSdfMethods.__init__(self, projection_method='sphere', device=device, dtype=dtype)

        self.G1_batch = None
        self.G2_batch = None
        self.G3_batch = None
        self.G2_batch_flat = None
        self.G3_batch_t = None

        self.centroids_batch = None
        self.scale_factors_batch = None

        self.max_n_func = None
        self.r1_max = None
        self.r2_max = None

        self._tt_cache = {}

    def init_robot_folder(self, ws_path, robot_name=''):
        CommonSdfMethods.init_robot_folder(self, ws_path, robot_name)
        self.ws_path = ws_path

    def train_links(self, link_names: Union[List[str], str], n_func: int, ranks: tuple, iters: int, robot_name='', debug=False):
        if isinstance(link_names, str):
            link_names = [link_names]
        cfg_w = TTLinkCfg(run=True, n_func=n_func, ranks=ranks, iters=iters)
        cfg = TrainConfig(links_to_train=link_names, debug=debug, tt_link=cfg_w)
        trainer = SDFTrain()
        trainer.init_robot_folder(self.ws_path, robot_name=robot_name)
        trainer.create_model(cfg, robot_name=robot_name)
        CommonSdfMethods.init_robot_folder(self, self.ws_path, robot_name=robot_name)

    def batch_to(self, device):
        self.device = device
        if self.G1_batch is not None:
            self.G1_batch = self.G1_batch.to(device=device, dtype=self.dtype)
        if self.G2_batch is not None:
            self.G2_batch = self.G2_batch.to(device=device, dtype=self.dtype)
        if self.G3_batch is not None:
            self.G3_batch = self.G3_batch.to(device=device, dtype=self.dtype)
        if self.G2_batch_flat is not None:
            self.G2_batch_flat = self.G2_batch_flat.to(device=device, dtype=self.dtype)
        if self.G3_batch_t is not None:
            self.G3_batch_t = self.G3_batch_t.to(device=device, dtype=self.dtype)
        if self.centroids_batch is not None:
            self.centroids_batch = self.centroids_batch.to(device=device, dtype=self.dtype)
        if self.scale_factors_batch is not None:
            self.scale_factors_batch = self.scale_factors_batch.to(device=device, dtype=self.dtype)

    def add_models(self, link_names: Union[List[str], str], namespace='', robot_name='', **kwargs):
        if isinstance(link_names, str):
            link_names = [link_names]
        for link_name in link_names:
            self.add_model(link_name, TtLinkModel.file_suffix, namespace, robot_name, **kwargs)

    def _pad_tt_cores(self, G1, G2, G3, N_max, r1_max, r2_max):
        device = G1.device
        dtype = G1.dtype
        N = G1.shape[1]
        r1 = G1.shape[2]
        r2 = G3.shape[0]

        G1_pad = torch.zeros((1, N_max, r1_max), device=device, dtype=dtype)
        G2_pad = torch.zeros((r1_max, N_max, r2_max), device=device, dtype=dtype)
        G3_pad = torch.zeros((r2_max, N_max, 1), device=device, dtype=dtype)

        G1_pad[:, :N, :r1] = G1
        G2_pad[:r1, :N, :r2] = G2
        G3_pad[:r2, :N, :] = G3

        return G1_pad, G2_pad, G3_pad

    def set_ordered_batch_params(self, link_names: List[str]):
        if isinstance(link_names, str):
            link_names = [link_names]

        per_link = []
        n_functions = []
        list_centroids = []
        list_scales = []

        self._tt_cache.clear()

        for i, link in enumerate(link_names):
            m: TtLinkModel = getattr(self, link + self.model_extension)

            if m.G1 is None or m.G2 is None or m.G3 is None:
                raise ValueError(f"TT cores missing for link '{link}' (G1/G2/G3).")

            G1 = self._to_device_dtype(m.G1)
            G2 = self._to_device_dtype(m.G2)
            G3 = self._to_device_dtype(m.G3)

            if G1.dim() != 3 or G2.dim() != 3 or G3.dim() != 3:
                raise ValueError(f"TT cores must be 3D tensors for link '{link}'.")
            if G1.shape[0] != 1 or G3.shape[2] != 1:
                raise ValueError(f"Unexpected TT core shapes for link '{link}'.")

            N = G1.shape[1]
            r1 = G1.shape[2]
            r2 = G3.shape[0]

            if G2.shape != (r1, N, r2):
                raise ValueError(
                    f"Incoherent TT shapes for link '{link}': G1{tuple(G1.shape)}, G2{tuple(G2.shape)}, G3{tuple(G3.shape)}"
                )

            n_func = int(m.n_func) if m.n_func is not None else int(N)
            n_functions.append(n_func)

            centroid = self._to_device_dtype(m.centroid_offset).reshape(-1)
            if centroid.numel() != 3:
                raise ValueError(f"centroid_offset must have 3 elements for link '{link}'.")
            list_centroids.append(centroid.view(1, 3))

            scale = self._to_device_dtype(m.scale_factor).reshape(-1)
            if scale.numel() != 1:
                raise ValueError(f"scale_factor must be scalar for link '{link}'.")
            list_scales.append(scale.view(1, 1, 1))

            self._tt_cache[link] = {
                "G1": G1,
                "G2": G2,
                "G3": G3,
                "N": int(N),
                "r1": int(r1),
                "r2": int(r2),
            }
            per_link.append((G1, G2, G3, int(N), int(r1), int(r2)))

            if i == 0:
                self.set_points_domain(m.domain_min, m.domain_max)

        self.max_n_func = max(n_functions)
        self.set_number_of_functions(self.max_n_func)

        r1_max = max(r1 for *_, r1, _ in per_link)
        r2_max = max(r2 for *_, _, r2 in per_link)

        G1_list = []
        G2_list = []
        G3_list = []

        for G1, G2, G3, N, r1, r2 in per_link:
            G1_pad, G2_pad, G3_pad = self._pad_tt_cores(G1, G2, G3, self.max_n_func, r1_max, r2_max)
            G1_list.append(G1_pad[0].unsqueeze(0))
            G2_list.append(G2_pad.unsqueeze(0))
            G3_list.append(G3_pad[..., 0].unsqueeze(0))

        self.G1_batch = torch.cat(G1_list, dim=0)
        self.G2_batch = torch.cat(G2_list, dim=0)
        self.G3_batch = torch.cat(G3_list, dim=0)
        self.G2_batch_flat = self.G2_batch.permute(0, 2, 1, 3).reshape(self.G2_batch.shape[0], self.G2_batch.shape[2], -1).contiguous()
        self.G3_batch_t = self.G3_batch.transpose(1, 2).contiguous()

        self.r1_max = r1_max
        self.r2_max = r2_max

        self.centroids_batch = torch.cat(list_centroids, dim=0).unsqueeze(1)
        self.scale_factors_batch = torch.cat(list_scales, dim=0)

        self.batch_to(self.device)

    def build_tt_from_model(self, link_names: List[str]):
        self.set_ordered_batch_params(link_names)

    @torch.no_grad()
    def inference_link(self, link_name: str, points: torch.Tensor, get_grad: bool = False, get_min: bool = False, fk_matrix: torch.Tensor = None):
        if fk_matrix is not None:
            points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=fk_matrix)

        model: TtLinkModel = getattr(self, link_name + self.model_extension)

        def _t(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=points.device, dtype=points.dtype)
            return torch.as_tensor(x, device=points.device, dtype=points.dtype)

        G1 = _t(model.G1)
        G2 = _t(model.G2)
        G3 = _t(model.G3)

        if G1 is None or G2 is None or G3 is None:
            raise ValueError(f"TT cores missing for link '{link_name}'.")

        N = G1.shape[1]
        self.set_number_of_functions(N)
        self.set_points_domain(domain_min=model.domain_min, domain_max=model.domain_max)

        centroid = _t(model.centroid_offset).reshape(1, 3)
        scale = _t(model.scale_factor).reshape(1, 1)

        pts = points.to(device=points.device, dtype=points.dtype).view(-1, 3)
        pts_scaled = (pts - centroid) / scale

        sphere_r1 = (self.domain_max - self.domain_min) / 2
        x_in, d_sphere, outside = spherical_projection(pts_scaled, r1=sphere_r1)

        p = self.normalize_points(x_in)
        Bx, dBx = self.build_bernstein_t(p[:, 0], use_derivative=get_grad)
        By, dBy = self.build_bernstein_t(p[:, 1], use_derivative=get_grad)
        Bz, dBz = self.build_bernstein_t(p[:, 2], use_derivative=get_grad)

        G1_0 = G1[0]
        G2_flat = G2.permute(1, 0, 2).reshape(G2.shape[1], -1).contiguous()
        G3_t = G3[..., 0].transpose(0, 1).contiguous()
        rank_r1 = G1_0.shape[1]
        r2 = G3_t.shape[1]

        U1 = torch.matmul(Bx, G1_0)
        Y = torch.matmul(By, G2_flat).reshape(-1, rank_r1, r2)
        U2 = (U1.unsqueeze(-1) * Y).sum(dim=1)
        Z = torch.matmul(Bz, G3_t)
        sdf_inner = (U2 * Z).sum(dim=-1)

        sdf = sdf_inner + d_sphere
        sdf = sdf * scale.view(1)

        if get_grad:
            dU1_dx = torch.matmul(dBx, G1_0)
            dU2_dx = (dU1_dx.unsqueeze(-1) * Y).sum(dim=1)

            Y_dy = torch.matmul(dBy, G2_flat).reshape(-1, rank_r1, r2)
            dU2_dy = (U1.unsqueeze(-1) * Y_dy).sum(dim=1)

            dZ_dz = torch.matmul(dBz, G3_t)
            dfdx = (dU2_dx * Z).sum(dim=-1)
            dfdy = (dU2_dy * Z).sum(dim=-1)
            dfdz = (U2 * dZ_dz).sum(dim=-1)

            g_in = torch.stack([dfdx, dfdy, dfdz], dim=-1)
            g_in = g_in / (self.domain_max - self.domain_min)

            d_sdf = correct_sphere_gradient(points_scaled=pts_scaled, g_in=g_in, outside=outside, r1=sphere_r1)
        else:
            d_sdf = torch.zeros((pts_scaled.shape[0], 3), device=points.device, dtype=points.dtype)

        if get_min:
            sdf_min, idx = torch.min(sdf, dim=0)
            return sdf_min, d_sdf[idx], points[idx]

        return sdf, d_sdf, points

    @torch.no_grad()
    def _sdf_kernel_for_tests(self, link_name: str, points: torch.Tensor, get_grad: bool = False, get_min: bool = False, fk_matrix: torch.Tensor = None):
        if fk_matrix is not None:
            points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=fk_matrix)

        model: TtLinkModel = getattr(self, link_name + self.model_extension)

        def _t(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=points.device, dtype=points.dtype)
            return torch.as_tensor(x, device=points.device, dtype=points.dtype)

        G1 = _t(model.G1)
        G2 = _t(model.G2)
        G3 = _t(model.G3)
        if G1 is None or G2 is None or G3 is None:
            raise ValueError(f"TT cores missing for link '{link_name}'.")

        self.set_number_of_functions(G1.shape[1])
        self.set_points_domain(domain_min=model.domain_min, domain_max=model.domain_max)
        centroid = _t(model.centroid_offset).reshape(1, 3)
        scale = _t(model.scale_factor).reshape(1, 1)
        pts_scaled = (points.to(device=points.device, dtype=points.dtype).view(-1, 3) - centroid) / scale
        p = self.normalize_points(pts_scaled)
        Bx, dBx = self.build_bernstein_t(p[:, 0], use_derivative=get_grad)
        By, dBy = self.build_bernstein_t(p[:, 1], use_derivative=get_grad)
        Bz, dBz = self.build_bernstein_t(p[:, 2], use_derivative=get_grad)
        G1_0 = G1[0]
        G2_flat = G2.permute(1, 0, 2).reshape(G2.shape[1], -1).contiguous()
        G3_t = G3[..., 0].transpose(0, 1).contiguous()
        rank_r1 = G1_0.shape[1]
        r2 = G3_t.shape[1]

        U1 = torch.matmul(Bx, G1_0)
        Y = torch.matmul(By, G2_flat).reshape(-1, rank_r1, r2)
        U2 = (U1.unsqueeze(-1) * Y).sum(dim=1)
        Z = torch.matmul(Bz, G3_t)
        sdf = (U2 * Z).sum(dim=-1) * scale.view(1)

        if get_grad:
            den = (self.domain_max - self.domain_min)
            dU1_dx = torch.matmul(dBx, G1_0)
            dU2_dx = (dU1_dx.unsqueeze(-1) * Y).sum(dim=1)

            Y_dy = torch.matmul(dBy, G2_flat).reshape(-1, rank_r1, r2)
            dU2_dy = (U1.unsqueeze(-1) * Y_dy).sum(dim=1)

            dZ_dz = torch.matmul(dBz, G3_t)
            d_sdf = torch.stack([
                (dU2_dx * Z).sum(dim=-1),
                (dU2_dy * Z).sum(dim=-1),
                (U2 * dZ_dz).sum(dim=-1),
            ], dim=-1) / den
        else:
            d_sdf = torch.zeros((pts_scaled.shape[0], 3), device=points.device, dtype=points.dtype)

        if get_min:
            sdf_min, idx = torch.min(sdf, dim=0)
            return sdf_min, d_sdf[idx], points[idx]

        return sdf, d_sdf, points

    @torch.no_grad()
    def sdf_kernel_for_tests(self, link_name: str, points: torch.Tensor, get_grad: bool = False, get_min: bool = False, fk_matrix: torch.Tensor = None):
        """Kernel minimale per benchmark/test: solo SDF, senza spherical projection correction."""
        return self._sdf_kernel_for_tests(
            link_name=link_name,
            points=points,
            get_grad=get_grad,
            get_min=get_min,
            fk_matrix=fk_matrix,
        )

    @torch.no_grad()
    def inner_kernel_for_tests(self, link_name: str, points: torch.Tensor, get_grad: bool = False, get_min: bool = False, fk_matrix: torch.Tensor = None):
        """Compat alias mantenuto per i benchmark esistenti."""
        return self._sdf_kernel_for_tests(
            link_name=link_name,
            points=points,
            get_grad=get_grad,
            get_min=get_min,
            fk_matrix=fk_matrix,
        )

    @torch.no_grad()
    def sdf_batch_kernel_for_tests(self, points: torch.Tensor, get_grad: bool = False, get_min: bool = False, forward_tensor: torch.Tensor = None):
        """Kernel batch SDF-only per benchmark: usa inference_link_batch senza gradienti."""
        return self.inference_link_batch(
            points=points,
            get_grad=get_grad,
            get_min=get_min,
            forward_tensor=forward_tensor,
        )

    @torch.no_grad()
    def predict_link_tt(self, link_name: str, points: torch.Tensor, get_gradient: bool = False, get_min: bool = False, fk_matrix: torch.Tensor = None):
        return self.inference_link(link_name, points, get_grad=get_gradient, get_min=get_min, fk_matrix=fk_matrix)

    @torch.no_grad()
    def inference_link_batch(self, points: torch.Tensor, get_grad: bool = False, get_min: bool = False, forward_tensor: torch.Tensor = None):
        if points.dim() != 3:
            raise ValueError("Expected points shape (L,P,3)")

        if self.G1_batch is None or self.G2_batch is None or self.G3_batch is None:
            raise ValueError("Batch TT cores not initialized. Call set_ordered_batch_params first.")
        if self.G2_batch_flat is None or self.G3_batch_t is None:
            raise ValueError("Batch TT contraction caches not initialized. Call set_ordered_batch_params first.")

        points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=forward_tensor) if forward_tensor is not None else points

        link_length = self.G1_batch.shape[0]
        point_length = points.shape[1]

        if points.shape[0] != link_length:
            raise ValueError(f"Mismatch: points has L={points.shape[0]} but model has L={link_length}")

        pts_scaled = ((points - self.centroids_batch) / self.scale_factors_batch).reshape(-1, 3)
        sphere_r1 = (self.domain_max - self.domain_min) / 2
        x_in, d_sphere, outside = spherical_projection(pts_scaled, r1=sphere_r1)

        p = self.normalize_points(x_in)
        Bx, dBx = self.build_bernstein_t(p[:, 0], use_derivative=get_grad)
        By, dBy = self.build_bernstein_t(p[:, 1], use_derivative=get_grad)
        Bz, dBz = self.build_bernstein_t(p[:, 2], use_derivative=get_grad)

        N = Bx.shape[1]
        Bx = Bx.reshape(link_length, point_length, N)
        By = By.reshape(link_length, point_length, N)
        Bz = Bz.reshape(link_length, point_length, N)

        G1b = self.G1_batch
        G2f = self.G2_batch_flat
        G3t = self.G3_batch_t
        rank_r1 = G1b.shape[2]
        r2 = G3t.shape[2]

        U1 = torch.matmul(Bx, G1b)
        Y = torch.matmul(By, G2f).reshape(link_length, point_length, rank_r1, r2)
        U2 = (U1.unsqueeze(-1) * Y).sum(dim=2)
        Z = torch.matmul(Bz, G3t)
        sdf_inner = (U2 * Z).sum(dim=-1)

        sdf = sdf_inner + d_sphere.reshape(link_length, point_length)
        sdf = sdf * self.scale_factors_batch.reshape(link_length, 1)

        if get_grad:
            dBx = dBx.reshape(link_length, point_length, N)
            dBy = dBy.reshape(link_length, point_length, N)
            dBz = dBz.reshape(link_length, point_length, N)

            dU1_dx = torch.matmul(dBx, G1b)
            dY_dx = Y
            dU2_dx = (dU1_dx.unsqueeze(-1) * dY_dx).sum(dim=2)

            dY_dy = torch.matmul(dBy, G2f).reshape(link_length, point_length, rank_r1, r2)
            dU2_dy = (U1.unsqueeze(-1) * dY_dy).sum(dim=2)

            dZ_dz = torch.matmul(dBz, G3t)
            dfdx = (dU2_dx * Z).sum(dim=-1)
            dfdy = (dU2_dy * Z).sum(dim=-1)
            dfdz = (U2 * dZ_dz).sum(dim=-1)

            g_in = torch.stack([dfdx, dfdy, dfdz], dim=-1)
            g_in = g_in / (self.domain_max - self.domain_min)

            g_corr = correct_sphere_gradient(
                points_scaled=pts_scaled,
                g_in=g_in.reshape(-1, 3),
                outside=outside,
                r1=sphere_r1,
            )
            d_sdf = g_corr.reshape(link_length, point_length, 3)
        else:
            d_sdf = torch.zeros((link_length, point_length, 3), device=points.device, dtype=points.dtype)

        if get_min:
            sdf_min, idx = torch.min(sdf, dim=1)
            ar = torch.arange(link_length, device=points.device)
            pts_min = points[ar, idx]
            grad_min = d_sdf[ar, idx] if get_grad else None
            return sdf_min, grad_min, pts_min

        return sdf, d_sdf, points

    @torch.no_grad()
    def predict_batch_tt(self, points: torch.Tensor, forward_tensor: torch.Tensor = None, get_min: bool = False, get_grad: bool = False):
        return self.inference_link_batch(points, get_grad=get_grad, get_min=get_min, forward_tensor=forward_tensor)

    @torch.no_grad()
    def gradient_descent_batch(self, points_link_frame: torch.Tensor, num_of_iteration: int = 5, epsilon: float = 1e-3):
        if points_link_frame.dim() != 3 or points_link_frame.shape[-1] != 3:
            raise ValueError(f"Expected points shape (L,P,3), got {tuple(points_link_frame.shape)}")

        pts = points_link_frame.clone()
        self.batch_to(pts.device)

        for _ in range(num_of_iteration):
            dist, grad, pts_min = self.inference_link_batch(pts, get_grad=True, get_min=True)

            gnorm = grad.norm(dim=1, keepdim=True).clamp_min(1e-9)
            step = dist.unsqueeze(1) * grad / gnorm
            pts_next = pts_min - step

            pts = pts_next.unsqueeze(1)

            if (dist.abs() <= epsilon).all():
                break

        return pts

    @torch.no_grad()
    def visualize_gradient_descent(
        self,
        initial_points: torch.Tensor,
        forward_dict: dict = None,
        num_of_iteration: int = 10,
        epsilon: float = 1e-3,
        mesh_link_names: List[str] = None,
        rdf_link_names: List[str] = None,
        mesh_color: str = "#302E2E",
        mesh_opacity: float = 1.0,
        rdf_color: str = "#c9c9bf",
        rdf_opacity: float = 0.3,
    ):
        if initial_points.dim() != 3 or initial_points.shape[-1] != 3:
            raise ValueError(f"Expected initial_points shape (L,P,3), got {tuple(initial_points.shape)}")

        pts = initial_points.clone()
        self.batch_to(pts.device)

        all_pts = [pts.clone()]

        for _ in range(num_of_iteration):
            dist, grad, pts_min = self.inference_link_batch(pts, get_grad=True, get_min=True)

            gnorm = grad.norm(dim=1, keepdim=True).clamp_min(1e-9)
            step = dist.unsqueeze(1) * grad / gnorm
            pts_next = (pts_min - step).unsqueeze(1)

            all_pts.append(pts_next.clone())
            pts = pts_next

            if (dist.abs() <= epsilon).all():
                break

        all_pts = torch.stack(all_pts, dim=0)
        all_pts = all_pts[:, :, 0:1, :].permute(1, 0, 2, 3).reshape(all_pts.shape[1], -1, 3)

        if forward_dict is None:
            raise ValueError("forward_dict is required for visualization")

        fk_list_t = []
        for v in forward_dict.values():
            if isinstance(v, torch.Tensor):
                fk_list_t.append(v.to(device=self.device, dtype=self.dtype))
            else:
                fk_list_t.append(torch.as_tensor(v, device=self.device, dtype=self.dtype))
        fk_tensor = torch.stack(fk_list_t, dim=0)

        all_pts_world = self.to_world_frame(all_pts, fk_tensor)

        self.visualize_scene(
            forward_as_dict=forward_dict,
            links_as_mesh=True,
            mesh_link_names=mesh_link_names,
            links_as_rdf=rdf_link_names is not None,
            rdf_link_names=rdf_link_names,
            gradient_descent_points=all_pts_world,
            mesh_color=mesh_color,
            mesh_opacity=mesh_opacity,
            rdf_color=rdf_color,
            rdf_opacity=rdf_opacity,
        )

    def to_mesh(self, model_name, type: Literal['pyvista', 'trimesh'], debug=False):
        from src.utils.MeshUtils import trimesh_to_pyvista
        from src.utils.sdf_utils import tt_sdf_to_mesh

        model: TtLinkModel = getattr(self, model_name + self.model_extension)

        n_func = int(model.n_func) if model.n_func is not None else None
        if n_func is None and model.G1 is not None:
            n_func = int(model.G1.shape[1])
        if n_func is not None:
            self.set_number_of_functions(n_func)
        self.set_points_domain(model.domain_min, model.domain_max)

        G1 = self._to_device_dtype(model.G1)
        G2 = self._to_device_dtype(model.G2)
        G3 = self._to_device_dtype(model.G3)

        mesh = tt_sdf_to_mesh(
            G1=G1, G2=G2, G3=G3,
            nbData=128,
            domain_min=model.domain_min, domain_max=model.domain_max,
            scaling_factor=model.scale_factor,
            centroid_offset=model.centroid_offset,
            sigma_smooth=1.0,
            batch_points=50_000,
            bernstein_matrix_1d=self.build_bernstein_t,
        )

        if mesh is None:
            return None

        if debug:
            mesh.show()
        return trimesh_to_pyvista(mesh, np.eye(4)) if type == 'pyvista' else mesh

    def visualize_scene(self,
                        forward_as_dict: dict,
                        links_as_mesh: bool = True,
                        mesh_link_names: List[str] = None,
                        links_as_rdf: bool = False,
                        rdf_link_names: List[str] = None,
                        additional_point_cloud: torch.Tensor = torch.empty((0,3), device='cpu'),
                        point_cloud_color: str = "#9E9E9E",
                        point_cloud_size: float = 8.0,
                        point_cloud_opacity: float = 1.0,
                        gradient_descent_points: torch.Tensor = None,
                        cartesian_frame_pose: np.ndarray = None,
                        mesh_color: str = "#302E2E",
                        mesh_opacity: float = 1.0,
                        rdf_color: str = "#c9c9bf",
                        rdf_opacity: float = 0.3):

        super().visualize_scene(
            forward_as_dict=forward_as_dict,
            links_as_mesh=links_as_mesh,
            mesh_link_names=mesh_link_names,
            additional_point_cloud=additional_point_cloud,
            point_cloud_color=point_cloud_color,
            point_cloud_size=point_cloud_size,
            point_cloud_opacity=point_cloud_opacity,
            cartesian_frame_pose=cartesian_frame_pose,
            xy_offset=(0.0, 0.0),
            mesh_color=mesh_color,
            mesh_opacity=mesh_opacity,
        )

        from src.utils.MeshUtils import trimesh_to_pyvista

        link_names = list(forward_as_dict.keys())
        rdf_link_set = None
        if rdf_link_names is not None:
            rdf_link_set = set(rdf_link_names)
            unknown = sorted(rdf_link_set.difference(link_names))
            if unknown:
                raise ValueError(f"rdf_link_names contains unknown links: {unknown}")

        if links_as_rdf:
            for link in link_names:
                if rdf_link_set is not None and link not in rdf_link_set:
                    continue
                if hasattr(self, link + self.model_extension):
                    m = self.to_mesh(link, type='trimesh')
                    if m is not None:
                        self._visualizer.add_mesh(
                            trimesh_to_pyvista(m, forward_as_dict[link]),
                            opacity=rdf_opacity,
                            color=rdf_color,
                        )

        # traiettorie GD
        if gradient_descent_points is not None and gradient_descent_points.numel() > 0:
            L, T, _ = gradient_descent_points.shape
            link_colors = ["red","blue","green","orange","magenta","cyan","yellow","purple","brown","pink"]
            for i in range(L):
                pts = gradient_descent_points[i]
                pts = pts.detach().cpu().numpy() if pts.is_cuda else pts.detach().numpy()
                color = link_colors[i % len(link_colors)]
                self._visualizer.add_pointcloud(pts, color=color, point_size=7)
                self._visualizer.add_polyline(pts, color=color, line_width=2)

        self._visualizer.show()
