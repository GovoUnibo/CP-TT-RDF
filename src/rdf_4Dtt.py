import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import List, Union

from src.core.common import CommonSdfMethods
from src.core.assets.ModelHandler import ModelHandler
from src.core.assets.entities.models import TtRobotModel
from src.core.train.train_config import TrainConfig, TTRobotCfg
from src.core.sdf_creator import SDFTrain
from src.core.math.projection_point import spherical_projection, correct_sphere_gradient
from src.utils.sdf_utils import tt4d_sdf_to_mesh_robot


class RDF_4D_TT(CommonSdfMethods):
    def __init__(self, device='cuda', dtype=torch.float32):
        CommonSdfMethods.__init__(self, projection_method='sphere', device=device, dtype=dtype)

        self.G0 = None
        self.G1 = None
        self.G2 = None
        self.G3 = None

        self.centroids_batch = None
        self.scale_factors_batch = None

        self.max_n_func = None
        self.L = None
        self.r0 = None
        self.r1 = None
        self.r2 = None

    def batch_to(self, device):
        self.device = device
        if self.G0 is not None:
            self.G0 = self.G0.to(device=device, dtype=self.dtype)
        if self.G1 is not None:
            self.G1 = self.G1.to(device=device, dtype=self.dtype)
        if self.G2 is not None:
            self.G2 = self.G2.to(device=device, dtype=self.dtype)
        if self.G3 is not None:
            self.G3 = self.G3.to(device=device, dtype=self.dtype)
        if self.centroids_batch is not None:
            self.centroids_batch = self.centroids_batch.to(device=device, dtype=self.dtype)
        if self.scale_factors_batch is not None:
            self.scale_factors_batch = self.scale_factors_batch.to(device=device, dtype=self.dtype)

    @staticmethod
    def _build_namespaced_dataset_views(base_ds_map: dict, base_link_names: List[str], namespaces: List[str]) -> List[dict]:
        ds_list = []
        for ns in namespaces:
            for link in base_link_names:
                ds_ns = dict(base_ds_map[link])
                ds_ns["file_name"] = ns + link
                ds_list.append(ds_ns)
        return ds_list

    def train_robot(
        self,
        link_names: Union[List[str], str],
        n_func: int,
        ranks: tuple,
        iters: int,
        batch_size: int = 65_536,
        robot_name='',
        debug=False,
        namespaces=None,
        method: str = "adam",
        lr: float = 5e-3,
        ridge: float = 1e-6,
        sample_weights=None,
        **dataset_kwargs,
    ):
        if isinstance(link_names, str):
            link_names = [link_names]
        if not link_names:
            raise ValueError("link_names must be a non-empty list")
        if namespaces is None:
            namespaces = []
        elif isinstance(namespaces, str):
            namespaces = [namespaces]
        if not namespaces:
            namespaces = [""]

        link_names_ns = [ns + ln for ns in namespaces for ln in link_names]

        cfg_w = TTRobotCfg(
            run=True,
            method=method,
            n_func=n_func,
            ranks=ranks,
            iters=iters,
            batch_size=batch_size,
            lr=lr,
            ridge=ridge,
            sample_weights=sample_weights,
        )
        cfg = TrainConfig(links_to_train=link_names_ns, debug=debug, tt_robot=cfg_w)
        trainer = SDFTrain(device=self.device, dtype=self.dtype)
        trainer.debug = debug
        trainer.init_robot_folder(self.ws_path, robot_name=robot_name)

        base_ds_map = {}
        for link in link_names:
            base_ds_map[link] = trainer.create_dataset(link, robot_name=robot_name, **dataset_kwargs)

        ds_list = self._build_namespaced_dataset_views(base_ds_map, list(link_names), list(namespaces))
        print(
            f"[DATASET] base datasets ready: {len(base_ds_map)} | "
            f"namespaced views in memory: {len(ds_list)}"
        )

        model_folder: ModelHandler = getattr(trainer, robot_name + trainer.folder_model)
        model_folder.create_tt_robot(
            list_ds=ds_list,
            cfg=cfg,
            model_name=robot_name,
            device=self.device,
            dtype=self.dtype,
        )

        CommonSdfMethods.init_robot_folder(self, self.ws_path, robot_name=robot_name)

    def add_robots(self, robot_model_name: Union[List[str], str], namespace='', robot_name='', **kwargs):
        if isinstance(robot_model_name, str):
            robot_model_name = [robot_model_name]
        for r_name in robot_model_name:
            self.add_model(r_name, TtRobotModel.file_suffix, namespace, robot_name=robot_name, **kwargs)

    def set_ordered_batch_params(self, robot_model_name: str):
        m: TtRobotModel = getattr(self, robot_model_name + self.model_extension)

        if m.n_func is None:
            raise ValueError("TtRobotModel.n_func is None")
        self.max_n_func = int(m.n_func)
        self.set_number_of_functions(self.max_n_func)
        self.set_points_domain(m.domain_min, m.domain_max)

        if m.G0 is None or m.G1 is None or m.G2 is None or m.G3 is None:
            raise ValueError("Missing TT robot cores (G0/G1/G2/G3).")

        self.G0 = self._to_device_dtype(m.G0)
        self.G1 = self._to_device_dtype(m.G1)
        self.G2 = self._to_device_dtype(m.G2)
        self.G3 = self._to_device_dtype(m.G3)

        if self.G0.dim() != 3 or self.G1.dim() != 3 or self.G2.dim() != 3 or self.G3.dim() != 3:
            raise ValueError("TT robot cores must be 3D tensors.")

        _, L, r0 = self.G0.shape
        r0_g1, N, r1 = self.G1.shape
        r1_g2, N2, r2 = self.G2.shape
        r2_g3, N3, one = self.G3.shape

        if one != 1 or r0_g1 != r0 or r1_g2 != r1 or r2_g3 != r2 or N != N2 or N2 != N3:
            raise ValueError("Incoherent TT robot core shapes.")

        self.L = int(L)
        self.r0 = int(r0)
        self.r1 = int(r1)
        self.r2 = int(r2)

        if len(m.links_centroids) != self.L or len(m.links_scale_factors) != self.L:
            raise ValueError(
                f"links_centroids/links_scale_factors length must be L={self.L}. "
                f"Got {len(m.links_centroids)} and {len(m.links_scale_factors)}"
            )

        cent = torch.stack(
            [torch.as_tensor(c, device=self.device, dtype=self.dtype).reshape(3) for c in m.links_centroids],
            dim=0,
        )

        sca = torch.stack(
            [torch.as_tensor(s, device=self.device, dtype=self.dtype).reshape(1) for s in m.links_scale_factors],
            dim=0,
        ).reshape(self.L, 1, 1)

        self.centroids_batch = cent.unsqueeze(1)
        self.scale_factors_batch = sca

        self.batch_to(self.device)

    @torch.no_grad()
    def inference_link_batch(self, points: torch.Tensor, get_grad: bool = False, get_min: bool = False,
                             forward_tensor: torch.Tensor = None):
        if points.dim() != 3:
            raise ValueError("Expected points shape (L,P,3)")

        if self.G0 is None or self.G1 is None or self.G2 is None or self.G3 is None:
            raise ValueError("TT robot model not initialized. Call set_ordered_batch_params.")

        points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=forward_tensor) if forward_tensor is not None else points

        link_length = self.G0.shape[1]
        point_length = points.shape[1]

        if points.shape[0] != link_length:
            raise ValueError(f"Mismatch: points has L={points.shape[0]} but model has L={link_length}")

        pts_scaled = ((points - self.centroids_batch) / self.scale_factors_batch).reshape(-1, 3)
        r1 = (self.domain_max - self.domain_min) / 2
        x_in, d_sphere, outside = spherical_projection(pts_scaled, r1=r1)

        p = self.normalize_points(x_in).clamp_(0.0, 1.0)
        Bx, dBx = self.build_bernstein_t(p[:, 0], use_derivative=get_grad)
        By, dBy = self.build_bernstein_t(p[:, 1], use_derivative=get_grad)
        Bz, dBz = self.build_bernstein_t(p[:, 2], use_derivative=get_grad)

        N = Bx.shape[1]
        Bx = Bx.reshape(link_length, point_length, N)
        By = By.reshape(link_length, point_length, N)
        Bz = Bz.reshape(link_length, point_length, N)

        g0 = self.G0[0]
        G1 = self.G1
        G2 = self.G2
        G3 = self.G3
        G3_red = G3[..., 0]

        Bx_core = torch.einsum('lpn,anr->lpar', Bx, G1)
        By_core = torch.einsum('lpn,anr->lpar', By, G2)
        Bz_core = torch.einsum('lpn,an->lpa', Bz, G3_red)

        T1 = torch.einsum('la,lpar->lpr', g0, Bx_core)
        T2 = torch.einsum('lpa,lpar->lpr', T1, By_core)

        sdf_inner = (T2 * Bz_core).sum(dim=-1)
        sdf = sdf_inner + d_sphere.reshape(link_length, point_length)
        sdf = sdf * self.scale_factors_batch.reshape(link_length, 1)

        if get_grad:
            dBx = dBx.reshape(link_length, point_length, N)
            dBy = dBy.reshape(link_length, point_length, N)
            dBz = dBz.reshape(link_length, point_length, N)

            dBx_core = torch.einsum('lpn,anr->lpar', dBx, G1)
            dT1_dx = torch.einsum('la,lpar->lpr', g0, dBx_core)
            dT2_dx = torch.einsum('lpa,lpar->lpr', dT1_dx, By_core)
            dfdx = (dT2_dx * Bz_core).sum(dim=-1)

            dBy_core = torch.einsum('lpn,anr->lpar', dBy, G2)
            dT2_dy = torch.einsum('lpa,lpar->lpr', T1, dBy_core)
            dfdy = (dT2_dy * Bz_core).sum(dim=-1)

            dBz_core = torch.einsum('lpn,an->lpa', dBz, G3_red)
            dfdz = (T2 * dBz_core).sum(dim=-1)

            g_in = torch.stack([dfdx, dfdy, dfdz], dim=-1)
            g_in = g_in / (self.domain_max - self.domain_min)

            g_corr = correct_sphere_gradient(
                points_scaled=pts_scaled,
                g_in=g_in.reshape(-1, 3),
                outside=outside,
                r1=r1,
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
    def visualize_gradient_descent(self, initial_points: torch.Tensor, forward_dict: dict = None,
                                   num_of_iteration: int = 10, epsilon: float = 1e-3,
                                   mesh_color: str = "#302E2E", mesh_opacity: float = 1.0,
                                   sdf_color: str = "#c9c9bf", sdf_opacity: float = 0.4):
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
            mesh_color=mesh_color,
            mesh_opacity=mesh_opacity,
            color=sdf_color,
            opacity=sdf_opacity,
            gradient_descent_points=all_pts_world,
        )

    def visualize_scene(self, forward_as_dict: dict = None, fk_tensor: torch.Tensor = None,
                        links_as_mesh: bool = True,
                        mesh_link_names: List[str] = None,
                        links_as_rdf: bool = True,
                        rdf_link_names: List[str] = None,
                        additional_point_cloud: torch.Tensor = torch.empty((0, 3), device='cpu'),
                        cartesian_frame_pose: np.ndarray = None,
                        nbData: int = 128,
                        sigma_smooth: float = 1.0,
                        opacity: float = 0.4,
                        color: str = "#c9c9bf",
                        mesh_color: str = "#302E2E",
                        mesh_opacity: float = 1.0,
                        gradient_descent_points: torch.Tensor = None):
        if self.G0 is None:
            raise ValueError("TT robot model not initialized. Call set_ordered_batch_params.")

        if forward_as_dict is None:
            if fk_tensor is None:
                raise ValueError("forward_as_dict or fk_tensor required")
            forward_as_dict = {f"link_{i}": fk_tensor[i] for i in range(fk_tensor.shape[0])}

        super().visualize_scene(
            forward_as_dict=forward_as_dict,
            links_as_mesh=links_as_mesh,
            mesh_link_names=mesh_link_names,
            additional_point_cloud=additional_point_cloud,
            cartesian_frame_pose=cartesian_frame_pose,
            xy_offset=(0.0, 0.0),
            mesh_color=mesh_color,
            mesh_opacity=mesh_opacity,
        )

        if fk_tensor is None:
            fk_list_t = []
            for v in forward_as_dict.values():
                if isinstance(v, torch.Tensor):
                    fk_list_t.append(v.to(device=self.device, dtype=self.dtype))
                else:
                    fk_list_t.append(torch.as_tensor(v, device=self.device, dtype=self.dtype))
            fk_tensor = torch.stack(fk_list_t, dim=0)

        T_list = []
        for t in fk_tensor:
            if isinstance(t, torch.Tensor):
                t_np = t.detach().cpu().numpy()
            else:
                t_np = np.asarray(t)
            T_list.append(t_np)

        link_names = list(forward_as_dict.keys())
        if len(link_names) != self.L:
            raise ValueError(
                f"forward_as_dict has {len(link_names)} links, but TT model has L={self.L}. "
                "Use a dictionary ordered and aligned with model links."
            )
        link_to_idx = {ln: i for i, ln in enumerate(link_names)}
        if rdf_link_names is None:
            rdf_indices = list(range(self.L))
        else:
            missing = [ln for ln in rdf_link_names if ln not in link_to_idx]
            if missing:
                raise ValueError(f"rdf_link_names contains unknown links: {missing}")
            rdf_indices = [link_to_idx[ln] for ln in rdf_link_names]

        if self.centroids_batch is None or self.scale_factors_batch is None:
            raise ValueError("Call set_ordered_batch_params() before building ds list.")
        cent = self.centroids_batch.reshape(self.L, 3)
        sca = self.scale_factors_batch.reshape(self.L)
        ds_list = [
            {
                "mesh_scale_factor": sca[i],
                "mesh_centroid_offset": cent[i],
            }
            for i in range(self.L)
        ]

        mesh_robot = None
        if links_as_rdf and len(rdf_indices) > 0:
            G0_sel = self.G0[:, rdf_indices, :]
            ds_sel = [ds_list[i] for i in rdf_indices]
            T_sel = [T_list[i] for i in rdf_indices]

            mesh_robot = tt4d_sdf_to_mesh_robot(
                G0=G0_sel,
                G1=self.G1,
                G2=self.G2,
                G3=self.G3,
                list_ds=ds_sel,
                T_list=T_sel,
                nbData=nbData,
                domain_min=float(self.domain_min),
                domain_max=float(self.domain_max),
                bernstein_matrix_1d=self.build_bernstein_t,
                sigma_smooth=sigma_smooth,
            )

        if mesh_robot is not None:
            from src.utils.MeshUtils import trimesh_to_pyvista
            mesh_pv = trimesh_to_pyvista(mesh_robot, np.eye(4))
            self._visualizer.add_mesh(mesh_pv, opacity=opacity, color=color)
            self._visualizer.incornicia_mesh(mesh_pv, linewidth=2, color='black')

        if gradient_descent_points is not None and gradient_descent_points.numel() > 0:
            L, T, _ = gradient_descent_points.shape
            link_colors = ["red", "blue", "green", "orange", "magenta", "cyan", "yellow", "purple", "brown", "pink"]
            for i in range(L):
                pts = gradient_descent_points[i]
                pts = pts.detach().cpu().numpy() if pts.is_cuda else pts.detach().numpy()
                color_i = link_colors[i % len(link_colors)]
                self._visualizer.add_pointcloud(pts, color=color_i, point_size=7)
                self._visualizer.add_polyline(pts, color=color_i, line_width=2)

        self._visualizer.show()
