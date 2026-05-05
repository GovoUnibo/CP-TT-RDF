import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import List, Union, Literal, Optional
from src.core.common import CommonSdfMethods
from src.core.assets.entities.models import CpLinkModel, CpRobotModel
from src.core.assets.ModelHandler import ModelHandler
from src.core.train.train_config import TrainConfig, CPLinkCfg, CPRobotCfg
from src.core.sdf_creator import SDFTrain
from src.core.math.projection_point import spherical_projection, correct_sphere_gradient
from src.utils.sdf_utils import cp4d_sdf_to_mesh_robot


class RDF_4D_CP(CommonSdfMethods):
    def __init__(self, device='cuda', dtype=torch.float32):
        CommonSdfMethods.__init__(self, projection_method='sphere', device=device, dtype=dtype)

        # cached tensors for batched inference
        self.A = None
        self.B = None
        self.C = None
        self.V = None
        self.cp_lambda = None
        self.gamma = None

        self.centroids_batch = None
        self.scale_factors_batch = None

        self.max_n_func = None
        self.L = None
        self.R = None
        self.link_names_ordered = None

    def batch_to(self, device):
        self.device = device
        if self.A is not None: self.A = self.A.to(device=device, dtype=self.dtype)
        if self.B is not None: self.B = self.B.to(device=device, dtype=self.dtype)
        if self.C is not None: self.C = self.C.to(device=device, dtype=self.dtype)
        if self.V is not None: self.V = self.V.to(device=device, dtype=self.dtype)
        if self.cp_lambda is not None: self.cp_lambda = self.cp_lambda.to(device=device, dtype=self.dtype)
        if self.centroids_batch is not None: self.centroids_batch = self.centroids_batch.to(device=device, dtype=self.dtype)
        if self.scale_factors_batch is not None: self.scale_factors_batch = self.scale_factors_batch.to(device=device, dtype=self.dtype)

        if self.gamma is not None: self.gamma = self.gamma.to(device=device, dtype=self.dtype)
    
    
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
        ranks: int,
        iters: int,
        batch_size: int = 65_536,
        robot_name: str = '',
        debug: bool = False,
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

        cfg_w = CPRobotCfg(
            run=True,
            method=method,
            n_func=n_func,
            iters=iters,
            rank=ranks,
            batch_size=batch_size,
            lr=lr,
            ridge=ridge,
            sample_weights=sample_weights,
        )
        cfg = TrainConfig(links_to_train=link_names_ns, debug=debug, cp_robot=cfg_w )
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
        model_folder.create_cp_robot(
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
            self.add_model(r_name, CpRobotModel.file_suffix, namespace, robot_name=robot_name, **kwargs)
            
    def set_ordered_batch_params(self, robot_model_name: str):
        m: CpRobotModel = getattr(self, robot_model_name + self.model_extension)
        m.to(self.device, self.dtype)

        # dominio + n_func
        if m.n_func is None:
            raise ValueError("CpRobotModel.n_func is None")
        self.max_n_func = int(m.n_func)
        super().set_number_of_functions(self.max_n_func)
        self.set_points_domain(m.domain_min, m.domain_max)

        # fattori CP (global + V per-link)
        if m.cp_A is None or m.cp_B is None or m.cp_C is None or m.cp_V is None or m.cp_lambda is None:
            raise ValueError("Missing CP robot tensors (cp_A/cp_B/cp_C/cp_V/cp_lambda).")

        self.A = m.cp_A
        self.B = m.cp_B
        self.C = m.cp_C
        self.V = m.cp_V
        self.cp_lambda = m.cp_lambda.reshape(-1)

        self.gamma = (self.V * self.cp_lambda.unsqueeze(0)).unsqueeze(1).contiguous()

        N, R = self.A.shape
        if self.B.shape != (N, R) or self.C.shape != (N, R):
            raise ValueError(f"Incoherent CP shapes: A{self.A.shape}, B{self.B.shape}, C{self.C.shape}")
        if self.cp_lambda.numel() != R:
            raise ValueError(f"cp_lambda must have R={R} elements, got {self.cp_lambda.numel()}")
        if self.V.dim() != 2 or self.V.shape[1] != R:
            raise ValueError(f"V must be (L,R). Got {tuple(self.V.shape)}")

        self.L = int(self.V.shape[0])
        self.R = int(R)

        # per-link centroid + scale (DEVONO essere nello stesso ordine di V)
        if len(m.links_centroids) != self.L or len(m.links_scale_factors) != self.L:
            raise ValueError(
                f"links_centroids/links_scale_factors length must be L={self.L}. "
                f"Got {len(m.links_centroids)} and {len(m.links_scale_factors)}"
            )

        cent = torch.stack(
            [torch.as_tensor(c, device=self.device, dtype=self.dtype).reshape(3) for c in m.links_centroids],
            dim=0
        )  # (L,3)

        sca = torch.stack(
            [torch.as_tensor(s, device=self.device, dtype=self.dtype).reshape(1) for s in m.links_scale_factors],
            dim=0
        ).reshape(self.L, 1, 1)  # (L,1,1)

        self.centroids_batch = cent.unsqueeze(1)         # (L,1,3)
        self.scale_factors_batch = sca                   # (L,1,1)

        # super().set_number_of_functions(self.max_n_func)

        self.batch_to(self.device)

    def repeat_robot_by_namespaces(
        self,
        namespaces: Union[List[str], str],
        base_link_names: Optional[Union[List[str], str]] = None,
    ) -> List[str]:
        if self.V is None or self.centroids_batch is None or self.scale_factors_batch is None:
            raise ValueError("Call set_ordered_batch_params before repeat_robot_by_namespaces.")

        if isinstance(namespaces, str):
            namespaces = [namespaces]
        namespaces = list(namespaces)
        if len(namespaces) == 0:
            raise ValueError("namespaces must be a non-empty list")

        if base_link_names is not None and isinstance(base_link_names, str):
            base_link_names = [base_link_names]
        if base_link_names is not None:
            base_link_names = list(base_link_names)
            if len(base_link_names) != int(self.V.shape[0]):
                raise ValueError(
                    f"base_link_names length must be {int(self.V.shape[0])}, got {len(base_link_names)}"
                )

        repeat_count = len(namespaces)
        self.V = self.V.repeat(repeat_count, 1).contiguous()
        self.centroids_batch = self.centroids_batch.repeat(repeat_count, 1, 1).contiguous()
        self.scale_factors_batch = self.scale_factors_batch.repeat(repeat_count, 1, 1).contiguous()
        self.L = int(self.V.shape[0])

        self.gamma = (self.V * self.cp_lambda.unsqueeze(0)).unsqueeze(1).contiguous()

        if base_link_names is not None:
            self.link_names_ordered = [ns + ln for ns in namespaces for ln in base_link_names]
        else:
            n_base = int(self.L // repeat_count)
            self.link_names_ordered = [f"{ns}link_{i}" for ns in namespaces for i in range(n_base)]

        return self.link_names_ordered



    # ------------------------------------------------------------
    # Robot batched inference (stessa API di inference_link_batch)
    # ------------------------------------------------------------
    @torch.no_grad()
    def inference_link_batch(self, points: torch.Tensor, get_grad=False, get_min=False, forward_tensor: torch.Tensor = None):
        """
        points: (L,P,3) (punti già nel frame link oppure world se passi forward_tensor)
        ritorna:
          - sdf: (L,P) o (L,)
          - grad: (L,P,3) o (L,3) o None
          - pts: (L,P,3) o (L,3)
        """
        if points.dim() != 3:
            raise ValueError("Expected points shape (L,P,3)")

        # self.batch_to(points.device)
        

        points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=forward_tensor) if forward_tensor is not None else points

        link_length  = self.V.shape[0]
        point_length = points.shape[1]

        if points.shape[0] != link_length:
            raise ValueError(f"Mismatch: points has L={points.shape[0]} but model has L={link_length}")

        # --- scale + projection ---
        pts_scaled = ((points - self.centroids_batch) / self.scale_factors_batch).reshape(-1, 3)
        r1 = (self.domain_max - self.domain_min) / 2
        x_in, d_sphere, outside = spherical_projection(pts_scaled, r1=r1)

        # --- basis 1D ---
        p = self.normalize_points(x_in).clamp_(0.0, 1.0)
        Bx, dBx = self.build_bernstein_t(p[:, 0], use_derivative=get_grad)  # (LP,N)
        By, dBy = self.build_bernstein_t(p[:, 1], use_derivative=get_grad)
        Bz, dBz = self.build_bernstein_t(p[:, 2], use_derivative=get_grad)

        N = Bx.shape[1]
        Bx = Bx.reshape(link_length, point_length, N)
        By = By.reshape(link_length, point_length, N)
        Bz = Bz.reshape(link_length, point_length, N)

        # --- CP contraction ---
        Sx = torch.matmul(Bx, self.A)   # (L,P,R)
        Sy = torch.matmul(By, self.B)
        Sz = torch.matmul(Bz, self.C)

        sdf = (Sx * Sy * Sz * self.gamma).sum(dim=-1)

        # ⚠️ usa SOLO se anche il tuo training usava questo termine
        sdf = sdf + d_sphere.reshape(link_length, point_length)

        sdf = sdf * self.scale_factors_batch.reshape(link_length, 1)

        if not get_grad:
            if get_min:
                sdf_min, idx = torch.min(sdf, dim=1)  # (L,)
                ar = torch.arange(link_length, device=points.device)
                pts_min = points[ar, idx]
                return sdf_min, None, pts_min
            return sdf, None, points

        # --- gradient ---
        dBx = dBx.reshape(link_length, point_length, N)
        dBy = dBy.reshape(link_length, point_length, N)
        dBz = dBz.reshape(link_length, point_length, N)

        dSx = torch.matmul(dBx, self.A)
        dSy = torch.matmul(dBy, self.B)
        dSz = torch.matmul(dBz, self.C)

        gx = (dSx * Sy  * Sz * self.gamma).sum(dim=-1)
        gy = (Sx  * dSy * Sz * self.gamma).sum(dim=-1)
        gz = (Sx  * Sy  * dSz * self.gamma).sum(dim=-1)

        g_in = torch.stack([gx, gy, gz], dim=-1)  # (L,P,3)

        den = torch.as_tensor((self.domain_max - self.domain_min), device=points.device, dtype=points.dtype)
        g_in = g_in / den

        g_corr = correct_sphere_gradient(
            points_scaled=pts_scaled,                 # (LP,3)
            g_in=g_in.reshape(-1, 3),                 # (LP,3)
            outside=outside,
            r1=r1
        ).reshape(link_length, point_length, 3)

        # chain scale: pts_scaled=(pts-c)/s -> d/dpts = (1/s)*d/dpts_scaled
        d_sdf = g_corr / self.scale_factors_batch     # (L,P,3)

        # se hai fatto sdf *= scale_factor, stessa convenzione sul grad
        d_sdf = d_sdf * self.scale_factors_batch

        if get_min:
            sdf_min, idx = torch.min(sdf, dim=1)  # (L,)
            ar = torch.arange(link_length, device=points.device)
            pts_min = points[ar, idx]
            grad_min = d_sdf[ar, idx]
            return sdf_min, grad_min, pts_min

        return sdf, d_sdf, points

    @torch.no_grad()
    def gradient_descent_batch(self, points_link_frame: torch.Tensor, num_of_iteration: int = 5, epsilon: float = 1e-3):
        """
        points_link_frame: (L, P, 3)  (punti in frame link)
        ritorna: (L, 1, 3) punti proiettati (uno per link)
        """
        if points_link_frame.dim() != 3 or points_link_frame.shape[-1] != 3:
            raise ValueError(f"Expected points shape (L,P,3), got {tuple(points_link_frame.shape)}")

        pts = points_link_frame.clone()
        self.batch_to(pts.device)

        for _ in range(num_of_iteration):
            dist, grad, pts_min = self.inference_link_batch(pts, get_grad=True, get_min=True)
            gnorm = grad.norm(dim=1, keepdim=True).clamp_min(1e-9)
            step = dist.unsqueeze(1) * grad / gnorm          # (L,3)
            pts_next = pts_min - step                        # (L,3)

            pts = pts_next.unsqueeze(1)  # prepara per iterazione successiva: (L,1,3)

            if (dist.abs() <= epsilon).all():
                break

        return pts

    @torch.no_grad()
    def visualize_gradient_descent(self, initial_points: torch.Tensor, forward_dict: dict = None,
                                   num_of_iteration: int = 10, epsilon: float = 1e-3,
                                   mesh_color: str = "#302E2E", mesh_opacity: float = 1.0,
                                   sdf_color: str = "#c9c9bf", sdf_opacity: float = 0.4):
        """
        initial_points: (L, P, 3) o (L,1,3)
        Visualizza la traiettoria in WORLD.
        """
        if initial_points.dim() != 3 or initial_points.shape[-1] != 3:
            raise ValueError(f"Expected initial_points shape (L,P,3), got {tuple(initial_points.shape)}")

        pts = initial_points.clone()
        self.batch_to(pts.device)

        all_pts = [pts.clone()]  # lista di (L,P,3)

        for _ in range(num_of_iteration):
            dist, grad, pts_min = self.inference_link_batch(pts, get_grad=True, get_min=True)

            gnorm = grad.norm(dim=1, keepdim=True).clamp_min(1e-9)
            step = dist.unsqueeze(1) * grad / gnorm
            pts_next = (pts_min - step).unsqueeze(1)  # (L,1,3)

            all_pts.append(pts_next.clone())
            pts = pts_next

            if (dist.abs() <= epsilon).all():
                break

        # (T,L,1,3) -> (L,T,3)
        all_pts = torch.stack(all_pts, dim=0)              # (T,L,1,3)
        all_pts = all_pts[:, :, 0:1, :]
        all_pts = all_pts.permute(1, 0, 2, 3).reshape(all_pts.shape[1], -1, 3)

        if forward_dict is None:
            raise ValueError("Serve forward_dict per visualizzare la GD.")
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

    def visualize_scene(self,
                        forward_as_dict: dict = None,
                        fk_tensor: torch.Tensor = None,
                        links_as_mesh: bool = True,
                        mesh_link_names: List[str] = None,
                        links_as_rdf: bool = True,
                        rdf_link_names: List[str] = None,
                        additional_point_cloud: torch.Tensor = torch.empty((0,3), device='cpu'),
                        cartesian_frame_pose: np.ndarray = None,
                        nbData: int = 128,
                        sigma_smooth: float = 1.0,
                        opacity: float = 0.4,
                        color: str = "#c9c9bf",
                        mesh_color: str = "#302E2E",
                        mesh_opacity: float = 1.0,
                        gradient_descent_points: torch.Tensor = None):
        """
        Visualizza mesh originali (stl) + mesh ricostruita via CP4D (SDF).
        """
        if self.A is None or self.V is None or self.cp_lambda is None:
            raise ValueError("Modello CP 4D non inizializzato. Chiama set_ordered_batch_params.")

        if forward_as_dict is None:
            if fk_tensor is None:
                raise ValueError("Serve forward_as_dict o fk_tensor per visualizzare.")
            forward_as_dict = {f"link_{i}": fk_tensor[i] for i in range(fk_tensor.shape[0])}

        # usa la scena comune (mesh, frame, punti aggiuntivi)
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

        # FK: prende da fk_tensor o dal dizionario
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
                f"forward_as_dict has {len(link_names)} links, but CP model has L={self.L}. "
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
            V_sel = self.V[rdf_indices, :]
            ds_sel = [ds_list[i] for i in rdf_indices]
            T_sel = [T_list[i] for i in rdf_indices]

            mesh_robot = cp4d_sdf_to_mesh_robot(
                A=self.A,
                B=self.B,
                C=self.C,
                lam=self.cp_lambda,
                V=V_sel,
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
            link_colors = ["red","blue","green","orange","magenta","cyan","yellow","purple","brown","pink"]
            for i in range(L):
                pts = gradient_descent_points[i]
                pts = pts.detach().cpu().numpy() if pts.is_cuda else pts.detach().numpy()
                color_i = link_colors[i % len(link_colors)]
                self._visualizer.add_pointcloud(pts, color=color_i, point_size=7)
                self._visualizer.add_polyline(pts, color=color_i, line_width=2)

        # Keep camera framing stable even when scene content/order changes.
        if hasattr(self._visualizer, "scene"):
            self._visualizer.scene.show_axes()
            self._visualizer.scene.reset_camera()

        self._visualizer.show()
