import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import List, Union, Literal, Optional
from src.core.common import CommonSdfMethods
from src.core.assets.entities.models import CpLinkModel, CpRobotModel
from src.core.train.train_config import TrainConfig, CPLinkCfg, CPRobotCfg
from src.core.sdf_creator import SDFTrain
from src.core.math.projection_point import spherical_projection, correct_sphere_gradient


class RDF_3D_CP(CommonSdfMethods):
    def __init__(self, device='cuda', dtype=torch.float32):
        CommonSdfMethods.__init__(self, projection_method='sphere', device=device, dtype=dtype)

        # single cache (per debug / access rapido)
        self._cp_cache = {}  # link_name -> dict(A,B,C,lam,N,R)

        # batch tensors
        self.A_batch   = None  # (L,N,Rmax)
        self.B_batch   = None
        self.C_batch   = None
        self.lam_batch = None  # (L,Rmax)

        self.centroids_batch     = None  # (L,1,3)
        self.scale_factors_batch = None  # (L,1,1)

        self.max_n_func = None
        self.R_max      = None

    def train_links(self, link_names:Union[List[str], str],  n_func:int, ranks:int, iters:int, robot_name='', debug=False):
        cfg_w = CPLinkCfg(run=True, n_func=n_func, iters=iters, rank=ranks)
        cfg = TrainConfig(links_to_train=link_names, debug=debug, cp_link=cfg_w )
        trainer = SDFTrain()
        trainer.init_robot_folder(self.ws_path, robot_name=robot_name)
        trainer.create_model(cfg, robot_name=robot_name)
        CommonSdfMethods.init_robot_folder(self, self.ws_path, robot_name=robot_name)
    

    # --------- util ----------
    def batch_to(self, device):
        self.device = device
        if self.A_batch is not None:   self.A_batch = self.A_batch.to(device)
        if self.B_batch is not None:   self.B_batch = self.B_batch.to(device)
        if self.C_batch is not None:   self.C_batch = self.C_batch.to(device)
        if self.lam_batch is not None: self.lam_batch = self.lam_batch.to(device)
        if self.centroids_batch is not None: self.centroids_batch = self.centroids_batch.to(device)
        if self.scale_factors_batch is not None: self.scale_factors_batch = self.scale_factors_batch.to(device)

    @staticmethod
    def _normalize_namespaces(namespace: str = "", namespaces: Optional[Union[List[str], str]] = None) -> List[str]:
        if namespaces is None:
            namespaces = [namespace]
        elif isinstance(namespaces, str):
            namespaces = [namespaces]
        else:
            namespaces = list(namespaces)

        if len(namespaces) == 0:
            namespaces = [""]

        return namespaces

    @staticmethod
    def compose_namespaced_link_names(link_names: Union[List[str], str], namespaces: Optional[Union[List[str], str]] = None) -> List[str]:
        if isinstance(link_names, str):
            link_names = [link_names]

        namespaces_l = RDF_3D_CP._normalize_namespaces(namespaces=namespaces)
        return [ns + ln for ns in namespaces_l for ln in link_names]

    def add_models(
        self,
        link_names: Union[List[str], str],
        namespace: str = "",
        namespaces: Optional[Union[List[str], str]] = None,
        robot_name: str = "",
        **kwargs,
    ):
        if isinstance(link_names, str):
            link_names = [link_names]

        namespaces_l = self._normalize_namespaces(namespace=namespace, namespaces=namespaces)
        for ns in namespaces_l:
            for link_name in link_names:
                self.add_model(link_name, CpLinkModel.file_suffix, ns, robot_name, **kwargs)
        


    def _pack_rank_pad(self, ABC_lam_list):
        # ABC_lam_list: [(A,B,C,lam), ...] con A:(N,Ri) e lam:(Ri,)
        Rmax = max(A.shape[1] for (A,_,_,_) in ABC_lam_list)
        A_b, B_b, C_b, lam_b = [], [], [], []

        for (A,B,C,lam) in ABC_lam_list:
            N, R = A.shape
            if R < Rmax:
                pad = Rmax - R
                z = torch.zeros((N, pad), device=self.device, dtype=self.dtype)
                A = torch.cat([A, z], dim=1)
                B = torch.cat([B, z], dim=1)
                C = torch.cat([C, z], dim=1)
                lam = torch.nn.functional.pad(lam, (0, pad))
            A_b.append(A.unsqueeze(0))
            B_b.append(B.unsqueeze(0))
            C_b.append(C.unsqueeze(0))
            lam_b.append(lam.unsqueeze(0))

        return torch.cat(A_b, 0), torch.cat(B_b, 0), torch.cat(C_b, 0), torch.cat(lam_b, 0), Rmax

    # ------------------------------------------------------------
    # CP single-link inference (analogo a inference_link weights)
    # ------------------------------------------------------------
    @torch.no_grad()
    def inference_link(self, link_name: str, points: torch.Tensor, get_grad: bool = False, get_min: bool = False, fk_matrix: torch.Tensor = None,):
        if fk_matrix is not None:
            points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=fk_matrix)

        model: CpLinkModel = getattr(self, link_name + self.model_extension)
        model.to(points.device, points.dtype)

        self.set_number_of_functions(model.n_func)
        self.set_points_domain(domain_min=model.domain_min, domain_max=model.domain_max)

        pts_scaled = (points - model.centroid_offset) / model.scale_factor
        r1 = (model.domain_max - model.domain_min) / 2

        x_in, d_sphere, outside = spherical_projection(pts_scaled, r1=r1)

        p = self.normalize_points(x_in)
        tx, ty, tz = p[:, 0], p[:, 1], p[:, 2]

        Bx, dBx = self.build_bernstein_t(tx, use_derivative=get_grad)
        By, dBy = self.build_bernstein_t(ty, use_derivative=get_grad)
        Bz, dBz = self.build_bernstein_t(tz, use_derivative=get_grad)

        Sx = Bx @ model.A
        Sy = By @ model.B
        Sz = Bz @ model.C

        lam = model.lamd.reshape(-1)
        sdf = (Sx * Sy * Sz) @ lam
        sdf = sdf + d_sphere
        sdf = sdf * model.scale_factor

        if get_grad:
            dSx = dBx @ model.A
            dSy = dBy @ model.B
            dSz = dBz @ model.C

            gx = (dSx * Sy  * Sz) @ lam
            gy = (Sx  * dSy * Sz) @ lam
            gz = (Sx  * Sy  * dSz) @ lam

            den = (model.domain_max - model.domain_min)
            den = torch.as_tensor(den, device=points.device, dtype=points.dtype)

            g_in = torch.stack([gx, gy, gz], dim=-1) / den
            d_sdf = correct_sphere_gradient(points_scaled=pts_scaled, g_in=g_in, outside=outside, r1=r1)
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

        model: CpLinkModel = getattr(self, link_name + self.model_extension)
        model.to(points.device, points.dtype)
        self.set_number_of_functions(model.n_func)
        self.set_points_domain(domain_min=model.domain_min, domain_max=model.domain_max)

        pts_scaled = (points - model.centroid_offset) / model.scale_factor
        p = self.normalize_points(pts_scaled)
        tx, ty, tz = p[:, 0], p[:, 1], p[:, 2]
        Bx, dBx = self.build_bernstein_t(tx, use_derivative=get_grad)
        By, dBy = self.build_bernstein_t(ty, use_derivative=get_grad)
        Bz, dBz = self.build_bernstein_t(tz, use_derivative=get_grad)
        Sx = Bx @ model.A
        Sy = By @ model.B
        Sz = Bz @ model.C
        lam = model.lamd.reshape(-1)
        sdf = ((Sx * Sy * Sz) @ lam) * model.scale_factor

        if get_grad:
            dSx = dBx @ model.A
            dSy = dBy @ model.B
            dSz = dBz @ model.C
            den = torch.as_tensor(model.domain_max - model.domain_min, device=points.device, dtype=points.dtype)
            d_sdf = torch.stack([
                (dSx * Sy * Sz) @ lam,
                (Sx * dSy * Sz) @ lam,
                (Sx * Sy * dSz) @ lam,
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

    # ------------------------------------------------------------
    # Batch params (analogo a RDF_Weights.set_ordered_batch_params)
    # ------------------------------------------------------------
    def set_ordered_batch_params(
        self,
        link_names: Union[List[str], str],
        namespaces: Optional[Union[List[str], str]] = None,
    ):
        if isinstance(link_names, str):
            link_names = [link_names]
        if namespaces is not None:
            link_names = self.compose_namespaced_link_names(link_names, namespaces=namespaces)

        ABC_lam_list = []
        n_functions  = []
        n_by_link = {}

        list_centroids = [
            getattr(self, link + self.model_extension).centroid_offset.unsqueeze(0)
            for link in link_names
        ]
        list_scales = [
            getattr(self, link + self.model_extension).scale_factor.unsqueeze(0)
            for link in link_names
        ]

        for link in link_names:
            m: CpLinkModel = getattr(self, link + self.model_extension)
            m.to(self.device, self.dtype)

            if m.A is None or m.B is None or m.C is None or m.lamd is None:
                raise ValueError(f"CP factors mancanti in '{link}' (A/B/C/lamd).")

            A = m.A
            B = m.B
            C = m.C
            lam = m.lamd.reshape(-1)

            N = A.shape[0]
            if B.shape[0] != N or C.shape[0] != N:
                raise ValueError(f"N incoerente tra A/B/C per link '{link}'.")

            n_f = int(m.n_func) if m.n_func is not None else int(N)
            n_functions.append(n_f)
            n_by_link[link] = n_f

            self._cp_cache[link] = {"A": A, "B": B, "C": C, "lam": lam, "N": N, "R": A.shape[1]}
            ABC_lam_list.append((A, B, C, lam))

        if len(set(n_functions)) != 1:
            raise ValueError(
                "set_ordered_batch_params richiede lo stesso n_func su tutti i link del batch. "
                f"Trovati: {n_by_link}. "
                "Raggruppa i link per n_func (es. un batch per N=16 e un batch per N=8)."
            )

        self.max_n_func = max(n_functions)
        super().set_number_of_functions(self.max_n_func)

        self.A_batch, self.B_batch, self.C_batch, self.lam_batch, self.R_max = self._pack_rank_pad(ABC_lam_list)

        self.centroids_batch = torch.cat(list_centroids, dim=0).unsqueeze(1)                # (L,1,3)
        self.scale_factors_batch = torch.cat(list_scales, dim=0).reshape(-1, 1, 1)          # (L,1,1)

        self.batch_to(self.device)
        
    # ------------------------------------------------------------
    # CP robot batched (analogo a inference_link_batch weights)
    # ------------------------------------------------------------
    @torch.no_grad()
    def inference_link_batch(self, points: torch.Tensor, get_grad=False, get_min=False, forward_tensor: torch.Tensor = None):
        if points.dim() != 3:
            raise ValueError("Expected points shape (L,P,3)")

        # self.batch_to(points.device)
        # super().set_number_of_functions(self.max_n_func)

        points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=forward_tensor) if forward_tensor is not None else points

        link_length  = self.A_batch.shape[0]
        point_length = points.shape[1]

        pts_scaled = ((points - self.centroids_batch) / self.scale_factors_batch).reshape(-1, 3)
        r1 = (self.domain_max - self.domain_min) / 2
        x_in, d_sphere, outside = spherical_projection(pts_scaled, r1=r1)

        p = self.normalize_points(x_in)
        Bx, dBx = self.build_bernstein_t(p[:, 0], use_derivative=get_grad)
        By, dBy = self.build_bernstein_t(p[:, 1], use_derivative=get_grad)
        Bz, dBz = self.build_bernstein_t(p[:, 2], use_derivative=get_grad)

        N = Bx.shape[1]
        Bx = Bx.reshape(link_length, point_length, N)
        By = By.reshape(link_length, point_length, N)
        Bz = Bz.reshape(link_length, point_length, N)

        Sx = torch.matmul(Bx, self.A_batch)   # (L,P,R)
        Sy = torch.matmul(By, self.B_batch)
        Sz = torch.matmul(Bz, self.C_batch)

        lam = self.lam_batch.unsqueeze(1)     # (L,1,R)
        sdf = (Sx * Sy * Sz * lam).sum(dim=-1)
        sdf = sdf + d_sphere.reshape(link_length, point_length)
        sdf = sdf * self.scale_factors_batch.reshape(link_length, 1)

        if get_grad:
            dBx = dBx.reshape(link_length, point_length, N)
            dBy = dBy.reshape(link_length, point_length, N)
            dBz = dBz.reshape(link_length, point_length, N)

            dSx = torch.matmul(dBx, self.A_batch)
            dSy = torch.matmul(dBy, self.B_batch)
            dSz = torch.matmul(dBz, self.C_batch)

            gx = (dSx * Sy  * Sz * lam).sum(dim=-1)
            gy = (Sx  * dSy * Sz * lam).sum(dim=-1)
            gz = (Sx  * Sy  * dSz * lam).sum(dim=-1)

            g_in = torch.stack([gx, gy, gz], dim=-1)  # (L,P,3)

            den = (self.domain_max - self.domain_min)
            den = torch.as_tensor(den, device=points.device, dtype=points.dtype)

            g_in = g_in / den
            g_corr = correct_sphere_gradient(points_scaled=pts_scaled, g_in=g_in.reshape(-1,3), outside=outside, r1=r1)
            d_sdf = g_corr.reshape(link_length, point_length, 3)
        else:
            d_sdf = torch.zeros((link_length, point_length, 3), device=points.device, dtype=points.dtype)

        if get_min:
            sdf_min, idx = torch.min(sdf, dim=1)  # (L,)
            ar = torch.arange(link_length, device=points.device)
            pts_min = points[ar, idx]
            grad_min = d_sdf[ar, idx] if get_grad else None
            return sdf_min, grad_min, pts_min

        return sdf, d_sdf, points
    


    @torch.no_grad()
    def gradient_descent_batch(self, points_link_frame: torch.Tensor, num_of_iteration: int = 5, epsilon: float = 1e-3):
        """
        points_link_frame: (L, P, 3)  (punti in frame link)
        ritorna: (L, 1, 3) punti proiettati (uno per link) come nel weights
        """
        if points_link_frame.dim() != 3 or points_link_frame.shape[-1] != 3:
            raise ValueError(f"Expected points shape (L,P,3), got {tuple(points_link_frame.shape)}")

        pts = points_link_frame.clone()
        self.batch_to(pts.device)

        for _ in range(num_of_iteration):
            dist, grad, pts_min = self.inference_link_batch(pts, get_grad=True, get_min=True)
            # dist: (L,), grad: (L,3), pts_min: (L,3)

            gnorm = grad.norm(dim=1, keepdim=True).clamp_min(1e-9)
            step = dist.unsqueeze(1) * grad / gnorm          # (L,3)
            pts_next = pts_min - step                        # (L,3)

            # prepara per iterazione successiva: (L,1,3)
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
        """
        initial_points: (L, P, 3) o (L,1,3). Nel tuo test usi (L,1,3)
        Visualizza la traiettoria in WORLD.
        """
        if initial_points.dim() != 3 or initial_points.shape[-1] != 3:
            raise ValueError(f"Expected initial_points shape (L,P,3), got {tuple(initial_points.shape)}")

        pts = initial_points.clone()
        self.batch_to(pts.device)

        all_pts = [pts.clone()]  # lista di (L,P,3) o (L,1,3)

        for _ in range(num_of_iteration):
            dist, grad, pts_min = self.inference_link_batch(pts, get_grad=True, get_min=True)

            gnorm = grad.norm(dim=1, keepdim=True).clamp_min(1e-9)
            step = dist.unsqueeze(1) * grad / gnorm
            pts_next = (pts_min - step).unsqueeze(1)  # (L,1,3)

            all_pts.append(pts_next.clone())
            pts = pts_next

            if (dist.abs() <= epsilon).all():
                break

        # stack: (T,L,1,3) -> (L,T,3)
        all_pts = torch.stack(all_pts, dim=0)              # (T,L,*,3)
        all_pts = all_pts[:, :, 0:1, :]                    # forza 1 punto per iter
        all_pts = all_pts.permute(1, 0, 2, 3)              # (L,T,1,3)
        all_pts = all_pts.reshape(all_pts.shape[0], -1, 3) # (L,T,3)

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
            mesh_link_names=mesh_link_names,
            links_as_rdf=rdf_link_names is not None,
            rdf_link_names=rdf_link_names,
            gradient_descent_points=all_pts_world,
            mesh_color=mesh_color,
            mesh_opacity=mesh_opacity,
            rdf_color=rdf_color,
            rdf_opacity=rdf_opacity,
        )

    def to_mesh(
        self,
        model_name,
        type: Literal['pyvista','trimesh'],
        debug=False,
        nbData: int = 128,
        sigma_smooth: float = 1.0,
        batch_points: int = 50_000,
        largest_component: bool = False,
    ):
        from src.utils.MeshUtils import trimesh_to_pyvista
        from src.utils.sdf_utils import cp_sdf_to_mesh  # dove l'hai messa tu

        model: CpLinkModel = getattr(self, model_name + self.model_extension)
        n_func = int(model.n_func) if model.n_func is not None else None
        if n_func is None and model.A is not None:
            n_func = int(model.A.shape[0])
        if n_func is not None:
            self.set_number_of_functions(n_func)
        self.set_points_domain(model.domain_min, model.domain_max)

        model.to(self.device, self.dtype)

        mesh = cp_sdf_to_mesh(
            A=model.A, B=model.B, C=model.C, lam=model.lamd,
            nbData=nbData,
            domain_min=model.domain_min, domain_max=model.domain_max,
            scaling_factor=model.scale_factor,
            centroid_offset=model.centroid_offset,
            sigma_smooth=sigma_smooth,
            batch_points=batch_points,
            bernstein_matrix_1d=self.build_bernstein_t,
        )

        if mesh is None:
            print(
                f"[WARN][CP] Nessun zero-level set trovato per '{model_name}' "
                f"(probabile collasso o shift del campo)."
            )
            if debug:
                input("CP mesh is None. Premi INVIO per continuare...")
            return None

        if largest_component:
            parts = mesh.split(only_watertight=False)
            if len(parts) > 1:
                mesh = max(parts, key=lambda part: len(part.faces))

        mesh.show() if debug else None
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
                        rdf_opacity: float = 0.3,
                        rdf_mesh_resolution: int = 128,
                        rdf_mesh_smoothing: float = 1.0,
                        rdf_mesh_resolution_by_link: dict = None,
                        rdf_mesh_smoothing_by_link: dict = None,
                        rdf_mesh_largest_component_by_link: dict = None):

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
                    nb_data = (
                        rdf_mesh_resolution_by_link.get(link, rdf_mesh_resolution)
                        if rdf_mesh_resolution_by_link is not None else rdf_mesh_resolution
                    )
                    sigma_smooth = (
                        rdf_mesh_smoothing_by_link.get(link, rdf_mesh_smoothing)
                        if rdf_mesh_smoothing_by_link is not None else rdf_mesh_smoothing
                    )
                    largest_component = (
                        rdf_mesh_largest_component_by_link.get(link, False)
                        if rdf_mesh_largest_component_by_link is not None else False
                    )
                    m = self.to_mesh(
                        link,
                        type='trimesh',
                        nbData=nb_data,
                        sigma_smooth=sigma_smooth,
                        largest_component=largest_component,
                    )
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
