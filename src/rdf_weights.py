import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mesh_to_sdf


import numpy as np
import torch
from src.visu.plots_3d import plot_robot_isosurfaces, RobotSene
from typing import List, Union, Literal
import time
import tensorly as tl
from tensorly.decomposition import parafac, tensor_train
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tt_tensor import tt_to_tensor
from src.core.common import CommonSdfMethods 
from src.core.assets.entities.models import WeightsLinkModel
from src.core.train.train_config import TrainConfig, Train_W
from src.core.math.projection_point import spherical_projection, ellipsoid_projection, correct_sphere_gradient, correct_ellipsoid_gradient

from src.core.sdf_creator import SDFTrain




class RDF_Weights(CommonSdfMethods):
    def __init__(self, device='cuda', dtype=torch.float32):

        CommonSdfMethods.__init__(self, projection_method='sphere', device=device, dtype=dtype)

    
    def init_robot_folder(self, ws_path, robot_name=''):
        CommonSdfMethods.init_robot_folder(self, ws_path, robot_name)
        self.ws_path = ws_path

    def train_links(self, link_names:Union[List[str], str],  n_func:int, iters:int, batch_near:int=1024, batch_rand:int=64, robot_name='', debug=False):
        cfg_w = Train_W(run=True, n_func=n_func, iters=iters, batch_near=batch_near, batch_rand=batch_rand)
        cfg = TrainConfig(links_to_train=link_names, debug=debug, classic=cfg_w )
        trainer = SDFTrain()
        trainer.init_robot_folder(self.ws_path, robot_name=robot_name)
        trainer.create_model(cfg, robot_name=robot_name)
        CommonSdfMethods.init_robot_folder(self, self.ws_path, robot_name=robot_name)

    def batch_to(self, device):
        self.device = device
        self.weights_batch = self.weights_batch.to(device)
        self.centroids_batch = self.centroids_batch.to(device)
        self.scale_factors_batch = self.scale_factors_batch.to(device)
        
    def add_models(self, link_names:Union[List[str], str], namespace='', robot_name='', **kwargs):
        for link_name in link_names:
            self.add_model(link_name, WeightsLinkModel.file_suffix, namespace, robot_name, **kwargs)

    def inference_link(self, link_name:str, points: torch.Tensor, get_grad: bool=False, get_min: bool=False, fk_matrix:torch.Tensor = None,):
        #DEBUGGATO CON ALTRO METODO PER 5 PUNTI E TORNANO STESSI RISULTATI, MA VALORI DIVERSI AL MILLIMETRO

        if fk_matrix is not None:
            points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=fk_matrix)

        model:WeightsLinkModel = getattr(self, link_name + self.model_extension)
        model.to(points.device, points.dtype)
        self.set_number_of_functions(model.n_func)
        self.set_points_domain(domain_min=model.domain_min, domain_max=model.domain_max)
        # print(f"Points device: {points.device}, model:", model.weights.device, model.centroid_offset.device, model.scale_factor.device)
        points_scaled = (points - model.centroid_offset) / model.scale_factor

        x_in, d_sphere, outside = spherical_projection(points_scaled, r1=(model.domain_max - model.domain_min) / 2)
        

        phi_p, d_phi_p = super().basis_function_from_3Dpoints(x_in, use_derivative=get_grad)
        sdf = torch.matmul(phi_p, model.weights) + d_sphere
 
        sdf = sdf * model.scale_factor
        # self.sdf = self.sdf + self.centroid_offset.norm()

        if get_grad:
            # grad in x_in-space
            g_in = torch.einsum('ijk,j->ik', d_phi_p, model.weights)  # (N,3)
            
            d_sdf = correct_sphere_gradient(points_scaled, g_in, outside, r1=(model.domain_max - model.domain_min) / 2)
        else:
            d_sdf = torch.zeros((points_scaled.shape[0], 3), device=self.device, dtype=self.dtype)

        if get_min:
            sdf_min, idx = torch.min(sdf, dim=0)
            return sdf_min, d_sdf[idx], points[idx]

        return sdf, d_sdf, points

    @torch.no_grad()
    def _sdf_kernel_for_tests(self, link_name: str, points: torch.Tensor, get_grad: bool = False, get_min: bool = False, fk_matrix: torch.Tensor = None):
        if fk_matrix is not None:
            points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=fk_matrix)

        model: WeightsLinkModel = getattr(self, link_name + self.model_extension)
        model.to(points.device, points.dtype)
        self.set_number_of_functions(model.n_func)
        self.set_points_domain(domain_min=model.domain_min, domain_max=model.domain_max)

        points_scaled = (points - model.centroid_offset) / model.scale_factor
        phi_p, d_phi_p = super().basis_function_from_3Dpoints(points_scaled, use_derivative=get_grad)
        sdf = torch.matmul(phi_p, model.weights) * model.scale_factor

        if get_grad:
            d_sdf = torch.einsum('ijk,j->ik', d_phi_p, model.weights)
        else:
            d_sdf = torch.zeros((points_scaled.shape[0], 3), device=self.device, dtype=self.dtype)

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

    def set_ordered_batch_params(self, link_names:List[str]):
        list_weights = []
        n_functions = []

        list_centroids = [getattr(self, link + self.model_extension).centroid_offset.unsqueeze(0) for link in link_names]
        list_scale_factors = [getattr(self, link + self.model_extension).scale_factor.unsqueeze(0) for link in link_names]
        
        for link in link_names:
            n_func = getattr(self, link + self.model_extension).n_func
            n_functions.append(n_func)
        
        self.max_n_func = max(n_functions)
        super().set_number_of_functions(self.max_n_func)
        for link in link_names:
            weights = getattr(self, link + self.model_extension).weights.unsqueeze(0)
            weights = weights.to(self.device)
            if  weights.shape[1] < self.max_n_func**3:
                padding = self.max_n_func**3 - weights.shape[1]
                list_weights.append(torch.nn.functional.pad(weights, (0, padding), mode='constant', value=0))
            else:
                list_weights.append(weights)


        #cat the lists
        self.centroids_batch = torch.cat(list_centroids, dim=0).unsqueeze(1)
        self.scale_factors_batch = torch.cat(list_scale_factors, dim=0).reshape(-1, 1, 1)
        self.weights_batch = torch.cat(list_weights, dim=0)

        self.batch_to(self.device)

    def get_weights_tensor(self, link_name: str) -> torch.Tensor:
        """
        Ritorna il tensore W di shape (N,N,N) del link nel modello weights.
        """
        model: WeightsLinkModel = getattr(self, link_name + self.model_extension)
        model.to(self.device, self.dtype)
        n_func = int(model.n_func)
        w = model.weights.reshape(-1)
        expected = n_func ** 3
        if w.numel() < expected:
            w = torch.nn.functional.pad(w, (0, expected - w.numel()))
        elif w.numel() > expected:
            w = w[:expected]
        return w.reshape(n_func, n_func, n_func)

    @staticmethod
    def _decomp_metrics(w_ref: np.ndarray, w_hat: np.ndarray) -> dict:
        err = w_hat - w_ref
        abs_err = np.abs(err)
        denom = max(float(np.linalg.norm(w_ref)), 1e-12)
        return {
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max_abs": float(abs_err.max()),
            "rel_fro": float(np.linalg.norm(err) / denom),
        }

    def decompose_weights_cp(
        self,
        link_name: str,
        rank: int,
        n_iter_max: int = 300,
        tol: float = 1e-8,
        random_state: int = 0,
    ) -> dict:
        """
        Decomposizione CP del tensore W (N,N,N) del modello weights.
        """
        if rank <= 0:
            raise ValueError("rank CP deve essere > 0")
        w_t = self.get_weights_tensor(link_name)
        w_np = w_t.detach().cpu().numpy()
        n = int(w_np.shape[0])

        prev_backend = tl.get_backend()
        tl.set_backend("numpy")
        try:
            cp = parafac(
                w_np,
                rank=int(rank),
                init="svd",
                n_iter_max=int(n_iter_max),
                tol=float(tol),
                random_state=int(random_state),
                normalize_factors=False,
            )
            w_hat = cp_to_tensor(cp)
        finally:
            tl.set_backend(prev_backend)

        metrics = self._decomp_metrics(w_np, w_hat)
        params = int(rank) * (3 * n + 1)  # 3 fattori Nxr + lambda (r)
        metrics.update({
            "method": "cp",
            "rank": int(rank),
            "params": params,
            "compression": float((n ** 3) / max(params, 1)),
        })
        return metrics

    def decompose_weights_tt(
        self,
        link_name: str,
        ranks: tuple,
    ) -> dict:
        """
        Decomposizione TT del tensore W (N,N,N) del modello weights.
        ranks: (r1, r2)
        """
        if not isinstance(ranks, (tuple, list)) or len(ranks) != 2:
            raise ValueError("ranks TT deve essere una tupla/lista (r1, r2)")
        r1, r2 = int(ranks[0]), int(ranks[1])
        if r1 <= 0 or r2 <= 0:
            raise ValueError("ranks TT devono essere > 0")

        w_t = self.get_weights_tensor(link_name)
        w_np = w_t.detach().cpu().numpy()
        n = int(w_np.shape[0])

        prev_backend = tl.get_backend()
        tl.set_backend("numpy")
        try:
            tt = tensor_train(w_np, rank=[1, r1, r2, 1], verbose=False)
            w_hat = tt_to_tensor(tt)
        finally:
            tl.set_backend(prev_backend)

        metrics = self._decomp_metrics(w_np, w_hat)
        params = n * r1 + r1 * n * r2 + r2 * n
        metrics.update({
            "method": "tt",
            "rank": (r1, r2),
            "params": int(params),
            "compression": float((n ** 3) / max(params, 1)),
        })
        return metrics

    def compare_weights_decompositions(
        self,
        link_name: str,
        cp_ranks: List[int],
        tt_ranks: List[tuple],
        cp_n_iter_max: int = 300,
        cp_tol: float = 1e-8,
        random_state: int = 0,
    ) -> dict:
        """
        Confronto CP/TT sull'approssimazione del tensore W al variare dei rank.
        """
        cp_results = []
        for r in cp_ranks:
            cp_results.append(
                self.decompose_weights_cp(
                    link_name=link_name,
                    rank=int(r),
                    n_iter_max=cp_n_iter_max,
                    tol=cp_tol,
                    random_state=random_state,
                )
            )

        tt_results = []
        for rr in tt_ranks:
            tt_results.append(self.decompose_weights_tt(link_name=link_name, ranks=rr))

        return {"cp": cp_results, "tt": tt_results}

    

    def inference_link_batch(self, points:torch.Tensor, get_grad=False, get_min=False, forward_tensor:torch.Tensor=None,) -> tuple:
        '''
        Convert a point cloud to a rdf representation
        '''
        if points.dim() not in (2, 3) or points.shape[-1] != 3:
            raise ValueError(f"Expected points shape (P,3) or (L,P,3), got {tuple(points.shape)}")
        
        self.batch_to(points.device)
        points = CommonSdfMethods.to_link_frame(points_w=points, H_wl=forward_tensor) if forward_tensor is not None else points
        super().set_number_of_functions(self.max_n_func)
        link_length  = self.weights_batch.shape[0]
        point_length = points.shape[1]

        # --- scala e proietta su sfera (flatten) ---
        point_scaled = ((points - self.centroids_batch) / self.scale_factors_batch).reshape(-1, 3)
        r1 = (self.domain_max - self.domain_min) / 2
        point_on_sphere, distances, outside = spherical_projection(point_scaled, r1=r1)

        phi_dt, d_phi_dt = super().basis_function_from_3Dpoints(point_on_sphere, use_derivative=get_grad)
        
        phi_dt = phi_dt.reshape(link_length, point_length, phi_dt.shape[1])
    
        sdf = torch.einsum('ijk,ik->ij', phi_dt, self.weights_batch)                    # (L,P)
        sdf = sdf + distances.reshape(link_length, point_length)                  # + distanza sferica
        sdf = sdf * self.scale_factors_batch.reshape(link_length, 1)               # scala in mondo

        d_sdf = None
        if get_grad:
            # g_in nel frame x_in (prima della correzione)
            d_phi_dt = d_phi_dt.reshape(link_length, point_length, d_phi_dt.shape[1], d_phi_dt.shape[2])
            g_in = torch.einsum('ijkl,ik->ijl', d_phi_dt, self.weights_batch)           # (L,P,3)

            # --- CORREZIONE GRADIENTE (SOLO SFERA) ---
            # flatten
            g_in_flat = g_in.reshape(-1, 3)                                       # (L*P,3)

            # usa il metodo già presente nella classe (definito sopra)
            g_corr_flat = correct_sphere_gradient(
                points_scaled=point_scaled,                                       # (L*P,3)
                g_in=g_in_flat,                                                   # (L*P,3)
                outside=outside,                                                  # (L*P,)
                r1=r1                                                             # scalar tensor
            )                                                                      # -> (L*P,3)

            # reshape al formato batch
            d_sdf = g_corr_flat.reshape(link_length, point_length, 3)             # (L,P,3)
            # Nota: il gradiente NON va ri-scalato.

        if get_min:
            sdf_min, min_indices = torch.min(sdf, dim=1)                          # (L,)
            if get_grad:
                d_sdf = d_sdf[torch.arange(d_sdf.size(0), device=d_sdf.device), min_indices]
            point = points[torch.arange(points.size(0), device=points.device), min_indices]
            return sdf_min, d_sdf, point

        return sdf, d_sdf, points

    def gradient_descent_link(self, points_link_frame:torch.Tensor, link_name:str, num_of_iteration:int = None):
        model:WeightsLinkModel = getattr(self, link_name + self.model_extension)
        self.n_func = model.classic_n_func
        points_debug = torch.zeros((num_of_iteration, points_link_frame.shape[0], 3), device=self.device, dtype=self.dtype)
        
        sdf, gradient, points_link_frame = self.inference_link(link_name, points_link_frame, get_gradient=True, get_min=False)
        for i in range(num_of_iteration):
            points_debug[i] = points_link_frame
            sdf, _, _ = self.inference_link(link_name, points_link_frame, get_gradient=True, get_min=False)
            step = step = sdf.unsqueeze(1) * gradient / (gradient.norm(dim=1, keepdim=True) + 1e-6)
            points_link_frame = points_link_frame - step
        
        return points_link_frame

    def gradient_descent_batch(self, points_link_frame:torch.Tensor, num_of_iteration:int = 5, epsilon:float=1e-3) -> torch.Tensor:
        '''
        Esegue la discesa del gradiente sui punti in frame link
        points_link_frame: (L,N,3)
        '''

        points_plojected = points_link_frame.clone()
        self.batch_to(points_link_frame.device)
        for i in range(num_of_iteration):
            # start_time = time.time()
            distances, gradient, points_plojected = self.inference_link_batch(points_plojected, get_grad=True, get_min=True)
            points_plojected = (points_plojected - distances.unsqueeze(1) * gradient / gradient.norm(dim=1, keepdim=True)).unsqueeze(1)

            active = distances.abs() > epsilon
            if not active.any():
                break  # tutti sotto soglia, stop
            # end_time = time.time()
            # print(f"Iteration {i+1}/{num_of_iteration} completed in {end_time - start_time:.4f} seconds, active points: {active.sum().item()}")
        
        return points_plojected
    
   

    def to_mesh(self, model_name, type:Literal['pyvista', 'trimesh'], debug=False):  
        from src.utils.sdf_utils import sdf_to_mesh
        from src.utils.MeshUtils import trimesh_to_pyvista   
        
        model:WeightsLinkModel = getattr(self, model_name + self.model_extension)
        if not hasattr(model, "weights") or model.weights is None:
            raise ValueError(f"Model '{model_name}' does not contain weights")

        prev_n_func = getattr(self, "n_func", None)
        prev_domain_min = getattr(self, "domain_min", None)
        prev_domain_max = getattr(self, "domain_max", None)

        if getattr(model, "n_func", None) is not None:
            model_n_func = int(model.n_func)
        else:
            model_n_func = int(round(float(model.weights.numel()) ** (1.0 / 3.0)))
            if model_n_func ** 3 != int(model.weights.numel()):
                raise ValueError(
                    f"Cannot infer n_func from {model.weights.numel()} weights for model '{model_name}'"
                )

        try:
            self.set_number_of_functions(model_n_func)
            self.set_points_domain(domain_min=model.domain_min, domain_max=model.domain_max)
            mesh = sdf_to_mesh(
                weights=model.weights,
                nbData=128,
                domain_max=model.domain_max,
                domain_min=model.domain_min,
                scaling_factor=model.scale_factor,
                centroid_offset=model.centroid_offset,
                basis_function_from_3Dpoints=super().basis_function_from_3Dpoints,
            )
        finally:
            if prev_n_func is not None and prev_domain_min is not None and prev_domain_max is not None:
                self.set_number_of_functions(prev_n_func)
                self.set_points_domain(domain_min=prev_domain_min, domain_max=prev_domain_max)

        mesh.show() if debug else None
        mesh = trimesh_to_pyvista(mesh, np.eye(4)) if type=='pyvista' else mesh
        return mesh

    def get_model_names(self):
        link_names = []
        for name in dir(self):
            if name.endswith(self.model_extension):
                link_names.append(name[:-len(self.model_extension)])

        return link_names
    
    def visualize_gradient_descent(
        self,
        initial_points: torch.Tensor,
        forward_dict: torch.Tensor = None,
        num_of_iteration: int = 10,
        epsilon: float = 1e-3,
        mesh_color: str = "#302E2E",
        mesh_opacity: float = 1.0,
        rdf_color: str = "#c9c9bf",
        rdf_opacity: float = 0.3,
    ):
        '''
        Visualizza la discesa del gradiente sui punti in frame mondo
        initial_points: (L,N,3)
        '''
        points_plojected = initial_points.clone()
        self.batch_to(initial_points.device)

        all_points = [points_plojected.clone()]

        for i in range(num_of_iteration):
            distances, gradient, points_plojected = self.inference_link_batch(points_plojected, get_grad=True, get_min=True)
            points_plojected = (points_plojected - distances.unsqueeze(1) * gradient / gradient.norm(dim=1, keepdim=True)).unsqueeze(1)

            all_points.append(points_plojected.clone())

            active = distances.abs() > epsilon
            if not active.any():
                break  # tutti sotto soglia, stop
        
        all_points = torch.stack(all_points, dim=0)  # (T,L,N,3)
        all_points = all_points.permute(1, 0, 2, 3)  # (L,T,N,3)
        #stack sulla dim T e N
        all_points = all_points.reshape(all_points.shape[0], -1, 3)

        if forward_dict is None:
            raise ValueError("Serve forward_dict per visualizzare la GD.")
        fk_list_t = []
        for v in forward_dict.values():
            if isinstance(v, torch.Tensor):
                fk_list_t.append(v.to(device=self.device, dtype=self.dtype))
            else:
                fk_list_t.append(torch.as_tensor(v, device=self.device, dtype=self.dtype))
        fk_tensor = torch.stack(fk_list_t, dim=0)

        all_points_world = self.to_world_frame(all_points, fk_tensor)
        

        # --- chiama visualize_scene, che si occuperà del plotting ---
        self.visualize_scene(
            forward_as_dict=forward_dict,
            links_as_mesh=True,
            gradient_descent_points=all_points_world,
            mesh_color=mesh_color,
            mesh_opacity=mesh_opacity,
            rdf_color=rdf_color,
            rdf_opacity=rdf_opacity,
        )

    def visualize_scene(
        self,
        forward_as_dict: dict,
        links_as_mesh: bool = True,
        mesh_link_names: List[str] = None,
        links_as_rdf: bool = False,
        rdf_link_names: List[str] = None,
        links_as_pc: bool = False,
        additional_point_cloud: torch.Tensor = torch.empty((0, 3), device="cpu"),
        point_cloud_color: str = "#9E9E9E",
        point_cloud_size: float = 8.0,
        point_cloud_opacity: float = 1.0,
        gradient_descent_points: torch.Tensor = None,  # trajectory of the gradient descent points debug
        points_on_link: torch.Tensor = torch.empty((0, 3), device="cpu"),  # points on the link after projection
        mesh_color: str = "#302E2E",
        mesh_opacity: float = 1.0,
        rdf_color: str = "#c9c9bf",
        rdf_opacity: float = 0.3,
    ):

        super().visualize_scene(
            forward_as_dict=forward_as_dict,
            links_as_mesh=links_as_mesh,
            mesh_link_names=mesh_link_names,
            additional_point_cloud=additional_point_cloud,
            point_cloud_color=point_cloud_color,
            point_cloud_size=point_cloud_size,
            point_cloud_opacity=point_cloud_opacity,
            mesh_color=mesh_color,
            mesh_opacity=mesh_opacity,
        )

        link_names = list(forward_as_dict.keys())
        rdf_link_set = None
        if rdf_link_names is not None:
            rdf_link_set = set(rdf_link_names)
            unknown = sorted(rdf_link_set.difference(link_names))
            if unknown:
                raise ValueError(f"rdf_link_names contains unknown links: {unknown}")
        from src.utils.MeshUtils import trimesh_to_pyvista

        for link in link_names:
            if links_as_rdf and (rdf_link_set is None or link in rdf_link_set):
                if hasattr(self, link + self.model_extension):
                    self._visualizer.add_mesh(
                        trimesh_to_pyvista(self.to_mesh(link, type="trimesh"), forward_as_dict[link]),
                        opacity=rdf_opacity,
                        color=rdf_color,
                    )
            


        # --- GRADIENT DESCENT EVOLUTION ---
        if gradient_descent_points is not None and gradient_descent_points.numel() > 0:
            L, T, _ = gradient_descent_points.shape
            link_colors = [ "red", "blue", "green", "orange", "magenta", "cyan", "yellow", "purple", "brown", "pink"]
            for i in range(L):
                pts = gradient_descent_points[i].cpu().numpy() if gradient_descent_points.is_cuda else gradient_descent_points[i].numpy()
                self._visualizer.add_pointcloud(pts, color=link_colors[i % len(link_colors)], point_size=7)
                color = link_colors[i % len(link_colors)]
                self._visualizer.add_polyline(pts, color=color, line_width=2)

        if additional_point_cloud.shape[0] > 0 and points_on_link.shape[0] > 0:
            if isinstance(additional_point_cloud, torch.Tensor):
                points_np = additional_point_cloud.detach().cpu().numpy()
            else:
                points_np = np.asarray(additional_point_cloud)

            if isinstance(points_on_link, torch.Tensor):
                proj_np = points_on_link.detach().cpu().numpy()
            else:
                proj_np = np.asarray(points_on_link)

            self._visualizer.visualize_gradient_descent(points=points_np, projected_points=proj_np)
        elif additional_point_cloud.shape[0] > 0 and points_on_link.shape[0] == 0:
            if isinstance(additional_point_cloud, torch.Tensor):
                points_np = additional_point_cloud.detach().cpu().numpy()
            else:
                points_np = np.asarray(additional_point_cloud)
            self._visualizer.add_pointcloud(points_np)

        self._visualizer.show()

    def visualize_isosurface_lvls(self, forward_dict:dict, spacing=0.02, n_isosurfaces=12,  clip_normal=[1,0,0], clip_half=True):

        plot_robot_isosurfaces(transforms=forward_dict, mesh_files={link:getattr(self, link + self.mesh_extension).path for link in self.get_mesh_names()}, 
                               spacing=spacing, n_isosurfaces=n_isosurfaces,  clip_normal=clip_normal, clip_half=clip_half, show=True)

    
    
