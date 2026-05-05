from typing import Dict, Optional, Iterable, Tuple
import torch
from src.core.mesh_pointcloud import MeshAndPointcloud
from src.core.assets.ModelHandler import ModelHandler
from typing import List
from src.visu.plots_3d import RobotSene
import numpy as np

from src.core.sdf_creator import SDFTrain
from src.core.train.train_config import TrainConfig
from src.core.assets.entities.models import SphereModel

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Iterable
import numpy as np
import torch

TLike = Union[torch.Tensor, np.ndarray]

class SphereOps:
    def __init__(self, centers: torch.Tensor, radii: torch.Tensor):
        self.centers = centers
        self.radii = radii

    @classmethod
    def from_model(cls, m: SphereModel) -> "SphereOps":
        return cls(m.centers, m.radii)

    def transform(self, T):
        import numpy as np
        if T is None:
            return SphereOps(self.centers.clone(), self.radii.clone())
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T)
        T = T.to(device=self.centers.device, dtype=self.centers.dtype)
        K = self.centers.shape[0]
        ones = torch.ones((K, 1), device=self.centers.device, dtype=self.centers.dtype)
        centers_h = torch.cat([self.centers, ones], dim=1)
        centers_w = (T @ centers_h.T).T[:, :3].contiguous()
        return SphereOps(centers_w, self.radii.clone())

    def distance_to_points(self, points: torch.Tensor) -> torch.Tensor:
        points = points.to(device=self.centers.device, dtype=self.centers.dtype)
        diff = points[:, None, :] - self.centers[None, :, :]
        d = torch.linalg.norm(diff, dim=-1) - self.radii[None, :]
        return d.min(dim=1).values

    def distance_to_other_spheres(self, other: "SphereOps") -> torch.Tensor:
        oc = other.centers.to(device=self.centers.device, dtype=self.centers.dtype)
        orr = other.radii.to(device=self.centers.device, dtype=self.centers.dtype)
        diff = self.centers[:, None, :] - oc[None, :, :]
        cd = torch.linalg.norm(diff, dim=-1)
        rs = self.radii[:, None] + orr[None, :]
        return cd - rs



class RobotSphere(MeshAndPointcloud):
    def __init__(self, device='cuda', dtype=torch.float32):
        MeshAndPointcloud.__init__(self, device=device, dtype=dtype)
        self.links: Dict[str]          # link_name -> SphereModel
        self.model_extension = "_" + 'sphere'
        self.folder_model = "Models"

        self.links = {}
        self.obstacles = {}

    def init_robot_folder(self, ws_path, robot_name='robot'):
        self.ws_path = ws_path
        self.robot_name = robot_name
        super().init_robot_folder(ws_path, robot_name)
        setattr(self, robot_name + self.folder_model, ModelHandler(ws_path))
    
    def create_models(self, cfg:TrainConfig, robot_name='robot'):
        trainer = SDFTrain(device='cuda')
        trainer.init_robot_folder(ws_path=self.ws_path, robot_name=robot_name)
        trainer.create_model(cfg, robot_name=robot_name)


    def add_single_model(self, link_name, namespace='', robot_name=None):
        if robot_name is None:
            robot_name = getattr(self, "robot_name", "robot")
        model_folder: ModelHandler = getattr(self, robot_name + self.folder_model)

        model_name = link_name + SphereModel().file_suffix  # oppure "_spheres"
        model: SphereModel = model_folder.load_model(model_name, device=self.device, dtype=self.dtype)

        _model_name = namespace + link_name + self.model_extension
        if model is None:
            print("\033[91m" + f'Model "{_model_name}" not found' + "\033[0m")
            exit()

        # qui: converti a SphereOps
        ops = SphereOps.from_model(model)

        # io ti consiglio di salvare ops come attributo “ufficiale”
        setattr(self, _model_name, ops)

        # se vuoi anche il modello dati per debug/IO:
        setattr(self, _model_name + "_raw", model)
    
    def add_robot_links(self, link_names, namespace='', robot_name=None, **kwargs):
        if robot_name is None:
            robot_name = getattr(self, "robot_name", "robot")
        for model_name in link_names:
            self.add_single_model(model_name, namespace, robot_name, **kwargs)
            self.links[namespace + model_name] = getattr(self, namespace + model_name + self.model_extension)

    def add_obstacles_links(self, obs_names: str, pose: List = None, belong_to='robot'):
        if pose is not None and len(pose) != len(obs_names):
            raise ValueError("pose deve avere la stessa lunghezza di obs_names")

        for i, obs_name in enumerate(obs_names):
            self.add_single_model(obs_name, robot_name=belong_to)

            obs_ops: SphereOps = getattr(self, obs_name + self.model_extension)  # nota: dipende dal tuo naming
            transformed = obs_ops.transform(pose[i]) if pose is not None else obs_ops
            self.obstacles[obs_name] = transformed



    def distances_to_obstacles(
        self,
        forward_as_dict: Dict[str, torch.Tensor],
        links: Optional[Iterable[str]] = None,
        obstacles: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Ritorna per-link un dict con:
        - dmin: minima distanza (scalar tensor)
        - obstacle: nome ostacolo che realizza dmin
        - pair: (i,j) indici sfere (link_i, obs_j)
        """
        link_keys = list(links) if links is not None else list(self.links.keys())
        obs_keys  = list(obstacles) if obstacles is not None else list(self.obstacles.keys())

        out: Dict[str, Dict[str, Any]] = {}

        for link in link_keys:
            T = forward_as_dict.get(link, None)
            if T is None:
                continue

            link_world = self.links[link].transform(T)

            best_d = None
            best_obs = None
            best_pair = None

            for obs_name in obs_keys:
                obs = self.obstacles[obs_name]  # già in world
                D = link_world.distance_to_other_spheres(obs)  # (K,M)
                dmin = D.min()
                flat = int(D.argmin().item())
                i = flat // D.shape[1]
                j = flat %  D.shape[1]

                if (best_d is None) or (dmin < best_d):
                    best_d = dmin
                    best_obs = obs_name
                    best_pair = (int(i), int(j))

            out[link] = {"dmin": best_d, "obstacle": best_obs, "pair": best_pair}

        return out

    

    def visualize_scene(
        self,
        forward_as_dict: Dict[str, torch.Tensor],
        links_as_meshes: bool = True,
        links_as_spheres: bool = True,
        obs_as_mesh: bool = True,
        obs_as_sphere: bool = True,
        cartesian_frame_pose: Optional[np.ndarray] = None,
        near_thresh: float = 0.02,   # 2 cm
    ):
        from src.utils.MeshUtils import trimesh_to_pyvista

        robot_mesh_color     = "#797976"
        robot_mesh_opacity   = 1.0
        robot_sphere_opacity = 0.18

        obs_mesh_color       = "orange"
        obs_mesh_opacity     = 0.85
        obs_sphere_color     = "red"
        obs_sphere_opacity   = 0.18

        visu = RobotSene()

        if cartesian_frame_pose is None:
            cartesian_frame_pose = np.eye(4)
        visu.add_cartesian_frame(cartesian_frame_pose)

        # --- calcola dmin per link rispetto agli ostacoli ---
        dist_info = self.distances_to_obstacles(forward_as_dict)

        # ---------- ROBOT ----------
        for link, T in forward_as_dict.items():

            # 1) mesh
            if links_as_meshes:
                mesh_name = link + self.mesh_extension
                if hasattr(self, mesh_name):
                    mesh = getattr(self, mesh_name)
                    pv_mesh = trimesh_to_pyvista(mesh.mesh, T)
                    visu.add_mesh(pv_mesh, opacity=robot_mesh_opacity, color=robot_mesh_color)
                    visu.incornicia_mesh(pv_mesh, linewidth=2, color="black")

            # 2) spheres (colorate in base a distanza)
            if links_as_spheres and (link in self.links):
                sm_world = self.links[link].transform(T)

                dmin = dist_info.get(link, {}).get("dmin", None)
                if dmin is None:
                    color = "cyan"
                else:
                    d = float(dmin.item())
                    if d < 0.0:
                        color = "red"
                    elif d < near_thresh:
                        color = "yellow"
                    else:
                        color = "cyan"

                visu.add_sphere_model(sm_world, color=color, opacity=robot_sphere_opacity)

        # ---------- OBSTACLES ----------
        if obs_as_mesh and hasattr(self, "get_mesh_names"):
            for attr in dir(self):
                if attr.endswith(self.mesh_extension) and attr.startswith("obstacle_"):
                    obs_mesh = getattr(self, attr)
                    pv_obs = trimesh_to_pyvista(obs_mesh.mesh, torch.eye(4, device=self.device, dtype=self.dtype))
                    visu.add_mesh(pv_obs, opacity=obs_mesh_opacity, color=obs_mesh_color)

        if obs_as_sphere and len(self.obstacles) > 0:
            for _, sm_obs in self.obstacles.items():
                visu.add_sphere_model(sm_obs, color=obs_sphere_color, opacity=obs_sphere_opacity)

        visu.show()
