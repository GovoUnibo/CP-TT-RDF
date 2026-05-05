from src.core.math.Bernstain_P import BersteinPoly
import torch
import numpy as np
from src.core.assets.ModelHandler import ModelHandler
from typing import List, Union
from src.core.mesh_pointcloud import MeshAndPointcloud

from src.visu.plots_3d import RobotSene

class CommonSdfMethods(MeshAndPointcloud, BersteinPoly):
    def __init__(self, projection_method = 'sphere', device='cpu', dtype=torch.float32):
        BersteinPoly.__init__(self, device=device, dtype=dtype)
        MeshAndPointcloud.__init__(self, device=device, dtype=dtype)
        self.method = projection_method
        self.show_cartesian_frame = True

        valid_methods = ['sphere', 'ellipsoid', 'none']
        if projection_method not in valid_methods:
            raise ValueError(f"Invalid projection method: {projection_method}. Choose from {valid_methods}")
        
        self.model_extension = "_" + 'pt'
        self.folder_model = 'Models'

        self.ws_path = None
    

    def init_robot_folder(self, ws_path, robot_name=''):
        self.ws_path = ws_path
        MeshAndPointcloud.init_robot_folder(self, ws_path, robot_name)
        setattr(self, robot_name + self.folder_model, ModelHandler(ws_path))

    def add_model(self, link_names:Union[List, str], file_suffix:str, namespace='', robot_name='', **kwargs):

        def add_single_model(link_name, file_suffix, namespace, robot_name):
            model_folder:ModelHandler = getattr(self, robot_name + self.folder_model)
            link_name_with_suffix = link_name + file_suffix

            model = model_folder.load_model(link_name_with_suffix, self.device, self.dtype)

            # print(f"Model {link_name} loaded: {model is not None}")
            # print(f"Namespace: {namespace}, Robot Name: {robot_name}")
            # print(f"Model Folder: {self.model_extension}")
            _model_name = namespace + link_name + self.model_extension
            if model is None:
                print("\033[91m" + f'Model "{_model_name}" not found' + "\033[0m")
                exit()
            else:
                setattr(self, _model_name, model)
                print("\033[92m" + f'Model "{_model_name}" loaded' + "\033[0m")

        if not hasattr(self, robot_name + self.folder_model):
            raise ValueError(f"Robot folder {robot_name} not found")
        

        if type(link_names) is list:
            for link_name in link_names:
                add_single_model(link_name, file_suffix, namespace, robot_name)
        else:
            add_single_model(link_names, file_suffix, namespace, robot_name)

    def _to_device_dtype(self, x):
        if isinstance(x, torch.Tensor):
            # detach to avoid gradients leaking in; clone if you need your own storage
            return x.detach().to(device=self.device, dtype=self.dtype)
        else:
            # numpy array, list, scalar...
            return torch.tensor(x, device=self.device, dtype=self.dtype)


    @staticmethod
    def to_link_frame(points_w: torch.Tensor, H_wl: torch.Tensor) -> torch.Tensor:
        """
        points_w : (N,3)      oppure (L,N,3)  in world
        H_wl     : (4,4)      oppure (L,4,4)  FK world←link
        ritorna  : (N,3)      oppure (L,N,3)  in link
        """
        single_pts = points_w.dim() == 2
        single_H   = H_wl.dim() == 2
        if single_pts: points_w = points_w.unsqueeze(0)
        if single_H:   H_wl     = H_wl.unsqueeze(0)
        assert points_w.shape[0] == H_wl.shape[0], "Batch L deve coincidere"

        R = H_wl[:, :3, :3]
        t = H_wl[:, :3, 3]
        p_l = torch.bmm(points_w - t[:, None, :], R)  # (p_w - t) @ R   perché R = R_wl^T nel cambio

        return p_l[0] if (single_pts and single_H) else p_l
    
    @staticmethod
    def to_world_frame(points_l: torch.Tensor, H_wl: torch.Tensor) -> torch.Tensor:
        """
        points_l : (N,3)      oppure (L,N,3)  in link
        H_wl     : (4,4)      oppure (L,4,4)  FK world←link
        ritorna  : (N,3)      oppure (L,N,3)  in world
        """
        single_pts = points_l.dim() == 2
        single_H   = H_wl.dim() == 2
        if single_pts: points_l = points_l.unsqueeze(0)
        if single_H:   H_wl     = H_wl.unsqueeze(0)
        assert points_l.shape[0] == H_wl.shape[0], "Batch L deve coincidere"

        R = H_wl[:, :3, :3]
        t = H_wl[:, :3, 3]
        p_w = torch.bmm(points_l, R.transpose(1, 2)) + t[:, None, :]   # p_l @ R^T + t

        return p_w[0] if (single_pts and single_H) else p_w

    def visualize_scene(self,   forward_as_dict:dict, 
                                links_as_mesh:bool=False,
                                mesh_link_names: List[str] = None,
                                additional_point_cloud:torch.Tensor=torch.empty((0,3), device='cpu'),
                                point_cloud_color: str = "#9E9E9E",
                                point_cloud_size: float = 8.0,
                                point_cloud_opacity: float = 1.0,
                                cartesian_frame_pose:np.ndarray=None,
                                xy_offset: np.ndarray = np.zeros(2),
                                mesh_color: str = "#302E2E",
                                mesh_opacity: float = 1.0):
        
        self._visualizer = RobotSene()


        dev = additional_point_cloud.device
        dt = additional_point_cloud.dtype
        def _apply_offset(T):
            if T is None:
                return None
            ox, oy = xy_offset
            if isinstance(T, torch.Tensor):
                t_new = T.to(device=dev, dtype=dt).clone()
                t_new[:3, 3] += torch.tensor([ox, oy, 0.0], device=dev, dtype=dt)
                return t_new
            else:
                t_new = np.array(T, copy=True)
                t_new[:3, 3] += np.array([ox, oy, 0.0], dtype=float)
                return t_new

        forward_as_dict = {
            k: _apply_offset(v)
            for k, v in forward_as_dict.items()
        }

        additional_point_cloud = additional_point_cloud.cpu().numpy() if additional_point_cloud.is_cuda else additional_point_cloud.numpy()
        if additional_point_cloud.shape[0] > 0:
            self._visualizer.add_pointcloud(
                additional_point_cloud,
                color=point_cloud_color,
                point_size=point_cloud_size,
                opacity=point_cloud_opacity,
            )
        if self.show_cartesian_frame:
            cartesian_frame_pose = np.eye(4) if cartesian_frame_pose is None else cartesian_frame_pose
            cartesian_frame_pose = _apply_offset(cartesian_frame_pose)
            self._visualizer.add_cartesian_frame(cartesian_frame_pose)

        link_names = list(forward_as_dict.keys())
        mesh_link_set = None
        if mesh_link_names is not None:
            mesh_link_set = set(mesh_link_names)
            unknown = sorted(mesh_link_set.difference(link_names))
            if unknown:
                raise ValueError(f"mesh_link_names contains unknown links: {unknown}")
        from src.utils.MeshUtils import trimesh_to_pyvista
        for link in link_names:
            if links_as_mesh and (mesh_link_set is None or link in mesh_link_set):
                mesh_name = link + self.mesh_extension
                if hasattr(self, mesh_name):
                    mesh = getattr(self, mesh_name)
                    self._visualizer.add_mesh(
                        trimesh_to_pyvista(mesh.mesh, forward_as_dict[link]),
                        opacity=mesh_opacity,
                        color=mesh_color,
                    )
                    self._visualizer.incornicia_mesh(trimesh_to_pyvista(mesh.mesh, forward_as_dict[link]), linewidth=2, color='black')
