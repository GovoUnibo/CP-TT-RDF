from src.core.assets.MeshHandler import MeshHandler
from src.core.assets.entities.MeshModel import Mesh
from src.core.math.transformations import homogeneous_matrix, direct_transform_points
import torch
import numpy as np
from typing import Union, List, Dict, Literal
from src.visu.debug_visu_utils import show_mesh_and_pointcloud, show_two_pc_mtplt


class MeshAndPointcloud():
    def __init__(self, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.pc_extension = "_" + 'pcd'
        self.mesh_extension  = "_" + 'stl'

        self.folder_mesh = 'Meshes'

        self.pointcloud_of_links = {}

        self.static_collision_points = {}
        self._dynamic_collision_points = {}

    def init_robot_folder(self, ws_path, robot_name='robot'):
        mesh_folder    = MeshHandler(ws_path)
        setattr(self, robot_name + self.folder_mesh, mesh_folder)

    def copy_file(self, old_path, offset=np.zeros(6), file_rename=None, robot_name='robot'):
        if not hasattr(self, robot_name + self.folder_mesh):
            raise ValueError(f"Robot folder {robot_name} not found")
        mesh_folder:MeshHandler = getattr(self, robot_name + self.folder_mesh)
        mesh_folder.copy_file(old_path, offset, file_rename)

    def add_mesh(self, link_names:List[str], namespace='', robot_name='robot'):
        def add_single_mesh(link_name, namespace, robot_name):
            mesh_folder:MeshHandler = getattr(self, robot_name + self.folder_mesh)
            mesh = mesh_folder.load(link_name)
            mesh_name = namespace + link_name + self.mesh_extension
            if mesh is None:
                return
            else:
                print("\033[92m" + f'Mesh "{mesh_name}" loaded' + "\033[0m")
                setattr(self, mesh_name, mesh)
        

        if not hasattr(self, robot_name + self.folder_mesh):
            raise ValueError(f"Robot folder {robot_name} not found")
        
        for name in link_names:
            add_single_mesh(name, namespace, robot_name)
                


    def add_pointcloud(self, link_name:Union[List[str],str], num_points, namespace='', robot_name='robot', debug = False):
        if not hasattr(self, robot_name + self.folder_mesh):
            raise ValueError(f"Robot folder {robot_name} not found")
        
        mesh_folder:MeshHandler = getattr(self, robot_name + self.folder_mesh)
        if link_name == "all":
            mesh_names = mesh_folder.get_file_names()
            for mesh_name in mesh_names:
                _pc_name = namespace + mesh_name + self.pc_extension
                self.pointcloud_of_links[namespace + mesh_name] = self.sample_points_from_mesh_surface(namespace + mesh_name, num_points, debug=debug)
                # setattr(self, _pc_name, self.sample_points_from_mesh_surface(link_name, num_points, debug=debug))
        else:
            
            self.pointcloud_of_links[namespace + link_name] = self.sample_points_from_mesh_surface(namespace + link_name, num_points, debug=debug)

    def add_static_collision_obj(self, obj_name:str, mesh_path:str, position:np.array=None, num_of_points=1000, debug=False  ):
        if isinstance(position, list):
            position = torch.tensor(position, dtype=self.dtype, device=self.device)
        elif isinstance(position, np.ndarray):
            position = torch.tensor(position, dtype=self.dtype, device=self.device)

        mesh = Mesh(mesh_path)
        
        points = direct_transform_points(homogeneous_matrix(position[0], position[1], position[2], position[3], position[4], position[5], device=self.device, dtype=self.dtype), mesh.surface_uniform_sampling(num_of_points, debug), device=self.device, dtype=self.dtype)
        
        self.static_collision_points[obj_name] = torch.tensor(points, dtype=self.dtype, device=self.device)

    def add_dynamic_collision_obj(self, obj_name:str, mesh_path:str, num_of_points=1000, debug=False):
        mesh = Mesh(mesh_path)
        self._dynamic_collision_points[obj_name] = torch.tensor(mesh.surface_uniform_sampling(num_of_points, debug), dtype=self.dtype, device=self.device)

    def get_static_collision_points(self, obj_list: List[str], as_dict: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if as_dict:
            return {
                name: self.static_collision_points[name].to(device=self.device, dtype=self.dtype)
                for name in obj_list
            }
        pts_list = [
            self.static_collision_points[name].to(device=self.device, dtype=self.dtype)
            for name in obj_list
        ]
        return torch.cat(pts_list, dim=0) if pts_list else torch.empty((0, 3), dtype=self.dtype, device=self.device)
    
    def get_dynamic_collision_obj(self, obj_name:List[str], position:List[torch.Tensor]):
        #check se hanno la stessa lunghezza allora batchare altrimenti cicllo for
        points = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        for i in range(len(obj_name)):
            points = torch.cat((points, direct_transform_points(homogeneous_matrix(position[i][0], position[i][1], position[i][2], position[i][3], position[i][4], position[i][5], device=self.device, dtype=self.dtype), self._dynamic_collision_points[obj_name[i]], device=self.device, dtype=self.dtype)), dim=0)
        return points
    
    def sample_points_from_mesh_surface(self, link_name, num_points, debug=False):
        '''
        Args:
            link_name: Name of the link to sample points from
            num_points: Number of points to sample
            debug: If True, prints the time taken to sample the points
        Returns:
            A tensor of shape (num_points, 3) containing the sampled points
        '''
        if getattr(self, link_name + self.mesh_extension) is None:
            return None
        mesh:Mesh =  getattr(self, link_name + self.mesh_extension)
        
        # point_cloud = mesh.surface_uniform_sampling(num_points)
        point_cloud = mesh.mixed_uniform_sampling(num_points, debug=False)
        # point_cloud = mesh.get_vertices()

        if debug:
            plot = 'matplotlib'
            if plot == "pyrender":
                show_mesh_and_pointcloud(mesh.mesh, point_cloud, title=link_name)
            elif plot == "matplotlib":
                show_two_pc_mtplt(mesh.mesh.vertices, point_cloud, title=link_name)

        return torch.tensor(point_cloud, device=self.device, dtype=self.dtype)
    
    def get_pc_tranformed(self, forward_dict: dict, as_dict: bool = False, link_names: List[str] = None) -> Union[torch.Tensor, dict]:
        if link_names is None:
            link_names = list(self.pointcloud_of_links.keys())
        else:
            link_names = [ln for ln in link_names if ln in self.pointcloud_of_links]

        # Keep only links that have both pointcloud and transform.
        filtered = []
        pcs = []
        for ln in link_names:
            pc = self.pointcloud_of_links.get(ln)
            if pc is None:
                continue
            if ln not in forward_dict:
                continue
            filtered.append(ln)
            pcs.append(pc)

        if not filtered:
            return {} if as_dict else torch.empty((0, 3), dtype=self.dtype, device=self.device)

        # Fast path: batch transform if all pointclouds have same number of points.
        try:
            n_points = {int(pc.shape[0]) for pc in pcs}
            can_batch = (len(n_points) == 1)
        except Exception:
            can_batch = False

        if can_batch:
            forward_tensors = torch.stack([forward_dict[ln] for ln in filtered], dim=0)
            device = forward_tensors.device
            pc_tensor = torch.stack([pc.to(device=device, dtype=self.dtype) for pc in pcs], dim=0)
            ones = torch.ones((pc_tensor.shape[0], pc_tensor.shape[1], 1), device=device, dtype=self.dtype)
            pc_h = torch.cat([pc_tensor, ones], dim=2)
            world = torch.bmm(pc_h, forward_tensors.transpose(-1, -2))[:, :, :3]
            if as_dict:
                return {ln: world[i] for i, ln in enumerate(filtered)}
            return world.reshape(-1, 3)

        # Robust path: per-link transform (supports variable point counts).
        out_dict = {}
        out_list = []
        for ln, pc in zip(filtered, pcs):
            T = forward_dict[ln]
            device = T.device
            pts = pc.to(device=device, dtype=self.dtype).reshape(-1, 3)
            ones = torch.ones((pts.shape[0], 1), device=device, dtype=self.dtype)
            pts_h = torch.cat([pts, ones], dim=1)
            world = (pts_h @ T.transpose(-1, -2))[:, :3]
            if as_dict:
                out_dict[ln] = world
            else:
                out_list.append(world)

        if as_dict:
            return out_dict
        return torch.cat(out_list, dim=0) if out_list else torch.empty((0, 3), dtype=self.dtype, device=self.device)

    def order_pc_by_link_names(self, link_names:List[str])-> torch.Tensor:
        return torch.cat([self.pointcloud_of_links[link].unsqueeze(0) for link in link_names], dim=0).to(self.device).to(self.dtype) # [num_links, num_points, 3]

    def offset_and_save_mesh(self, link_name:str, offset:np.ndarray,):
        #THIS OPERATION WILL OVERWRITE THE MESH FILE
        mesh_name = link_name + self.mesh_extension
        if not hasattr(self, mesh_name):
            raise ValueError(f"Mesh '{mesh_name}' not found")
        
        mesh:Mesh = getattr(self, mesh_name)
        mesh.apply_offset(offset)
        mesh.save(path=mesh.path)

    def get_mesh_names(self):
        link_names = []
        for name in dir(self):
            if name.endswith(self.mesh_extension):
                link_names.append(name[:-len(self.mesh_extension)])
        return link_names

    def get_pointcloud_names(self):
        return list(self.pointcloud_of_links.keys())
    

    def get_robot_meshes(self,  mesh_names: List[str] = None,  type_of_mesh: Literal['trimesh', 'rdf', 'pyvista'] = 'trimesh') -> List:
        from src.utils.MeshUtils import trimesh_to_pyvista
        meshes = []
        names = mesh_names if mesh_names is not None else self.get_mesh_names()
        for link in names:
            if type_of_mesh == 'trimesh':
                if hasattr(self, link + self.mesh_extension):
                    mesh = getattr(self, link + self.mesh_extension).mesh
                    meshes.append(mesh)

            elif type_of_mesh == 'rdf':
                if hasattr(self, link + self.model_extension):
                    mesh = super().to_mesh(link)
                    meshes.append(mesh)

            elif type_of_mesh == 'pyvista':
                # supporta sia i mesh salvati che i modelli
                if hasattr(self, link + self.mesh_extension):
                    mesh = getattr(self, link + self.mesh_extension).mesh
                    meshes.append(trimesh_to_pyvista(mesh))
                elif hasattr(self, link + self.model_extension):
                    mesh = super().to_mesh(link)
                    meshes.append(trimesh_to_pyvista(mesh))

        return meshes
