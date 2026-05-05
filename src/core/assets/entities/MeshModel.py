import os
import trimesh
import trimesh.exchange
from trimesh.exchange.export import export_mesh
from trimesh import load, Trimesh, scene
from os import listdir
from os.path import isfile, join
from src.utils.MeshUtils import overlap_origins
import torch
import numpy as np
import time
from pymeshfix import _meshfix #pip install pymeshfix pip install pyvista
#pip install pycollada becouse trimesh can't read .dae files by default

from src.visu.debug_visu_utils import show_pointcloud_matplot
from src.core.math.transformations import homogeneous_matrix
 

class Mesh():
    name = None
    mesh = None
    path = None
    
    def __init__(self, mesh):
        

        if isinstance(mesh, Trimesh):
            self.mesh = mesh
        elif isinstance(mesh, str):
            # print("\033[1m" + f'LOADING MESH FILE: {mesh}' + "\033[0m")
            self.path = mesh
            self.mesh = load(mesh)
            try:
                self.mesh.metadata['file_name'] = self.mesh.metadata['file_name'].split('.')[0]
            except:
                self.mesh.metadata['file_name'] = os.path.basename(mesh.split('.')[0])

    def save(self, path, file_rename=None):       
        if file_rename is None:
            unified_path = path
        else:
            unified_path = os.path.join(path, file_rename)
        
        mesh_name = os.path.basename(unified_path)
            
        print("\033[1m" + f'SAVING MESH FILE: {mesh_name} in --> {path}' + "\033[0m")

        export_mesh(self.mesh, unified_path)
        print("\033[92m" + f'[SAVED] MESH FILE: {mesh_name} in --> {path}' + "\033[0m")
        print("\033[1m" + f'---------------------' + "\033[0m")
        time.sleep(0.5)
    
    @staticmethod
    def mesh_fix(mesh_path) -> Trimesh:
        # print("\033[91m" + "Mesh is not watertight, attempting to repair..." + "\033[0m")
        tin = _meshfix.PyTMesh()
        tin.load_file(mesh_path)
        for i in range(10):
            
            print('There are {:d} boundaries'.format(tin.boundaries()))
            if tin.boundaries() > 0:
                tin.fill_small_boundaries(0.001)
        
            # tin.clean(max_iters=10, inner_loops=3)
        tin.save_file(mesh_path)

        # return trimesh.load(mesh_path)

    @staticmethod
    def dae_to_stl(old_path, new_path, offset=np.zeros(6)):
        # print("\033[93m" + f'Loading file: {old_path}' + "\033[0m")
        mesh = trimesh.load(old_path, force='mesh')
 

        # print("\033[92m" + f'COPYING FILE: {old_path} to {new_path}' + "\033[0m")
        print("\033[93m" + f'Converting from .dae to .stl' + "\033[0m")
        if isinstance(offset, np.ndarray):
            offset = torch.tensor(offset)
        print("\033[93m" + f'Applying offset: {offset}' + "\033[0m")
        transform_matrix = homogeneous_matrix(offset[0], offset[1], offset[2], offset[3], offset[4], offset[5])
        if isinstance(offset, torch.Tensor):
            transform_matrix = transform_matrix.cpu().numpy()
        mesh.apply_transform(transform_matrix)        
        mesh.export(file_obj=new_path, file_type='stl')
        time.sleep(1)
        print("\033[92m" + f'COPYING FILE: {old_path} to {new_path} [DONE]' + "\033[0m")

        if not mesh.is_watertight:
             print("\033[91m" + "Mesh SAVED, but is not watertight, attempting to repair..." + "\033[0m")
             decision = input("Do you want to repair the mesh? [y/n]: ")
             if decision == 'y':
                Mesh.mesh_fix(new_path)
                # print("\033[92m" + "Mesh repaired!" + "\033[0m")


    def set(self, mesh):
        if not isinstance(mesh, Trimesh):
            raise ValueError("\033[91m" + "The mesh must be a trimesh object" + "\033[0m")
        self.mesh = mesh

    def get(self):
        return self.mesh
    
    def apply_offset(self, tf_matrix):
        if isinstance(tf_matrix, torch.Tensor):
            tf_matrix = tf_matrix.cpu().numpy() if tf_matrix.is_cuda else tf_matrix.numpy()
        self.mesh.apply_transform(tf_matrix)

    def create_box(self, size, origin = [0, 0, 0], path = None):
        mesh = trimesh.creation.box(size)
        mesh.metadata = self.mesh.metadata
        self.mesh = mesh.apply_translation(origin)
        if path is not None:
            self.save(path)
    
    @staticmethod
    def to_cylinder(mesh_path:str, overlap_frames=True, save_path:str=None):
        # Fit a bounding cylinder to the mesh
        mesh = trimesh.load(mesh_path)
        cylinder_params = trimesh.bounds.minimum_cylinder(mesh)
        
        # Generate vertices for the cylinder
        cylinder_vertices  = trimesh.creation.cylinder(radius=cylinder_params['radius'], height=cylinder_params['height'], sections=100)
        

        if overlap_frames:
            cylinder_vertices = overlap_origins(mesh, cylinder_vertices)
        
        cylinder_vertices.metadata = mesh.metadata

        mesh = cylinder_vertices
        
        if save_path is not None:
            mesh.export(file_obj=save_path, file_type='stl')
            print("\033[92m" + f'[SAVED] MESH FILE: {mesh_path} in --> {save_path}' + "\033[0m")
    
    def get_vertices(self):
        return self.mesh.vertices


    def mixed_uniform_sampling(self, num_points, surface_ratio=0.7, debug=False, random_seed=0):
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise ValueError("\033[91m" + "The mesh must be una Trimesh" + "\033[0m")
        
        num_surface = int(num_points * surface_ratio)
        surface_points = self.surface_uniform_sampling(num_surface, show=False, random_seed=random_seed)
        # print(f"\033[92m" + f'Sampled {surface_points.shape[0]} points from mesh surface' + "\033[0m")
        # show_pointcloud_matplot(surface_points)
        num_volume = num_points - num_surface

        volume_points = self.random_volume_sampling(num_volume, show=False, random_seed=random_seed)
        # print(f"\033[92m" + f'Sampled {volume_points.shape[0]} points from mesh volume' + "\033[0m")
        # show_pointcloud_matplot(volume_points)

        if volume_points.shape[0] > num_volume: # se ci sono più punti del necessario, ne seleziono un sottoinsieme casuale
            np.random.seed(random_seed)
            selected_indices = np.random.choice(volume_points.shape[0], num_volume, replace=False)
            volume_points = volume_points[selected_indices]
        elif volume_points.shape[0] < num_volume: # se ci sono meno punti del necessario, ne aggiungo altri casuali
            additional_points = self.surface_uniform_sampling(num_volume - volume_points.shape[0], show=False, random_seed=random_seed+1)
            volume_points = np.vstack((volume_points, additional_points))

        points = np.vstack((surface_points, volume_points))
        if debug:
            print("\033[92m" + f'Showing pointcloud' + "\033[0m")
            show_pointcloud_matplot(points)
        return points

    def random_volume_sampling(self, num_points, show=False, random_seed=0):
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise ValueError("\033[91m" + "The mesh must be una Trimesh" + "\033[0m")
        
        rng = np.random.default_rng(random_seed)
        mins, maxs = self.mesh.bounds

        # oversample 3x per sicurezza, poi filtriamo
        candidates = rng.random((num_points * 3, 3))
        candidates = mins + (maxs - mins) * candidates

        solid = self.mesh.convex_hull
        mask = solid.contains(candidates)
        inside = candidates[mask]

        # se bastano → prendi i primi num_points
        if inside.shape[0] >= num_points:
            points = inside[:num_points]
        else:
            # fallback: continua a generare finché hai abbastanza punti
            while inside.shape[0] < num_points:
                extra = rng.random((num_points, 3))
                extra = mins + (maxs - mins) * extra
                inside = np.vstack([inside, extra[solid.contains(extra)]])
            points = inside[:num_points]

        if show:
            print(f"\033[92mShowing {points.shape[0]} random interior points\033[0m")
            show_pointcloud_matplot(points)

        return points

    def surface_uniform_sampling(self, num_points, show=False, random_seed=0):
        
        if not isinstance(self.mesh, Trimesh):
            raise ValueError("\033[91m" + "The mesh must be a trimesh object" + "\033[0m")
        from sklearn.cluster import KMeans
        """
        Restituisce num_points distribuiti in maniera uniforme e deterministica
        sulla superficie della mesh, usando Poisson-disk sampling.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            points, _ = trimesh.sample.sample_surface_even(self.mesh, num_points, seed=random_seed)
        
        if points.shape[0] < num_points:
            remaining_points = num_points - points.shape[0]
            remaining_points = trimesh.sample.sample_surface(self.mesh, remaining_points)[0]
            points = np.vstack((points, remaining_points))

        kmeans = KMeans(n_clusters=num_points, n_init=10, random_state=random_seed)
        kmeans.fit(points)
        points = kmeans.cluster_centers_


        # print(f"\033[92m" + f'Sampled {points.shape[0]} points from mesh surface' + "\033[0m")

        if show:
            print("\033[92m" + f'Showing pointcloud' + "\033[0m")
            show_pointcloud_matplot(points)

        return points
    
    
    def __repr__(self) -> str:
        return f'Mesh: {self.name}'
        


    

