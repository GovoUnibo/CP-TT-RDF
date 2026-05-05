from src.core.assets.FolderManage import FolderManage
from src.core.math.transformations import homogeneous_matrix
from src.core.assets.entities.MeshModel import Mesh
# from rdf.utils.MeshUtils import transform_to_voxel, surface_rdm_uniform_sampling
import os, torch
from trimesh.exchange.export import export_mesh


import numpy as np

def create_mesh_class(class_name):
    # Creazione della classe utilizzando type() e ereditando dalla classe template
    dynamic_class = type(class_name, (Mesh,), {})
    return dynamic_class

class MeshHandler(FolderManage):
    debug = False
    def __init__(self, ws_path, extention='stl'):
        super().__init__(ws_path, 'Meshes', extention)

    @staticmethod
    def _mesh_alias_candidates(mesh_name: str):
        """
        Build filename candidates to handle common link-vs-file naming mismatches.
        Examples:
        - ur_left_shoulder_link -> shoulder_link
        - left_summit_xls_base_link -> summit_xls_base_link
        """
        candidates = [mesh_name]

        if mesh_name.startswith("ur_left_"):
            candidates.append(mesh_name[len("ur_left_"):])
        if mesh_name.startswith("ur_right_"):
            candidates.append(mesh_name[len("ur_right_"):])

        if mesh_name.startswith("left_summit_xls_"):
            candidates.append("left_summit_" + mesh_name[len("left_summit_xls_"):])
            candidates.append("summit_xls_" + mesh_name[len("left_summit_xls_"):])
        if mesh_name.startswith("right_summit_xls_"):
            candidates.append("right_summit_" + mesh_name[len("right_summit_xls_"):])
            candidates.append("summit_xls_" + mesh_name[len("right_summit_xls_"):])
        if mesh_name.startswith("left_summit_"):
            candidates.append("summit_xls_" + mesh_name[len("left_summit_"):])
        if mesh_name.startswith("right_summit_"):
            candidates.append("summit_xls_" + mesh_name[len("right_summit_"):])

        if mesh_name in ("ur_left_base_link", "ur_right_base_link", "base_link"):
            candidates.append("base_link_inertia")

        # Preserve order, remove duplicates/empty.
        out = []
        seen = set()
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out
    
    def load(self, mesh_name):
        '''
        Get all the meshes loaded
        '''
        # print("Loading mesh: ", mesh_name)
        mesh_path = None
        for candidate in self._mesh_alias_candidates(mesh_name):
            mesh_path = super().get_file_path(candidate, debug=False)
            if mesh_path is not None:
                break

        if mesh_path is None:
            raise ValueError(f"033[91m" + f'Mesh "{mesh_name}" not found' + "\033[0m")
        
        return Mesh(mesh_path)

    def copy_file(self, old_path, offset=np.zeros(6), file_rename=None):
        
        file_path, extension = os.path.splitext(old_path)
        file_name = os.path.basename(old_path).split('.')[0]
        file_rename = file_rename + ".stl" if file_rename is not None else file_name + ".stl"
        new_path = os.path.join(self.get_path(), file_rename)
        if super().check_file_existence(new_path, debug=False): # if already exists, do not copy
            return
        
        if extension == ".dae":
            Mesh.dae_to_stl(old_path, new_path, offset)
            return
        
        mesh = Mesh(old_path)

        if np.any(offset != 0):
            if isinstance(offset, np.ndarray):
                offset = torch.tensor(offset)
            
            mesh.apply_offset(homogeneous_matrix(offset[0], offset[1], offset[2], offset[3], offset[4], offset[5]))

        if not mesh.mesh.is_watertight:
            print("\033[91m" + "Mesh SAVED, but is not watertight, attempting to repair..." + "\033[0m")
            decision = input("Do you want to repair the mesh? [y/n]: ")
            if decision == 'y':
                mesh.mesh.export(file_obj=new_path, file_type='stl')
                Mesh.mesh_fix(new_path)#salva il file riparato
                return 

        
        mesh.mesh.export(file_obj=new_path, file_type='stl')
            

        # return super().copy_file(old_path, new_path)

    def save(self, mesh, mesh_name):
        '''
        Save the mesh in the folder
        '''
        from trimesh import Trimesh
        if not isinstance(mesh, Trimesh):
            raise ValueError("\033[91m" + "The mesh must be a trimesh object" + "\033[0m")
        folder_path = self.get_path()
        path = os.path.join(folder_path, mesh_name + ".stl")
        export_mesh(mesh, path)



                
