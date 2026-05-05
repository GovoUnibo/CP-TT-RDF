
import os
import sys
import numpy as np
import torch

try:
    from src.core.assets.ModelHandler import ModelHandler
    from src.core.assets.MeshHandler import MeshHandler
    from src.core.assets.DatasetHandler import DatasetHandler
    from src.core.train.train_config import TrainConfig
except ModuleNotFoundError:
    # Allow standalone usage by adding RDF root to sys.path
    _rdf_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _rdf_root not in sys.path:
        sys.path.append(_rdf_root)
    from src.core.assets.ModelHandler import ModelHandler
    from src.core.assets.MeshHandler import MeshHandler
    from src.core.assets.DatasetHandler import DatasetHandler
    from src.core.train.train_config import TrainConfig


class SDFTrain:

    def __init__(self, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        self.mesh_extension  = "_" + 'stl'
        self.model_extension = "_" + 'pt'
        self.dataset_extension = "_" + 'npy'
        self.pc_extension = "_" + 'pcd'

        self.folder_mesh = 'Meshes'
        self.folder_model = 'Models'
        self.folder_dataset = 'Datasets'
        self.debug = False   
        
    
    def init_robot_folder(self, ws_path, robot_name):
        mesh_folder    = MeshHandler(ws_path)
        dataset_folder = DatasetHandler(ws_path)
        model_folder   = ModelHandler(ws_path)
        setattr(self, robot_name + self.folder_mesh, mesh_folder)
        setattr(self, robot_name + self.folder_model, model_folder)
        setattr(self, robot_name + self.folder_dataset, dataset_folder)

    def copy_file(self, old_path, offset=np.zeros(6), file_rename=None, robot_name='robot'):
        getattr(self, robot_name + self.folder_mesh).copy_file(old_path, offset, file_rename)

    def create_dataset(self, link_name, robot_name='robot', **kwargs): #link name dataset name
        if not hasattr(self, robot_name + self.folder_dataset):
            raise ValueError(f"Robot folder {robot_name} not found")
        dataset_folder:DatasetHandler = getattr(self, robot_name + self.folder_dataset)
        ds = dataset_folder.load(link_name)
        if ds is None:
            mesh_folder:MeshHandler = getattr(self, robot_name + self.folder_mesh)
            mesh = mesh_folder.load(link_name)
            ds = dataset_folder.create(link_name, mesh, debug=self.debug, **kwargs)
        
        return ds

    def create_model(self, cfg: TrainConfig, robot_name="robot", **kwargs):
        self.debug = cfg.debug

        create_per_link  = cfg.classic.run or cfg.cp_link.run or cfg.tt_link.run
        create_per_robot = cfg.cp_robot.run or cfg.tt_robot.run

        # -------- per-link --------
        if create_per_link:
            model_folder: ModelHandler = getattr(self, robot_name + self.folder_model)

            for link_name in cfg.links_to_train:
                ds = self.create_dataset(link_name, robot_name=robot_name, **kwargs)
                if ds is None:
                    raise ValueError(f"Dataset None for link {link_name}")

                if cfg.classic.run:
                    model_folder.create_weights(ds, cfg, device=self.device, dtype=self.dtype)

                if cfg.cp_link.run:
                    model_folder.create_cp(ds, cfg, device=self.device, dtype=self.dtype)

                if cfg.tt_link.run:
                    model_folder.create_tt(ds, cfg, device=self.device, dtype=self.dtype)

        # -------- per-robot (4D) --------
        if create_per_robot:
            ds_list = [self.create_dataset(ln, robot_name=robot_name, **kwargs) for ln in cfg.links_to_train]
            if any(ds is None for ds in ds_list):
                missing = [ln for ln, ds in zip(cfg.links_to_train, ds_list) if ds is None]
                raise ValueError(f"Missing datasets for links: {missing}")

            model_folder: ModelHandler = getattr(self, robot_name + self.folder_model)

            if cfg.cp_robot.run:
                model_folder.create_cp_robot(ds_list, cfg, model_name=robot_name, device=self.device, dtype=self.dtype)

            if cfg.tt_robot.run:
                model_folder.create_tt_robot(ds_list, cfg, model_name=robot_name, device=self.device, dtype=self.dtype)

        
        if cfg.sphere.run:

            mesh_folder:MeshHandler = getattr(self, robot_name + self.folder_mesh)
            model_folder:ModelHandler = getattr(self, robot_name + self.folder_model)
            if len(cfg.links_to_train) != len(cfg.sphere.n_points) or len(cfg.links_to_train) != len(cfg.sphere.n_spheres):
                raise ValueError("cfg.sphere.n_points and cfg.sphere.n_spheres must have the same length as cfg.links_to_train")
            for i in range(len(cfg.links_to_train)):
                link_name = cfg.links_to_train[i]
                n_points = cfg.sphere.n_points[i]
                n_spheres = cfg.sphere.n_spheres[i]
                mesh = mesh_folder.load(link_name)
                mesh.name = link_name
                model_folder.create_spheres(mesh, n_points=n_points, n_spheres=n_spheres, debug=cfg.debug, device=self.device, dtype=self.dtype)
