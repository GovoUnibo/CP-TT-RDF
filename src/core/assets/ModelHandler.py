from src.core.assets.FolderManage import FolderManage
from torch import load as torch_load
from torch import save as torch_save
from src.core.assets.train_wrapper import TrainWrapper
import os
from src.core.train.train_config import TrainConfig
import torch
from typing import Union
from src.core.assets.entities.models import CpLinkModel, TtLinkModel, WeightsLinkModel, CpRobotModel, TtRobotModel
from src.core.assets.load_model_wrapper import *

def _infer_kind(d: dict) -> str:
    # robot TT 4D
    if "G0" in d and "G1" in d and "G2" in d and "G3" in d:
        return "robot_tt"
    # robot CP 4D
    if "cp_V" in d and "cp_A" in d and "cp_B" in d and "cp_C" in d:
        return "robot_cp"
    # link TT
    if ("G1" in d and "G2" in d and "G3" in d) or ("tt_core_G1" in d and "tt_core_G2" in d and "tt_core_G3" in d):
        return "link_tt"
    # link CP
    if ("A" in d and "B" in d and "C" in d) or ("cp_A" in d and "cp_B" in d and "cp_C" in d):
        return "link_cp"
    # link weights
    if "weights" in d:
        return "link_w"
    
    if "centers" in d and "radii" in d:
        return "link_sphere"

    raise ValueError(f"Tipo modello non riconosciuto. Chiavi trovate: {list(d.keys())[:40]}")

ModelT = Union[CpLinkModel, TtLinkModel, WeightsLinkModel, CpRobotModel, TtRobotModel]


class ModelHandler(FolderManage):
    def __init__(self, ws_path, extention='pt'):
        super().__init__(ws_path, 'Models', extention)
    
    def load_model(self, model_name, device, dtype) -> ModelT:
        '''
        Load a model from the model folder using the model name
        '''
     
        path = super().get_file_path(model_name, debug=True)
       
        if path is None:
            raise FileNotFoundError(
                "\033[91m"
                + (
                    f'Model "{model_name}" not found in {self.get_path()}.\n'
                    "No fallback is performed.\n"
                    "Fix: point the RDF workspace (ws_path) to the folder that contains the expected Models/, "
                    "or copy/rename the model files so the expected filename exists."
                )
                + "\033[0m"
            )
        
        model_dict = torch_load(path, map_location=torch.device(device))

        kind = _infer_kind(model_dict)

        if kind == "robot_tt":
            return load_robot_tt_model(model_dict, device=device, dtype=dtype)
        if kind == "robot_cp":
            return load_robot_cp_model(model_dict, device=device, dtype=dtype)
        if kind == "link_tt":
            return load_link_tt_model(model_dict, device=device, dtype=dtype)
        if kind == "link_cp":
            return load_link_cp_model(model_dict, device=device, dtype=dtype)
        if kind == "link_w":
            return load_link_weight_model(model_dict, device=device, dtype=dtype)
        if kind == "link_sphere":
            return load_link_sphere_model(model_dict, device=device, dtype=dtype)

        

        raise RuntimeError("Error in loading model")
        
    
    def save(self, model_pt):
        '''
        Save a model to the model folder
        '''
        path = super().get_path()
        model_name = model_pt['file_name'] + model_pt.get('file_suffix', '')
        path = os.path.join(path, model_name + '.' + self.extension)
        # print("\033[1m" + f'SAVING MODEL FILE: {model_name} in --> {path}' + "\033[0m")

        # print("\33[90m" + "Using the foolwing domain [Min, Max]: " + f"[{self.model_pt['sdf_domain_min']}, {self.model_pt['sdf_domain_max']}]" + "\033[0m")
        # print("\33[90m" + "Using the foolwing number of functions: " + f"{self.n_func}" + "\033[0m")

        torch_save(model_pt, path)
        print("\033[92m" + f'[SAVED] FILE: {model_name} in --> {path}' + "\033[0m")
    
    def create_weights(self, dataset:dict, cfg:TrainConfig , device, dtype) -> dict:
        '''
        Train a model using a dataset
        '''
        print("\033[1m" + f'CREATING MODEL FILE: {dataset["file_name"]}' + "\033[0m")
        model = TrainWrapper(device=device, dtype=dtype)
        model.debug = cfg.debug
        model.set_weight_model()
        model.initialize_model(dataset)
        dataset = model.filter_dataset(dataset)
        points_inside = dataset['near_points'][dataset['near_sdf'] < 0]
        model.fit_ellipsoid(points_inside)

        model.train(dataset, cfg.classic.n_func, epoches=cfg.classic.iters, sample_near=cfg.classic.batch_near, sample_rand=cfg.classic.batch_rand)


        self.save(model.get_model_pt())
    
    def create_cp(self, dataset:dict, cfg:TrainConfig , device, dtype) -> dict:
        '''
        Train a model using a dataset
        '''
        print("\033[1m" + f'CREATING MODEL FILE: {dataset["file_name"]}' + "\033[0m")
        model = TrainWrapper(device=device, dtype=dtype)
        model.debug = cfg.debug
        model.set_cp_model() 
        model.initialize_model(dataset)
        dataset = model.filter_dataset(dataset)
        points_inside = dataset['near_points'][dataset['near_sdf'] < 0]
        model.fit_ellipsoid(points_inside)

        model.train_cp( dataset,n_func=cfg.cp_link.n_func,rank=cfg.cp_link.rank,ridge=cfg.cp_link.ridge,lr=cfg.cp_link.lr,iters=cfg.cp_link.iters,batch_size=cfg.cp_link.batch_size,method=cfg.cp_link.method,weights=cfg.cp_link.sample_weights)
        
        self.save(model.get_model_pt())

    
    def create_tt(self, dataset:dict, cfg:TrainConfig , device, dtype) -> dict:
        '''
        Train a model using a dataset
        '''
        print("\033[1m" + f'CREATING MODEL FILE: {dataset["file_name"]}' + "\033[0m")
        model = TrainWrapper(device=device, dtype=dtype)
        model.debug = cfg.debug
        model.set_tt_model() 
        model.initialize_model(dataset)
        dataset = model.filter_dataset(dataset)
        points_inside = dataset['near_points'][dataset['near_sdf'] < 0]
        model.fit_ellipsoid(points_inside)
        
        model.train_tt( dataset,
                            n_func=cfg.tt_link.n_func,
                            tt_ranks=cfg.tt_link.ranks,
                            ridge=cfg.tt_link.ridge,
                            lr=cfg.tt_link.lr,
                            iters=cfg.tt_link.iters,
                            batch_size=cfg.tt_link.batch_size,
                            method=cfg.tt_link.method,
                            weights=cfg.tt_link.sample_weights)



        self.save(model.get_model_pt())
    
    def create_cp_robot(self, list_ds, cfg:TrainConfig,  model_name='robot', device='cuda', dtype=torch.float32) -> dict:

        model = TrainWrapper(device=device, dtype=dtype)
        model.set_robot_cp_model()

        model.initialize_robot_model(list_ds, model_name)
        model.debug = cfg.debug

        model.train_4Dcp(   list_ds,
                            method=cfg.cp_robot.method,
                            n_func=cfg.cp_robot.n_func,
                            rank=cfg.cp_robot.rank,
                            ridge=cfg.cp_robot.ridge,
                            lr=cfg.cp_robot.lr,
                            iters=cfg.cp_robot.iters,
                            weights=cfg.cp_robot.sample_weights,
                            batch_size=cfg.cp_robot.batch_size, 
                            T_list=cfg.fk_matrices,
                            )

        self.save(model.get_robot_model_pt())


    def create_tt_robot(self, list_ds, cfg:TrainConfig,  model_name='robot', device='cuda', dtype=torch.float32) -> dict:
        model = TrainWrapper(device=device, dtype=dtype)
        model.set_robot_tt_model()
        model.initialize_robot_model(list_ds, model_name)
        model.debug = cfg.debug

        model.train_4Dtt(   list_ds,
                            method=cfg.tt_robot.method,
                            n_func=cfg.tt_robot.n_func,
                            tt_ranks=cfg.tt_robot.ranks,
                            ridge=cfg.tt_robot.ridge,
                            lr=cfg.tt_robot.lr,
                            iters=cfg.tt_robot.iters,
                            weights=cfg.tt_robot.sample_weights,
                            batch_size=cfg.tt_robot.batch_size, 
                            T_list=cfg.fk_matrices,
                            )


        self.save(model.get_robot_model_pt())

    def create_spheres(self, mesh, n_points, n_spheres, debug, device='cuda', dtype=torch.float32) -> dict:
        '''
        Create a sphere model for a given mesh
        '''
        model = TrainWrapper(device=device, dtype=dtype)
        
        model.debug = debug
        model.set_sphere_model()
        model.train_sphere(mesh, n_points=n_points, n_spheres=n_spheres)
        
        self.save(model.get_model_pt())
    
