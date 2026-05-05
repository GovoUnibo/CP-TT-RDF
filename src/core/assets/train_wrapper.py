
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch import save as torch_save
import numpy as np
import torch
import cvxpy as cp
from src.visu.debug_visu_utils import show_mesh, plot_ellipsoid, show_cp_sdf_level_curves
from src.core.train.weight_train import BernsteinWeightsTrain
from src.core.train.cp_train import BernsteinCPTrain
from src.core.train.tt_train import BernsteinTTrain
from src.core.train.sphere_train import build_spheres_from_mesh
from src.core.math.fit_ellipsoid import compute_ellipsoid_parameters
from src.utils.sdf_utils import sdf_to_mesh, cp_sdf_to_mesh, tt_sdf_to_mesh, cp4d_sdf_to_mesh_robot, tt4d_sdf_to_mesh_robot
from src.visu.pv_mesh_spheres_viewer import show_mesh_with_spheres
from typing import Optional, Union, Any, List, Tuple


from src.core.assets.entities.models import CpLinkModel, TtLinkModel, WeightsLinkModel, CpRobotModel, TtRobotModel, SphereModel



class TrainWrapper():
    def __init__(self, n_func=8, domain_min=-1, domain_max=1, device='cuda', dtype=torch.float32):
        self._domain_min = domain_min
        self._domain_max = domain_max
        self.n_func = n_func
        self._berstein_train    = BernsteinWeightsTrain(n_func=n_func, domain_min=domain_min, domain_max=domain_max, device=device, dtype=dtype)
        self._berstein_train_cp = BernsteinCPTrain(n_func=n_func, domain_min=domain_min, domain_max=domain_max, device=device, dtype=dtype)
        self._berstein_train_tt = BernsteinTTrain(n_func=n_func, domain_min=domain_min, domain_max=domain_max, device=device, dtype=dtype)
        self.device = device
        self.dtype  = dtype

        self.name = 'generc_model'
        self.instance_type = 'Model'    # For a lack of a better solution I used this to check the type of the object
        self.debug = False

        self.model_pt = None
        self.r_model_pt = None

    def set_cp_model(self):
        self.model_pt = CpLinkModel()
    def set_tt_model(self):
        self.model_pt = TtLinkModel()
    def set_weight_model(self):
        self.model_pt = WeightsLinkModel()
    
    def set_robot_cp_model(self):
        self.r_model_pt = CpRobotModel()
    def set_robot_tt_model(self):
        self.r_model_pt = TtRobotModel()
    
    def set_sphere_model(self):
        self.model_pt = SphereModel()


    def initialize_model(self, dataset: dict):
        self.model_pt.file_name       = dataset['file_name']
        self.model_pt.domain_min      = dataset['sdf_domain_min']
        self.model_pt.domain_max      = dataset['sdf_domain_max']
        self.model_pt.scale_factor    = dataset['mesh_scale_factor']
        self.model_pt.centroid_offset = dataset['mesh_centroid_offset']
    
    def initialize_robot_model(self, list_ds: list, model_name: str):
        self.r_model_pt.file_name = model_name
        self.r_model_pt.domain_min = min(ds['sdf_domain_min'] for ds in list_ds)
        self.r_model_pt.domain_max = max(ds['sdf_domain_max'] for ds in list_ds)

        self.r_model_pt.links_scale_factors = [ds['mesh_scale_factor'] for ds in list_ds]
        self.r_model_pt.links_centroids     = [ds['mesh_centroid_offset'] for ds in list_ds]

    def fit_ellipsoid(self, points_inside:np.ndarray):
        N = 5000

        if len(points_inside) > N:
            indices = np.random.choice(len(points_inside), size=N, replace=False)
            points_inside = points_inside[indices]
        A, c, eigen_vector = compute_ellipsoid_parameters(points_inside)
        self.model_pt.axes_ellipsoid = A
        self.model_pt.center_ellipsoid = c
        self.model_pt.eigen_vector_ellipsoid = eigen_vector

        if self.debug:
            plot_ellipsoid(center=self.model_pt.center_ellipsoid,
                        axes_lengths=self.model_pt.axes_ellipsoid,
                        eigenvectors=self.model_pt.eigen_vector_ellipsoid,
                        points=points_inside)
            
    def filter_dataset(self, dataset:dict):
        near_points = dataset['near_points']
        near_sdf = dataset['near_sdf']
        near_dist = np.linalg.norm(near_points, axis=1)
        near_mask = near_dist <= 1

        dataset['near_points'] = near_points[near_mask]
        dataset['near_sdf'] = near_sdf[near_mask]

        return dataset

    def train(self, dataset:dict, n_func,  epoches:int=200, sample_near:int=1024, sample_rand:int=64):
        if not isinstance(dataset, dict):
            raise ValueError("\033[91m" + "The dataset must be a dictionary" + "\033[0m")
        
        for key in ['near_points', 'near_sdf', 'query_points', 'query_sdf', 'mesh_scale_factor', 'mesh_centroid_offset', 'file_name']:
            if key not in dataset.keys():
                raise ValueError("\033[91m" + f"The dataset must have the key: {key}" + "\033[0m")
        

        print('\033[95m Training Bernstain function for ' + dataset['file_name'] + " Using Device: " + self.device + '\033[0m')

        self._berstein_train.set_points_domain(dataset['sdf_domain_min'], dataset['sdf_domain_max'])
        self._berstein_train.set_number_of_functions(n_func)

        self.weights = self._berstein_train.train(  dataset['near_points'], 
                                                    dataset['near_sdf'], 
                                                    dataset['query_points'], 
                                                    dataset['query_sdf'], 
                                                    epoches=epoches,
                                                    sample_near=sample_near,
                                                    sample_rand=sample_rand)
        
        self.model_pt.n_func = n_func
        self.model_pt.weights = self.weights



        self.model_pt.weights = self.weights
        
        if self.debug: 
            mesh = sdf_to_mesh(weights=self.weights,
                                nbData=128,
                                domain_max=self._domain_max, domain_min=self._domain_min,
                                scaling_factor=self.model_pt.scale_factor,
                                centroid_offset=self.model_pt.centroid_offset,
                                basis_function_from_3Dpoints=self._berstein_train.basis_function_from_3Dpoints)
            show_mesh(mesh)

    
    def train_cp(self, dataset:dict, n_func:int, rank:int=8, iters:int=20, lr: float = 5e-3, ridge:float=1e-6,
                batch_size: int = 65_536,
                method: str = "adam",
                weights: Optional[Union[torch.Tensor, np.ndarray]] = None,  ):
        if not isinstance(dataset, dict):
            raise ValueError("\033[91mIl dataset deve essere un dict\033[0m")
        for key in ['near_points','near_sdf','sdf_domain_min','sdf_domain_max','mesh_scale_factor','mesh_centroid_offset','file_name']:
            if key not in dataset: raise ValueError("\033[91mManca la chiave: "+key+"\033[0m")

        print('\033[95m' + 'Training CP Bernstain function for ' + dataset['file_name'] + " Using Device: " + self.device + '\033[0m')
        # dominio per le basi
        self._berstein_train_cp.set_points_domain(dataset['sdf_domain_min'], dataset['sdf_domain_max'])
        self._berstein_train_cp.set_number_of_functions(n_func)


        # opzionale: pesi per campione
        sw = weights
        if weights is not None:
            sw = torch.as_tensor(weights, device=self.device, dtype=self.dtype).reshape(-1)

        method_l = str(method).lower()
        if method_l == "adam":
            A, B, C, lam = self._berstein_train_cp.train_cp(
                points_near=dataset['near_points'],
                sdf_near=dataset['near_sdf'],
                points_rand=dataset['query_points'],
                sdf_rand=dataset['query_sdf'],
                rank=rank,
                iters=iters,
                lr=lr,
                ridge=ridge,
                batch_size=batch_size,
                weights=sw,
            )
        elif method_l == "als":
            A, B, C, lam = self._berstein_train_cp.train_cp_als(
                points_near=dataset['near_points'],
                sdf_near=dataset['near_sdf'],
                points_rand=dataset['query_points'],
                sdf_rand=dataset['query_sdf'],
                rank=rank,
                iters=iters,
                ridge=ridge,
                batch_size=batch_size,
                weights=sw,
            )
        else:
            raise ValueError(f"CP method non supportato: {method}. Usa 'adam' o 'als'.")

        if self.debug:
            print("\033[93m" + "Visualizing CP mesh for debugging..." + "\033[0m")
            visualize_sdf_in_3d = True
            sdf_3d_res = 72
            sdf_cp_mesh = cp_sdf_to_mesh(
                A=A, B=B, C=C, lam=lam,
                nbData=128,
                domain_min=self._domain_min, domain_max=self._domain_max,
                scaling_factor=self.model_pt.scale_factor,
                centroid_offset=self.model_pt.centroid_offset,
                bernstein_matrix_1d=self._berstein_train_cp.build_bernstein_t,
            )
            show_cp_sdf_level_curves(
                A=A,
                B=B,
                C=C,
                lam=lam,
                domain_min=self._domain_min,
                domain_max=self._domain_max,
                scaling_factor=self.model_pt.scale_factor,
                centroid_offset=self.model_pt.centroid_offset,
                bernstein_matrix_1d=self._berstein_train_cp.build_bernstein_t,
                visualize_3d=visualize_sdf_in_3d,
                res_3d=sdf_3d_res,
            )
            if sdf_cp_mesh is None:
                input("CP mesh is None. Premi INVIO per continuare...")
            else:
                show_mesh(sdf_cp_mesh)

        self.model_pt.A    = A.detach().cpu().numpy()
        self.model_pt.B    = B.detach().cpu().numpy()
        self.model_pt.C    = C.detach().cpu().numpy()
        self.model_pt.lamd = lam.detach().cpu().numpy()
        self.model_pt.rank = int(rank)
        self.model_pt.n_func = int(n_func)
    

    
    def train_4Dcp(self, list_ds, n_func: int, rank: int = 8, iters: int = 20, ridge: float = 1e-6, lr: float = 5e-3, batch_size: int = 65_536,
        method: str = "adam",
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
        T_list: Optional[List[np.ndarray]] = None,):

        if not isinstance(list_ds, (list, tuple)) or len(list_ds) == 0:
                raise ValueError("\033[91m list_ds deve essere una lista non vuota di dataset \033[0m")

        # controlla chiavi minime
        required_keys = [
            'near_points', 'near_sdf',
            'query_points', 'query_sdf',
            'sdf_domain_min', 'sdf_domain_max',
            'mesh_scale_factor', 'mesh_centroid_offset', 'file_name'
        ]

        for i, ds in enumerate(list_ds):
            if not isinstance(ds, dict):
                raise ValueError(f"\033[91m list_ds[{i}] non è un dict \033[0m")
            for k in required_keys:
                if k not in ds:
                    raise ValueError(f"\033[91m Manca la chiave '{k}' nel dataset[{i}] \033[0m")

        print('\033[95m' + f"Training CP *robot-wide* su {len(list_ds)} link, device={self.device}" + '\033[0m')
        
        self._berstein_train_cp.set_points_domain(
            domain_min=min(ds['sdf_domain_min'] for ds in list_ds),
            domain_max=max(ds['sdf_domain_max'] for ds in list_ds)
        )
        self._berstein_train_cp.set_number_of_functions(n_func)

        points_near_list = [ds['near_points']  for ds in list_ds]
        sdf_near_list    = [ds['near_sdf']     for ds in list_ds]
        points_rand_list = [ds['query_points'] for ds in list_ds]
        sdf_rand_list    = [ds['query_sdf']    for ds in list_ds]

        method_l = str(method).lower()
        if method_l == "adam":
            V, A, B, C, lam = self._berstein_train_cp.train_cp_robot(
                points_near_list=points_near_list,
                sdf_near_list=sdf_near_list,
                points_rand_list=points_rand_list,
                sdf_rand_list=sdf_rand_list,
                rank=rank,
                iters=iters,
                lr=lr,
                ridge=ridge,
                batch_size=batch_size,
                weights_near_list=[None] * len(list_ds),
                weights_rand_list=[None] * len(list_ds),
            )
        elif method_l == "als":
            V, A, B, C, lam = self._berstein_train_cp.train_cp_robot_als(
                points_near_list=points_near_list,
                sdf_near_list=sdf_near_list,
                points_rand_list=points_rand_list,
                sdf_rand_list=sdf_rand_list,
                rank=rank,
                iters=iters,
                ridge=ridge,
                batch_size=batch_size,
                weights_near_list=[None] * len(list_ds),
                weights_rand_list=[None] * len(list_ds),
            )
        else:
            raise ValueError(f"CP-4D method non supportato: {method}. Usa 'adam' o 'als'.")

        self.r_model_pt.cp_V      = V.detach().cpu().numpy()
        self.r_model_pt.cp_A      = A.detach().cpu().numpy()
        self.r_model_pt.cp_B      = B.detach().cpu().numpy()
        self.r_model_pt.cp_C      = C.detach().cpu().numpy()
        self.r_model_pt.cp_lambda = lam.detach().cpu().numpy()
        self.r_model_pt.rank      = int(rank)       # occhio: nel tuo dataclass è rank (non cp_rank)
        self.r_model_pt.n_func    = int(n_func)
    


        if T_list is not None and self.debug:
            mesh_robot = cp4d_sdf_to_mesh_robot(
                A=A, B=B, C=C, lam=lam, V=V,
                list_ds=list_ds,
                T_list=T_list,               # da fk_panda(q)
                nbData=128,
                domain_min=self._domain_min,
                domain_max=self._domain_max,
                bernstein_matrix_1d=self._berstein_train_cp.build_bernstein_t,
            )
            if mesh_robot is not None:
                show_mesh(mesh_robot)
        
            
    def train_tt(self, dataset:dict, n_func:int, tt_ranks:tuple=(8,8), iters:int=20, lr: float = 5e-3, ridge:float=1e-6,
                batch_size: int = 65_536,
                method: str = "adam",
                weights: Optional[Union[torch.Tensor, np.ndarray]] = None,  ):
        if not isinstance(dataset, dict):
            raise ValueError("\033[91mIl dataset deve essere un dict\033[0m")
        for key in ['near_points','near_sdf','sdf_domain_min','sdf_domain_max','mesh_scale_factor','mesh_centroid_offset','file_name']:
            if key not in dataset: raise ValueError("\033[91mManca la chiave: "+key+"\033[0m")
        
        print('\033[95m' + 'Training TT Bernstain function for ' + dataset['file_name'] + " Using Device: " + self.device + '\033[0m')
        # dominio per le basi
        self._berstein_train_tt.set_points_domain(dataset['sdf_domain_min'], dataset['sdf_domain_max'])
        self._berstein_train_tt.set_number_of_functions(n_func)

        # opzionale: pesi per campione
        sw = weights
        if weights is not None:
            sw = torch.as_tensor(weights, device=self.device, dtype=self.dtype).reshape(-1)
        method_l = str(method).lower()
        if method_l == "adam":
            core_G1, core_G2, core_G3 = self._berstein_train_tt.train_tt(
                points_near=dataset['near_points'],
                sdf_near=dataset['near_sdf'],
                points_rand=dataset['query_points'],
                sdf_rand=dataset['query_sdf'],
                tt_ranks=tt_ranks,
                iters=iters,
                lr=lr,
                ridge=ridge,
                batch_size=batch_size,
                weights=sw,
            )
        elif method_l == "mals":
            core_G1, core_G2, core_G3 = self._berstein_train_tt.train_tt_mals(
                points_near=dataset['near_points'],
                sdf_near=dataset['near_sdf'],
                points_rand=dataset['query_points'],
                sdf_rand=dataset['query_sdf'],
                tt_ranks=tt_ranks,
                iters=iters,
                lr=lr,
                ridge=ridge,
                batch_size=batch_size,
                weights=sw,
            )
        elif method_l == "als":
            raise ValueError("TT method 'als' è deprecato. Usa 'mals' (consigliato) o 'adam'.")
        else:
            raise ValueError(f"TT method non supportato: {method}. Usa 'adam' o 'mals'.")

        self.model_pt.G1 = core_G1.detach().cpu().numpy()
        self.model_pt.G2 = core_G2.detach().cpu().numpy()
        self.model_pt.G3 = core_G3.detach().cpu().numpy()
        self.model_pt.ranks = (int(core_G1.shape[2]), int(core_G2.shape[2]))
        self.model_pt.n_func = int(n_func)

        if self.debug:
            sdf_tt_mesh = tt_sdf_to_mesh(
                G1=core_G1, G2=core_G2, G3=core_G3,
                nbData=128,
                domain_min=self._domain_min, domain_max=self._domain_max,
                scaling_factor=self.model_pt.scale_factor,
                centroid_offset=self.model_pt.centroid_offset,
                bernstein_matrix_1d=self._berstein_train_tt.build_bernstein_t
            )

            show_mesh(sdf_tt_mesh)


        
    def train_4Dtt(
        self,
        list_ds,
        n_func: int,
        tt_ranks: Tuple[int, int, int] = (4, 8, 4),
        iters: int = 20,
        ridge: float = 1e-6,
        lr: float = 5e-3,
        batch_size: int = 65_536,
        method: str = "adam",
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
        T_list: Optional[List[np.ndarray]] = None,
    ):
        """
        Training TT 4D *robot-wide* (tutti i link) sullo stesso dominio,
        gemello di train_4Dcp ma usando la decomposizione TT 4D con core
        G0,G1,G2,G3.

        Salva i core in self.r_model_pt:
        - 'G0','G1','G2','G3'
        - 'tt_ranks','tt_n_of_func'
        E opzionalmente mostra la mesh TT del robot alla posa data da T_list.
        """
        if not isinstance(list_ds, (list, tuple)) or len(list_ds) == 0:
            raise ValueError("\033[91m list_ds deve essere una lista non vuota di dataset \033[0m")

        required_keys = [
            'near_points', 'near_sdf',
            'query_points', 'query_sdf',
            'sdf_domain_min', 'sdf_domain_max',
            'mesh_scale_factor', 'mesh_centroid_offset', 'file_name'
        ]

        for i, ds in enumerate(list_ds):
            if not isinstance(ds, dict):
                raise ValueError(f"\033[91m list_ds[{i}] non è un dict \033[0m")
            for k in required_keys:
                if k not in ds:
                    raise ValueError(f"\033[91m Manca la chiave '{k}' nel dataset[{i}] \033[0m")

        print('\033[95m' + f"Training TT *robot-wide* su {len(list_ds)} link, device={self.device}" + '\033[0m')

        # dominio globale (come in train_4Dcp)
        dom_min = min(ds['sdf_domain_min'] for ds in list_ds)
        dom_max = max(ds['sdf_domain_max'] for ds in list_ds)

        self._berstein_train_tt.set_points_domain(
            domain_min=dom_min,
            domain_max=dom_max
        )
        self._berstein_train_tt.set_number_of_functions(n_func)

        # estrazione liste per-link
        points_near_list = [ds['near_points']  for ds in list_ds]
        sdf_near_list    = [ds['near_sdf']     for ds in list_ds]
        points_rand_list = [ds['query_points'] for ds in list_ds]
        sdf_rand_list    = [ds['query_sdf']    for ds in list_ds]

        method_l = str(method).lower()
        if method_l == "adam":
            G0, G1, G2, G3 = self._berstein_train_tt.train_tt_robot(
                points_near_list=points_near_list,
                sdf_near_list=sdf_near_list,
                points_rand_list=points_rand_list,
                sdf_rand_list=sdf_rand_list,
                tt_ranks=tt_ranks,
                iters=iters,
                lr=lr,
                ridge=ridge,
                batch_size=batch_size,
                weights_near_list=[None] * len(list_ds),
                weights_rand_list=[None] * len(list_ds),
            )
        elif method_l == "mals":
            G0, G1, G2, G3 = self._berstein_train_tt.train_tt_robot_mals(
                points_near_list=points_near_list,
                sdf_near_list=sdf_near_list,
                points_rand_list=points_rand_list,
                sdf_rand_list=sdf_rand_list,
                tt_ranks=tt_ranks,
                iters=iters,
                lr=lr,
                ridge=ridge,
                batch_size=batch_size,
                weights_near_list=[None] * len(list_ds),
                weights_rand_list=[None] * len(list_ds),
            )
        elif method_l == "als":
            raise ValueError("TT-4D method 'als' è deprecato. Usa 'mals' (consigliato) o 'adam'.")
        else:
            raise ValueError(f"TT-4D method non supportato: {method}. Usa 'adam' o 'mals'.")

        # salvataggio nel modello (versione numpy)
        self.r_model_pt.G0 = G0.detach().cpu().numpy()
        self.r_model_pt.G1 = G1.detach().cpu().numpy()
        self.r_model_pt.G2 = G2.detach().cpu().numpy()
        self.r_model_pt.G3 = G3.detach().cpu().numpy()
        self.r_model_pt.ranks = (int(G0.shape[2]), int(G1.shape[2]), int(G2.shape[2]))
        self.r_model_pt.n_func = int(n_func)


        if T_list is not None:
            mesh_robot = tt4d_sdf_to_mesh_robot(
                G0=G0,
                G1=G1,
                G2=G2,
                G3=G3,
                list_ds=list_ds,
                T_list=T_list,
                nbData=128,  # come in train_4Dcp
                domain_min=self._domain_min,   # o dom_min se preferisci
                domain_max=self._domain_max,   # o dom_max
                bernstein_matrix_1d=self._berstein_train_tt.build_bernstein_t,
            )

            if mesh_robot is not None:
                show_mesh(mesh_robot)
    
    def train_sphere(self, mesh, n_points:int=1000, n_spheres:int=50):

        sphere_dict = build_spheres_from_mesh(mesh, n_points=n_points, n_spheres=n_spheres)
        
        self.model_pt.centers = sphere_dict['centers']
        self.model_pt.radii   = sphere_dict['radii']

        self.model_pt.file_name = mesh.name 

        if self.debug:
            show_mesh_with_spheres(mesh_path=mesh.path, centers=self.model_pt.centers, radii=self.model_pt.radii, mesh_opacity=0.9, sphere_opacity=0.6)



    def get_model_pt(self):
        return self.model_pt.to_dict()

    def get_robot_model_pt(self):
        return self.r_model_pt.to_dict()



    # @staticmethod
    # def load(model_pt, device='cuda', dtype=torch.float32) -> Model:
    #     model_param:Model = Model()
    #     # print(model_pt)

    #     model_param.file_name = model_pt['file_name']
    #     model_param.domain_min = model_pt['domain_min']
    #     model_param.domain_max = model_pt['domain_max']
    #     model_param.scale_factor = torch.tensor(model_pt['scale_factor'], device=device, dtype=dtype)
    #     model_param.centroid_offset = torch.tensor(model_pt['centroid_offset'], device=device, dtype=dtype)
    #     model_param.center_ellipsoid = torch.tensor(model_pt['center_ellipsoid'], device=device, dtype=dtype) if model_pt['center_ellipsoid'] is not None else None
    #     model_param.axes_ellipsoid = torch.tensor(model_pt['axes_ellipsoid'], device=device, dtype=dtype) if model_pt['axes_ellipsoid'] is not None else None
    #     model_param.eigen_vector_ellipsoid = torch.tensor(model_pt['eigen_vector_ellipsoid'], device=device, dtype=dtype) if model_pt['eigen_vector_ellipsoid'] is not None else None
        
    #     print("\033[38;2;255;165;0m" + f'LOADING MODEL BERNSTEIN: {model_param.file_name}' + "\033[0m")
        
    #     if 'weights' in model_pt and model_pt['weights'] is not None:
    #         model_param.weights = torch.tensor(model_pt['weights'], device=device, dtype=dtype)
    #         model_param.classic_n_func = model_pt['classic_n_func']
    #         print("\33[38;2;255;165;0m" + f'Number of WEIGHT functions: {model_param.classic_n_func}' + "\033[0m")

    #     if 'cp_A' in model_pt and model_pt['cp_A'] is not None:
    #         model_param.cp_A      = torch.tensor(model_pt['cp_A'], device=device, dtype=dtype)
    #         model_param.cp_B      = torch.tensor(model_pt['cp_B'], device=device, dtype=dtype)
    #         model_param.cp_C      = torch.tensor(model_pt['cp_C'], device=device, dtype=dtype)
    #         model_param.cp_lambda = torch.tensor(model_pt['cp_lambda'], device=device, dtype=dtype)
    #         model_param.cp_rank   = int(model_pt['cp_rank'])
    #         model_param.cp_n_of_func = model_pt['cp_n_of_func']
    #         print("\033[38;2;255;165;0m" + f'CP RANK: {model_param.cp_rank}, WEIGHT functions: {model_pt["cp_n_of_func"]}' + "\033[0m")
        
    #     if 'tt_core_G1' in model_pt and model_pt['tt_core_G1'] is not None:
    #         model_param.tt_core_G1 = torch.tensor(model_pt['tt_core_G1'], device=device, dtype=dtype)
    #         model_param.tt_core_G2 = torch.tensor(model_pt['tt_core_G2'], device=device, dtype=dtype)
    #         model_param.tt_core_G3 = torch.tensor(model_pt['tt_core_G3'], device=device, dtype=dtype)
    #         model_param.tt_ranks   = tuple(model_pt['tt_ranks'])
    #         model_param.tt_n_of_func = model_pt['tt_n_of_func']
    #         print("\033[38;2;255;165;0m" + f'TT RANKS: {model_param.tt_ranks}, WEIGHT functions: {model_pt["tt_n_of_func"]}' + "\033[0m")

    #     # print("\033[38;2;255;165;0m" + f'ellipsoid CENTER: {self.ellipsoid_center}' + "\033[0m")
    #     # print("\033[38;2;255;165;0m" + f'ellipsoid AXES: {self.ellipsoid_semi_axes}' + "\033[0m")
    #     # print("\033[38;2;255;165;0m" + f'ellipsoid EIGEN VECTORS: {self.ellipsoid_eigen_vector}' + "\033[0m")
        
    #     return model_param
        
    
