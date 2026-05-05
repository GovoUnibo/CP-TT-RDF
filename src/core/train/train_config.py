from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

@dataclass
class Train_W:
    run: bool = False
    n_func: int = 8
    iters: int = 200
    batch_near: int = 1024
    batch_rand: int = 64

# ---- CP ----
@dataclass
class CPLinkCfg:
    run: bool = False
    method: str = "adam"  # {"adam", "als"}
    n_func: int = 8
    rank: int = 8
    iters: int = 20
    batch_size: int = 65_536
    ridge: float = 1e-6
    lr: float = 5e-3
    sample_weights: Optional[ArrayLike] = None

@dataclass
class CPRobotCfg(CPLinkCfg):
    run: bool = False
    batch_size: int = 65_536   # tipico del robot-wide

# ---- TT ----
@dataclass
class TTLinkCfg:
    run: bool = False
    method: str = "adam"  # {"adam", "mals"}
    n_func: int = 8
    ranks: Tuple[int, int] = (8, 8)
    iters: int = 20
    batch_size: int = 65_536
    ridge: float = 1e-6
    lr: float = 5e-3
    sample_weights: Optional[ArrayLike] = None

@dataclass
class TTRobotCfg:
    run: bool = False
    method: str = "adam"  # {"adam", "mals"}
    n_func: int = 8
    ranks: Tuple[int, int, int] = (4, 8, 4)  # 4D TT ranks
    iters: int = 20
    ridge: float = 1e-6
    lr: float = 5e-3
    sample_weights: Optional[ArrayLike] = None
    batch_size: int = 65_536

# ---- Spheres ----
@dataclass
class Train_Sphere:
    run: bool = False
    n_points: int = 3000
    n_spheres: int = 3

@dataclass
class TrainConfig:
    links_to_train: List[str]
    fit_ellipsoid: bool = True
    debug: bool = False
    fk_matrices: Optional[List[ArrayLike]] = None

    classic: Train_W = field(default_factory=Train_W)

    cp_link:  CPLinkCfg  = field(default_factory=CPLinkCfg)
    cp_robot: CPRobotCfg = field(default_factory=CPRobotCfg)

    tt_link:  TTLinkCfg  = field(default_factory=TTLinkCfg)
    tt_robot: TTRobotCfg = field(default_factory=TTRobotCfg)

    sphere: Train_Sphere = field(default_factory=Train_Sphere)

    @property
    def train(self) -> bool:
        return (
            self.classic.run
            or self.cp_link.run or self.cp_robot.run
            or self.tt_link.run or self.tt_robot.run
            or self.sphere.run
        )
