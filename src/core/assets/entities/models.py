from dataclasses import dataclass, asdict, field, fields
from typing import Any, Optional, Union, Dict, List, Tuple
import numpy as np
import torch

ArrayLike = Union[float, np.ndarray, torch.Tensor, list]

@dataclass
class BaseLinkModel:
    file_name: str = ""
    file_suffix: str = field(default="", init=False)   # <-- chiave
    domain_min: float = 0.0
    domain_max: float = 0.0
    scale_factor: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    centroid_offset: ArrayLike = 0.0

    center_ellipsoid: torch.Tensor = field(default_factory=lambda: torch.empty((3,)))
    axes_ellipsoid: Optional[Any] = None
    scales_ellipsoid: Optional[Any] = None
    eigen_vector_ellipsoid: Optional[Any] = None

    n_func: Optional[int] = None

    device: str = field(default="cpu", init=False)
    dtype: torch.dtype = field(default=torch.float32, init=False)

    def to_dict(self) -> Dict[str, Any]:
        # salva solo campi che il costruttore accetta (init=True)
        # return {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        allowed = {f.name for f in fields(cls) if f.init}
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(**clean)

@dataclass
class CpLinkModel(BaseLinkModel):
    file_suffix: str = field(default="_cp", init=False)
    A: Optional[Any] = None
    B: Optional[Any] = None
    C: Optional[Any] = None
    lamd: Optional[Any] = None
    rank: Optional[int] = None

    def to(self, device, dtype):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        cur = torch.device(self.device) if isinstance(self.device, str) else self.device

        if (cur == dev) and (self.dtype == dtype):
            return self
        self.A = self.A.to(device=device, dtype=dtype)
        self.B = self.B.to(device=device, dtype=dtype)
        self.C = self.C.to(device=device, dtype=dtype)
        self.lamd = self.lamd.to(device=device, dtype=dtype)
        self.centroid_offset = self.centroid_offset.to(device=device, dtype=dtype)
        self.scale_factor = self.scale_factor.to(device=device, dtype=dtype)

        self.device = str(dev)
        self.dtype = dtype
        return self

@dataclass
class TtLinkModel(BaseLinkModel):
    file_suffix: str = field(default="_tt", init=False)
    G1: Optional[Any] = None
    G2: Optional[Any] = None
    G3: Optional[Any] = None
    ranks: Optional[Tuple[int, ...]] = None

@dataclass
class WeightsLinkModel(BaseLinkModel):
    file_suffix: str = field(default="_w", init=False)
    weights: torch.Tensor = field(default_factory=list)
    
    def to(self, device, dtype):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        cur = torch.device(self.device) if isinstance(self.device, str) else self.device

        if (cur == dev) and (self.dtype == dtype):
            return self
        
        self.weights = self.weights.to(device=device, dtype=dtype)
        self.centroid_offset = self.centroid_offset.to(device=device, dtype=dtype)
        self.scale_factor = self.scale_factor.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype


        self.device = str(dev)
        self.dtype = dtype
        return self

@dataclass
class BaseRobotModel(BaseLinkModel):
    # qui hai ancora ereditarietà: ok se “robot ha anche meta di dominio”
    links_scale_factors: List[Any] = field(default_factory=list)
    links_centroids: List[Any] = field(default_factory=list)

@dataclass
class CpRobotModel(BaseRobotModel):
    file_suffix: str = field(default="_robot_cp", init=False)
    cp_V: Optional[Any] = None
    cp_A: Optional[Any] = None
    cp_B: Optional[Any] = None
    cp_C: Optional[Any] = None
    cp_lambda: Optional[Any] = None
    rank: Optional[Tuple[int, ...]] = None

    def to(self, device, dtype):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        cur = torch.device(self.device) if isinstance(self.device, str) else self.device

        if (cur == dev) and (self.dtype == dtype):
            return self
        self.cp_V = self.cp_V.to(device=device, dtype=dtype)
        self.cp_A = self.cp_A.to(device=device, dtype=dtype)
        self.cp_B = self.cp_B.to(device=device, dtype=dtype)
        self.cp_C = self.cp_C.to(device=device, dtype=dtype)
        self.cp_lambda = self.cp_lambda.to(device=device, dtype=dtype)
        self.centroid_offset = self.centroid_offset.to(device=device, dtype=dtype)
        self.scale_factor = self.scale_factor.to(device=device, dtype=dtype)

        self.device = str(dev)
        self.dtype = dtype

        return self

@dataclass
class TtRobotModel(BaseRobotModel):
    file_suffix: str = field(default="_robot_tt", init=False)
    G0: Optional[Any] = None
    G1: Optional[Any] = None
    G2: Optional[Any] = None
    G3: Optional[Any] = None
    ranks: Optional[Tuple[int, ...]] = None

@dataclass
class SphereModel:
    centers: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3)))
    radii: torch.Tensor   = field(default_factory=lambda: torch.empty((0,)))
    file_name: str = ""
    file_suffix: str = field(default="_spheres", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        allowed = {f.name for f in fields(cls) if f.init}
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(**clean)
