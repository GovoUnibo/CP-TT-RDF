from src.core.assets.entities.models import CpLinkModel, TtLinkModel, WeightsLinkModel, CpRobotModel, TtRobotModel, SphereModel
import torch

import numpy as np

_C_METHOD = {
    "CP3D": "\033[38;2;255;140;0m",
    "CP4D": "\033[38;2;214;176;76m",
    "TT3D": "\033[38;2;80;170;255m",
    "TT4D": "\033[38;2;0;191;255m",
    "W3D": "\033[38;2;120;220;120m",
    "SPHERE": "\033[38;2;210;210;210m",
}
_C_RESET = "\033[0m"


def _method_print(method: str, text: str):
    print(_C_METHOD.get(method, "\033[38;2;255;165;0m") + text + _C_RESET)


def _log_loading(method: str, file_name: str, details: str = ""):
    _method_print(method, f"LOADING[{method}] MODEL: {file_name}")
    if details:
        _method_print(method, details)


def _to_tensor(x, device, dtype) -> torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, (np.ndarray, list, float, int)):
        return torch.as_tensor(x, device=device, dtype=dtype)
    return x  # fallback: str, dict, ecc.

    

def load_link_cp_model(model_dict, device="cuda", dtype=torch.float32) -> CpLinkModel:
    d = dict(model_dict)

    # backward compat (se esistono file vecchi)
    if "cp_A" in d: d["A"] = d.pop("cp_A")
    if "cp_B" in d: d["B"] = d.pop("cp_B")
    if "cp_C" in d: d["C"] = d.pop("cp_C")
    if "cp_lambda" in d: d["lamd"] = d.pop("cp_lambda")
    if "cp_rank" in d: d["rank"] = d.pop("cp_rank")
    if "cp_n_of_func" in d: d["n_func"] = d.pop("cp_n_of_func")

    # to torch
    for k in ["scale_factor", "centroid_offset",
              "center_ellipsoid", "axes_ellipsoid", "scales_ellipsoid", "eigen_vector_ellipsoid",
              "A", "B", "C", "lamd"]:
        if k in d:
            d[k] = _to_tensor(d[k], device, dtype)

    _log_loading("CP3D", d["file_name"], f"Number of BASIS functions: {d['n_func']}, Number of RANK: {d['rank']}")

    return CpLinkModel.from_dict(d)


def load_link_tt_model(model_dict, device="cuda", dtype=torch.float32) -> TtLinkModel:
    d = dict(model_dict)

    # backward compat
    if "tt_core_G1" in d: d["G1"] = d.pop("tt_core_G1")
    if "tt_core_G2" in d: d["G2"] = d.pop("tt_core_G2")
    if "tt_core_G3" in d: d["G3"] = d.pop("tt_core_G3")
    if "tt_ranks" in d: d["ranks"] = d.pop("tt_ranks")
    if "tt_n_of_func" in d: d["n_func"] = d.pop("tt_n_of_func")

    for k in ["scale_factor", "centroid_offset",
              "center_ellipsoid", "axes_ellipsoid", "scales_ellipsoid", "eigen_vector_ellipsoid",
              "G1", "G2", "G3"]:
        if k in d:
            d[k] = _to_tensor(d[k], device, dtype)

    if "ranks" in d and d["ranks"] is not None:
        d["ranks"] = tuple(int(r) for r in d["ranks"])

    _log_loading("TT3D", d["file_name"], f"Number of BASIS functions: {d['n_func']}, RANKS: {d.get('ranks')}")
    return TtLinkModel.from_dict(d)


def load_link_weight_model(model_dict, device="cuda", dtype=torch.float32) -> WeightsLinkModel:
    d = dict(model_dict)

    for k in ["scale_factor", "centroid_offset",
            "center_ellipsoid", "axes_ellipsoid", "scales_ellipsoid", "eigen_vector_ellipsoid",
            "weights"]:
        if k in d:
            d[k] = _to_tensor(d[k], device, dtype)

    _log_loading("W3D", d["file_name"], f"Number of BASIS functions: {d['n_func']}")


    return WeightsLinkModel.from_dict(d)

def load_robot_cp_model(model_dict, device="cuda", dtype=torch.float32) -> CpRobotModel:
    d = dict(model_dict)


    for k in ["scale_factor", "centroid_offset",
              "cp_V","cp_A","cp_B","cp_C","cp_lambda"]:
        if k in d:
            d[k] = _to_tensor(d[k], device, dtype)
            
    _log_loading("CP4D", d["file_name"], f"Number of BASIS functions: {d['n_func']}, Number of RANK: {d['rank']}")
    return CpRobotModel.from_dict(d)

def load_robot_tt_model(model_dict, device="cuda", dtype=torch.float32) -> TtRobotModel:
    d = dict(model_dict)
    for k in ["scale_factor", "centroid_offset", "G0","G1","G2","G3"]:
        if k in d:
            d[k] = _to_tensor(d[k], device, dtype)
    if "ranks" in d and d["ranks"] is not None:
        d["ranks"] = tuple(int(r) for r in d["ranks"])
    _log_loading("TT4D", d["file_name"], f"Number of BASIS functions: {d['n_func']}, RANKS: {d.get('ranks')}")
    return TtRobotModel.from_dict(d)

def load_link_sphere_model(model_dict, device="cuda", dtype=torch.float32) -> SphereModel:
    d = dict(model_dict)

    for k in ["centers", "radii"]:
        if k in d:
            d[k] = _to_tensor(d[k], device, dtype)

    _log_loading("SPHERE", d["file_name"])
    return SphereModel.from_dict(d)
