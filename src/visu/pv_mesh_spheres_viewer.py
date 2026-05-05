#!/usr/bin/env python3
"""
Funzioni di utilità per visualizzare una mesh e le sue sfere con PyVista.
"""

import numpy as np
import torch
import pyvista as pv


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def show_mesh_with_spheres(
    mesh_path: str,
    centers,
    radii,
    mesh_opacity: float = 0.4,
    sphere_opacity: float = 0.5,
    mesh_color="lightgray",
    sphere_color="lightblue",
):
    """
    Visualizza una singola mesh con le sfere approssimanti.

    Args:
        mesh_path: path alla mesh (es. .stl / .obj ...)
        centers: (K,3) torch.Tensor o np.ndarray con i centri delle sfere
        radii:   (K,)   torch.Tensor o np.ndarray con i raggi
    """
    centers_np = _to_numpy(centers)
    radii_np = _to_numpy(radii).reshape(-1)

    # Carica mesh con pyvista (supporta STL, OBJ, ecc.)
    mesh = pv.read(mesh_path)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, opacity=mesh_opacity, show_edges=False, color=mesh_color)

    for c, r in zip(centers_np, radii_np):
        sphere = pv.Sphere(radius=float(r), center=c)
        plotter.add_mesh(sphere, opacity=sphere_opacity, color=sphere_color)

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
