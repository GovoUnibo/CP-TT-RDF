from typing import Dict
import numpy as np
import torch

from src.core.assets.entities.MeshModel import Mesh




def fit_spheres_to_points(points: np.ndarray,
                          n_spheres: int = 3,
                          min_radius: float = 0.015):
    """
    Approssima un insieme di punti con n_spheres sfere.
    Ritorna:
        centers: (K, 3) np.array
        radii:   (K,)   np.array
    """
    assert points.ndim == 2 and points.shape[1] == 3

    if n_spheres <= 1 or points.shape[0] < n_spheres:
        # Una sola sfera: centro = centroide, r = percentile 90
        center = points.mean(axis=0)
        d = np.linalg.norm(points - center, axis=1)
        r = float(np.percentile(d, 90))
        r = max(r, min_radius)
        return center[None, :], np.array([r], dtype=np.float32)

    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_spheres, random_state=0, n_init=10)
        labels = kmeans.fit_predict(points)
        centers = []
        radii = []

        for k in range(n_spheres):
            cluster_pts = points[labels == k]
            if cluster_pts.size == 0:
                # cluster vuoto, skip
                continue

            c = cluster_pts.mean(axis=0)
            d = np.linalg.norm(cluster_pts - c, axis=1)
            r = float(np.percentile(d, 90) * 1.05)  # piccolo margine
            r = max(r, min_radius)

            centers.append(c)
            radii.append(r)

        if len(centers) == 0:
            # fallback brutale
            center = points.mean(axis=0)
            d = np.linalg.norm(points - center, axis=1)
            r = float(np.percentile(d, 90))
            r = max(r, min_radius)
            return center[None, :], np.array([r], dtype=np.float32)

        return np.vstack(centers), np.array(radii, dtype=np.float32)

    except ImportError:
        # fallback semplice: distribuisci lungo l’asse principale
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        dims = maxs - mins
        axis = np.argmax(dims)

        centers = []
        radii = []

        for i in range(n_spheres):
            alpha = (i + 0.5) / n_spheres
            c = mins.copy()
            c[axis] = mins[axis] + alpha * dims[axis]
            d = np.linalg.norm(points - c, axis=1)
            r = float(np.percentile(d, 70))
            r = max(r, min_radius)
            centers.append(c)
            radii.append(r)

        return np.vstack(centers), np.array(radii, dtype=np.float32)


# ----------------- MAIN LOGIC -----------------

def build_spheres_from_mesh(mesh:Mesh, 
                            n_points: int = 3000, 
                            n_spheres: int = 3):
    """
    Campiona una singola mesh e ritorna il DIZIONARIO per quella mesh:
    {
        "centers": (K,3) FloatTensor,
        "radii":   (K,)  FloatTensor,
        "bounds_min": (3,) FloatTensor,
        "bounds_max": (3,) FloatTensor
    }
    """
    # pts = mesh.mixed_uniform_sampling(n_points, surface_ratio=0.8)

    pts = mesh.surface_uniform_sampling(n_points)
    if pts is None:
        raise ValueError("Nessun punto campionato dalla mesh")

    if isinstance(pts, torch.Tensor):
        pts_np = pts.cpu().numpy()
    else:
        pts_np = np.asarray(pts)

    if pts_np.shape[0] == 0:
        raise ValueError("Point cloud vuota")

    centers_np, radii_np = fit_spheres_to_points(
        pts_np,
        n_spheres=n_spheres
    )

    mins = pts_np.min(axis=0)
    maxs = pts_np.max(axis=0)

    d = {"centers": centers_np,"radii": radii_np,}

    return d

