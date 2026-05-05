from scipy.spatial import ConvexHull
import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




import numpy as np

def compute_ellipsoid_parameters(points, sphere_radius=1.0):
    """
    Calcola centro, lunghezze degli assi e autovettori dell'elissoide che approssima i punti.
    Scala l'elissoide per essere inscritto in una sfera di raggio `sphere_radius` centrata sull'elissoide stesso.
    Inoltre, esegue un test per verificare che l'elissoide risultante sia effettivamente inscritto.
    """
    center = points.mean(axis=0)
    cov = np.cov(points - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axes_lengths = np.sqrt(eigenvalues) * 2  # diametri

    # Calcolo dei punti estremi per la scala
    semi_axes = axes_lengths / 2
    extreme_points = []
    for i in range(3):
        for sign in [-1, 1]:
            p = center + sign * semi_axes[i] * eigenvectors[:, i]
            extreme_points.append(p)

    max_dist = max(np.linalg.norm(p - center) for p in extreme_points)

    # Scala per inscrivere l'elissoide nella sfera
    scale = sphere_radius / max_dist
    axes_lengths *= scale

    # ==========================
    # Verifica di iscrizione
    # ==========================
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = axes_lengths[0] / 2 * np.outer(np.cos(u), np.sin(v))
    y = axes_lengths[1] / 2 * np.outer(np.sin(u), np.sin(v))
    z = axes_lengths[2] / 2 * np.outer(np.ones_like(u), np.cos(v))

    ellipsoid = np.dot(np.stack([x.flatten(), y.flatten(), z.flatten()]).T, eigenvectors.T)
    ellipsoid += center  # traslazione nel centro

    distances = np.linalg.norm(ellipsoid - center, axis=1)
    max_test_dist = np.max(distances)

    if max_test_dist <= sphere_radius + 1e-6:  # con tolleranza numerica
        print("✅ L'elissoide è inscritto nella sfera di raggio", sphere_radius)
    else:
        print("❌ L'elissoide NON è inscritto. Distanza massima:", max_test_dist)

    return axes_lengths, center, eigenvectors



def solve_mvce_fixed_center_in_sphere(points: np.ndarray, R: float = 1.0):
    
    c = points.mean(axis=0)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    dim = points.shape[1]
    A = cp.Variable((dim, dim), PSD=True)

    constraints = []
    for p in hull_points:
        constraints.append(cp.quad_form(p - c, A) <= 1)

    # Vincolo che l'ellissoide stia dentro sfera di raggio R centrata in 0:
    # norm(c) + 1/sqrt(lambda_min(A)) <= R
    # constraints.append(cp.norm(c, 2) + 1 / cp.sqrt(cp.lambda_min(A)) <= R)

    prob = cp.Problem(cp.Minimize(-cp.log_det(A)), constraints)
    prob.solve(qcp=True)

    

    return A.value, c

def solve_mvce_fixed_center_in_sphere(points, R=1.0):
    c = points.mean(axis=0)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    dim = points.shape[1]
    A = cp.Variable((dim, dim), PSD=True)

    constraints = []
    for p in hull_points:
        constraints.append(cp.quad_form(p - c, A) <= 1)

    # Vincolo che l'ellissoide stia dentro sfera di raggio R centrata in 0:
    # norm(c) + 1/sqrt(lambda_min(A)) <= R
    # constraints.append(cp.norm(c, 2) + 1 / cp.sqrt(cp.lambda_min(A)) <= R)

    prob = cp.Problem(cp.Minimize(-cp.log_det(A)), constraints)
    prob.solve(qcp=True)

    return A.value, c