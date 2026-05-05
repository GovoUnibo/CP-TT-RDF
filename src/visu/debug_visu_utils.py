import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import trimesh
import torch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


def show_mesh(sdf_mesh):
    if sdf_mesh is None:
        print("[WARN] show_mesh ricevuto None, salto la visualizzazione.")
        return
    if not isinstance(sdf_mesh, trimesh.Trimesh):
        print("[WARN] show_mesh richiede una trimesh.Trimesh, salto la visualizzazione.")
        return
    if sdf_mesh.vertices is None or len(sdf_mesh.vertices) == 0:
        print("[WARN] show_mesh ricevuto una mesh vuota, salto la visualizzazione.")
        return
    scene = trimesh.Scene()
    scene.add_geometry(sdf_mesh)
    scene.show()


def _to_axis_scale_and_offset(scaling_factor, centroid_offset):
    sf = scaling_factor.detach().cpu().numpy() if torch.is_tensor(scaling_factor) else np.asarray(scaling_factor)
    co = centroid_offset.detach().cpu().numpy() if torch.is_tensor(centroid_offset) else np.asarray(centroid_offset)
    sf = np.atleast_1d(sf).astype(float)
    co = np.atleast_1d(co).astype(float)
    return sf, co


def _axis_value(values, axis):
    if values.size == 1:
        return float(values[0])
    if values.size >= 3:
        return float(values[axis])
    return float(values.ravel()[0])


@torch.no_grad()
def _eval_cp_sdf_points(
    A,
    B,
    C,
    lam,
    points,
    domain_min,
    domain_max,
    bernstein_matrix_1d,
    batch_points,
):
    device = A.device
    dtype = A.dtype
    _, rank = A.shape
    lam = lam.view(rank).to(device=device, dtype=dtype)
    pts = torch.as_tensor(points, device=device, dtype=dtype).reshape(-1, 3)
    denom = float(domain_max - domain_min)
    sdf_vals = torch.empty(pts.shape[0], device=device, dtype=dtype)

    for start in range(0, pts.shape[0], batch_points):
        end = min(start + batch_points, pts.shape[0])
        p_batch = pts[start:end]

        tx = (p_batch[:, 0] - domain_min) / denom
        ty = (p_batch[:, 1] - domain_min) / denom
        tz = (p_batch[:, 2] - domain_min) / denom

        Phi_x, _ = bernstein_matrix_1d(tx, use_derivative=False)
        Phi_y, _ = bernstein_matrix_1d(ty, use_derivative=False)
        Phi_z, _ = bernstein_matrix_1d(tz, use_derivative=False)

        Sx = Phi_x @ A
        Sy = Phi_y @ B
        Sz = Phi_z @ C
        sdf_vals[start:end] = (Sx * Sy * Sz) @ lam

    return sdf_vals


@torch.no_grad()
def _show_cp_sdf_level_curves_3d(
    A,
    B,
    C,
    lam,
    domain_min,
    domain_max,
    bernstein_matrix_1d,
    scaling_factor,
    centroid_offset,
    res,
    batch_points,
    n_levels,
    cmap,
    sigma_smooth,
):
    try:
        import pyvista as pv
    except ImportError:
        print("[WARN] PyVista non disponibile, salto la visualizzazione 3D delle isosuperfici SDF.")
        return

    sf, co = _to_axis_scale_and_offset(scaling_factor, centroid_offset)
    grid_res = int(res)
    domain = np.linspace(domain_min, domain_max, grid_res, dtype=np.float64)
    X, Y, Z = np.meshgrid(domain, domain, domain, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    sdf_vals = _eval_cp_sdf_points(
        A=A,
        B=B,
        C=C,
        lam=lam,
        points=pts,
        domain_min=domain_min,
        domain_max=domain_max,
        bernstein_matrix_1d=bernstein_matrix_1d,
        batch_points=batch_points,
    )
    sdf_grid = sdf_vals.reshape(grid_res, grid_res, grid_res).detach().cpu().numpy()

    if sigma_smooth and sigma_smooth > 0:
        sdf_grid = gaussian_filter(sdf_grid, sigma=sigma_smooth)

    grid_min = float(np.min(sdf_grid))
    grid_max = float(np.max(sdf_grid))
    if np.isclose(grid_min, grid_max):
        print("[WARN] SDF quasi costante: impossibile estrarre isosuperfici 3D utili.")
        return

    levels = np.linspace(grid_min, grid_max, int(n_levels))
    levels = np.unique(levels)

    spacing_sdf = float(domain_max - domain_min) / max(grid_res - 1, 1)
    spacing = (
        spacing_sdf * abs(_axis_value(sf, 0)),
        spacing_sdf * abs(_axis_value(sf, 1)),
        spacing_sdf * abs(_axis_value(sf, 2)),
    )
    origin = (
        domain_min * _axis_value(sf, 0) + _axis_value(co, 0),
        domain_min * _axis_value(sf, 1) + _axis_value(co, 1),
        domain_min * _axis_value(sf, 2) + _axis_value(co, 2),
    )

    grid = pv.ImageData()
    grid.dimensions = (grid_res, grid_res, grid_res)
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data["sdf"] = sdf_grid.flatten(order="F")

    contours = grid.contour(isosurfaces=levels.tolist(), scalars="sdf")
    zero_surface = None
    if grid_min <= 0.0 <= grid_max:
        zero_surface = grid.contour(isosurfaces=[0.0], scalars="sdf")

    pl = pv.Plotter()
    if contours.n_points > 0:
        pl.add_mesh(contours, scalars="sdf", cmap=cmap, opacity=0.22, show_scalar_bar=True)
    if zero_surface is not None and zero_surface.n_points > 0:
        pl.add_mesh(zero_surface, color="black", opacity=0.55)
    pl.add_axes()
    pl.show()


@torch.no_grad()
def show_cp_sdf_level_curves(
    A,
    B,
    C,
    lam,
    domain_min,
    domain_max,
    bernstein_matrix_1d,
    scaling_factor=1.0,
    centroid_offset=(0.0, 0.0, 0.0),
    slice_value=None,
    res=220,
    n_levels=21,
    batch_points=50_000,
    cmap="coolwarm",
    visualize_3d=False,
    res_3d=72,
    sigma_smooth=1.0,
):
    if visualize_3d:
        _show_cp_sdf_level_curves_3d(
            A=A,
            B=B,
            C=C,
            lam=lam,
            domain_min=domain_min,
            domain_max=domain_max,
            bernstein_matrix_1d=bernstein_matrix_1d,
            scaling_factor=scaling_factor,
            centroid_offset=centroid_offset,
            res=res_3d,
            batch_points=batch_points,
            n_levels=n_levels,
            cmap=cmap,
            sigma_smooth=sigma_smooth,
        )
        return

    device = A.device
    dtype = A.dtype

    sf, co = _to_axis_scale_and_offset(scaling_factor, centroid_offset)
    slice_value = float((domain_min + domain_max) * 0.5) if slice_value is None else float(slice_value)
    domain = np.linspace(domain_min, domain_max, int(res), dtype=np.float64)

    plane_specs = [
        ("xy", 0, 1, 2),
        ("xz", 0, 2, 1),
        ("yz", 1, 2, 0),
    ]

    slice_results = []
    global_min = np.inf
    global_max = -np.inf

    for plane_name, axis_u, axis_v, axis_fixed in plane_specs:
        U, V = np.meshgrid(domain, domain, indexing="xy")
        pts = np.empty((U.size, 3), dtype=np.float64)
        pts[:, axis_u] = U.reshape(-1)
        pts[:, axis_v] = V.reshape(-1)
        pts[:, axis_fixed] = slice_value

        sdf_vals = _eval_cp_sdf_points(
            A=A,
            B=B,
            C=C,
            lam=lam,
            points=pts,
            domain_min=domain_min,
            domain_max=domain_max,
            bernstein_matrix_1d=bernstein_matrix_1d,
            batch_points=batch_points,
        )
        sdf_grid = sdf_vals.reshape(int(res), int(res)).detach().cpu().numpy()
        global_min = min(global_min, float(np.min(sdf_grid)))
        global_max = max(global_max, float(np.max(sdf_grid)))

        U_plot = U * _axis_value(sf, axis_u) + _axis_value(co, axis_u)
        V_plot = V * _axis_value(sf, axis_v) + _axis_value(co, axis_v)
        fixed_plot = slice_value * _axis_value(sf, axis_fixed) + _axis_value(co, axis_fixed)

        slice_results.append((plane_name, U_plot, V_plot, sdf_grid, axis_u, axis_v, axis_fixed, fixed_plot))

    if np.isclose(global_min, global_max):
        global_min -= 1e-6
        global_max += 1e-6
    levels = np.linspace(global_min, global_max, int(n_levels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axis_names = ["x", "y", "z"]

    for ax, (plane_name, U_plot, V_plot, sdf_grid, axis_u, axis_v, axis_fixed, fixed_plot) in zip(axes, slice_results):
        contour_fill = ax.contourf(U_plot, V_plot, sdf_grid, levels=levels, cmap=cmap)
        ax.contour(U_plot, V_plot, sdf_grid, levels=levels, colors="k", linewidths=0.35, alpha=0.35)
        if global_min <= 0.0 <= global_max:
            ax.contour(U_plot, V_plot, sdf_grid, levels=[0.0], colors="k", linewidths=1.8)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(axis_names[axis_u])
        ax.set_ylabel(axis_names[axis_v])
        ax.set_title(f"{plane_name.upper()} | {axis_names[axis_fixed]}={fixed_plot:.3f}")

    fig.colorbar(contour_fill, ax=axes.ravel().tolist(), shrink=0.9, label="SDF")
    fig.suptitle("Curve di livello SDF CP")
    plt.tight_layout()
    plt.show()

def show_sdf_pointcloud(points, sdf_points, filter_interval=0.01):
    import pyrender
    
    colors = np.zeros(points.shape)
    colors[sdf_points < -filter_interval, 2] = 1
    colors[sdf_points > filter_interval, 0] = 1

    cloud = pyrender.Mesh.from_points(points, colors=colors)

    scene = pyrender.Scene()
    scene.add(cloud)

    # Visualizza la scena
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)



def relate_meshes(mesh1, mesh2, title=''):
    ver1 = mesh1.vertices
    ver2 = mesh2.vertices

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ver1[:, 0], ver1[:, 1], ver1[:, 2], c='b', marker='.')
    ax.scatter(ver2[:, 0], ver2[:, 1], ver2[:, 2], c='r', marker='.')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #set legend
    ax.legend(['mesh1', 'mesh2'])
    #display origin
    ax.scatter(0,0,0, c='g', marker='o')
    #show origin axes
    ax.quiver(0,0,0,1,0,0, color='r')
    ax.quiver(0,0,0,0,1,0, color='g')
    ax.quiver(0,0,0,0,0,1, color='b')
    #add title
    ax.set_title(title)
    plt.show()



def show_mesh_unit_spere(mesh):
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    sphere.visual.vertex_colors = [0.0, 0.0, 1.0, 0.1]
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(sphere)
    scene.show()

#matplot
def show_mesh_mtplt(mesh, title=''): #pc stands for pointcloud
    vertices = mesh.vertices
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #display origin
    ax.scatter(0,0,0, c='g', marker='o')
    #show origin axes
    ax.quiver(0,0,0,1,0,0, color='r')
    ax.quiver(0,0,0,0,1,0, color='g')
    ax.quiver(0,0,0,0,0,1, color='b')
    #add title
    ax.set_title(title)
    plt.show()

def show_two_pc_mtplt(pc1, pc2, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='b', marker='.')
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='r', marker='.')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #display origin
    ax.scatter(0,0,0, c='g', marker='o')
    #show origin axes
    ax.quiver(0,0,0,1,0,0, color='r')
    ax.quiver(0,0,0,0,1,0, color='g')
    ax.quiver(0,0,0,0,0,1, color='b')
    #add title
    ax.set_title(title)
    plt.show()

def show_mesh_with_pyrender(mesh):
    import pyrender
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(pyrender_mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def show_sdf_pointcloud_with_mesh(points, mesh, sdf_points):
    import pyrender
    colors = np.zeros(points.shape)
    colors[sdf_points < 0, 2] = 1
    colors[sdf_points > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(cloud)
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def show_mesh_and_pointcloud(mesh, pointcloud, title=''):
    import pyrender
    mesh = pyrender.Mesh.from_trimesh(mesh)
    # Imposta i punti della pointcloud in rosso
    red_color = np.array([1.0, 0.0, 0.0])  # RGB per rosso
    colors = np.tile(red_color, (pointcloud.shape[0], 1))  # Replica il colore per ogni punto
    # Crea il mesh della pointcloud con colori
    pointcloud = pyrender.Mesh.from_points(pointcloud, colors=colors)
    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(pointcloud)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)

def show_pointcloud_matplot(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()





# def plot_mvie_ellipsoid(points, A, c):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot original points
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.2, color='gray')

#     # Plot ellipsoid
#     U, s, _ = np.linalg.svd(A)
#     radii = 1.0 / np.sqrt(s)
#     ellipsoid_transform = U @ np.diag(radii)

#     u = np.linspace(0, 2 * np.pi, 50)
#     v = np.linspace(0, np.pi, 25)
#     x = np.outer(np.cos(u), np.sin(v))
#     y = np.outer(np.sin(u), np.sin(v))
#     z = np.outer(np.ones_like(u), np.cos(v))
#     sphere = np.stack((x, y, z), axis=-1)
#     ellipsoid = sphere @ ellipsoid_transform.T + c

#     # Actual surface
#     ax.plot_surface(
#         ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
#         rstride=2, cstride=2, color='red', alpha=0.4
#     )

#     # Wireframe reference unit sphere (optional)
#     us = np.linspace(0, 2 * np.pi, 30)
#     vs = np.linspace(0, np.pi, 15)
#     xs = np.outer(np.cos(us), np.sin(vs))
#     ys = np.outer(np.sin(us), np.sin(vs))
#     zs = np.outer(np.ones_like(us), np.cos(vs))
#     ax.plot_wireframe(xs, ys, zs, color='blue', alpha=0.2, linewidth=0.5)

#     # Manual legend
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Punti',
#                markerfacecolor='gray', markersize=8, alpha=0.5),
#         Patch(facecolor='red', edgecolor='r', label='MVIE', alpha=0.4)
#     ]
#     ax.legend(handles=legend_elements)

#     ax.set_title("MVIE - Ellissoide di massimo volume dentro il convesso")
#     ax.set_box_aspect([1, 1, 1])
#     plt.tight_layout()
#     plt.show()

def plot_ellipsoid(center, axes_lengths, eigenvectors, points=None):
    """
    Plotta un elissoide definito da centro, assi e autovettori.
    Se forniti, mostra anche i punti originali.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Mostra i punti originali, se presenti
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    # Parametri sferici
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = axes_lengths[0] / 2 * np.outer(np.cos(u), np.sin(v))
    y = axes_lengths[1] / 2 * np.outer(np.sin(u), np.sin(v))
    z = axes_lengths[2] / 2 * np.outer(np.ones_like(u), np.cos(v))

    # Rotazione e traslazione
    ellipsoid = np.dot(np.stack([x.flatten(), y.flatten(), z.flatten()]).T, eigenvectors.T)
    x_rot = ellipsoid[:, 0].reshape(x.shape) + center[0]
    y_rot = ellipsoid[:, 1].reshape(y.shape) + center[1]
    z_rot = ellipsoid[:, 2].reshape(z.shape) + center[2]

    # Plotta la superficie dell’elissoide
    ax.plot_surface(x_rot, y_rot, z_rot, rstride=4, cstride=4, color='c', alpha=0.3)

    plt.show()
