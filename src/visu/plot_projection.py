
import numpy as np
import matplotlib.pyplot as plt

import torch
import pyvista as pv
import numpy as np
from src.core.math.projection_point import ellipsoid_projection, spherical_projection

import imageio
import os


def debug_gradient_descent(points_link_frame, mesh):
    """
    points_link_frame: tensor (num_iter, num_points, 3)
    mesh: trimesh.Trimesh (verrà convertita in PyVista)
    """
    from src.utils.MeshUtils import trimesh_to_pyvista
    import torch
    import pyvista as pv
    from src.visu.plots_3d import RobotSene
    # --- prepara mesh ---
    

    # --- prepara scena ---
    r_scene = RobotSene()
    r_scene.add_mesh(mesh)

    # --- porta i punti su numpy ---
    if isinstance(points_link_frame, torch.Tensor):
        points_np = points_link_frame.detach().cpu().numpy()
    else:
        points_np = np.asarray(points_link_frame)

    assert points_np.ndim == 3 and points_np.shape[2] == 3, \
        "points_link_frame deve essere (num_iter, num_points, 3)"

    num_iter, num_points, _ = points_np.shape

    # === 1) traiettorie: una polilinea per punto iniziale ===
    # (se vuoi un colore unico per tutte le traiettorie, usa una stringa fissa in polyline_color)
    polyline_color = 'deepskyblue'      # colore delle traiettorie
    chord_color    = 'tomato'           # segmento iniziale->finale
    finals_color   = 'yellow'           # colore delle proiezioni finali (tutte uguali come richiesto)

    # opzionale: se vuoi colori diversi per ogni path, attiva questa cm
    # cmap = plt.cm.get_cmap('viridis', num_points)

    for j in range(num_points):
        path = points_np[:, j, :]                 # (num_iter, 3) per il punto j
        r_scene.add_polyline(path, color=polyline_color, line_width=2)

        # segmento diretto start->final
        start = path[0]
        end   = path[-1]
        seg = np.vstack([start, end])
        seg_line = pv.lines_from_points(seg, close=False)
        r_scene.scene.add_mesh(seg_line, color=chord_color, line_width=1)

        # se preferisci un colore per path, sostituisci polyline_color con:
        # rgb = np.array(cmap(j)[:3])  # viridis restituisce RGBA [0..1]
        # r_scene.add_polyline(path, color=tuple((rgb*255).astype(np.uint8)))

    # === 2) nuvola dei punti finali, tutti dello stesso colore ===
    finals = points_np[-1]                          # (num_points, 3)
    r_scene.add_pointcloud(finals, color=finals_color, point_size=10)

    # (opzionale) disegna anche i punti iniziali, più piccoli e semitrasparenti
    # PyVista non gestisce direttamente alpha con color string -> usa RGB se serve
    r_scene.add_pointcloud(points_np[0], color='white', point_size=6)

    r_scene.show()








def plot_points_and_projection_on_ellipsoid(points: torch.Tensor,
                                            model_mesh_pv: pv.PolyData,
                                            center: torch.Tensor,
                                            scale: torch.Tensor,
                                            ellipsoid_center: torch.Tensor,
                                            ellipsoid_axes: torch.Tensor,
                                            ellipsoid_eigenvectors: torch.Tensor,
                                            project: bool = True,
                                            device='cpu'):
    """
    Plot della mesh del modello, dell'elissoide, dei punti originali e delle loro proiezioni.
    """

    points = points.to(device)
    center = center.to(device)
    scale = scale.to(device)
    ellipsoid_center = ellipsoid_center.to(device)
    ellipsoid_axes = ellipsoid_axes.to(device)
    ellipsoid_eigenvectors = ellipsoid_eigenvectors.to(device)

    # Normalizzazione (come nel training)
    points_scaled = (points - center) / scale

    # Maschera: solo punti fuori dalla sfera
    norms = torch.linalg.norm(points_scaled, dim=1)
    mask = norms > 1.0

    # Proiezione
    if project:
        projected, _ = ellipsoid_projection(points_scaled, center=ellipsoid_center,
                                            axes=ellipsoid_eigenvectors, D=ellipsoid_axes, device=device)
    else:
        projected = points_scaled.clone()

    # Ritorna nello spazio originale (mesh)
    projected_world = projected * scale + center
    points_scaled_world = points_scaled * scale + center

    # Crea scena PyVista
    p = pv.Plotter()

    # ➤ Aggiungi la mesh del modello
    p.add_mesh(model_mesh_pv, color='lightgrey', opacity=0.4, show_edges=False)

    # ➤ Aggiungi i punti originali (solo quelli esterni)
    p.add_points(points_scaled_world[mask].cpu().numpy(), color='red', render_points_as_spheres=True, point_size=10, label='Original points')

    # ➤ Aggiungi i punti proiettati
    p.add_points(projected_world[mask].cpu().numpy(), color='blue', point_size=10, render_points_as_spheres=True, label='Projected points')

    # ➤ Aggiungi linee tra ogni punto e la sua proiezione
    for p1, p2 in zip(points_scaled_world[mask], projected_world[mask]):
        line = np.stack([p1.cpu().numpy(), p2.cpu().numpy()])
        p.add_lines(line, color='black', width=1)

    # ➤ Disegna ellissoide come mesh
    ellipsoid = _create_ellipsoid_mesh(
        center=(ellipsoid_center * scale + center).cpu().numpy(),
        semi_axes=(ellipsoid_axes * scale).cpu().numpy(),
        rotation_matrix=ellipsoid_eigenvectors.cpu().numpy()
    )
    p.add_mesh(ellipsoid,  color='green', opacity=0.25, line_width=1.5, label='Ellipsoid')

    p.add_legend()
    p.show()


def _create_ellipsoid_mesh(center, semi_axes, rotation_matrix, resolution=100):
    """
    Crea una mesh solida dell'elissoide come pyvista PolyData
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = semi_axes[0] * np.outer(np.cos(u), np.sin(v))
    y = semi_axes[1] * np.outer(np.sin(u), np.sin(v))
    z = semi_axes[2] * np.outer(np.ones_like(u), np.cos(v))

    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    xyz_rotated = xyz @ rotation_matrix.T + center
    x_r = xyz_rotated[:, 0].reshape(x.shape)
    y_r = xyz_rotated[:, 1].reshape(y.shape)
    z_r = xyz_rotated[:, 2].reshape(z.shape)

    # Da griglia strutturata a superficie poligonale
    grid = pv.StructuredGrid(x_r, y_r, z_r)
    return grid.extract_surface()



def save_projection_ellipsoidal_proj_gif(points: torch.Tensor,
                        model_mesh_pv: pv.PolyData,
                        center: torch.Tensor,
                        scale: torch.Tensor,
                        ellipsoid_center: torch.Tensor,
                        ellipsoid_axes: torch.Tensor,
                        ellipsoid_eigenvectors: torch.Tensor,
                        gif_path: str = "./projection_interp.gif",
                        project: bool = True,
                        device: str = "cpu",
                        fps: int = 24,
                        duration_s: float = 3.0,     # durata animazione
                        hold_start_frames: int = 8,  # pausa iniziale
                        hold_end_frames: int = 12,   # pausa finale
                        orbit_degrees: float = 360.0,
                        only_outside: bool = True,
                        point_size: int = 10,
                        view: str = "iso",
                        easing: str = "ease_in_out", # "linear" | "ease_in_out"
                        show_paths: bool = True,
                        show_targets: bool = True):
    """
    Crea una GIF in cui TUTTI i punti si muovono contemporaneamente dall'origine
    alla proiezione sull'elissoide, mentre la camera ruota.

    - gif_path: percorso di salvataggio (la funzione crea la cartella se manca)
    - fps, duration_s: controllano fluidità e durata del movimento
    - easing: 'linear' o 'ease_in_out' (più naturale)
    - show_paths: disegna le linee origine→proiezione come guida
    - show_targets: mostra anche i target (proiezioni) in blu durante l'animazione
    """

    # --- Prepara path
    out_dir = os.path.dirname(os.path.abspath(gif_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # --- Tensors su device
    points = points.to(device)
    center = center.to(device)
    scale = scale.to(device)
    ellipsoid_center = ellipsoid_center.to(device)
    ellipsoid_axes = ellipsoid_axes.to(device)
    ellipsoid_eigenvectors = ellipsoid_eigenvectors.to(device)

    # --- Normalizzazione come nel training
    points_scaled = (points - center) / scale

    # --- Mask (solo punti esterni se richiesto)
    if only_outside:
        norms = torch.linalg.norm(points_scaled, dim=1)
        mask = norms > 1.0
    else:
        mask = torch.ones(points_scaled.shape[0], dtype=torch.bool, device=device)

    # --- Proiezione
    if project:
        projected_scaled, _ = ellipsoid_projection(points_scaled, center=ellipsoid_center,
                                                   axes=ellipsoid_eigenvectors, D=ellipsoid_axes, device=device)
    else:
        projected_scaled = points_scaled.clone()

    # --- Torna nello spazio world
    orig_world = (points_scaled * scale + center).detach().cpu().numpy()
    proj_world = (projected_scaled * scale + center).detach().cpu().numpy()

    idxs = torch.nonzero(mask, as_tuple=False).squeeze(1).cpu().numpy()
    if idxs.size == 0:
        raise ValueError("Nessun punto da animare (mask vuota).")

    orig = orig_world[idxs]
    targ = proj_world[idxs]

    # --- Plotter off-screen
    p = pv.Plotter(off_screen=True)
    p.set_background("white")

    # Mesh del modello
    p.add_mesh(model_mesh_pv, color='lightgrey', opacity=0.35, show_edges=False)

    # Ellissoide solido
    ellipsoid = _create_ellipsoid_mesh(
        center=(ellipsoid_center * scale + center).detach().cpu().numpy(),
        semi_axes=(ellipsoid_axes * scale).detach().cpu().numpy(),
        rotation_matrix=ellipsoid_eigenvectors.detach().cpu().numpy()
    )
    p.add_mesh(ellipsoid, color='green', opacity=0.25, smooth_shading=True, show_edges=False, label='Ellipsoid')

    # Linee guida origine→target (opzionali, create una sola volta)
    if show_paths:
        # costruiamo una singola PolyData con tante linee 2-vertici
        line_pts = np.vstack([np.column_stack((orig, targ)).reshape(-1, 3)][0])
        # connettività: per ogni segmento: [2, i, i+1]
        n_lines = len(orig)
        cells = np.empty(n_lines * 3, dtype=np.int32)
        cells[0::3] = 2
        cells[1::3] = np.arange(0, 2*n_lines, 2, dtype=np.int32)
        cells[2::3] = np.arange(1, 2*n_lines, 2, dtype=np.int32)
        lines_mesh = pv.PolyData()
        lines_mesh.points = line_pts
        lines_mesh.lines = cells
        p.add_mesh(lines_mesh, color='black', line_width=1)

    # Punti target (blu) opzionali
    if show_targets:
        p.add_points(targ,
             color='blue', point_size=10,
             render_points_as_spheres=True,
             label='Original points')

    # Punti in movimento (rossi) – aggiornati in-place ad ogni frame
    moving = pv.PolyData(orig.copy())
    p.add_points(moving, color='red', point_size=point_size,
             render_points_as_spheres=True,
             label='Original→Projected')

    p.add_legend()

    # Inquadratura
    if view in ("iso", "xy", "xz", "yz"):
        p.camera_position = view
    else:
        p.camera_position = "iso"

    p.open_gif(gif_path, fps=fps)

    # Frame counts
    move_frames = max(2, int(duration_s * fps))
    total_frames = hold_start_frames + move_frames + hold_end_frames
    deg_per_frame = float(orbit_degrees) / max(1, total_frames)

    # Easing
    if easing == "ease_in_out":
        # t_ease = 0.5 * (1 - cos(pi * t))
        def ease_fn(t):  # t in [0,1]
            return 0.5 * (1.0 - np.cos(np.pi * t))
    else:
        def ease_fn(t):
            return t
    p.reset_camera()
    # Hold iniziale
    p.render()
    for _ in range(hold_start_frames):
        try: p.camera.azimuth(deg_per_frame)
        except Exception: pass
        p.write_frame()

    # Movimento simultaneo
    for f in range(move_frames):
        t = f / (move_frames - 1)
        tt = ease_fn(t)
        new_pos = (1.0 - tt) * orig + tt * targ
        # aggiorna i punti in-place
        moving.points = new_pos
        # p.camera.zoom(1.0 + 0.05 * t)
        try: p.camera.azimuth(deg_per_frame)
        except Exception: pass

        p.render()
        p.write_frame()

    # Hold finale
    for _ in range(hold_end_frames):
        try: p.camera.azimuth(deg_per_frame)
        except Exception: pass
        p.render()
        p.write_frame()

    p.close()
    return gif_path



def plot_points_and_projection_on_sphere(points: torch.Tensor,
                                         model_mesh_pv: pv.PolyData,
                                         center: torch.Tensor,
                                         scale: torch.Tensor,
                                         sphere_center: torch.Tensor,
                                         r1: float = 1.0,
                                         project: bool = True,
                                         device: str = 'cpu',
                                         point_size: int = 10):
    """
    Plot della mesh del modello, della sfera (in spazio normalizzato), dei punti originali e delle proiezioni.
    - points: Nx3 (world)
    - center, scale: tensori 3D usati per normalizzare (come nel training)
    - sphere_center: centro sfera in spazio normalizzato
    - r1: raggio sfera in spazio normalizzato
    """

    # Sposta su device
    points = points.to(device)
    center = center.to(device)
    scale = scale.to(device)
    sphere_center = sphere_center.to(device)

    # Normalizzazione (come nel training)
    points_scaled = (points - center) / scale

    # Proiezione su sfera (in spazio normalizzato)
    if project:
        projected_scaled, _, outside = spherical_projection(points_scaled, r1=r1, center=sphere_center)
    else:
        projected_scaled = points_scaled.clone()
        r = torch.norm(points_scaled - sphere_center, dim=1)
        outside = r > r1

    # Torna in world
    points_world = (points_scaled * scale + center).detach().cpu().numpy()
    projected_world = (projected_scaled * scale + center).detach().cpu().numpy()
    mask = outside.detach().cpu().numpy()

    # Plotter
    p = pv.Plotter()
    p.set_background("white")

    # Mesh del modello
    p.add_mesh(model_mesh_pv, color='lightgrey', opacity=0.4, show_edges=False)

    # Punti originali (solo esterni)
    p.add_points(points_world[mask], color='red', point_size=point_size,
                 render_points_as_spheres=True, label='Original points')

    # Punti proiettati
    p.add_points(projected_world[mask], color='blue', point_size=point_size,
                 render_points_as_spheres=True, label='Projected points')

    # Linee
    for p1, p2 in zip(points_world[mask], projected_world[mask]):
        line = np.vstack([p1, p2])
        p.add_lines(line, color='black', width=1)

    # Sfera come mesh (costruita in spazio normalizzato e poi portata in world)
    sphere_mesh = _create_sphere_mesh(
        center=(sphere_center * scale + center).detach().cpu().numpy(),
        r_world=None,                   # se None usiamo r1 con lo scaling anisotropo (vedi sotto)
        r_normalized=r1,
        scale=scale.detach().cpu().numpy(),
        resolution=100
    )
    p.add_mesh(sphere_mesh, color='green', opacity=0.25, smooth_shading=True, show_edges=False, label='Sphere')

    p.add_legend()
    p.show()


def _create_sphere_mesh(center, r_world=None, r_normalized=1.0, scale=None, resolution=100):
    """
    Crea una sfera:
      - se r_world è fornito: sfera in world di r_world (ignora r_normalized/scale)
      - altrimenti: sfera di r_normalized nello spazio normalizzato, poi world con 'scale' (può essere scalare o 3D)
    Nota: con scale anisotropo, la sfera normalizzata diventa ellissoide in world.
    """
    import numpy as np
    import pyvista as pv

    # Normalizza i parametri in array 1D
    c = np.asarray(center, dtype=float).reshape(3,)
    if r_world is None:
        if scale is None:
            raise ValueError("Per usare r_normalized serve 'scale'.")
        s_arr = np.asarray(scale, dtype=float)
        if s_arr.ndim == 0:  # scalare isotropo
            s_arr = np.array([s_arr.item(), s_arr.item(), s_arr.item()], dtype=float)
        else:
            s_arr = s_arr.reshape(-1)
            if s_arr.size != 3:
                raise ValueError(f"'scale' deve essere scalare o 3D; ricevuto shape {s_arr.shape}")
    else:
        s_arr = None  # non usato in modalità r_world

    # Parametrizzazione sfera unit
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    if r_world is not None:
        # Sfera direttamente in world
        X = r_world * x + c[0]
        Y = r_world * y + c[1]
        Z = r_world * z + c[2]
    else:
        # Sfera in normalizzato -> world via scale (scalare o 3D)
        Xn = r_normalized * x
        Yn = r_normalized * y
        Zn = r_normalized * z
        X = Xn * s_arr[0] + c[0]
        Y = Yn * s_arr[1] + c[1]
        Z = Zn * s_arr[2] + c[2]

    return pv.StructuredGrid(X, Y, Z).extract_surface().triangulate().clean()




def save_spherical_projection_gif(points: torch.Tensor,
                                  model_mesh_pv: pv.PolyData,
                                  center: torch.Tensor,
                                  scale: torch.Tensor,
                                  sphere_center: torch.Tensor,
                                  r1: float = 1.0,
                                  gif_path: str = "./projection_sphere.gif",
                                  device: str = "cpu",
                                  fps: int = 24,
                                  duration_s: float = 3.0,
                                  hold_start_frames: int = 8,
                                  hold_end_frames: int = 12,
                                  orbit_degrees: float = 360.0,
                                  only_outside: bool = True,
                                  point_size: int = 10,
                                  view: str = "iso",
                                  easing: str = "ease_in_out",
                                  show_paths: bool = True,
                                  show_targets: bool = True):
    """
    Salva una GIF in cui TUTTI i punti si muovono contemporaneamente verso la loro
    proiezione sulla sfera (in spazio normalizzato), mentre la camera ruota.

    - sphere_center, r1: centro e raggio della sfera in spazio normalizzato
    - gif_path: percorso file GIF (la funzione crea la cartella se manca)
    """

    # Path
    out_dir = os.path.dirname(os.path.abspath(gif_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Tensors su device
    points = points.to(device)
    center = center.to(device)
    scale = scale.to(device)
    sphere_center = sphere_center.to(device)

    # Normalizzazione
    points_scaled = (points - center) / scale

    # Maschera
    if only_outside:
        norms = torch.norm(points_scaled - sphere_center, dim=1)
        mask = norms > r1
    else:
        mask = torch.ones(points_scaled.shape[0], dtype=torch.bool, device=device)

    # Proiezione sferica (in spazio normalizzato)
    projected_scaled, _, _ = spherical_projection(points_scaled, r1=r1, center=sphere_center)

    # Torna in world
    orig_world = (points_scaled * scale + center).detach().cpu().numpy()
    proj_world  = (projected_scaled * scale + center).detach().cpu().numpy()

    idxs = torch.nonzero(mask, as_tuple=False).squeeze(1).cpu().numpy()
    if idxs.size == 0:
        raise ValueError("Nessun punto da animare (mask vuota).")

    orig = orig_world[idxs]
    targ = proj_world[idxs]

    # Plotter off-screen
    p = pv.Plotter(off_screen=True)
    p.set_background("white")

    # Modello
    p.add_mesh(model_mesh_pv, color='lightgrey', opacity=0.35, show_edges=False)

    # Sfera (costruita da normalizzato → world)
    sphere_mesh = _create_sphere_mesh(
        center=(sphere_center * scale + center).detach().cpu().numpy(),
        r_world=None,
        r_normalized=r1,
        scale=scale.detach().cpu().numpy(),
        resolution=100
    )
    p.add_mesh(sphere_mesh, color='green', opacity=0.25, smooth_shading=True, show_edges=False, label='Sphere')

    # Linee guida (opzionali)
    if show_paths:
        line_pts = np.vstack([np.column_stack((orig, targ)).reshape(-1, 3)][0])
        n_lines = len(orig)
        cells = np.empty(n_lines * 3, dtype=np.int32)
        cells[0::3] = 2
        cells[1::3] = np.arange(0, 2*n_lines, 2, dtype=np.int32)
        cells[2::3] = np.arange(1, 2*n_lines, 2, dtype=np.int32)
        lines_mesh = pv.PolyData()
        lines_mesh.points = line_pts
        lines_mesh.lines = cells
        p.add_mesh(lines_mesh, color='black', line_width=1)

    # Target (blu)
    if show_targets:
        p.add_points(targ, color='blue', point_size=point_size,
                     render_points_as_spheres=True, label='Projected points')

    # Punti in movimento (rossi)
    moving = pv.PolyData(orig.copy())
    # NB: add_points accetta direttamente pv.PolyData
    p.add_points(moving, color='red', point_size=point_size,
                 render_points_as_spheres=True, label='Original→Projected')

    p.add_legend()

    # Camera
    if view in ("iso", "xy", "xz", "yz"):
        p.camera_position = view
    else:
        p.camera_position = "iso"

    p.open_gif(gif_path, fps=fps)

    # Frame/easing
    move_frames = max(2, int(duration_s * fps))
    total_frames = hold_start_frames + move_frames + hold_end_frames
    deg_per_frame = float(orbit_degrees) / max(1, total_frames)

    if easing == "ease_in_out":
        def ease_fn(t): return 0.5 * (1.0 - np.cos(np.pi * t))
    else:
        def ease_fn(t): return t

    # Hold iniziale
    p.render()
    for _ in range(hold_start_frames):
        try: p.camera.azimuth(deg_per_frame)
        except Exception: pass
        p.write_frame()

    # Animazione simultanea
    for f in range(move_frames):
        t = f / (move_frames - 1)
        tt = ease_fn(t)
        new_pos = (1.0 - tt) * orig + tt * targ
        moving.points = new_pos

        try: p.camera.azimuth(deg_per_frame)
        except Exception: pass

        p.render()
        p.write_frame()

    # Hold finale
    for _ in range(hold_end_frames):
        try: p.camera.azimuth(deg_per_frame)
        except Exception: pass
        p.render()
        p.write_frame()

    p.close()
    return gif_path
