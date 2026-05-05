
import trimesh
import numpy as np
import pyvista as pv
from typing import List
from vtkmodules.vtkFiltersCore import vtkAppendPolyData

class RobotSene:
    def __init__(self) -> None:

        self.scene = pv.Plotter()
        self.list_of_meshes = []
    
    # def add_pointcloud(self, points: np.ndarray, color='red', point_size=10):
    #     assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
    #     # PyVista accetta direttamente Nx3 con add_mesh per punti se passiamo i flag sotto
    #     self.scene.add_mesh(points, render_points_as_spheres=True, point_size=point_size, color=color)

    def add_pointcloud(self, points: np.ndarray, color='red', point_size=10, opacity=1.0):
        assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
        self.scene.add_mesh(points, render_points_as_spheres=True,
                            point_size=point_size, color=color, opacity=opacity)

    def add_polyline(self, pts: np.ndarray, color='deepskyblue', line_width=2):
        """pts: (M,3) ordinati lungo la traiettoria"""
        assert pts.ndim == 2 and pts.shape[1] == 3, "polyline points must be (M,3)"
        line = pv.lines_from_points(pts, close=False)
        self.scene.add_mesh(line, color=color, line_width=line_width)
    
    def incornicia_mesh(self, mesh, linewidth=2, color='black'):
        edges = mesh.extract_feature_edges(
            boundary_edges=True, feature_edges=True, manifold_edges=True, feature_angle=30
        )
        if getattr(edges, "n_points", 0) == 0 or getattr(edges, "n_cells", 0) == 0:
            return
        self.scene.add_mesh(edges, color=color, line_width=linewidth)
        # self.scene.add_mesh(mesh, style='wireframe', line_width=linewidth, color=color)

    def add_mesh(self, mesh, opacity=0.5, color='lightgray'):
        self.scene.add_mesh(mesh, color=color, opacity=opacity, show_edges=False)
    
    def add_cartesian_frame(self, tf: np.ndarray=np.eye(4),
                            axis_length: float = 0.2,
                            shaft_radius: float = 0.005):
        frame = make_cartesian_frame_polydata_pv(
            tf=tf,
            axis_length=axis_length,
            shaft_radius=shaft_radius,
            head_length_ratio=0.25,
            head_radius_ratio=2.0,
            sections=24
        )
        self.scene.add_mesh(frame, scalars='RGBA', rgb=True)
    
    def show(self):
        self.scene.show()

    def visualize_gradient_descent(
        self,
        points: np.ndarray,
        projected_points: np.ndarray,
        color_points: str = 'red',
        color_proj: str = 'blue',
        line_color: str = 'deepskyblue',
        arrow_color: str = 'deepskyblue',
        arrow_scale: float = 0.3,
    ):
        """points: (N,3), projected_points: (N,3)"""
        assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
        assert projected_points.ndim == 2 and projected_points.shape[1] == 3, "projected_points must be (N,3)"
        assert points.shape[0] == projected_points.shape[0], "points and projected_points must have same length"

        # nuvole di punti
        self.add_pointcloud(points,           color=color_points, point_size=20)
        self.add_pointcloud(projected_points, color=color_proj,   point_size=20)

        for p, pp in zip(points, projected_points):
            # segmento tra p e pp (solo se ti serve ancora)
            seg = np.vstack([p, pp])
            self.add_polyline(seg, color=line_color, line_width=2)

            # freccia che parte da projected point e va oltre,
            # nella stessa direzione pp - p (cioè "via" da p)
            direction = pp - p
            arrow = pv.Arrow(
                start=pp,          # parte dal projected point
                direction=direction,
                scale=arrow_scale  # fattore di scala sulla lunghezza
            )
            self.scene.add_mesh(arrow, color=arrow_color)

    def add_sphere_model(
        self,
        sm,
        color: str = "dodgerblue",
        opacity: float = 0.35,
        resolution: int = 18,
        show_wire: bool = False,
        wire_color: str = "black",
        wire_width: float = 1.0,
    ):
        """
        Disegna tutte le sfere contenute in uno SphereModel (centers,radii).
        Nessuna trasformazione applicata.
        """
        centers = sm.centers.detach().cpu().numpy() if hasattr(sm.centers, "detach") else np.asarray(sm.centers)
        radii   = sm.radii.detach().cpu().numpy()   if hasattr(sm.radii, "detach")   else np.asarray(sm.radii)

        for c, r in zip(centers, radii):
            sph = pv.Sphere(radius=float(r), center=(float(c[0]), float(c[1]), float(c[2])),
                            theta_resolution=resolution, phi_resolution=resolution)
            self.scene.add_mesh(sph, color=color, opacity=opacity, smooth_shading=True)
            if show_wire:
                self.scene.add_mesh(sph.extract_all_edges(), color=wire_color, line_width=wire_width)



def _append_polydata(meshes):
    app = vtkAppendPolyData()
    for m in meshes:
        app.AddInputData(m)
    app.Update()
    return pv.wrap(app.GetOutput()).clean().triangulate()

def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4)
    M[1,1] =  c; M[1,2] = -s
    M[2,1] =  s; M[2,2] =  c
    return M

def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4)
    M[0,0] =  c; M[0,2] =  s
    M[2,0] = -s; M[2,2] =  c
    return M

def make_cartesian_frame_polydata_pv(
    tf: np.ndarray = None,
    axis_length: float = 0.2,
    shaft_radius: float = 0.005,
    head_length_ratio: float = 0.25,
    head_radius_ratio: float = 2.0,
    sections: int = 24
) -> pv.PolyData:
    """Crea X/Y/Z come frecce (cilindro+cono) direttamente in PyVista e restituisce un'unica PolyData.
       Colori: X=rosso, Y=verde, Z=blu (RGBA per-vertice). Compatibile con vecchie versioni di PyVista."""
    head_len   = axis_length * head_length_ratio
    shaft_len  = axis_length - head_len
    head_rad   = shaft_radius * head_radius_ratio

    def _arrow_along_z(color_rgba):
        shaft = pv.Cylinder(center=(0, 0, shaft_len/2.0),
                            direction=(0, 0, 1),
                            radius=shaft_radius,
                            height=shaft_len,
                            resolution=sections)
        head  = pv.Cone(center=(0, 0, shaft_len + head_len/2.0),
                        direction=(0, 0, 1),
                        height=head_len,
                        radius=head_rad,
                        resolution=sections)
        arrow = _append_polydata([shaft, head])
        rgba = np.tile(np.array(color_rgba, dtype=np.uint8), (arrow.n_points, 1))
        arrow.point_data['RGBA'] = rgba
        return arrow

    # Z (blu)
    z_arrow = _arrow_along_z([0, 0, 255, 255])

    # Rotazioni per ricavare Y e X da Z
    y_arrow = z_arrow.copy(deep=True)
    y_arrow.transform(_rot_x(-np.pi/2), inplace=True)
    y_arrow.point_data['RGBA'][:] = np.array([0, 255, 0, 255], dtype=np.uint8)  # verde

    x_arrow = z_arrow.copy(deep=True)
    x_arrow.transform(_rot_y(+np.pi/2), inplace=True)
    x_arrow.point_data['RGBA'][:] = np.array([255, 0, 0, 255], dtype=np.uint8)  # rosso

    frame = _append_polydata([x_arrow, y_arrow, z_arrow])

    if tf is not None:
        if not (isinstance(tf, np.ndarray) and tf.shape == (4, 4)):
            raise ValueError("tf deve essere una matrice omogenea 4x4 (np.ndarray).")
        frame.transform(tf, inplace=True)

    return frame





def plot_robot_isosurfaces(mesh_files, transforms, spacing=0.02, n_isosurfaces=12,
                           cmap="turbo", clip_half=False, clip_normal=[1,0,0],
                           padding=0.10, show=True, show_robot=True):
    """
    Visualizza isosuperfici multilivello attorno al robot, con opzionale robot mesh.
    """
    import pyvista as pv
    try:
        from pyvista.core import UniformGrid
    except ImportError:
        try:
            from pyvista import UniformGrid
        except ImportError:
            UniformGrid = pv.ImageData
    # 1) carica + trasforma e unisci
    all_links = []
    for link, path in mesh_files.items():
        m = pv.read(path)
        T = transforms[link]
        mt = m.copy()
        mt.transform(T)
        all_links.append(mt)

    robot_mesh = all_links[0]
    for m in all_links[1:]:
        robot_mesh = robot_mesh.merge(m)

    robot_mesh = robot_mesh.triangulate().clean()

    # 2) bounding box + padding
    xmin, xmax, ymin, ymax, zmin, zmax = robot_mesh.bounds
    xr = xmax - xmin; yr = ymax - ymin; zr = zmax - zmin
    xmin -= padding * xr; xmax += padding * xr
    ymin -= padding * yr; ymax += padding * yr
    zmin -= padding * zr; zmax += padding * zr

    nx = int(np.ceil((xmax - xmin)/spacing)) + 1
    ny = int(np.ceil((ymax - ymin)/spacing)) + 1
    nz = int(np.ceil((zmax - zmin)/spacing)) + 1

    # 3) griglia volumetrica
    grid = UniformGrid()
    grid.origin = (xmin, ymin, zmin)
    grid.spacing = (spacing, spacing, spacing)
    grid.dimensions = (max(nx, 2), max(ny, 2), max(nz, 2))

    # 4) campo distanza
    dist_field = grid.compute_implicit_distance(robot_mesh)

    # 5) isosuperfici
    contours = dist_field.contour(isosurfaces=n_isosurfaces, scalars="implicit_distance")

    if clip_half:
        contours = contours.clip(normal=clip_normal,
                                 origin=((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2))

    # 6) show
    if show:
        pl = pv.Plotter()
        if show_robot:
            pl.add_mesh(robot_mesh, color="#2C2A2A", opacity=1.0, show_edges=False)
        pl.add_mesh(contours, cmap=cmap, opacity=0.7)
        pl.show()

    return robot_mesh, contours
