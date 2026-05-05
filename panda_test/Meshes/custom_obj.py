import trimesh
import numpy as np
import trimesh
from trimesh.creation import cylinder, box
from trimesh.boolean import difference

class CustomOBJMeshes:
    @staticmethod
    def create_letter_C():
        # Parametri "a piacere"
        width  = 2.0   # larghezza complessiva della C
        height = 2.0   # altezza complessiva
        thick  = 0.4   # spessore dei tratti
        depth  = 0.2   # profondità (asse z)

        # --- BARRETTA VERTICALE SINISTRA ---
        vert = trimesh.creation.box(extents=(thick, height, depth))
        # la porto sulla sinistra
        vert.apply_translation((
            - (width / 2.0 - thick / 2.0),  # x
            0.0,                            # y
            0.0                             # z
        ))

        # --- BARRETTA ORIZZONTALE SUPERIORE ---
        top = trimesh.creation.box(extents=(width, thick, depth))
        top.apply_translation((
            0.0,                             # x
            + (height / 2.0 - thick / 2.0),  # y
            0.0                              # z
        ))

        # --- BARRETTA ORIZZONTALE INFERIORE ---
        bottom = trimesh.creation.box(extents=(width, thick, depth))
        bottom.apply_translation((
            0.0,                             # x
            - (height / 2.0 - thick / 2.0),  # y
            0.0                              # z
        ))

        # Unisco le tre mesh in una sola "C"
        C_mesh = trimesh.util.concatenate([vert, top, bottom])

        return C_mesh
    @staticmethod
    def create_rounded_C():
        from trimesh.creation import cylinder, box
        from trimesh.boolean import difference

        # Parametri "a piacere"
        R_outer = 1.0   # raggio esterno dell'anello
        R_inner = 0.6   # raggio interno (cavo)
        height  = 0.3   # spessore lungo z
        sections = 128  # più alto = più liscio

        # --- Anello pieno (donut "pieno") ---
        outer = cylinder(radius=R_outer, height=height, sections=sections)
        inner = cylinder(radius=R_inner, height=height, sections=sections)

        # Anello = cilindro grande - cilindro piccolo
        ring = difference([outer, inner])

        # --- Taglio per aprire la C (gap a destra) ---
        gap_width  = R_outer * 1.3   # quanto "aprire" la C
        gap_height = R_outer * 2.2   # altezza del box di taglio

        gap_box = box(extents=(gap_width, gap_height, height * 1.2))
        # lo sposto a destra per tagliare il lato dell'anello
        gap_box.apply_translation((
            R_outer * 0.6,  # verso +x
            0.0,
            0.0
        ))

        # --- C arrotondata = anello - box di taglio ---
        C_rounded = difference([ring, gap_box])

        return C_rounded
    @staticmethod
    def create_cube(size=1.0):
        cube = trimesh.creation.box(extents=(size, size, size))
        cube.apply_translation([0.0, 0.0, 0.0])  # lo sposto a sinistra giusto per separarlo
        return cube
    @staticmethod
    def create_parallelepiped(a=1.0, b=2.0, c=0.5):
        box_mesh = trimesh.creation.box(extents=(a, b, c))
        box_mesh.apply_translation([0.0, 0.0, 0.0])  # spostato a destra
        return box_mesh
    @staticmethod
    def create_sphere(radius=0.75):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        sphere.apply_translation([0.0, 0.0, 0.0])  # lasciata al centro
        return sphere
# Esempi di utilizzo:

C_mesh = CustomOBJMeshes.create_letter_C()
C_rounded_mesh = CustomOBJMeshes.create_rounded_C()
cube_mesh = CustomOBJMeshes.create_cube(size=1.0)
box_mesh = CustomOBJMeshes.create_parallelepiped(a=1.0, b=2.0, c=0.5)
sphere_mesh = CustomOBJMeshes.create_sphere(radius=0.75)

# Salvataggio delle mesh in file .stl
C_mesh.export('letter_C.stl')
C_rounded_mesh.export('rounded_C.stl')
cube_mesh.export('cube.stl')
box_mesh.export('parallelepiped.stl')
sphere_mesh.export('sphere.stl')