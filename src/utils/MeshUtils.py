import trimesh
import torch
from trimesh.base import Trimesh
import numpy as np


def voxelize( mesh, voxel_resolution, voxel_mesh_name=None):
    if not isinstance(mesh, Trimesh):
        raise ValueError("\033[91m" + "The mesh must be a trimesh object" + "\033[0m")
    name = mesh.metadata['file_name']
    voxelized_mesh = transform_to_voxel(mesh, voxel_resolution)


    voxelized_mesh.metadata['file_name'] = name

    voxelized_mesh = overlap_origins(mesh, voxelized_mesh)
    
    voxel_mesh_name.metadata['file_name'] = name

    return voxelized_mesh

def surface_rdm_uniform_sampling(mesh, num_points):

    if not isinstance(mesh, Trimesh):
        raise ValueError("\033[91m" + "The mesh must be a trimesh object" + "\033[0m")
    
    points = []
    areas = mesh.area_faces
    cumulative_areas = np.cumsum(areas)
    total_area = cumulative_areas[-1]

    for _ in range(num_points):
        # Genera un punto casuale uniforme tra 0 e l'area totale
        rand_area = np.random.rand() * total_area
        
        # Trova il triangolo corrispondente all'area generata casualmente
        index = np.searchsorted(cumulative_areas, rand_area)
        
        # Campiona casualmente un punto all'interno del triangolo trovato
        point = mesh.triangles[index].mean(axis=0)
        points.append(point)

    return np.array(points)


def apply_transform(mesh, translation, quaternion):
    # print(translation, quaternion)
    mesh.apply_transform(trimesh.transformations.quaternion_matrix(quaternion))
    mesh.apply_translation(translation)
    return mesh

def concatenate_meshes(mesh_list, debug=True):
    
    for i in range(len(mesh_list)):
        # print(mesh_list[i].metadata['file_name'])  
        mesh_list[i] = apply_transform(mesh_list[i], mesh_list[i].metadata['translation'], mesh_list[i].metadata['quaternion']) 

    mesh = trimesh.util.concatenate(mesh_list)
    mesh.metadata = mesh_list[0].metadata
    if debug:
        mesh.show()
    
    return mesh

def overlap_origins(original_mesh, disaligned_mesh):
    # Get the bounding box centroid of the original mesh
    original_centroid = original_mesh.bounding_box.centroid
    
    # Get the bounding box centroid of the disaligned mesh
    disaligned_centroid = disaligned_mesh.bounding_box.centroid
    
    # Calculate the translation vector
    translation = original_centroid - disaligned_centroid
    
    # Apply the translation to the mesh
    disaligned_mesh.vertices += translation
    
    return disaligned_mesh


def denormalize_mesh(mesh, scale, offset): # porta la mesh dal frame unitario al frame originale
    scale = np.asarray(scale, dtype=float)
    offset = np.asarray(offset, dtype=float)

    if scale.ndim == 0:
        scale = np.repeat(float(scale), 3)
    if offset.ndim == 0:
        offset = np.repeat(float(offset), 3)

    mesh.vertices = mesh.vertices * scale + offset
    return mesh

# def normalize_mesh(mesh, domain_min, domain_max): # porta la mesh da un intervallo [min,max] a [0,1]
#         '''
#             Takes a mesh and normalizes its vertices from the interval [min, max] to [0, 1]
#         '''
#         mesh.vertices = (mesh.vertices - domain_min)/(domain_max-domain_min)
#         return mesh

def normalize_mesh(mesh):  # mesh centrata affinchè la dimensione più grande sia 1
    # Calcola il centroide
    centroid = mesh.centroid

    # Trasla i vertici
    mesh.vertices -= centroid

    # Calcola la massima distanza dal centroide
    max_distance = np.max(np.linalg.norm(mesh.vertices, axis=1))

    # Ridimensiona la mesh
    mesh.vertices /= max_distance

    return mesh

def subdivide_size(mesh, subdevide_size=0.01, show=False):    
    vertices,faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.01, max_iter=10, return_index=False)
    mesh = trimesh.Trimesh(vertices,faces)
    center = np.mean(mesh.vertices,axis=0)
    verts = torch.from_numpy(mesh.vertices-center)
    normals = torch.from_numpy(mesh.vertex_normals)
    cosine = torch.cosine_similarity(verts,normals)
    normals[cosine<0] = -normals[cosine<0]
    normals = normals.numpy()
    ray_visualize = trimesh.load_path(np.hstack((mesh.vertices, mesh.vertices + normals / 100)).reshape(-1, 2, 3))
    new_verts,new_faces = trimesh.remesh.subdivide_to_size(verts, faces, max_edge=0.1, max_iter=10, return_index=False)
    new_mesh = trimesh.Trimesh(new_verts,new_faces)
    # new_mesh = new_mesh.simplify_quadratic_decimation(500)
    if show:
        scene = trimesh.Scene()
        scene.add_geometry(mesh)
        scene.add_geometry(ray_visualize)
    
    return mesh

def transform_to_voxel(mesh, voxel_resolution=128, show=True):
    import skimage, mesh_to_sdf
    voxels = mesh_to_sdf.mesh_to_voxels(mesh, voxel_resolution=voxel_resolution,
                                        surface_point_method='scan',  # or 'sample'
                                        sign_method='normal',
                                        scan_count=250,
                                        scan_resolution=1000,
                                        sample_point_count=10000000,
                                        normal_sample_count=11, # with 11 or 5 there are some residual meshes outside of gripper 
                                        pad=True,
                                        check_result=False)
    
    spacing = max(mesh.extents) / voxel_resolution # Spacing for all links

    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0, spacing=(spacing, spacing, spacing))
    
    mesh_voxelized = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    if show:
        scene = trimesh.Scene()
        scene.add_geometry(mesh_voxelized)
        scene.show()
    
    return mesh_voxelized


def translate_in_positive_axis(mesh):
    # Calcola il minimo lungo ciascuna dimensione
    min_coords = np.min(mesh.vertices, axis=0)

    # Calcola la traslazione necessaria
    translation = -min_coords

    # Trasla i vertici
    mesh.vertices += translation

    return mesh, translation


def trimesh_to_pyvista(tm_mesh, T):
    import pyvista
    import numpy as np

    if isinstance(T, torch.Tensor):
        T = T.cpu().numpy() if T.is_cuda else T.numpy()

    vertices = tm_mesh.vertices
    faces = tm_mesh.faces

    faces_pv = np.hstack((np.full((faces.shape[0], 1), 3), faces)).flatten()
    mesh = pyvista.PolyData(vertices, faces_pv).copy()

    # Apply transform in place
    mesh.transform(T)
    return mesh
