import numpy as np
 # requires python3 -m pip install setuptools==61.0 --> python3 -m pip install setuptools==61.0 --> pip3 install mesh-to-sdf
from trimesh import Trimesh
import os
#pip3 install rtree
def create_cylinder_pointcloud(radius, height, points_count):
    points = np.random.uniform(low=-2.0, high=2.0, size=(points_count, 3))
    mask_cylinder = (points[:, 0]**2 + points[:, 1]**2 <= radius**2) & (np.abs(points[:, 2]) <= height/2)
    points_in_cylinder = points[mask_cylinder]
    return points_in_cylinder


class Dataset():
    name = 'DatasetCreator'
    sdf_method = 'scan'
    sample_point_count = 10000000
    normal_sample_count = 100
    number_of_points = 800000

    def __init__(self):   
        self.name = ''
        self._point_sampling_doamin_min = 0
        self._point_sampling_doamin_max = 0
        self.dataset = {}


        self.scan_count = 100
        self.scan_resolution = 400

        self.min_size = 0.01
        self.return_gradients = False

    
    # def set_sdf_N_point_sampled(self, number_of_points):
    #     self.number_of_points = number_of_points
    
    def set_sampling_domain(self, domain_min, domain_max):
        self._point_sampling_doamin_min = domain_min
        self._point_sampling_doamin_max = domain_max

    def set_mesh(self, mesh):
        if not isinstance(mesh, Trimesh):
            raise ValueError("\033[91m" + "The mesh must be a trimesh object" + "\033[0m")
            
        self.mesh = mesh
        self.name = mesh.metadata['file_name']
        print("\033[95m" + f'Mesh "{self.name}" loaded' + "\033[0m")
    
    def _get_rdm_train_points(self):
        return np.random.uniform(self._point_sampling_doamin_min, self._point_sampling_doamin_max, size=(self.number_of_points, 3))
    
    def _get_data_sdf(self, show=False): 
        import mesh_to_sdf
        #returns points, sdf--> points and sdf are points near the surface, query_sdf --> the sdf of the query_points

        near_points, near_sdf= mesh_to_sdf.sample_sdf_near_surface(self.mesh, 
                                                        number_of_points=self.number_of_points, # the number N of points to be sampled, including surface points and uniform points
                                                        surface_point_method=self.sdf_method, 
                                                        sign_method='normal', 
                                                        scan_count=self.scan_count, 
                                                        scan_resolution=self.scan_resolution,
                                                        sample_point_count=self.sample_point_count,
                                                        normal_sample_count=self.normal_sample_count,
                                                        min_size=0.015, # The fraction of uniformly sampled that should be inside the shape. If this is 0.015 and less than 1.5% of uniformly sampled points have negative SDFs, an exception is thrown. This can be used to detect bad meshes.
                                                        return_gradients=False) 
        
        query_points = self._get_rdm_train_points()
        query_sdf = mesh_to_sdf.mesh_to_sdf(self.mesh,
                                            query_points, 
                                            surface_point_method=self.sdf_method, 
                                            sign_method='normal', 
                                            bounding_radius=None, 
                                            scan_count=self.scan_count,
                                            scan_resolution=self.scan_resolution,
                                            sample_point_count=self.sample_point_count,
                                            normal_sample_count=self.normal_sample_count,
                                            )

        def filter_negative_sdf_points_out_from_mesh_bbox(points, sdf, mesh):
            points = np.asarray(points)

            # Bounding box reale della mesh (già scalata in unit sphere)
            min_bounds = mesh.bounds[0]
            max_bounds = mesh.bounds[1]

            # Maschere logiche
            sdf_negative = sdf < 0
            sdf_positive = ~sdf_negative

            # Controlla se i punti negativi sono dentro la bounding box della mesh
            in_mesh_bbox = np.all((points >= min_bounds) & (points <= max_bounds), axis=1)

            # Valid mask:
            # - Tieni tutti i punti con sdf >= 0
            # - Tieni solo quelli negativi che sono dentro la bounding box
            valid_mask = sdf_positive | (sdf_negative & in_mesh_bbox)

            # Log
            num_total = len(points)
            num_filtered_out = np.sum(~valid_mask)

            print(f'\033[93m[Filter] Removed {num_filtered_out} / {num_total} points (SDF<0 and outside mesh bounding box)\033[0m')

            return points[valid_mask], sdf[valid_mask]
        
        near_points, near_sdf = filter_negative_sdf_points_out_from_mesh_bbox(near_points, near_sdf, self.mesh)
        query_points, query_sdf = filter_negative_sdf_points_out_from_mesh_bbox(query_points, query_sdf, self.mesh)

        if show:
            from src.visu.debug_visu_utils import show_sdf_pointcloud
            distance = 0.0
            show_sdf_pointcloud(near_points, near_sdf, filter_interval=distance)
            show_sdf_pointcloud(query_points, query_sdf, filter_interval=distance)

        return near_points, near_sdf, query_points, query_sdf

    def get_boundigBox_maxDimention(self):
        # Calcola il centroide
        bb_center = self.mesh.bounding_box.centroid
        #print limit points of the bounding box
        print(f'\033[95m' + f'Mesh bounding box: {self.mesh.bounds}' + '\033[0m') 
        
        # Calcola la max grandezza della self.mesh
        max_distance = np.max(np.linalg.norm(self.mesh.vertices - bb_center, axis=1))

            
        return bb_center, max_distance
    
    def create_inside_unit_sphere(self, debug=True):
        import mesh_to_sdf
        '''
        This function takes a mesh and returns 2 dataset limited in a bounding box [-1,1]
        '''
        print('\033[95m'+'Collecting sdf data inside unit sphere for ' + self.name + '\033[0m')
        self.set_sampling_domain(-1, 1)
        centroid_offset, scale_factor = self.get_boundigBox_maxDimention()
        print(f'\033[95m' + f'Mesh centroid offset: {centroid_offset}\033[0m')
        print(f'\033[95m' + f'Mesh scale factor: {scale_factor}\033[0m')
        
        # trasformo la mesh a stare dentro un intervallo [-1,1]
        
        self.mesh = mesh_to_sdf.scale_to_unit_sphere(self.mesh) #non DOVREBBE deformare la geometria mesh ?!? non credo ma bho
        # self.mesh = subdivide_size(self.mesh, 0.1)
        # show_mesh_mtplt(self.mesh)
        
        


        self.mesh.metadata['file_name'] = self.name
        '''
            nominally the operation should not deform the geometry, but it is not guaranteed
            the mesh is scaled to fit inside a unit sphere, so the maximum distance from the centroid is 1
            the mesh is centered in the origin
            it means that i don't have to apply formulations to bring to parameters to the origin and scale 
            otherwise i have to apply the following operations:
            scaled_centroid_offset,scaled_max_dim = self.get_boundigBox_maxDimention(mesh_sphere, debug=debug)
            centroid_offset = centroid_offset + scaled_centroid_offset
            scale_factor = scale_factor/scaled_max_dim
            to bring the mesh back to the original from [-1,1] i have to multiply for scale_factor and add the centroid

        '''

        near_points, near_sdf, query_points, query_sdf = self._get_data_sdf(show=debug)

        mask_inside_unit_box = np.all((near_points >= -1) & (near_points <= 1), axis=1)
        mask_positive_sdf = near_sdf >= 0
        valid_mask = mask_inside_unit_box | mask_positive_sdf

        # Applica il filtro
        near_points = near_points[valid_mask]
        near_sdf = near_sdf[valid_mask]

        
        self.dataset['file_name'] = self.mesh.metadata['file_name']
        self.dataset['near_points']= near_points
        self.dataset['near_sdf']= near_sdf
        self.dataset['query_points']= query_points
        self.dataset['query_sdf']= query_sdf
            
        #add the centroid offset and scale factor to the dataset
        self.dataset['mesh_centroid_offset'] = centroid_offset
        self.dataset['mesh_scale_factor'] = scale_factor
        self.dataset['sdf_domain_min'] = self._point_sampling_doamin_min
        self.dataset['sdf_domain_max'] = self._point_sampling_doamin_max
        

        return self.dataset
        

    def save(self, path):
        path = os.path.join(path, self.name + '.npy')

        print("\033[1m" + f'[SAVING] DATASET FILE: {self.name} in --> {path}' + "\033[0m")
        print("\033[90m" + f'In the following domain [Min, Max] = [{self._point_sampling_doamin_min}, {self._point_sampling_doamin_max}]' + "\033[0m")
        print("\033[90m" + f'Number of points sampled: {self.number_of_points}' + "\033[0m")
        print("\033[90m" + f'Mesh centroid offset: {self.dataset["mesh_centroid_offset"]}' + "\033[0m")
        print("\033[90m" + f'Mesh scale factor: {self.dataset["mesh_scale_factor"]}' + "\033[0m")

        np.save(path, self.dataset, allow_pickle=True)
        
        print("\033[92m" + f'[SAVED] FILE: {self.name} in --> {path}' + "\033[0m")
        print("\033[1m" + f'---------------------' + "\033[0m")
    


        

   
    
