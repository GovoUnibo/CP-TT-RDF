import numpy as np
from src.core.assets.FolderManage import FolderManage
from src.core.assets.entities.Dataset import Dataset

class DatasetHandler(FolderManage):
    def __init__(self, ws_path, extention='npy'):
        super().__init__(ws_path, 'Dataset', extention)
        self.debug = True
        

    def load(self, dataset_name):
        ''' load a dictionary from a file'''
        if not isinstance(dataset_name, str):
            raise ValueError("The dataset name must be a string")

        dataset_name = super().check_name_extension(dataset_name)
        
        path = super().get_file_path(dataset_name)
        if path is None:
            return None
        
        print("\033[1m" + f'LOADING DATASET FILE in --> {path}' + "\033[0m")
        print("\033[1m" + f'---------------------' + "\033[0m")
        dataset = np.load(path, allow_pickle=True).item()
        name = dataset['file_name']
        print("\033[92m" + f'[LOADED] FILE: {name} in --> {path}' + "\033[0m")
        print("\033[1m" + f'---------------------' + "\033[0m")
        return dataset
    
    def create(self, dataset_name, mesh, debug=False, **kwargs):   
        method           = kwargs.get('method', Dataset.sdf_method)
        sample_points    = kwargs.get('sample_points', Dataset.sample_point_count)
        number_of_points = kwargs.get('number_of_points', Dataset.number_of_points)

        dataset = Dataset()
        dataset.sdf_method         = method
        dataset.sample_point_count = sample_points
        dataset.number_of_points   = number_of_points
        dataset.set_mesh(mesh.get())
        
        dataset.create_inside_unit_sphere(debug=debug)

        dataset.save(self.get_path())


        return dataset.dataset
    


    

    

