import os, shutil, time


def create_folder_path(abs_path, folder_name):
    folder_path = os.path.join(abs_path, folder_name)
   

    return folder_path

class FolderManage:
    def __init__(self, ws_path, rel_path, extension, debug_content = False):
        #directory of alla files is the name of the folder that contains the components that u want to model
        
        if not isinstance(extension, str):
            raise ValueError("\033[91m" + "The body subfolder folder must be a string" + "\033[0m")
        
        self.extension = extension
        self.ws_path    = ws_path
        self.rel_path   = rel_path
        # print("\033[1m" + f'[INFO] WORKSPACE PATH: {ws_path}' + "\033[0m")
        self.folder_path  = os.path.join(ws_path, rel_path)
        # print("\033[1m" + f'[INFO] WORKSPACE PATH: {self.folder_path}' + "\033[0m")
        # print("\033[1m" + f'[INFO] WORKSPACE PATH: {self.folder_path}' + "\033[0m")

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print("\033[93m [Created]:\033[0m" + "\033[1m" + f'{self.folder_path}' + '\033[0m')

        self.scan_files(debug_content)
        self.folder_is_protected    = False
        

    def scan_files(self, debug = True):
        if debug:
            print('\033[1m'+ f'[INFO] SCAN For All Files in PATH: "{self.folder_path}"' + "\033[0m")
        files = os.listdir(self.folder_path)
        self.folder_path_list = []
        
        for file in files:
            if file.endswith(self.extension):
                if debug:
                    print('\033[94m' + f'[INFO:] FOUNDED FILE: {file} in --> {self.folder_path}' + "\033[0m")
                
                self.folder_path_list.append(os.path.join(self.folder_path, file))
        
        if len(self.folder_path_list) == 0:
            if debug:
                print("\033[1m" + f'EMPTY FOLDER: {self.folder_path}' + "\033[0m")
        if debug:
            print("\033[1m" + f'---------------------' + "\033[0m") 
        
        return self.folder_path_list
    
    def add_file(self, file_path):
        self.folder_path_list.append(file_path)
        
    def wanna_save(self):
        overwrite = input(f"Do you want to overwrite it? [y/n]").lower()
        if not overwrite == 'y':
            return False
        return True
    
    def check_name_extension(self, name, debug = False):
        dataset_extension = name.split(".")[-1]
        if self.extension != dataset_extension:
            if debug:
                print("\033[33m" + f"The file searched hasn't an extension '{self.extension}' ADDING..." + "\033[0m")
            name = name + f".{self.extension}"
        return name
    
    def check_file_existence(self, searched_file, debug=True): #attenzione richiede che non ci sia estensione
        self.scan_files(debug=False)
        # searched_file = self.check_name_extension(searched_file, debug=False)
        # print(self.folder_path_list)
        # print(searched_file)

        if searched_file in self.folder_path_list:
        # for file in self.folder_path_list:
            # print(searched_file)
            # if searched_file == os.path.basename(file).rsplit('.', 1)[0]: # FIXED: == instead of in beacause it checks two strings
                if debug:
                    print("\033[92m" + f'FILE: "{searched_file}" EXISTS in "{self.folder_path}"' + "\033[0m")
                return True
        if debug:
            print("\033[91m" + f'FILE: "{searched_file}" NOT FOUND in "{self.folder_path}"' + "\033[0m")
        
        return False
    
    def delete_file(self, file_name):
        file_name = self.check_name_extension(file_name, debug=False)
        file_path = self.get_file_path(file_name, debug=False)
        if file_path is None:
            return
        os.remove(file_path)
        self.scan_files(debug=False)
        # print("\033[91m" + f'[DELETED] FILE: {file_path}' + "\033[0m")
        # print("\033[1m" + f'---------------------' + "\033[0m")

    def delete_content(self):
        if self.folder_is_protected:
            print("\033[91m" + f'ERROR --> {self.folder_path} FOLDER IS PROTECTED, CANNOT DELETE' + "\033[0m")
            return

        for file in self.folder_path_list:
            os.remove(file)
            print("\033[91m" + f'[DELETED] FILE: {file}' + "\033[0m")
            print("\033[1m" + f'---------------------' + "\033[0m")
        self.folder_path_list = []

        print("\033[93m" + f'[DELETED] FOLDER: {self.folder_path} now is EMPTY' + "\033[0m")

    
    def set_protect_folder(self, value = True):
        self.folder_is_protected = value
    
    def get_workspace_folders_path(self):
        return self.ws_path

    def get_ws_path(self):
        return self.ws_path

    def get_path(self):
        return self.folder_path

    def get_extension(self):
        return self.extension
    
    def get_file_path(self, expected_file, debug = True):
        self.scan_files(debug=False)
        expected_file = self.check_name_extension(expected_file, debug=False)
        for path in self.folder_path_list:
            if os.path.basename(path) == expected_file:
                return path
        if debug:
            print("\033[91m" + f'FILE: "{expected_file}" NOT FOUND in "{self.folder_path}"' + "\033[0m")
        return None

    def get_file_paths(self):
        self.scan_files()
        return self.folder_path_list 
    
    def get_file_names(self):
         file_names = [os.path.splitext(os.path.basename(path))[0] for path in self.folder_path_list]
         return file_names
    
    def copy_file(self, old_path, new_path):
        self.scan_files(debug=False)
        # print("\033[92m" + f'COPYING FILE: {old_path} to {new_path}' + "\033[0m")
        dest = shutil.copyfile(old_path, new_path)
        self.scan_files(debug=False) # update file list of the folder, after copying
        print("\033[92m" + f'FILE: "{dest}" COPIED' + "\033[0m")
        time.sleep(0.5)
    
    

        
if __name__ == "__main__":

    ws_path = os.getcwd()
    robot_folder = "tiago"
    necessary_folder = "pippo"
    sub_folder = "pluto"
    folder = FolderManage(ws_path, robot_folder, necessary_folder, sub_folder, "stl")

    # print(folder.delete_content())
