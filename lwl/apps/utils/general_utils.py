import os
import pickle


def try_int(value):
    try:
        return int(value)
    except ValueError:
        return value

def numerical_sort(value):
    try:
        return int(value)
    except ValueError:
        return int(''.join(c for c in value if c.isdigit())) # keep only digits from string

def list_and_order_directories(root_path, full_path=True):
    # list all directories in the root path
    directories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    # sort directories alphabetically
    sorted_directories = sorted(directories, key=numerical_sort)
    if full_path is True:
        full_path_sorted_directories = list()
        for directory in sorted_directories:
            full_path_sorted_directories.append(os.path.join(root_path, directory))
        return full_path_sorted_directories
    # else 
    return sorted_directories

def list_and_order_files(dir_path, full_path=True, remove_from_files=None):
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    if remove_from_files in files:
        files.remove(remove_from_files)
    # sort the list of files alphabetically
    sorted_files = sorted(files, key=numerical_sort)
    if full_path is True:
        full_path_sorted_files = list()
        for directory in sorted_files:
            full_path_sorted_files.append(os.path.join(dir_path, directory))
        return full_path_sorted_files
    return sorted_files

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_last_part_of_path(path):
    # check if the string contains a "/"
    if os.path.sep in path:
        # extract the last part of the path
        return os.path.basename(path)
    else:
        return path

def get_single_filename(folder_path, target_extension=None):

    # if there are multiple files
    def find_file_recursive(folder_path, target_extension):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(target_extension):
                    return os.path.join(root, file)
                
    if(target_extension != None):
        return find_file_recursive(folder_path, target_extension)

    # if there is only one files per folder
    
    # list all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # ensure there is only one file in the folder
    if len(files) != 1:
        # raise ValueError("get_single_filename|there should be exactly one file in the folder!")
        print("WARNING get_single_filename|there should be exactly one file in the folder - n files: {}".format(len(files)))
        return None
    
    file_name = files[0]
    file_path = os.path.join(folder_path, file_name)
    return file_path

def find_file_in_folders(root_path, folder_names, local_file_path):
    file_paths = []
    for folder_name in folder_names:
        file_path = os.path.join(root_path, folder_name, local_file_path)
        if os.path.exists(file_path):
            file_paths.append(file_path)    
        else:
            print(f"find_file_in_folders|file '{file_path}' does not exist!")
            exit(-1)
    return file_paths

def find_folder_with_name(root_path, folder_names, target_folder_name):
    file_paths = []
    for folder_name in folder_names:
        directory = os.path.join(root_path, folder_name)
        folder_name = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder)) and folder.startswith(target_folder_name)]
        assert len(folder_name) == 1
        file_path = os.path.join(directory, folder_name[0])
        file_paths.append(file_path)
    return file_paths
        
import numpy as np
def fsigmoid(x):
    return x / (1.0 + np.abs(x))