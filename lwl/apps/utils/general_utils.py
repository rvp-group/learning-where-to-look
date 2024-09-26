import os
import pickle


##############################################################################################
################################### GENERAL UTILS ############################################
##############################################################################################

def try_int(value):
    """
    Attempts to convert the given value to an integer.

    Parameters:
    value (any): The value to be converted to an integer.

    Returns:
    int or any: The integer representation of the value if conversion is successful;
                otherwise, returns the original value.
    """
    try:
        return int(value)
    except ValueError:
        return value

def numerical_sort(value):
    """
    Sorts a given value numerically. If the value is not directly convertible to an integer,
    it extracts and uses only the digit characters from the value.

    Args:
        value (str): The value to be sorted numerically.

    Returns:
        int: The numerical representation of the value.

    Raises:
        ValueError: If the value cannot be converted to an integer and does not contain any digits.
    """
    try:
        return int(value)
    except ValueError:
        return int(''.join(c for c in value if c.isdigit())) # keep only digits from string

def list_and_order_directories(root_path, full_path=True):
    """
    Lists and orders directories within the specified root path.

    Args:
        root_path (str): The path to the root directory where subdirectories will be listed.
        full_path (bool, optional): If True, returns the full path of each directory. 
                                    If False, returns only the directory names. Defaults to True.

    Returns:
        list: A list of directories sorted alphabetically. The list contains full paths if 
              full_path is True, otherwise it contains only directory names.
    """
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
    """
    List and order files in a directory.

    Args:
        dir_path (str): The path to the directory containing the files.
        full_path (bool, optional): If True, return the full path of the files. Defaults to True.
        remove_from_files (str, optional): A filename to remove from the list of files. Defaults to None.

    Returns:
        list: A list of files in the directory, optionally with their full paths, sorted alphabetically.
    """
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
    """
    Load and return the data from a pickle file.

    Args:
        file_path (str): The path to the pickle file to be loaded.

    Returns:
        Any: The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def make_dir(dir_path):
    """
    Creates a directory if it does not already exist.

    Args:
        dir_path (str): The path of the directory to create.

    Returns:
        None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_last_part_of_path(path):
    """
    Extracts the last part of a given file path.

    This function checks if the provided path contains the system's path separator.
    If it does, it returns the last part of the path. Otherwise, it returns the path itself.

    Args:
        path (str): The file path from which to extract the last part.

    Returns:
        str: The last part of the file path or the path itself if no separator is found.
    """
    # check if the string contains a "/"
    if os.path.sep in path:
        # extract the last part of the path
        return os.path.basename(path)
    else:
        return path
    

def get_single_filename(folder_path, target_extension=None):
    """
    Retrieves a single file from a specified folder. If a target extension is provided, 
    it searches recursively for the first file with that extension. If no extension is 
    provided, it ensures there is exactly one file in the folder.
    Args:
        folder_path (str): The path to the folder where the file is located.
        target_extension (str, optional): The file extension to search for. Defaults to None.
    Returns:
        str: The full path to the file if found, otherwise None.
    Raises:
        ValueError: If no target extension is provided and there is not exactly one file in the folder.
    """

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
    """
    Searches for a specific file within multiple folders and returns the paths to the found files.

    Args:
        root_path (str): The root directory path where the search will begin.
        folder_names (list of str): A list of folder names to search within the root directory.
        local_file_path (str): The relative path of the file to search for within each folder.

    Returns:
        list of str: A list of file paths where the specified file was found.

    Raises:
        SystemExit: If the file does not exist in any of the specified folders, the function will print an error message and exit the program.
    """
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
    """
    Finds folders with a specific name within given directories.

    Args:
        root_path (str): The root directory path where the search will begin.
        folder_names (list): A list of folder names within the root directory to search in.
        target_folder_name (str): The prefix of the folder name to search for.

    Returns:
        list: A list of paths to the folders that match the target folder name.

    Raises:
        AssertionError: If there is not exactly one folder matching the target name in each directory.
    """
    file_paths = []
    for folder_name in folder_names:
        directory = os.path.join(root_path, folder_name)
        folder_name = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder)) and folder.startswith(target_folder_name)]
        assert len(folder_name) == 1
        file_path = os.path.join(directory, folder_name[0])
        file_paths.append(file_path)
    return file_paths


def load_dataset(path_to_dataset):
    """
    Loads a dataset from a specified file path.

    Args:
        path_to_dataset (str): The file path to the dataset to be loaded.

    Returns:
        dict: The loaded dataset.

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
    """
    print("loading data {}".format(path_to_dataset))
    dataset = dict()
    with open(path_to_dataset, 'rb') as file:
        dataset = pickle.load(file) 
    print("serialized data...")
    return dataset


##############################################################################################
################################# GEOMETRICAL UTILS ##########################################
##############################################################################################

import numpy as np

def transform(T, pt):
    """
    Applies a transformation matrix to a point.

    Parameters:
    T (numpy.ndarray): A 4x4 transformation matrix.
    pt (numpy.ndarray): A 3D point represented as a numpy array.

    Returns:
    numpy.ndarray: The transformed 3D point.
    
    Raises:
    AssertionError: If the shape of T is not (4, 4).
    """
    assert T.shape == (4, 4)
    return np.dot(T[0:3, 0:3], pt) + T[0:3, 3]

def invertT(T):
    """
    Inverts a 4x4 transformation matrix.

    This function takes a 4x4 transformation matrix `T` and returns its inverse.
    The transformation matrix `T` is assumed to be composed of a 3x3 rotation 
    matrix and a 3x1 translation vector.

    Parameters:
    T (numpy.ndarray): A 4x4 transformation matrix.

    Returns:
    numpy.ndarray: The inverse of the input 4x4 transformation matrix.
    """
    invT = np.eye(4)
    R = T[0:3, 0:3]
    invT[0:3, 0:3] = np.transpose(R)
    invT[0:3, 3] = -np.transpose(R)@T[0:3, 3]
    return invT
        
def fsigmoid(x):
    """
    Computes the fsigmoid function, which is a variant of the sigmoid function.

    The fsigmoid function is defined as:
        fsigmoid(x) = x / (1.0 + abs(x))

    Parameters:
    x (float or array-like): Input value or array of values.

    Returns:
    float or array-like: The fsigmoid of the input value(s).
    """
    return x / (1.0 + np.abs(x))