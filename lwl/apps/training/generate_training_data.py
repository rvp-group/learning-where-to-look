import argparse
import numpy as np
from scipy.spatial.transform import Rotation

from lwl.apps.utils.general_utils import *
from lwl.apps.utils.dataset_generator import Data, InputDims
from lwl.apps.utils.colmap.colmap_read_write_model import read_images_binary
from lwl.apps.utils.seed import *

COLMAP_CAMERAS_FILE = "sparse/0/images.bin"
LABELS_DIR = "labels_directions"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for training, this script puts the data in correct format to speed up data loading during training. '
                                                 'In addition does some processing, for instance dividing set with positive and negative samples, discretizing img in bins, etc')
    parser.add_argument('--data_path', type=str, required=True, help='path to your data file')
    parser.add_argument('--positive_percentage', type=float, required=True, help='percentage of positive samples during required for training, i.e. 0.5 is balanced between positive and negative samples, for test set use 0 (no need to balance)', default=0.5)
    parser.add_argument('--rows', type=int, help='image rows', default=480)
    parser.add_argument('--cols', type=int, help='image cols', default=640)
    parser.add_argument('--num_bin_per_dim', type=int, help='number of bins per dimension', default=30)
    parser.add_argument('--output_data', type=str, help='output data filename, no extension required', required=True)
    parser.add_argument('--parallel', type=int, help='process data in parallalel to speed up computation, 0 required or max num of cpu used, 1 single cpu, other number is custum', default=0)
    args = parser.parse_args()

    ############### start processing folders ###############
    # get root path and path to each folder representing a camera bucket in the grid world
    folders = [f for f in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, f))]
    print("parsing the following folder with data", folders)
    
    print("looking for colmap camera poses obtained during mapping")
    camera_poses_files = find_file_in_folders(args.data_path, folders, COLMAP_CAMERAS_FILE)
    camera_poses_list_of_list = list()
    for c_file_path in camera_poses_files:
        camera_poses_list = list()
        camera_poses = read_images_binary(c_file_path)
        # colmap format qw, qx, qy, qz, tx, ty, tz
        for pose in camera_poses.values():
            Tcw = np.identity(4)
            rot = Rotation.from_quat([pose.qvec[1], pose.qvec[2], pose.qvec[3], pose.qvec[0]])
            Tcw[0:3, 0:3] = rot.as_matrix()
            Tcw[0:3, 3] = pose.tvec
            camera_poses_list.append(Tcw)
        camera_poses_list_of_list.append(camera_poses_list)
    
    
    label_folders = find_folder_with_name(args.data_path, folders, LABELS_DIR)
    print("camera poses: ", camera_poses_files)
    print("label folders: ", label_folders)

    ordered_dirs_list = list()
    folder_labels_counter = 0
    for d in label_folders:
        ordered_dirs = list_and_order_directories(d)
        folder_labels_counter += len(ordered_dirs)
        ordered_dirs_list.append(ordered_dirs)

    file_path_tmp = get_single_filename(ordered_dirs_list[0][0])
    bucket_tmp = load_pickle(file_path_tmp)
    viewing_directions_indices_tmp = [key for key in bucket_tmp.keys() if str(key).isdigit()]
    num_samples_per_bucket = len(viewing_directions_indices_tmp)
    tot_num_samples = num_samples_per_bucket * folder_labels_counter
    print("tot num of buckets {} | tot num of samples per bucket {} | tot num of samples {}".format(folder_labels_counter, num_samples_per_bucket, tot_num_samples))
    
    # ############### start processing folders ###############

    if(args.num_bin_per_dim > 0):
        assert args.rows > 0
        assert args.cols > 0

    print("discretizing image with buckets {} of dimensions {} {}".format(args.num_bin_per_dim, args.rows, args.cols))
    
    dims = InputDims()
    print("processing data...")
    dataset = Data(dims, (ordered_dirs_list, folders), args.positive_percentage, num_bins_per_dim=args.num_bin_per_dim, rows=args.rows, cols=args.cols, include_poses=camera_poses_list_of_list, num_cpu=args.parallel)

    # strip extension from filename
    directory = os.path.dirname(args.output_data)
    filename, extension = os.path.splitext(os.path.basename(args.output_data))
    with open(os.path.join(directory, filename) +".pickle", 'wb') as datafile:
        pickle.dump(dataset.data, datafile)

    # with open(os.path.join(directory, filename)+"_normalizer.pickle", 'wb') as datafile:
    #     pickle.dump(dataset.normalizer, datafile)
    
    print("data deserialized in {}".format(directory))
    
   