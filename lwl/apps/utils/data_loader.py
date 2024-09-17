import torch
from torch.utils.data import Dataset
import sys, os
import random
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from lwl.apps.utils.general_utils import *
# from seed import *

MAX_T = 10 # max translation in meters
MAX_R = 20 # mar rotation in deg

SEED = 42
# set the seed for all random stuff
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# TODO need some refactoring here

class InputDims():
    def __init__(self):
        self.pose_dim = 3 # 3d pose
        self.orientation_dim = 4 # quaternion
        self.feature_dim = 2 # 2d feature
        self.land_dim = 3 # 3d landmark
        self.land_err_dim = 1 # colmap error
        self.weight = 1
        self.num_poses = 10

    def size(self, max_features, use_poses=True):
        # only features
        return self.feature_dim * max_features
        # dim = self.pose_dim + self.orientation_dim + max_features * self.feature_dim + max_features * self.land_dim + max_features * self.land_err_dim
        # # # TODO missing weight
        # if(use_poses):
        #     return dim + self.num_poses + self.num_poses  
        # dim = self.feature_dim * max_features + self.land_dim * max_features + self.num_poses + self.num_poses
        # return dim      


def precompute_binmap(num_bins_per_dim, rows, cols):
    # set num bins along each dimensions
    bin_size_cols = cols // num_bins_per_dim 
    bin_size_rows = rows // num_bins_per_dim 
    pixel_to_bin_img = np.zeros((rows, cols), dtype=np.int32)
    for r in range(0, rows):
        r_bin = np.floor(r / bin_size_rows) * num_bins_per_dim
        for c in range(0, cols): 
            bin_idx = np.floor(r_bin + c / bin_size_cols)
            pixel_to_bin_img[r, c] = bin_idx  

    # find max bin idx
    max_r_bin = np.floor(rows / bin_size_rows) * num_bins_per_dim
    max_bin_idx = int(np.floor(max_r_bin + cols / bin_size_cols))
    return pixel_to_bin_img, max_bin_idx

def make_poses_differences(pose, mapping_poses, use_N_poses=10):
    # this assumes to have mapping poses as Tcw format (colmap) and pose in Twc format, so we don't need to invert it internally
    # T_diff = np.einsum('ijk,ikl->ijl', pose, mapping_poses) # this if we have two batches
    T_diff = np.einsum('ijk,kl->ijl', mapping_poses, pose)
    R_diff = T_diff[:, :3, :3]
    traces = np.trace(R_diff, axis1=1, axis2=2)
    angles_diff = np.arccos((traces - 1) / 2) / np.pi # angle diff, normalize between 0 a 1 for net input
    t_diff = T_diff[:, :3, 3]
    # make some normalization in translation to be in the same range of rotation diff
    t_diff_normalized = np.linalg.norm(t_diff, axis=1)
    t_diff_normalized = t_diff_normalized/np.max(t_diff_normalized) 
    # sum errors for sorting
    sum_of_errors =  t_diff_normalized + angles_diff
    
    sort_indices = np.argsort(sum_of_errors)
    t_diff = t_diff[sort_indices]
    angles_diff = angles_diff[sort_indices]
    # print("t diff", t_diff)
    # print("r diff", angles_diff)
    
    assert t_diff.shape[0] == t_diff.shape[0]

    t_diff, angles_diff = t_diff[:use_N_poses], angles_diff[:use_N_poses]
    # permute stuff to avoid biasing learning with "best" pose at the beginning
    perm_indices = np.random.permutation(use_N_poses)
    angles_diff = angles_diff[perm_indices]
    t_diff = t_diff[perm_indices]

    return t_diff, angles_diff

def get_total_set(X, Y, PX):
    PY = 1 - PX
    sets = list(zip([X, Y], [PX, PY]))

    min_cardinality = min(len(X), len(Y))
    if (len(sets[0][0]) != min_cardinality) :
        temp = sets[0]
        sets[0] = sets[1]
        sets[1] = temp

    select_from_max = len(sets[0][0]) * sets[1][1] / sets[0][1]
    select_from_min = len(sets[1][0]) * sets[0][1] / sets[1][1]

    Z = {}
    if(select_from_max < len(sets[1][0])):
        Z = list(sets[0][0]) + list(sets[1][0])[0:int(select_from_max)]
        # print(select_from_max, " selected total min")
    else:
        Z = list(sets[1][0]) + list(sets[0][0])[0:int(select_from_min)]
        # print(select_from_min, " selected total max")
    return Z


# generic data processor
class Data(Dataset):
    def __init__(self,
                 input_dims,
                 dir_list,
                 num_positive_percentage,
                 max_features=900,
                 max_translation_error=0.05, 
                 max_rotation_error=0.4,
                 num_bins_per_dim=30, # these 3 args are required to have a normalized input
                 rows=480, # img dims
                 cols=640,
                 is_test=False,
                 include_poses=None):
        
        self.input_dims = input_dims 
        self.data, self.normalizer = dict(), dict()
        self.is_test = is_test
        # self.curr_samples = 0
        self.max_features = max_features # max number of features used for each instance, less we pad
        self.max_translation_error = max_translation_error # localization accuracy for true label
        self.max_rotation_error = max_rotation_error # localization accuracy for true label
        self.num_positive_samples = 0 # labels
        self.num_negative_samples = 0 # labels
        self.num_positive_percentage = num_positive_percentage 
        self.include_poses = include_poses

        # max magnitude required for normalization
        self.initVariables()
        assert self.input_dims.land_err_dim == 1
        self.is_img_binning = False
        
        # create map for image binning
        if(num_bins_per_dim > 0 and rows > 0 and cols > 0):
            self.is_img_binning = True
            self.pixel_to_bin_img, self.max_bin_idx = precompute_binmap(num_bins_per_dim, rows, cols)
            self.initializeBinnedFeatures()
            self.max_features = num_bins_per_dim * num_bins_per_dim
            print("overriding max num of features, binning/avg mode | max num: {}".format(self.max_features))
        
        self.instances_counter = 0
        
        folders, names = None, None
        if isinstance(dir_list, tuple) and len(dir_list) == 2:
            folders, names = dir_list
        else:
            folders = dir_list
        
        # for each dir
        for dataset_id, dirs in enumerate(folders):
            self.initVariables()
            print("="*30)
            data_name = dataset_id
            if names != None:
                data_name = names[dataset_id]
            data, curr_samples, curr_num_positive, curr_num_negative = self.processData(dirs, dataset_id, self.include_poses[dataset_id], data_name)
            # store normalization data to use after
            self.normalizer[dataset_id] = dict()
            self.normalizer[dataset_id]["max_pose"] = self.max_pose
            self.normalizer[dataset_id]["max_pt_img"] = self.max_pt_img
            if(rows > 0 and cols > 0):
                self.normalizer[dataset_id]["max_pt_img"] = np.array([rows, cols])
            self.normalizer[dataset_id]["max_pt_3d"] = self.max_pt_3d
            self.normalizer[dataset_id]["max_error_pt_3d"] = self.max_error_pt_3d

            print(self.max_pose, self.max_pt_3d, self.max_error_pt_3d)

            # make keys unique, copy to global dict
            assert curr_samples == len(data.keys())
            print("reindexing data...")
            for _, v in data.items():
                self.data[self.instances_counter] = v
                self.instances_counter += 1
            
            self.num_positive_samples += curr_num_positive
            self.num_negative_samples += curr_num_negative
            print("="*30)
            print("total positive labels {} | negative labels {} | total samples {}".format(self.num_positive_samples, self.num_negative_samples, self.instances_counter))


    def initVariables(self):
        self.max_pose = np.ones((self.input_dims.pose_dim)) * sys.float_info.min
        self.max_pt_img = np.ones((self.input_dims.feature_dim)) * sys.float_info.min
        self.max_pt_3d = np.ones((self.input_dims.land_dim)) * sys.float_info.min
        self.max_error_pt_3d = sys.float_info.min


    def initializeBinnedFeatures(self):
        self.binned_features = [{"features" :  list(), "pts3d" :  list(), "err" : list()} for _ in range(self.max_bin_idx)]

    def processData(self, dirs, dataset_id, mapping_poses=None, data_name=None):
        # TODO for simplicity get two random elements, TODO is permuting
        samples_counter = {0: 0, 1: 0}
        negative_samples_key, positive_samples_key = list(), list()
        instances_counter = 0 

        data = dict()
        
        for d in tqdm(dirs, desc="pre-processing buckets"):    
            file_path = get_single_filename(d)
            if file_path is None:
                continue
            bucket = load_pickle(file_path)
            # pose = bucket['pose'] TODO test
            viewing_directions_indices = [key for key in bucket.keys() if str(key).isdigit()]
            bucket_id = bucket['bucket_idx']
            # print(len(viewing_directions_indices))
            for idx_viewpoint in viewing_directions_indices:
                pose = bucket[idx_viewpoint]['pose']
                quat = bucket[idx_viewpoint]["quat"]
                view_data = bucket[idx_viewpoint]["view_data"]
                view_data_size = len(view_data)
                # initialize some np array for the reprojection
                pts_img = np.zeros((view_data_size, self.input_dims.feature_dim), dtype=np.float32)
                pts_3d = np.zeros((view_data_size, self.input_dims.land_dim), dtype=np.float32)
                # rgb = np.zeros((view_data_size, 3), dtype=np.float32)
                errors_pt_3d = np.zeros((view_data_size), dtype=np.float32)
                weights = None # just for binning
                # init label
                label = None
                if(view_data_size > 0): # only if there is some viewing data
                    for i, idx_feature in enumerate(view_data): # i the incremental index, idx_feature the id associated to the colmap model
                        pts_img[i, :] = view_data[idx_feature]["pt_img"]
                        pts_3d[i, :] = view_data[idx_feature]["pt_3d"]
                        errors_pt_3d[i] = view_data[idx_feature]["error_pt_3d"]
                        # rgb[i, :] = view_data[idx_feature]["rgb"] # TODO RGB not used 
                    
                    # some magnitude maximum required for input net normalization later
                    self.max_pt_img = np.maximum(self.max_pt_img, np.max(np.abs(pts_img), axis=0))
                    self.max_pt_3d = np.maximum(self.max_pt_3d, np.max(np.abs(pts_3d), axis=0))
                    self.max_error_pt_3d = np.maximum(self.max_error_pt_3d, np.max(np.abs(errors_pt_3d)))

                    # process labels
                    # all conditions according to https://www.visuallocalization.net/benchmark/ : (0.25m, 2°) / (0.5m, 5°) / (5m, 10°)
                    # since is indoor environment and good mapping we scale to (0.05m, 0.4°)
                    if (bucket[idx_viewpoint]["err_t"] < self.max_translation_error and bucket[idx_viewpoint]["err_r"] < self.max_rotation_error): 
                        positive_samples_key.append(instances_counter)
                        label = 1
                    else:
                        negative_samples_key.append(instances_counter)
                        label = 0

                    
                    # if no discretization chop to max number of features
                    if(self.is_img_binning == False): 
                        # shuffle viewing data, features maybe ordered somehow
                        perm = np.random.permutation(view_data_size)
                        pts_img = pts_img[perm] 
                        pts_3d = pts_3d[perm] 
                        errors_pt_3d = errors_pt_3d[perm] 
                        # rgb = rgb[perm]

                        # keep only a fixed number
                        # if smaller than max features not a problem we will padd later
                        pts_img = pts_img[:self.max_features] 
                        pts_3d = pts_3d[:self.max_features] 
                        errors_pt_3d = errors_pt_3d[:self.max_features]
                        # rgb = rgb[:self.max_features]

                    else:
                        # import time
                        # start_time = time.time()
                        # import copy as copy
                        # binned_features = copy.copy(self.binned_features)
                        # print(binned_features)
                        for (f, pt3d, e) in zip(pts_img, pts_3d, errors_pt_3d):
                            bin_idx = self.pixel_to_bin_img[int(f[0]), int(f[1])]
                            self.binned_features[bin_idx]["features"].append(f.flatten())
                            self.binned_features[bin_idx]["pts3d"].append(pt3d.flatten())
                            self.binned_features[bin_idx]["err"].append(e.flatten())

                        # calculate means
                        binned_features_means, binned_pts_mean, binned_err_mean, weights = list(), list(), list(), list()
                        for elements in self.binned_features:
                            features_per_bin = np.asarray(elements["features"])
                            num_features_per_bin = features_per_bin.shape[0] 
                            if(num_features_per_bin > 0):
                                binned_features_means.append(np.mean(features_per_bin, axis=0))
                                binned_pts_mean.append(np.mean(np.asarray(elements["pts3d"]), axis=0))
                                binned_err_mean.append(np.mean(np.asarray(elements["err"]), axis=0))
                                weights.append(fsigmoid(num_features_per_bin))
                                # print("mean {} {} {} | weight {} | num features {}".format(features_mean, pts3d_mean, err_mean, weight, num_features_per_bin))
                                # print(10*"=")
                        pts_img = np.asarray(binned_features_means)
                        pts_3d = np.asarray(binned_pts_mean)
                        errors_pt_3d = np.asarray(binned_err_mean)
                        weights = np.asarray(weights)
                        view_data_size = pts_img.shape[0]
                                        
                        # import matplotlib.pyplot as plt
                        # plt.imshow(self.pixel_to_bin_img)
                        # plt.scatter(pts_img[:, 1], pts_img[:, 0], marker='o', s=6, c='b', alpha=0.4)
                        # plt.scatter(binned_features_means[:, 1], binned_features_means[:, 0], marker='o', s=10, c='r', alpha=0.2)
                        # plt.title('binned features')
                        # # plt.colorbar()  
                        # plt.show()
                        
                        self.initializeBinnedFeatures()
                        # end_time = time.time()
                        # print(f"Elapsed Time: {end_time-start_time} seconds")
                        
                else: # if no features are predicted we still want the network to predict zero
                    pts_img = np.zeros((self.max_features, self.input_dims.feature_dim), dtype=np.float32)
                    pts_3d = np.zeros((self.max_features, self.input_dims.land_dim), dtype=np.float32)
                    # rgb = np.zeros((self.max_features, 3), dtype=np.float32)
                    errors_pt_3d = np.zeros((self.max_features), dtype=np.float32)
                    view_data_size = self.max_features # this is required to make slicing working later for empty array
                    negative_samples_key.append(instances_counter)
                    label = 0
                    
                # insert in nested dictionary, here if features are less than max_features may be less in container
                data[instances_counter] = dict()
                data[instances_counter]["bucket_id"] = bucket_id
                data[instances_counter]["map"] = dataset_id 
                if(data_name != None):
                    data[instances_counter]["map"] = (dataset_id, data_name) 
                data[instances_counter]["label"] = label # (0, 1) binary
                samples_counter[label] += 1
                data[instances_counter]["pose"] = pose # (3 x 1) translation
                data[instances_counter]["quat"] = quat # (4 x 1) quaternion 
                data[instances_counter]["valid_num_features"] = np.min([self.max_features, view_data_size]) # clipping to max number of features
                data[instances_counter]["pts_img"] = pts_img # (2 x max_features) img points
                data[instances_counter]["pts_3d"] = pts_3d # (3 x max_features) world points
                data[instances_counter]["errors_pt_3d"] = errors_pt_3d.flatten() # (max_features) errors
                data[instances_counter]["err_t"] = bucket[idx_viewpoint]["err_t"]
                data[instances_counter]["err_r"] = bucket[idx_viewpoint]["err_r"]
                if(self.is_img_binning == True):
                    data[instances_counter]["weights"] = weights
                if(mapping_poses != None):
                    Twc = np.eye(4) 
                    Twc[0:3, 0:3] = Rotation.from_quat(quat).as_matrix()    
                    Twc[0:3, 3] = pose
                    data[instances_counter]["t_diff"], data[instances_counter]["angles_diff"] = make_poses_differences(Twc, mapping_poses)
                    
                # for normalization
                # TODO for each or not?
                self.max_pose = np.maximum(self.max_pose, np.abs(pose))

                # update counter for global instace, each direction is a sample
                instances_counter += 1
                
                # break
            
        print("\tcurrent positive labels {} | negative labels {}".format(samples_counter[1], samples_counter[0])) 

        # keep an eye and balance amount of positive and negative labels
        if(self.is_test == False):

            random.Random(SEED).shuffle(positive_samples_key)
            random.Random(SEED).shuffle(negative_samples_key)

            if(self.num_positive_percentage > 0):
                print("\tresampling based on new percentage positive {} and negative {}".format(self.num_positive_percentage, 1-self.num_positive_percentage))
                indices_to_keep = get_total_set(set(positive_samples_key), set(negative_samples_key), self.num_positive_percentage)
                data = {index: data[index] for index in indices_to_keep}
                instances_counter = len(indices_to_keep)

            num_negative_samples = sum(1 for key in data if data[key].get('label') == 0)
            num_positive_samples = sum(1 for key in data if data[key].get('label') == 1)
                        
            # select items to be removed      
            print("\tshuffled and trimmed | positive labels {} | negative labels {}".format(num_positive_samples, num_negative_samples))

            # data = dict(enumerate(v for _, v in sorted(data.items())))
            # print("\treindexing data")

        return data, instances_counter, num_positive_samples, num_negative_samples