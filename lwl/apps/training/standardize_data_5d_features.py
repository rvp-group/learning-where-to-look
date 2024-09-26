import torch
from torch.utils.data import random_split

import argparse
import numpy as np

from lwl.apps.training.data_loader import Data
from lwl.apps.utils.general_utils import *
from lwl.apps.utils.seed import *


def make_set(instance):
    valid_num_features = instance['valid_num_features']
    label = torch.tensor(instance['label'], dtype=torch.long).flatten()  # assuming binary classification
    elements = list()
    for idx in range(valid_num_features):
        element = np.zeros((5, ))
        element[0:2] += instance['pts_img'][idx]
        element[2:] += instance['pts_cam'][idx]
        elements.append(element)
    elements = torch.tensor(np.array(elements), dtype=torch.float32)
    return Data(x=elements, y=label, num_features=valid_num_features)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessor, calculate mean, std to standardize data during training and evaluation.' 
                                                 'Parameterize as 5d features (2d landmark reprojection and 3d point in camera coordinates) for training. Other information is discarded.')
    parser.add_argument('--data_path', type=str, required=True, help='path to your data file, this must be a unique pickle containing all data')
    parser.add_argument('--test_data_path', type=str, required=True, help='path to your independent test file, this must be a unique pickle containing all data')
    parser.add_argument('--val_split', type=float, default=0.05, help='percentage of validation data wrt to train data, i.e. 0.05')
    parser.add_argument('--output_data_path', type=str, help='output data name', default='data.pickle')    
    args = parser.parse_args()

    train_dataset = load_dataset(args.data_path)
    test_dataset = load_dataset(args.test_data_path)
          
    total_train_size = len(train_dataset)
    test_size = len(test_dataset)
    
    # splitting train and test data
    generator = torch.Generator().manual_seed(SEED)

    # getting the validation from train set and balance sample making train set even 
    # this is required for batch normalization
    validation_size = int(args.val_split * total_train_size + 0.5)
    train_size = total_train_size - validation_size
    if(train_size % 2 != 0):
        train_size += 1
        validation_size -= 1
    
    train_data, validation_data = random_split(train_dataset, [train_size, validation_size], generator=generator)
    assert train_size+validation_size == total_train_size
    
    # find mean and std
    count = 0
    summation = torch.zeros(5)
    train_graphs = list()
    for sample in train_data:
        features = make_set(sample)
        summation += features.x.sum(dim=0)
        count += features.num_features
        train_graphs.append(features)  
    # calculate the mean
    mean = summation / count
    
    sum_squares = torch.zeros(5)
    for sample in train_graphs:
        sum_squares += ((features.x - mean) ** 2).sum(dim=0)
    
    # calculate the standard deviation
    std_dev = torch.sqrt(sum_squares / count)

    print("calculating some stats on training data distribution")
    print("total num of features: {} mean: {} std: {}".format(count, mean, std_dev))

    def standardize_data(data, mean, std):
        stand_data = data
        stand_data.x = stand_data.x - mean
        stand_data.x = stand_data.x / std
        return stand_data
    
    standardized_data_dict = {"train" : list(), "val" : list(), "test" : list(), "mean" : mean, "std" : std_dev}

    for sample in train_graphs: # this was already preprocessed
        standardized_data_dict["train"].append(sample)
    print("completed training...")
    for sample in validation_data:
        sample = make_set(sample)
        standardized_data_dict["val"].append(sample)
    print("completed validation...")
    for sample in test_dataset.values():
        sample = make_set(sample)
        standardized_data_dict["test"].append(sample)
    print("completed test...")

    # dump pickle
    with open(args.output_data_path, 'wb') as pickle_file:
        pickle.dump(standardized_data_dict, pickle_file)