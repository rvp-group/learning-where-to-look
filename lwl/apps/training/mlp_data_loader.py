import torch
from torch.utils.data import Dataset
import numpy as np

from lwl.apps.utils.general_utils import *
from lwl.apps.utils.seed import *
    
class Data():
    def __init__(self, x, y, num_features):
        self.x = x
        self.y = y
        self.num_features = num_features

# for some data augmentation during training
IMG_NOISE_STD = 1
PT_NOISE_STD = 0.02

class DataMLPTrain(Dataset):
    def __init__(self, 
                 data,
                 device,
                 rows=480, 
                 cols=640,
                 scale_ratio=3,
                 max_features=900, 
                 normalizer = (1, 1),
                 is_training=True):
        self.data = data
        self.size = len(data)
        self.is_training = is_training
        self.max_features = max_features
        self.device = device
        self.rows = rows
        self.cols = cols
        self.target_img_size = (rows // scale_ratio, cols // scale_ratio)
        # normalizer 
        self.mean_normalizer = normalizer[0]
        self.std_normalizer = normalizer[1]
        # 2 classes and 5d features
        self.num_classes = 2
        self.feature_dim = 5
        # self.feature_dim = 3
        self.num_entities = self.feature_dim * self.max_features

    def __len__(self):
        return self.size

    # # get item 2d
    # def __getitem__(self, idx):
    #     # get the correct normalizer for each map is different
    #     instance = self.data[idx]
    #     x = torch.zeros((self.max_features, self.feature_dim))
    #     x[:instance.num_features] = instance.x[:, :2]
    #     if(self.is_training):
    #         # torch size (Nxfeature_dim)
    #         tensor_dim = x[:instance.num_features].shape
    #         img_noise = torch.normal(mean=0, std=IMG_NOISE_STD, size=tensor_dim)
    #         x[:instance.num_features] += img_noise
        
    #     # normalize data        
    #     x[:instance.num_features] -= self.mean_normalizer[:2]
    #     x[:instance.num_features] /= self.std_normalizer[:2]
        
    #     # x[:instance.num_features] = instance.x
    #     return {"input" : x.flatten(), "label" : instance.y}

    # get item 5d
    def __getitem__(self, idx):
        # get the correct normalizer for each map is different
        instance = self.data[idx]
        x = torch.zeros((self.max_features, self.feature_dim))
        # print(instance.x)
        # exit(0)
        x[:instance.num_features] = instance.x
        # x[:instance.num_features, :2] = instance.x[:, :2] # pt 2d
        # x[:instance.num_features, 2] = instance.x[:, 4] # z 3d
        if(self.is_training):
            # torch size (Nxfeature_dim)
            img_noise = torch.normal(mean=0, std=IMG_NOISE_STD, size=(instance.num_features, 2))
            pt_noise = torch.normal(mean=0, std=PT_NOISE_STD, size=(instance.num_features, 3))
            # pt_noise = torch.normal(mean=0, std=PT_NOISE_STD, size=(instance.num_features, 1))
            x[:instance.num_features] += torch.cat([img_noise, pt_noise], dim=1)
        
        # normalize data        
        x[:instance.num_features] -= self.mean_normalizer
        x[:instance.num_features] /= self.std_normalizer
        # x[:instance.num_features, :2] -= self.mean_normalizer[:2]
        # x[:instance.num_features, 2] -= self.mean_normalizer[4]

        # x[:instance.num_features, :2] /= self.std_normalizer[:2]
        # x[:instance.num_features, 2] /= self.std_normalizer[4]
        
        # x[:instance.num_features] = instance.x
        return {"input" : x.flatten(), "label" : instance.y}