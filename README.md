<div align="center">
    <h1>Learning Where to Look: Self-supervised Viewpoint Selection for Active Localization using Geometrical Information</h1>
    <h3>ECCV 2024</h3>
    <div align="center">
        <a href="https://github.com/rvp-group/learning-where-to-look"><img src="assets/pipeline.png"/></a>   
    </div>
    Given a Structure-from-Motion model, we aim to learn the camera viewpoint that can be employed to maximize the accuracy in visual localization. 
    Our methodology requires first sampling the camera locations and orientation, calculating the best visibility orientation for each location,
    and learning active viewpoint through a Multi-layer Perceptron encoder. The illustration above shows our full pipeline predicting active viewpoints for visual 
    localization embedded into a planning framework.
    <br />   
    <br />   
    We will soon release the code; we need time to clean it and make it understandable! For now you can check our <a href="https://arxiv.org/abs/2407.15593">preprint</a>
  - accepted ECCV 2024.
</div>


# Dependencies
The following external dependencies are required
| Dependency                                                        | Version(s) known to work |
| ----------------------------------------------------------------- | ------------------------ |
| [CUDA](https://developer.nvidia.com/cuda-12-1-0-download-archive) | >12.1                    |

>[!IMPORTANT]
 >CUDA is used both during training by `torch` and to efficiently process viewpoints vibility. 


# Download some data (training and test set)
train: ```wget ftp://anonymous:@151.100.59.119/learning_where_to_look/train_data_10_meshes.pickle```

test: ```wget ftp://anonymous:@151.100.59.119/learning_where_to_look/test_data_2_meshes.pickle```


# Install (local) using `pip`

First download this repo and `cd learning-where-to-look`. Once inside the folder, you can install `learning-where-to-look` using `pip`
```bash
pip install .
```

# Training
python3 lwl/apps/training/mlp_train.py --data_path /media/ldg/T71/train_data_3_5_24/mlp_100p.pickle --test_data_path /media/ldg/T71/test_MH3D_2/matterport_data_test_2.pickle --checkpoint_path models/tmp



