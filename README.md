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
    <a href="https://arxiv.org/abs/2407.15593">preprint</a>
</div>


# Dependencies
The following external dependencies are required
| Dependency                                                        | Version(s) known to work |
| ----------------------------------------------------------------- | ------------------------ |
| [CUDA](https://developer.nvidia.com/cuda-12-1-0-download-archive) | <12.1                    |

>[!IMPORTANT]
 >CUDA is used both during training by `torch` and to efficiently process viewpoints visibility. 

# Install (local) via `pip`

First download this repo and `cd learning-where-to-look`. Once inside the folder, you can build/install `learning-where-to-look` using `pip`
```bash
pip install .
```

# Training

## Download some data

```bash
./download.sh 
```

The script will download some data that you can use to play with:
```
├── train_MH3D_10_scene --> contains all the preprocessed meshes used for training, SfM models and precomputed viewpoints
├── test_MH3D_2_scene --> contains all the preprocessed meshes used for testing, SfM models and precomputed viewpoints
├── train_data_10_meshes_with_preprocessed_test.pickle --> contains data that can be
|                                                          loaded directly for learning, with essential (observed landmarks 
|                                                          reprojections in image, landmarks in camera frame) train, 
|                                                          validation, test, mean and std (of training set)
├── test_raw_data_2_meshes.pickle --> contains all preprocessed data used 
|                                     for testing with other information (i.e., locations used for evaluation)
└── raw_MH3D_00017.pickle --> preprocessed test mesh you can use to visualize results
```

## Run

Run training with the following script; the default is 300 epochs

```bash
python3 lwl/apps/training/mlp_train.py --data_path data/train_data_10_meshes_with_preprocessed_test.pickle --test_data_path data/test_raw_data_2_meshes.pickle --checkpoint_path data/mymodels/tmp_training
```

## Inference
Evaluate trained model numerically

```bash
python3 lwl/apps/inference/evaluate_model.py --train_data_path data/train_data_10_meshes_with_preprocessed_test.pickle --evaluate_data_path data/raw_MH3D_00017.pickle --model_dir data/model/
```

Evaluate trained model visually, showing best predicted viewpoints for each location and their observed landmarks
```bash
python3 lwl/apps/inference/compute_active_map.py --train_data_path data/train_data_10_meshes_with_preprocessed_test.pickle --evaluate_data_path data/raw_MH3D_00017.pickle --model_dir data/model/ --enable_viz --config_path configurations/sampler_matterport_1.cfg --landmarks data/test_MH3D_2_scene/MH3D_00017/sparse/0/points3D.txt
```

<div align="center">
        <a href="https://github.com/rvp-group/learning-where-to-look"><img src="assets/active_map.gif"/></a>   
</div>

# Cite us
If you use any of this code, please cite our <a href="https://arxiv.org/abs/2407.15593">paper</a> - accepted ECCV 2024:

```
@article{di2024learning,
  title={Learning Where to Look: Self-supervised Viewpoint Selection for Active Localization using Geometrical Information},
  author={Di Giammarino, Luca and Sun, Boyang and Grisetti, Giorgio and Pollefeys, Marc and Blum, Hermann and Barath, Daniel},
  journal={arXiv preprint arXiv:2407.15593},
  year={2024}
}
```

# What's Missing
The repo is currently under updates; you can keep track here

| Feature/Component         | Status        |
| ------------------------- | ------------- |
| CUDA/C++ compilation      | ✅ Completed   |
| Unit tests                | ✅ Completed   |
| Pybidings                 | ✅ Completed   |
| Training                  | ✅ Completed   |
| Documentation             | ⚠️ In Progress |
| Preprocessing             | ⚠️ In Progress |
| Custom data setup         | ⚠️ In Progress |
| Inference/plot active map | ✅ In Progress |
