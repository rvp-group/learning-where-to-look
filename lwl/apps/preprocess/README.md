# Preprocess data

>[!IMPORTANT]
This README is not completed yet

We expect an SfM model processed using COLMAP. Given a set of images (and eventually a set of known poses) you can generate your SfM model following this guide **TODO**. Once computed your folder with the model should have the following files/folders:

```
├── database.db --> database
├── sparse --> SfM model
├── images --> set of images
├── img_name_to_colmap_Tcw.txt
└── img_nm_to_colmap_cam.txt
```

In order to construct a voxel grid over the 3d sparse map you can use `active_viewpoints_computation_reprojection.py` that is based on some CUDA code to speed up visibility checks. Usually the SfM map is noisy, in order to reconstruct a precise voxel grid over the map we assume to have a mesh. The data used in our experiments belongs to Matterport (**TODO**), but you should be able to load any kind of mesh with `Open3D`. Once you have the mesh and your SfM COLMAP folder you can run:

```bash
python3 active_viewpoints_computation_reprojection.py --config_path <path-to-conf-file.cfg> \
                                                      --landmarks <colmap_model_folder/sparse/0/points3D.txt> \
                                                      --mesh_path <path-to-mesh> \ 
                                                      --output_file <path-to-output-viewpoints>
```

>[!IMPORTANT]
 >configurations are inside the `configurations` folder, `num-max-directions` indicates how many "best" viewpoints need to be computed for each camera location in the voxel grid. The output is descending order for each location (from the one with more visible landmarks to the one with less - might be zero). 


Now convert the grid from `.txt` file to `.pickle` for faster processing (dict representation)
```bash
python3 grid_converter.py 
```


