#!/bin/bash

# some global variables
CONFIGURATION_PATH=../../../configurations
CONFIGURATION_FILENAMES=("sampler_matterport_1.cfg" "sampler_matterport_10.cfg" "sampler_matterport_50.cfg" "sampler_matterport_100.cfg")
L_FILENAME_PREFIX=points3D # landmark file prefix, this reflect colmap stuff
EXT_MESH="glb" # mesh extension to look for in the folder


# check if a directory is provided as an argument
if [ $# -ne 1 ]; then
    echo "usage: $0 <folder-with-sfm-and-meshes>"
    exit 1
fi

# get the folder from the argument
DATA_PATH=$1

# verify if the argument is a valid directory
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: $DATA_PATH is not a valid directory."
    exit 1
fi

# list all dirs in specific path
directories=($(find "$DATA_PATH" -mindepth 1 -maxdepth 1 -type d))

# Iterate through the array of directory names
echo "processing matterport folders:"
for dir in "${directories[@]}"; do
    DATA_PATH=$dir
    echo processing folder: $DATA_PATH
    
    sfm_folder=$(basename "$dir")
    matterport_mesh_id=$(echo $sfm_folder | cut -d'_' -f2 | awk '{gsub(/[^0-9]/,""); print}')

    # find "$DATA_PATH" -type f -name "$L_FILENAME_PREFIX*" -print
    landmark_path=$(find "$DATA_PATH" -type f -name "$L_FILENAME_PREFIX*")
    echo "->  landmarks found: "$landmark_path

    mesh_path=$(find "$DATA_PATH" -type f -name "*$EXT_MESH")
    echo "->  mesh found: "$mesh_path

    # check if mesh match with filepath names
    mesh_path_no_ext=${mesh_path%.*}
    if [[ "$mesh_path_no_ext" != *"$sfm_folder"* ]]; then
        echo "filename '$sfm_folder' is not part of the path string '$mesh_path_no_ext' | cannot process the current SfM model!"
        continue
    fi

    # create output folder
    dir_folder_path=$DATA_PATH/sampling_directions
    if [ ! -d "$dir_folder_path" ]; then
        mkdir $dir_folder_path
    fi

    for configuration_name in "${CONFIGURATION_FILENAMES[@]}"; do
        echo "--->  processing current configuration: $configuration_name"

        # build output file filename 
        num_directions="${configuration_name#*_}"
        num_directions="${num_directions//[^0-9]/}"

        output_filename=$dir_folder_path/sampling_directions_$num_directions.dat
        echo "--->  writing output to $output_filename"
        
        configuration_fullpath=$CONFIGURATION_PATH/$configuration_name
        python3 active_viewpoints_computation_reprojection.py --config_path $configuration_fullpath --landmarks $landmark_path --output_file $output_filename --mesh_path $mesh_path > /dev/null 2>&1
        python3 grid_converter.py --grid_file $output_filename > /dev/null 2>&1
        echo "--->  converted output to bin"
    done
done
echo processed successfully!
