# this are the only required input
# MATTERPORT_PATH=/media/ldg/T71/MH3D_10_scene/
MATTERPORT_PATH=/media/ldg/T71/MH3D_2_test/

MESHES_DATABASE_PATH=/media/ldg/T71/hm3d-train-glb-v0.2
CONFIGURATION_PATH=../../../configurations
CONFIGURATION_FILENAMES=("sampler_matterport_1.cfg" "sampler_matterport_10.cfg" "sampler_matterport_50.cfg" "sampler_matterport_100.cfg")

THIS_PATH=$(pwd)
CONVERT_PATH=${HOME}/source/eth/voxgraph/scripts/learning/

# list all dirs in specific path
# directories=($(find "$MATTERPORT_PATH" -maxdepth 1 -type d))
directories=($(find "$MATTERPORT_PATH" -mindepth 1 -maxdepth 1 -type d))

# Iterate through the array of directory names
echo "processing matterport folders:"
for dir in "${directories[@]}"; do
    MATTERPORT_PATH=$dir
    echo processing folder: $MATTERPORT_PATH
    
    SFM_FOLDER=$(basename "$dir")
    matterport_mesh_id=$(echo $SFM_FOLDER | cut -d'_' -f2 | awk '{gsub(/[^0-9]/,""); print}')

    # retrieve mesh based on id in folder database
    mesh_path=$MESHES_DATABASE_PATH/$matterport_mesh_id-*/

    unset mesh_name
    file_count=$(find $mesh_path -maxdepth 1 -type f | wc -l)
    if [ "$file_count" -eq 1 ]; then
        # get the filename
        mesh_name=$(basename "$(find $mesh_path -maxdepth 1 -type f)")
        echo found mesh file: $mesh_name in $mesh_path
        mesh_name=$mesh_path/$mesh_name

    else
        echo the folder $mesh_path does not contain exactly one file
        exit 0
    fi

    # create output folder
    dir_folder_path=$MATTERPORT_PATH/sampling_directions
    mkdir $dir_folder_path

    for configuration_name in "${CONFIGURATION_FILENAMES[@]}"; do
        echo "processing current configuration: $configuration_name"

        # build output file filename 
        # configuration_name=$(basename "$CONFIGURATION_PATH")
        num_directions="${configuration_name#*_}"
        num_directions="${num_directions//[^0-9]/}"

        output_filename=$dir_folder_path/sampling_directions_$num_directions.dat
        echo writing output to $output_filename
        
        configuration_fullpath=$CONFIGURATION_PATH/$configuration_name
        python3 active_viewpoints_computation_reprojection.py --config_path $configuration_fullpath --landmarks $MATTERPORT_PATH/sparse/0/points3D.txt --output_file $output_filename --mesh_path $mesh_name
        
        # change path to run other script
        cd $CONVERT_PATH
        python3 ${HOME}/source/eth/voxgraph/scripts/learning/txt2bin_grid.py --grid_file $output_filename
        echo converted output to bin
        cd $THIS_PATH
    done
    # exit 0
done
echo processed successfully!
