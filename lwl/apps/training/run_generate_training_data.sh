#!/bin/bash

# SCRIPT ONLY TO USE FOR BATCH PROCESSING, 
# IF YOU WANT TO PROCESS MULTIPLE MAPS AT THE SAME TIME
# AND INCLUDE EVERYTHING IN ONE SINGLE DATASET

# check if three arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <data_path> <output_data> <train|test>"
  exit 1
fi

DATA_PATH=$1 # path to folder with data, takes multiple map folders at a time
OUTPUT_DATA=$2
MODE=$3 # "train" or "test"

# set POSITIVE_PERCENTAGE based on mode
if [ "$MODE" == "train" ]; then
  POSITIVE_PERCENTAGE=0.5 # train, balance training data with 50% positive samples (and 50% negative samples)
elif [ "$MODE" == "test" ]; then
  POSITIVE_PERCENTAGE=0 # test, take all samples
else
  echo "Invalid mode. Please specify 'train' or 'test'."
  exit 1
fi

echo "Mode: $MODE, Positive percentage: $POSITIVE_PERCENTAGE"

# define other fixed parameters
ROWS=480 # img rows
COLS=640 # img cols
NUM_BIN_PER_DIM=30 # how many bins per img dimension, i.e. 30x30 grid (900 bins in total)

# run the Python preprocessing script with the provided and fixed parameters
python3 generate_training_data.py --data_path "$DATA_PATH" \
                                  --positive_percentage $POSITIVE_PERCENTAGE \
                                  --rows $ROWS \
                                  --cols $COLS \
                                  --num_bin_per_dim $NUM_BIN_PER_DIM \
                                  --output_data "$OUTPUT_DATA"

# notify user upon completion
if [ $? -eq 0 ]; then
  echo "Preprocessing completed successfully."
else
  echo "Error during preprocessing."
fi

# local example
# python3 preprocess.py --data_path /media/ldg/T71/MH3D_10_scene/ --positive_percentage 0.5 --rows 480 --cols 640 --num_bin_per_dim 30 --output_data /media/ldg/T71/matterport_data_50 >& /media/ldg/T71/matterport_data_50_out.txt;
# python3 preprocess.py --data_path /media/ldg/T71/MH3D_2_test/ --positive_percentage 0 --rows 480 --cols 640 --num_bin_per_dim 30 --output_data /media/ldg/T71/matterport_data_test