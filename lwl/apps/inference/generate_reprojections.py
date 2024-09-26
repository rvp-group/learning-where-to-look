import torch
from torch.utils.data import DataLoader

import argparse, sys

from lwl.apps.training.data_loader import DataMLPTrain
from lwl.apps.training.trainer import MLPTrainer, MODEL_NAME
from lwl.apps.training.standardize_data_5d_features import make_set

from lwl.apps.utils.general_utils import *
from lwl.apps.utils.seed import *
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate reprojections from a given model based on MLP probabilities')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to your train data file, important for standardizing the data (mean and std)')
    parser.add_argument('--evaluate_data_path', type=str, required=True, help='Path to your raw data you want to use to compute the active map')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to torch model, start from existing model if available')
    parser.add_argument('--max_features', type=int, help='Number of features to use, if num bin used is num_bins*num_bins', default=900)
    parser.add_argument('--output_data_path', type=str, required=True, help='Path to save the output data')
    parser.add_argument('--config_path', type=str, required=True, help='Path to YAML configuration file, if not provided, cameras will be plotted as directions, not frustum')
    args = parser.parse_args()

    # torch.multiprocessing.set_start_method('spawn') # multiprocessing option

    # check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    dataset = load_dataset(args.train_data_path)
    normalizer = (dataset["mean"], dataset["std"])
    print("loaded data from {} | using this just to standardize dataset to evaluate with | mean {} std {}".format(args.train_data_path, normalizer[0], normalizer[1]))
       
    raw_test_dataset = load_dataset(args.evaluate_data_path)
    print("loaded raw evaluation data from {}".format(args.evaluate_data_path))
    test_dataset = dict()      
    test_dataset = [make_set(sample) for sample in raw_test_dataset.values()]
    print("completed loaded and preprocessing")
    test_data = DataMLPTrain(test_dataset, device, max_features=args.max_features, normalizer=normalizer, is_training=False) 
    
    NUM_WORKERS = 1
    PIN_MEMORY = False
    BATCH_SIZE = 128 # only to speed up inference
    
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # initialize the MLP model
    print("loading model from {}".format(args.model_dir))

    _, model, _ = MLPTrainer.start_from_existing_model(args.model_dir, MODEL_NAME)
    print("model ", model)

    model.eval() 
    pred_test = list()               
    for test_batch in test_loader: 
        pred = model(test_batch['input'].to(device=device, non_blocking=PIN_MEMORY))
        pred_test.append(pred.squeeze())
        
    # MLPTrainer.evaluate_benchmark(raw_test_dataset, pred_test)

    predictions_list_of_list = [t.tolist() for t in pred_test]
    predictions = [item for sublist in predictions_list_of_list for item in sublist]
    best_viewpoints_per_bucket = dict()
    # order all_preds per bucket
    for idx, pred in enumerate(predictions):
        pose_idx = raw_test_dataset[idx]['bucket_id']
        if (pose_idx not in best_viewpoints_per_bucket.keys()):
            best_viewpoints_per_bucket[pose_idx] = list()
        pose = raw_test_dataset[idx]['pose']
        quat = raw_test_dataset[idx]['quat']
        pts_img = raw_test_dataset[idx]['pts_img']
        best_viewpoints_per_bucket[pose_idx].append((idx, pred, pose, quat, pts_img))


    # parse configuration
    import yaml
    infile = open(args.config_path, 'r')
    configuration_file = yaml.safe_load(infile)
    
    # getting some parameters
    f_length           = configuration_file["focal-length"]
    rows               = configuration_file["rows"]
    cols               = configuration_file["cols"]
    max_mapping_error  = configuration_file["colmap-error"]

    import matplotlib.pyplot as plt
    best_direction_dict = {key: sorted(value, key=lambda x: x[1], reverse=True) for key, value in best_viewpoints_per_bucket.items()}
   
    main_folder = f"{args.output_data_path}"
    os.makedirs(main_folder, exist_ok=True)
    scale_factor = 0.01  # adjust this for different figure sizes

    for pose_idx, value in best_direction_dict.items():
        subfolder = os.path.join(main_folder, f"pose_{pose_idx}")
        os.makedirs(subfolder, exist_ok=True)
        for idx, pred, pose, quat, pts_img in value:
            fig, ax = plt.subplots(figsize=(cols * scale_factor, rows * scale_factor))

            # set axis limits to match the number of rows and columns
            ax.set_xlim(0, cols)  
            ax.set_ylim(0, rows)  

            # move x-axis to the top
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()

            # set background color to white
            ax.set_facecolor('white')

            # create scatter plot of the features
            ax.scatter(pts_img[:, 0], pts_img[:, 1], c='red', marker='o', label='reprojected landmarks')

            # invert y-axis to match image coordinates
            ax.invert_yaxis()

            # save the image to the subfolder, named based on prob value and image number
            pred_str = f"{pred:.3f}"
            image_filename = str(idx)+"_"+pred_str+".png"
            image_path = os.path.join(subfolder, image_filename)
            plt.savefig(image_path)

            # close the plot to free memory
            plt.close()

        print(f"saved images in {subfolder}")
