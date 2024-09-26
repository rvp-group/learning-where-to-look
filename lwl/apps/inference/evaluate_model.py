import torch
from torch.utils.data import DataLoader

import argparse

from lwl.apps.training.data_loader import DataMLPTrain
from lwl.apps.training.trainer import MLPTrainer, MODEL_NAME
from lwl.apps.training.standardize_data_5d_features import make_set

from lwl.apps.utils.general_utils import *
from lwl.apps.utils.seed import *
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on dataset based on Visual Localization Benchmark')
    parser.add_argument('--train_data_path', type=str, required=True, help='path to your train data file, this is important only to standardize the data for evaluation since contains mean and std')
    parser.add_argument('--evaluate_data_path', type=str, required=True, help='path to your raw data you want the active map')
    parser.add_argument('--model_dir', type=str, required=True, help='path to torch model, if already exists start from existing one')
    parser.add_argument('--max_features', type=int, help='if num bin used is num_bins*num_bins', default=900)
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
    pred_test = list(), list()               
    for test_batch in test_loader: 
        pred = model(test_batch['input'].to(device=device, non_blocking=PIN_MEMORY))
        pred_test.append(pred.squeeze())
        
    MLPTrainer.evaluate_benchmark(raw_test_dataset, pred_test)