import torch
from torch.utils.data import DataLoader

import argparse
from datetime import date

from lwl.apps.training.data_loader import DataMLPTrain
from lwl.apps.training.trainer import MLPTrainer
from lwl.apps.training.mlp import MLPClassifier

from lwl.apps.utils.general_utils import *
from lwl.apps.utils.seed import *
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train MLP - active viewpoint classifier - this script requires also the not processed test set to perform evaluation once training is completed')
    parser.add_argument('--data_path', type=str, required=True, help='path to your data file, this must be a unique pickle containing all data')
    parser.add_argument('--test_data_path', type=str, required=True, help='path to your data file, this must be a unique pickle containing all data')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs required for training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--max_features', type=int, help='if num bin used is num_bins*num_bins', default=900)
    parser.add_argument('--checkpoint_path', type=str, help='path to torch model, if already exists start from existing one')
    args = parser.parse_args()

    # torch.multiprocessing.set_start_method('spawn') # multiprocessing option

    # check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    dataset = load_dataset(args.data_path)
    
    train_dataset = dataset["train"]
    validation_dataset = dataset["val"]
    test_dataset = dataset["test"]
    normalizer = (dataset["mean"], dataset["std"])
       
    # splitting train and test data
    generator = torch.Generator().manual_seed(SEED)

    NUM_WORKERS = 1
    PIN_MEMORY = False
    print("num processors involved", NUM_WORKERS)

    print("using n samples training {}, validation {}, testing {}".format(len(train_dataset), len(validation_dataset), len(test_dataset)))    

    # train_data = DataMLPTrain(dims, train_dataset, device, max_features=args.max_features, normalizer=normalizer, is_training=True) 
    # validation_data = DataMLPTrain(dims, validation_dataset, device, max_features=args.max_features, normalizer=normalizer, is_training=False) 
    # test_data = DataMLPTrain(dims, test_dataset, device, max_features=args.max_features, normalizer=normalizer, is_training=False) 
    
    train_data = DataMLPTrain(train_dataset, device, max_features=args.max_features, normalizer=normalizer, is_training=True) 
    validation_data = DataMLPTrain(validation_dataset, device, max_features=args.max_features, normalizer=normalizer, is_training=False) 
    test_data = DataMLPTrain(test_dataset, device, max_features=args.max_features, normalizer=normalizer, is_training=False) 
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # initialize the MLP model
    model = MLPClassifier(input_size=train_data.num_entities, output_size=1)
    model = model.to(device)
        
    print("model\n", model)
    
    # create model to save weights
    checkpoint_path = args.checkpoint_path 
    if(checkpoint_path == None):
        checkpoint_path = "model_" + date.today().strftime("%d_%b_%Y_%H%M%S")
        make_dir(checkpoint_path)
    else:
        make_dir(checkpoint_path)

    # train
    tr = MLPTrainer(model=model, epochs=args.epochs, device=device)
    original_test_dataset = load_dataset(args.test_data_path)
    tr.train((train_loader, validation_loader, test_loader), checkpoint_path, test_data=original_test_dataset, pin_memory=PIN_MEMORY, batch_size=args.batch_size)