import numpy as np
import torch

import torch.nn as nn
from torcheval.metrics.functional import binary_f1_score, binary_precision, binary_recall

from tqdm import tqdm

from lwl.apps.utils.general_utils import *
from lwl.apps.utils.calculate_grid_accuracy import calculate_accuracy, get_accuracies_of_n_elements

SAVE_INTERVAL = 5 # save interval for checkpoint
MODEL_NAME = "model_architecture.torch"

import wandb
# this is just for tuning/logger configuration
WANDB_CONFIG={
    "learning_rate": 1e-4,
    "architecture": "MLP",
    "train-dataset": "mlp_100p",
    "epochs": 300,
    }

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class MLPTrainer():

    def __init__(self, model, epochs, device=None):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
        self.loss_fn = nn.BCELoss() # binary cross entropy loss        
        self.epochs = epochs
        self.save_model = True
        self.writer = None

    def compute_and_write_stats(self, epoch, pred_list, y_list, name="train"):
        # list of tensors to tensor
        preds = torch.cat(pred_list, dim=0)
        y = torch.cat(y_list, dim=0)
        rounded_preds = preds.round()

        size = len(y)

        # calculate accuracy
        acc = int((rounded_preds == y).sum())/size
        self.writer.log({'accuracy/'+name: acc}, step=epoch)
        
        # rest of metric
        f1_score = binary_f1_score(rounded_preds, y, threshold=1)
        rec = binary_recall(rounded_preds, y, threshold=1)
        prec = binary_precision(rounded_preds, y, threshold=1)
        
        # other metrics
        self.writer.log({'f1_score/'+name: f1_score}, step=epoch)
        self.writer.log({'recall/'+name: rec}, step=epoch)
        self.writer.log({'precision/'+name: prec}, step=epoch)

        return acc


    def predict(self, batch, pin_memory=True):
        x = batch['input'].to(device=self.device, non_blocking=pin_memory)
        predictions = self.model(x)
        y = batch['label'].to(device=self.device, non_blocking=pin_memory)
        return predictions, y

    def train(self, data_loader, checkpoint_path, num_tests=50, test_data=None, pin_memory=True, batch_size=128): # threshold is max for regression accuracy calculation
        
        # check if exists model and params
        bkp_epoch = self.start_from_existing_model(checkpoint_path, MODEL_NAME)
        if(bkp_epoch == 0):
            print("starting training from scratch, no model provided")

        # each test epoch
        test_epoch = int(self.epochs / num_tests) + 1
        print("run test every {} epochs".format(test_epoch))
        self.epochs = bkp_epoch + self.epochs
        # get each loader
        train_loader, val_loader, test_loader = data_loader  

        # get summary write name from checkpoint path
        summary_writer_name = get_last_part_of_path(checkpoint_path)
        self.writer = wandb.init(project="learning-where-to-look", config=WANDB_CONFIG, name=summary_writer_name)
        
        # this is used here just for final benchmark evaluation
        # test_data, pred_test = None, None 

        # TODO
        # early_stopper = EarlyStopper(patience=3, min_delta=1) #0.1
        for epoch in tqdm(range(bkp_epoch, self.epochs)):
            tot_train_loss = 0.0
            y_train, pred_train = list(), list()
            self.model.train()
            if(epoch == 0):
                print("epoch 0, evaluating everything with randomly initialized weights")
                self.model.eval()
            for train_batch in train_loader:
                pred, y = self.predict(train_batch, pin_memory=pin_memory)
                loss_train = self.loss_fn(pred, y.float())  
                        
                # optimize
                if(epoch > 0):
                    loss_train.backward()  # derive gradients
                    self.optimizer.step()  # update parameters based on gradients
                    self.optimizer.zero_grad()  # clear gradients
                
                # calculate accuracy and loss
                tot_train_loss += loss_train.item()
                pred_train.append(pred.squeeze())
                y_train.append(y.squeeze())

            num_train_batches = len(train_loader.dataset)/batch_size
            train_loss = tot_train_loss/num_train_batches
            self.writer.log({"loss/train": train_loss}, step=epoch)

            train_acc = self.compute_and_write_stats(epoch, pred_train, y_train, "train")
            
            # validation each end of training epoch
            tot_val_loss = 0.0
            y_val, pred_val = list(), list()

            self.model.eval()
            for val_batch in val_loader: 
                pred, y = self.predict(val_batch, pin_memory=pin_memory)
                loss_val = self.loss_fn(pred, y.float())  
                # calculate accuracy and loss
                tot_val_loss += loss_val.item()
                # store predictions and labels 
                pred_val.append(pred.squeeze())
                y_val.append(y.squeeze())
            
            num_val_batches = len(val_loader.dataset)/batch_size
            val_loss = tot_val_loss/num_val_batches
            self.writer.log({"loss/val": val_loss}, step=epoch)

            val_acc = self.compute_and_write_stats(epoch, pred_val, y_val, "val")

            # output some stuff
            print(f"epoch {epoch + 1}/{self.epochs}, train [ loss: {train_loss} acc: {train_acc} ] val [ loss: {val_loss} acc: {val_acc} ]")
            
            if(epoch % test_epoch == 0):
                # validation each end of training epoch
                tot_test_loss  = 0.0
                y_test, pred_test = list(), list()
                
                self.model.eval()                
                for test_batch in test_loader: 
                    pred, y = self.predict(test_batch, pin_memory=pin_memory)
                    loss_test = self.loss_fn(pred, y.float())  
                    # calculate accuracy and loss
                    tot_test_loss += loss_test.item()
                    # store predictions and labels 
                    pred_test.append(pred.squeeze())
                    y_test.append(y.squeeze())

                # compute loss
                num_test_batches = len(test_loader.dataset)/batch_size
                test_loss = tot_test_loss/num_test_batches
                self.writer.log({"loss/test": test_loss}, step=epoch)

                # compute other stats
                test_acc = self.compute_and_write_stats(epoch, pred_test, y_test, "test")

                print(f"epoch {epoch + 1}/{self.epochs}, test [ loss: {test_loss} acc: {test_acc} ]")
                # self.evaluate_benchmark(test_data, pred_test)
                
            # test once in a lifetime
            # if(epoch % test_epoch == 0):
            #     self.evaluate_test(test_loader, test_data, writer, epoch, pin_memory)
                        
            if(epoch % SAVE_INTERVAL == 0): # save interval for checkpoint:    
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }
                checkpoint_filename = f'{epoch}_checkpoint.pth'            
                torch.save(checkpoint, os.path.join(checkpoint_path, checkpoint_filename))
                if(self.save_model):
                    torch.save(self.model, os.path.join(checkpoint_path, MODEL_NAME))
                    self.save_model = False
        
        # evaluate benchmark with final model
        print(10*"=")
        print("benchmark evaluation on test set")
        self.evaluate_benchmark(test_data, pred_test)
        print(10*"=")
        
        self.writer.finish()
        
        return self.model
        
    def start_from_existing_model(self, checkpoint_path, remove=None):
        checkpoint_files = list_and_order_files(checkpoint_path, True, remove_from_files=remove)
        epoch = 0
        if(len(checkpoint_files) != 0):
            last_checkpoint_filename = checkpoint_files[-1]
            print("loading existing model {}".format(last_checkpoint_filename))
            dir, rest = os.path.split(checkpoint_path)
            print("looking for architecture in the folder {}".format(MODEL_NAME))
            self.model = torch.load(os.path.join(dir, rest, MODEL_NAME))
            print("model overriden with")
            print(self.model)
            checkpoint = torch.load(last_checkpoint_filename)
            epoch = checkpoint['epoch']
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print("model {} changed, init from another folder otherwise current model will be overriden!".format(last_checkpoint_filename))
                exit(0)
        return epoch
    
    # TODO
    def evaluate_benchmark(self, data, predictions):
        predictions_list_of_list = [t.tolist() for t in predictions]
        predictions = [item for sublist in predictions_list_of_list for item in sublist]
        best_viewpoints_per_bucket = dict()
        # order all_preds per bucket
        for idx, pred in enumerate(predictions):
            # since we have multiple maps make unique id from 3d pose and map id
            bucket_data = np.zeros((4, 1))
            bucket_data[0:3] = data[idx]['pose'].reshape(-1, 1)
            bucket_data[-1] = data[idx]['map'][0]
            pose_idx = bucket_data.tobytes()
            # initialize value
            if (pose_idx not in best_viewpoints_per_bucket.keys()):
                best_viewpoints_per_bucket[pose_idx] = list() 
            best_viewpoints_per_bucket[pose_idx].append((idx, pred))
        
        best_direction_dict = None
        # sort from one with best probability to last one
        # print("sorting in descending order {}".format(self.modality))
        best_direction_dict = {key: sorted(value, key=lambda x: x[1], reverse=True) for key, value in best_viewpoints_per_bucket.items()}
        # print("evaluating benchmark", data) 
        values = [1, 5, 10]
        for v in values:
            et, er = get_accuracies_of_n_elements(data, best_direction_dict, v, modality='classification')                
            # sanity check, these might not be populated raise a warning
            if not et:
                continue 
            errors = calculate_accuracy(et, er)
            print("total num of samples {} using n {} best directions".format(errors["total_num"], v))
            # print("percentage cases of success {:.3f}".format(errors["succes_rate"]))
            print("localization benchmark results: ")
            integer_keys = [key for key in errors.keys() if isinstance(key, int)]
            for k in integer_keys:
                (k, err_t_thresholds, err_r_thresholds, valid_ratio) = errors[k]
                print("\t{} ({} m, {} deg) : {:.3f}".format(k, err_t_thresholds, err_r_thresholds, valid_ratio)) 
                