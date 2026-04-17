import sys

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_data_merged
from utils import act_plot, loss_plot, act_plot_custom, plot_XYZ, plot_XY, TSTime
from train_test_openloop import train, test
from utils import CustomDatasetForDataLoader

from progDense_block import ProgDenseBlock, ProgLstmBlock1, ProgLstmBlock2
from progColumn import ProgColumn
from progNet import ProgNet
from progBlock_Column_template import ProgColumnGenerator



    
# Initialize the PNN model
class PNN_model(ProgColumnGenerator):
    def __init__(self, input_dim, out_dim, hidden_size, lat_connect, device):
        self.ids = 0
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.lat_connect = lat_connect
        self.device = device

    def generateColumn(self, parentCols, msg = None): 
        b1 = ProgLstmBlock1(inSize=self.input_dim, hidden_size=self.hidden_size, numLaterals=0, lat_connect=self.lat_connect)
        b2 = ProgLstmBlock2(self.hidden_size, self.hidden_size, len(parentCols), lat_connect=self.lat_connect, args=args)
        b3 = ProgDenseBlock(self.hidden_size, self.out_dim, len(parentCols), activation = None, args=args)
        c = ProgColumn(self.__genID(), [b1, b2, b3], device = self.device, parentCols = parentCols)
        return c

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

def incremental_call(model, X, Y, batch_size, task_id, noise_var, args):

    if args.mode == "train" or args.mode == "test":        
        # (train 70%, validation 10%, test 20%)
        total_size = X.shape[0]
        train_size = int(0.7 * total_size)
        val_size = int(0.1 * total_size)
        test_size = int(0.2 * total_size)

        # train
        train_X, train_Y = X[:train_size], Y[:train_size]
        # val
        val_X, val_Y = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
        # test
        test_size = batch_size * (test_size // batch_size - (test_size % batch_size == 0))
        test_X, test_Y = X[-test_size:], Y[-test_size:]

        train_stream = CustomDatasetForDataLoader(train_X, train_Y)
        val_stream = CustomDatasetForDataLoader(val_X, val_Y)
        test_stream = CustomDatasetForDataLoader(test_X, test_Y)
     
    elif args.mode == "test_custom":
        test_stream = CustomDatasetForDataLoader(X, Y)

    if args.mode == "train":
        print("------------------------->Starting training for task-id<-------------------------", task_id)
        train_loss, val_loss = train(model=model, train_stream=train_stream,
                                     val_stream=val_stream, task_id=task_id, 
                                     noise_var=noise_var, args=args)
        loss_plot(args, train_loss, val_loss, task_id)

        # test the model after training
        print("------------------------->Starting testing for task-id<-------------------------", task_id)
        pred, gt, fm_pred, fm_gt, _, _ = test(model=model, test_size=test_size, test_stream=test_stream,
                        task_id=task_id, noise_var=noise_var, args=args)
        act_plot(args, pred, gt, task_id, samples=50)
    
    elif args.mode == "test":
        test_stream = CustomDatasetForDataLoader(X, Y)
        test_size=1500
        print("------------------------->Starting testing for task-id<-------------------------", task_id)
        pred, gt, fm_pred, fm_gt, _, _ = test(model=model, test_size=test_size, test_stream=test_stream,
                        task_id=task_id, noise_var=noise_var, args=args)
        act_plot(args, pred, gt, task_id, samples=50) 
        TSTime(args, fm_pred, fm_gt, task_id, samples=50)

    elif args.mode == "test_custom":
        print("------------------------->Starting Custom testing for task id<-------------------------", task_id)
        pred, gt, fm_pred, fm_gt, _, _ = test(model=model, test_size=X.shape[0], test_stream=test_stream,
                        task_id=task_id, noise_var=noise_var, args=args)
        # np.save("pred_seg0.npy", np.concatenate(pred)) if task_id == 0 else None
        act_plot_custom(args, pred, gt, task_id, samples=1000)
        # plot_XY(args, fm_pred, fm_gt, task_id, samples=1000)
        TSTime(args, fm_pred, fm_gt, task_id, samples=500)
        # plot_XYZ(args, fm_pred, fm_gt, task_id, samples=1000)


def main(args):
    fb_dim = 6
    feature_dim = 6
    target_act_dim = 4
    noise_var = 0.01
    batch_size = args.batch_size
    device = args.device
    n_seg = args.n_seg

    # model init
    model = ProgNet(colGen = PNN_model(input_dim=feature_dim+4, out_dim=target_act_dim, hidden_size=args.hidden_size, 
                                       device=device, lat_connect=args.lat_connect))
    print("args mode ", args.mode)

    if args.mode == "test" or args.mode == "test_custom":
        for col_idx in range(3): # for 3 modules
            model.addColumn()

    # data loader
    X, Y = load_data_merged(args, feature_dim=feature_dim, target_act_dim=target_act_dim)
    for i in range(n_seg):
        incremental_call(model, X, Y, batch_size, i, noise_var, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test", "test_custom"], help="train or test")
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM layer")
    parser.add_argument("--hidden_size", type=int, default=32, help="hidden size")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="max training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--file_path", type=str, default="../../3cable_3mods/dataset", help="files path") 
    parser.add_argument("--patience", type=int, default=10, help="Early Stopping patience") 
    parser.add_argument("--time_step", type=int, default=15, help="time step for LSTM")
    parser.add_argument("--n_seg", type=int, default=1, help="number of segments")
    parser.add_argument("--use_orien", type=str, default="true", help="use orientation or not")
    parser.add_argument("--add_noise", type=str, default="false", help="Add external noise to the data")
    parser.add_argument("--device", type=str, default="cuda", help="training device")
    parser.add_argument("--shape_type", type=str, default="circle", choices=["circle", "rect", "spiral"])
    parser.add_argument("--lat_connect", type=str, default="true", choices=["true", "false"], help="lateral connection")

    args = parser.parse_args()

    main(args)

