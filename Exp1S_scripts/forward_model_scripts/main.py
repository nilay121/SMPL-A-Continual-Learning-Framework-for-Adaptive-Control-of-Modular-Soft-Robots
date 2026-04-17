import sys
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMModel_FM
from data_loader import load_data_merged
from utils import task_plot, loss_plot, task_space_inv_transform, task_plot_orien



"""
Guidlines to train the model:

1. To generate the forward model for the first module, set n_seg=1 and use_orien="true" or "false"
2. Similarly, for the second module, set n_seg=2 and use_orien="true" or "false"

"""


def train(args):

    current_seg = args.n_seg # Only train the forward model for the end effcetor module
    train_loss_collector = []
    val_loss_collector = []
    
    # params
    if current_seg == 1:
        feature_dim  = 14 
    elif current_seg == 2:
        feature_dim  = 16 
    elif current_seg == 3:
        feature_dim  = 18
    elif current_seg == 4:
        feature_dim = 20
    elif current_seg == 5:
        feature_dim = 22

    target_act_dim = 6 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data loade

    X, Y = load_data_merged(args, feature_dim=feature_dim, target_act_dim=target_act_dim, current_seg=current_seg)

    # (train 70%, validation 10%, test 20%)
    total_size = X.shape[0]
    train_size = int(0.9 * total_size)
    val_size = int(0.1 * total_size)

    train_X, train_Y = X[:train_size], Y[:train_size]
    val_X, val_Y = X[train_size:], Y[train_size:]

    print(f"Training size: {train_X.shape[0]}, Validation size: {val_X.shape[0]}")

    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=args.batch_size, shuffle=False)

    # model init
    model = LSTMModel_FM(input_size=feature_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=target_act_dim)
    model.to(device)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping setting
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        flag = True
        train_loss = 0
        for X_batch, Y_batch in train_loader:

            X_batch, Y_batch = X_batch[:, :, 0:(2*current_seg)].to(device), Y_batch.to(device)

            # initial random feedback

            if flag:
                s0_pos = torch.zeros((X_batch.shape[0], X_batch.shape[1], 6)).to(device)
                s1_pos = torch.zeros((X_batch.shape[0], X_batch.shape[1], 6)).to(device)
                flag = False
            X_batch_fb = torch.cat((s0_pos[-X_batch.shape[0]:, :, :], s1_pos[-X_batch.shape[0]:, :, :], X_batch), dim=2)

            optimizer.zero_grad()
            outputs = model(X_batch_fb)
            
            s1_pos = s0_pos.detach()
            s0_pos = outputs.detach()
            
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_collector.append(train_loss)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            flag_val = True
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch[:, :, 0:(2*current_seg)].to(device), Y_batch.to(device)
                
                if flag_val:
                    s0_pos = torch.zeros((X_batch.shape[0], X_batch.shape[1], 6)).to(device)
                    s1_pos = torch.zeros((X_batch.shape[0], X_batch.shape[1], 6)).to(device)
                    flag_val = False

                X_batch_fb = torch.cat((s0_pos[-X_batch.shape[0]:, :, :], s1_pos[-X_batch.shape[0]:, :, :], X_batch), dim=2)
                outputs = model(X_batch_fb)

                s1_pos = s0_pos.detach()
                s0_pos = outputs.detach()

                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_loss_collector.append(val_loss)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save the best model
            torch.save(model.state_dict(),  '../forward_models/FM_seg' + str(current_seg)+ "_Mod"+ str(args.n_seg) + '.pt')
            print(f"Best model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break
    loss_plot(args, train_loss_collector, val_loss_collector, current_seg=current_seg)

    return train_loss_collector, val_loss_collector
    


def test(args):

    current_seg = args.n_seg
    predictions = []
    ground_truth = []

    # params
    if current_seg == 1:
        feature_dim  = 14 
    elif current_seg == 2:
        feature_dim  = 16 
    elif current_seg == 3:
        feature_dim  = 18
    elif current_seg == 4:
        feature_dim = 20
    elif current_seg == 5:
        feature_dim = 22
    
    target_act_dim = 6
    current_robot_length = 0.05 * current_seg ## in meters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = load_data_merged(args, feature_dim, target_act_dim, current_seg=current_seg)

    batch_size = args.batch_size

    # last 20% as test
    test_size = int(X.shape[0])
    test_size = batch_size * (test_size // batch_size - (test_size % batch_size == 0))
    test_X, test_Y = X[-test_size:], Y[-test_size:]
    test_loader = DataLoader(TensorDataset(test_X, test_Y), batch_size=args.batch_size, shuffle=False)

    # load model
    model = LSTMModel_FM(input_size=feature_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=target_act_dim)

    model.load_state_dict(torch.load('../forward_models/FM_seg' + str(current_seg)+ "_Mod"+ str(args.n_seg) + '.pt', map_location=device))
    model.to(device)
    model.eval()

    test_err = np.zeros([test_size, args.time_step, target_act_dim])
    it = 0
    batch = args.batch_size
    flag_val = True
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch[:, :, 0:(2*current_seg)].to(device), Y_batch.to(device)
            
            if flag_val:
                s0_pos_val = torch.zeros((X_batch.shape[0], X_batch.shape[1], 6)).to(device)
                s1_pos_val = torch.zeros((X_batch.shape[0], X_batch.shape[1], 6)).to(device)
                flag_val = False

            X_batch_fb = torch.cat((s0_pos_val[-X_batch.shape[0]:, :, :], s1_pos_val[-X_batch.shape[0]:, :, :], X_batch), dim=2)
            
            outputs = model(X_batch_fb)

            s1_pos_val = s0_pos_val.detach()
            s0_pos_val = outputs.detach()
            
            outputs_np = outputs.detach().cpu().numpy()
            Y_np = Y_batch.detach().cpu().numpy()
            predictions.append(outputs_np[:, -1, :])
            ground_truth.append(Y_np[:, -1, :])
            test_err[it * batch: (it + 1) * batch] = np.abs(outputs_np - Y_np)
            it += 1


    ## Task space error in mm
    predictions_inv = task_space_inv_transform(np.concatenate(predictions)[:, 0:3], current_seg=current_seg, args=args)
    ground_truth_inv = task_space_inv_transform(np.concatenate(ground_truth)[:, 0:3], current_seg=current_seg, args=args)

    final_error_ts = np.linalg.norm((predictions_inv - ground_truth_inv), axis=1) ## task space prediction error
    print("Task space error for the last time step:", np.mean(final_error_ts)*1000, "mm")

    ## Relative task space error for each module with respect to the length of the module

    relative_error_percent = (np.mean(final_error_ts) / current_robot_length) * 100
    print("Relative error w.r.t robot length:", relative_error_percent, "%")

    task_plot(args, predictions_inv, ground_truth_inv, current_seg, 75)
    task_plot_orien(args, np.concatenate(predictions)[:, 3:6], np.concatenate(ground_truth)[:, 3:6], current_seg, 75)
        
    
    return predictions_inv, ground_truth_inv



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], help="train or test")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM layer")
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=800, help="max training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--file_path", type=str, default="../dataset", help="files path")
    parser.add_argument("--patience", type=int, default=10, help="Early Stopping patience") #15
    parser.add_argument("--time_step", type=int, default=15, help="time step for LSTM")# 15
    parser.add_argument("--n_seg", type=int, default=1, help="number of segments")
    parser.add_argument("--use_orien", type=str, default="true", help="use orientation or not")

    args = parser.parse_args()

    # if args.mode == "train":
    #     train_loss, val_loss = train(args)
    #     # test the model after training
    #     # pred, gt = test(args)
        
    
    # elif args.mode == "test":
    #     pred, gt = test(args)

    ## ---------> Complete Training <---------
    print("Starting training on 1 segment MSR")
    args.mode = "train"
    args.n_seg = 1
    train_loss, val_loss = train(args)
    
    ## phase 2
    print("Starting training on 2 segment MSR")
    args.mode = "train"
    args.n_seg = 2
    train_loss, val_loss = train(args)

    ## phase 3
    print("Starting training on 3 segment MSR")
    args.mode = "train"
    args.n_seg = 3
    train_loss, val_loss = train(args)

    ## phase 4
    print("Starting training on 4 segment MSR")
    args.mode = "train"
    args.n_seg = 4
    train_loss, val_loss = train(args)

    ## phase 5
    print("Starting training on 5 segment MSR")
    args.mode = "train"
    args.n_seg = 5
    train_loss, val_loss = train(args)

    