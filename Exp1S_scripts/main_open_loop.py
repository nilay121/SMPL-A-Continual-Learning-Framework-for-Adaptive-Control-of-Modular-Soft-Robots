import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_data_merged
from utils import act_plot, loss_plot, act_plot_custom, plot_XYZ, plot_XY, pos_plot, orien_plot
from train_test_openloop_with_fm import train, test
from utils import CustomDatasetForDataLoader, extract_and_rearrange_sim_pred

from progDense_block import ProgDenseBlock, ProgLstmBlock1, ProgLstmBlock2
from progColumn import ProgColumn
from progNet import ProgNet
from progBlock_Column_template import ProgColumnGenerator
from replay_buffer import ReplayBuffer
from simulator.sim_env import sim


    
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
        b1 = ProgLstmBlock1(inSize=self.input_dim, hidden_size=self.hidden_size, numLaterals=0, lat_connect=self.lat_connect, dropout=0.5)
        b2 = ProgLstmBlock2(self.hidden_size, self.hidden_size, len(parentCols), lat_connect=self.lat_connect, dropout=0.5, args=args)
        b3 = ProgDenseBlock(self.hidden_size, self.out_dim, len(parentCols), activation = None, args=args)
        c = ProgColumn(self.__genID(), [b1, b2, b3], device = self.device, parentCols = parentCols)
        return c

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id
    

## Train on the current segment
def incremental_train_call(model, X, Y, batch_size, task_id, noise_var, out_dim, act_feedback_dim, args):
    
    total_size = X.shape[0]
    train_size = int(0.9 * total_size)
    # train
    train_X, train_Y = X[:train_size], Y[:train_size]  # based on babbling data 8k
    # val
    val_X, val_Y = X[train_size:], Y[train_size:]

    train_stream = CustomDatasetForDataLoader(train_X, train_Y)
    val_stream = CustomDatasetForDataLoader(val_X, val_Y)

    print(f"------------------------->Starting training on Total Module Size {args.train_total_segments}, current number of segments<-------------------------", task_id+1)
    train_loss, val_loss = train(model=model, train_stream=train_stream,
                                    val_stream=val_stream, task_id=task_id, 
                                    noise_var=noise_var, out_dim=out_dim, 
                                    act_feedback_dim=act_feedback_dim, args=args)
    loss_plot(args, train_loss, val_loss, task_id)

    ## if incremental test is on
    """
    Here the model is evaluated after training on the current segment, i.e after training on the 2nd segment, we test on the 1st and 2nd segment
    """
    if args.incremental_test == "yes":
        args.mode = "test"
        args.shape_type = "babbling"
        print("Incremental test is turned on, testing on the current and all the previous segments!!....")
        for current_seg in range(task_id+1):
            ## loop over all the prevous segments and test on each of them
            incremental_test_call(feature_dim=6, target_dim=2, task_id=current_seg, model=model, act_feedback_dim=act_feedback_dim, total_seg_increment_test=task_id+1, args=args)


## Test on current and all the previous segments
def incremental_test_call(feature_dim, target_dim, task_id, model, act_feedback_dim, total_seg_increment_test, args):
    """
    First check if the model has been trained on the first segment
    """
    if args.incremental_test == "no":
        X_test, Y_test = load_data_merged(args, feature_dim=feature_dim, target_act_dim=target_dim, mode="test", seg_numb=task_id+1)
        test_stream = CustomDatasetForDataLoader(X_test, Y_test)
        print(f"------------------------->Starting Inference on Total Module Size {args.test_total_segments}, current number of segments<-------------------------", task_id+1)
        pred, gt_act, _, _ = test(model=model,
                                        task_id=task_id, 
                                        noise_var=None, 
                                        test_stream=test_stream,
                                        act_feedback_dim=act_feedback_dim,
                                        args=args)
    
        # ----------------------- call the simulator and pass the predicted actuation -----------------------
        print("Calling the simulator with the predicted actuation to get the pos and ori data!!....")
        ctrl_step = pred.shape[0] 
        pred_reorg = pred.reshape(ctrl_step, args.test_total_segments, 2)

        print("ctrl step is ", ctrl_step, "pred reorg shape is ", pred_reorg.shape)
        pred_pos_list, pred_ori_list, _ = sim(ctrl_step=ctrl_step, num_seg=args.test_total_segments, act_list=pred_reorg)
        # extract the position and orientation information
        pred_pos_list, pred_ori_list = extract_and_rearrange_sim_pred(pos_list=pred_pos_list, ori_list=pred_ori_list, seg_numb=args.test_total_segments, args=args)

    elif args.incremental_test == "yes":
        X_test, Y_test = load_data_merged(args, feature_dim=feature_dim, target_act_dim=target_dim, mode="test", seg_numb=task_id+1)
        test_stream = CustomDatasetForDataLoader(X_test, Y_test)
        print(f"------------------------->Starting Inference on Total Module Size {total_seg_increment_test}, current number of segments<-------------------------", task_id+1)
        pred, gt_act, _, _ = test(model=model,
                                        task_id=task_id, 
                                        noise_var=None, 
                                        test_stream=test_stream,
                                        act_feedback_dim=act_feedback_dim,
                                        args=args)
    
        # ----------------------- call the simulator and pass the predicted actuation -----------------------
        print("Calling the simulator with the predicted actuation to get the pos and ori data!!....")
        ctrl_step = pred.shape[0] 
        pred_reorg = pred.reshape(ctrl_step, task_id+1, 2)

        print("ctrl step is ", ctrl_step, "pred reorg shape is ", pred_reorg.shape)
        pred_pos_list, pred_ori_list, _ = sim(ctrl_step=ctrl_step, num_seg=task_id+1, act_list=pred_reorg)
        # extract the position and orientation information
        pred_pos_list, pred_ori_list = extract_and_rearrange_sim_pred(pos_list=pred_pos_list, ori_list=pred_ori_list, seg_numb=task_id+1, args=args)
   
    ## ----------------------- Save the position, orientation and actuation values in memory for predictions -----------------------
    print("Saving the predicted actuation array in memory!!....")
    # # predictions
    # np.save(f"saved_predicted_data/pred_oloop_act_shape_{args.shape_type}_totalSegSize{total_seg_increment_test}_currentSeg{task_id+1}", 
    #         pred)
    # np.save(f"saved_predicted_data/pred_oloop_pos_shape_{args.shape_type}_totalSegSize{total_seg_increment_test}_currentSeg{task_id+1}",
    #         pred_pos_list)
    # np.save(f"saved_predicted_data/pred_oloop_ori_shape_{args.shape_type}_totalSegSize{total_seg_increment_test}_currentSeg{task_id+1}",
    #         pred_ori_list)

    # predictions
    np.save(f"saved_predicted_data_after_all_training_run2/pred_oloop_act_shape_{args.shape_type}_totalSegSize{total_seg_increment_test}_currentSeg{task_id+1}", 
            pred)
    np.save(f"saved_predicted_data_after_all_training_run2/pred_oloop_pos_shape_{args.shape_type}_totalSegSize{total_seg_increment_test}_currentSeg{task_id+1}",
            pred_pos_list)
    np.save(f"saved_predicted_data_after_all_training_run2/pred_oloop_ori_shape_{args.shape_type}_totalSegSize{total_seg_increment_test}_currentSeg{task_id+1}",
            pred_ori_list)

    ## ---------------- save the ground truth actuation, position and orientation values in memory by sending the ground truth actuation -----------------------
  
    # print("Calling the simulator with the ground truth actuation to get the pos and ori data!!....")
    # ctrl_step = gt_act.shape[0] 
    # gt_reorg = gt_act.reshape(ctrl_step, args.test_total_segments, 2)
    # print("ctrl step is ", ctrl_step, "gt reorg shape is ", gt_reorg.shape)
    # gt_pos_list, gt_ori_list, _ = sim(ctrl_step=ctrl_step, num_seg=args.test_total_segments, act_list=gt_reorg)
    # # extract the position and orientation information
    # gt_pos_list, gt_ori_list = extract_and_rearrange_sim_pred(pos_list=gt_pos_list, ori_list=gt_ori_list, seg_numb=args.test_total_segments, args=args)
    # # ----------------------- Save the position, orientation and actuation values in memory -----------------------
    # print("Saving the ground truth actuation array in memory!!....")

    # np.save(f"saved_predicted_data/gt_oloop_act_shape_{args.shape_type}_totalSegSize{args.test_total_segments}", 
    #         gt_act)
    # np.save(f"saved_predicted_data/gt_oloop_pos_shape_{args.shape_type}_totalSegSize{args.test_total_segments}",
    #         gt_pos_list)
    # np.save(f"saved_predicted_data/gt_oloop_ori_shape_{args.shape_type}_totalSegSize{args.test_total_segments}",
    #         gt_ori_list)

    ## ----------------------- Plot the actuation and task space predictions -----------------------
    # act_plot(args, pred, gt_act, task_id+1, samples=500) # actuation plot
    # pos_plot(args, gt=gt_pos_list, pred=pred_pos_list, task_id+1, samples=500) # position wrt time plot
    # plot_XY(args=args, gt=gt_pos_list, pred=pred_pos_list, task_id=task_id, samples=500) # XY plot 
    # orien_plot(args, gt=gt_ori_list, pred=pred_ori_list, task_id+1, samples=500) # orientation wrt time plot



def main(args):
    feature_dim = 6
    noise_var = 0.01
    batch_size = args.batch_size
    device = args.device
    test_n_seg = args.test_total_segments
    total_segments = args.train_total_segments
    act_feedback_dim = 2*total_segments

    # model init
    model = ProgNet(colGen = PNN_model(input_dim=feature_dim+(act_feedback_dim), out_dim=2, hidden_size=args.hidden_size, 
                                       device=device, lat_connect=args.lat_connect)) # target action dim will change for each module
    print("args mode ", args.mode)

    # data loader
    if args.mode == "train":
        for current_total_segments in range(total_segments):
            current_total_segments += 1
            out_dim = 2*current_total_segments

            X, Y = load_data_merged(args, feature_dim=feature_dim, target_act_dim=2, mode=args.mode, seg_numb=current_total_segments) 
            incremental_train_call(model, X, Y, batch_size, current_total_segments-1, noise_var, out_dim, act_feedback_dim, args)
    
    elif args.mode == "test":

        for i in range(5): # for 3 modules
            model.addColumn(out_dim=(i+1)*2)

        X, Y = load_data_merged(args, feature_dim=feature_dim, target_act_dim=2, mode=args.mode, seg_numb=test_n_seg)
        incremental_test_call(feature_dim, 2, test_n_seg-1, model, act_feedback_dim, None, args)




## Parameters to provide from command line : n_seg, mode, epochs, incremental_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test", "test_custom"], help="train or test")
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM layer") ## initial value is 1
    parser.add_argument("--hidden_size", type=int, default=32, help="hidden size") ## initial value is 32
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=800, help="max training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--file_path", type=str, default="../../3cable_3mods/dataset", help="files path") 
    parser.add_argument("--patience", type=int, default=10, help="Early Stopping patience") 
    parser.add_argument("--time_step", type=int, default=15, help="time step for LSTM")
    parser.add_argument("--train_total_segments", type=int, default=5, help="number of segments")
    parser.add_argument("--test_total_segments", type=int, default=1, help="test with how many segments")
    parser.add_argument("--use_orien", type=str, default="true", help="use orientation or not")
    parser.add_argument("--add_noise", type=str, default="false", help="Add external noise to the data")
    parser.add_argument("--device", type=str, default="cuda", help="training device")
    parser.add_argument("--shape_type", type=str, default="babbling", choices=["babbling","circle", "rect", "spiral"])
    parser.add_argument("--incremental_training", type=str, default="yes", choices=["yes", "no"], help="Incremental training or not")
    parser.add_argument("--incremental_test", type=str, default="no", choices=["yes", "no"], help="Incremental test or not")
    parser.add_argument("--lat_connect", type=str, default="true", choices=["true", "false"], help="lateral connection")
    parser.add_argument("--lr_patience", type=int, default=10, help="LR scheduler patience")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="LR scheduler factor")

    args = parser.parse_args() 
    main(args=args)
    
    """
    Train --> Parameters to provide from command line : mode, epochs, incremental_training, train_total_segments

    python main_open_loop_with_fm.py --mode train --incremental_training yes --train_total_segments 5
    python3 main_open_loop_with_fm.py --mode train --incremental_training yes --train_total_segments 5 --incrmental_test yes

    Test
    python main_open_loop_with_fm.py --mode test --test_total_segments N --shape_type circle
    """

    # ## Run all the shapes and segmets one by one
    # shape_types = ["babbling", "circle", "rect", "spiral"]
    # for shape in shape_types:
    #     for seg in range(1, 6):
    #         print(f" ----------------------- Running for shape {shape} with total segments {seg}!! -----------------------")
    #         args.shape_type = shape
    #         args.mode = "test"
    #         args.test_total_segments = seg
    #         main(args)