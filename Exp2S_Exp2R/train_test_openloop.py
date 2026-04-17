import torch
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import save_dictionary, load_dictionary



import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_data
from utils import act_plot, loss_plot, load_forward_model, task_space_inv_transform

warnings.filterwarnings("ignore")

def train(model, train_stream, val_stream, task_id, noise_var, args):
    
    train_loss_collector = []
    val_loss_collector = []

    # feature dim
    if task_id == 0:
        feature_dim = 6
        target_dim = 4
    elif task_id == 1:
        feature_dim = 6
        target_dim = 4
    elif task_id == 2:
        feature_dim = 6  
        target_dim = 4  

    # params
    device = args.device

    # model init
    model.addColumn()
    model.to(device)

    train_loader = DataLoader(train_stream, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_stream, batch_size=args.batch_size, shuffle=False)

    # loss and optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping setting
    best_val_loss = float('inf')
    patience_counter = 0

    ## adding random noise to the position values of the next time step (i.e S2) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True,)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        flag = True
        for features, targets in train_loader:

            X_batch, Y_batch = features[:, :, :, 0:feature_dim].to(device), targets[:, :, :, :].to(device)

            # Add the feedback from the forward model
            if flag:
                # initial actuatons are 0
                act0_init = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)
                act1_init = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)
                act2_init = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)
                flag = False
    
            if task_id==0:            
                X_batch = torch.concat((features[:, 0, :, 0:feature_dim].to(device), act0_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                Y_batch = targets[:, 0, :, 0:target_dim].to(device)
          
            elif task_id==1:
                # current model
                X_batch = torch.concat((features[:, 1, :, 0:feature_dim].to(device), act1_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                Y_batch = targets[:, 1, :, 0:target_dim].to(device)
          
            elif task_id==2:
                # current model
                X_batch = torch.concat((features[:, 2, :, 0:feature_dim].to(device), act2_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                Y_batch = targets[:, 2, :, 0:target_dim].to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(X_batch, 0, 0, task_id)
            
            if task_id == 0:
                # actuation
                act0_init = outputs.detach()
            elif task_id == 1:
                # actuation
                act1_init = outputs.detach()
            elif task_id == 2:
                # actuation
                act2_init = outputs.detach()

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

            for features, targets in val_loader:
                # phase 1
                X_batch, Y_batch = features[:, :, :, 0:feature_dim].to(device), targets[:, :, :, :].to(device)

                # Add the feedback from the forward model
                if flag_val:
                    # initial actuatons are 0
                    act0_init_val = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)
                    act1_init_val = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)
                    act2_init_val = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)

                    flag_val = False

                if task_id==0:
                    X_batch = torch.concat((features[:, 0, :, 0:feature_dim].to(device), act0_init_val[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                    Y_batch = targets[:, 0, :, 0:target_dim].to(device)
            
                elif task_id==1:
                    # # current model
                    X_batch = torch.concat((features[:, 1, :, 0:feature_dim].to(device), act1_init_val[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                    Y_batch = targets[:, 1, :, 0:target_dim].to(device)
            
                elif task_id==2:
                    # current model
                    X_batch = torch.concat((features[:, 2, :, 0:feature_dim].to(device), act2_init_val[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                    Y_batch = targets[:, 2, :, 0:target_dim].to(device)
              
                # current model predictions
                outputs, _, _ = model(X_batch, 0, 0, task_id)   

                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()

                if task_id == 0:
                    act0_init_val = outputs.detach()
                elif task_id == 1:
                    act1_init_val = outputs.detach()
                elif task_id == 2:
                    act2_init_val = outputs.detach()

        val_loss /= len(val_loader)
        val_loss_collector.append(val_loss)

        # Adjust the learning rate
        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save the best model
            torch.save(model.state_dict(),  'saved_models_open_loop/model' + '.pt')
            print(f"Best model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break

    # freeze the current column
    model.freezeAllColumns()
    return train_loss_collector, val_loss_collector




def test(model, test_size, test_stream, task_id, noise_var, args):
    predictions = []
    ground_truth = []
    forward_mod_pred = []
    forward_mod_gt = []
    error_output_arr = []

    # params
    # feature dim
    if task_id == 0:
        feature_dim = 6
        target_dim = 4
    elif task_id == 1:
        feature_dim = 6
        target_dim = 4
    elif task_id == 2:
        feature_dim = 6  
        target_dim = 4  

    #args.n_seg = task_id ## will affect the lateral saving format
    device = args.device

   
    test_batch_size = 1#args.batch_size
    test_loader = DataLoader(test_stream, batch_size=test_batch_size, shuffle=False)
    model.load_state_dict(torch.load('saved_models_open_loop/model' + '.pt', map_location=device))
    model.to(device)
    model.eval()

    # load the forward model
    forward_model_seg0 = load_forward_model(n_seg=0, use_orien=args.use_orien,
                                       feature_dim=16, 
                                       target_dim=6,
                                       device=device)
    forward_model_seg1 = load_forward_model(n_seg=1, use_orien=args.use_orien,
                                       feature_dim=20, 
                                       target_dim=6,
                                       device=device)
    forward_model_seg2 = load_forward_model(n_seg=2, use_orien=args.use_orien,
                                       feature_dim=24, 
                                       target_dim=6,
                                       device=device)
    forward_model_seg0.eval()
    forward_model_seg1.eval()
    forward_model_seg2.eval()
    flag_test = True
  
    fm_feedback_train_seg1 = None
    fm_feedback_train_seg2 = None

    with torch.no_grad():

        for features, targets in test_loader:    
            X_batch, Y_batch = features[:, :, :, 0:feature_dim].to(device), targets[:, :, :, :].to(device) # exclude the actutaions
            

            # Add the feedback from the forward model
            if flag_test:
                
                # intial errors are 0
                error_fb_seg0 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device) # orientation is true
                error_fb_seg1 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)
                error_fb_seg2 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)
                
                # initial actuatons are 0
                act0_init = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)
                act1_init = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)
                act2_init = torch.zeros((X_batch.shape[0], args.time_step, 4)).to(device)

                s0_pose_mod1 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)
                s1_pose_mod1 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)

                s0_pose_mod2 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)
                s1_pose_mod2 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)

                s0_pose_mod3 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)
                s1_pose_mod3 = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)

                
                flag_test = False
    
            if task_id==0:
                X_batch = torch.concat((features[:, 0, :, 0:feature_dim].to(device), act0_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                Y_batch = targets[:, 0, :, 0:target_dim].to(device)
          
            elif task_id==1:
                # needs actuation from the first module to send it to the forward model of module 2
                # previous seg prediction
                seg0_fb_input = torch.concat((features[:, 0, :, 0:feature_dim].to(device), act0_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                out_seg0, _, _ = model(seg0_fb_input, 0, 0, 0) # last is task id
                # current model
                X_batch = torch.concat((features[:, 1, :, 0:feature_dim].to(device), act1_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                Y_batch = targets[:, 1, :, 0:target_dim].to(device)
          
            elif task_id==2:
                # needs actuation from the first module, second module to send it to the forward model of module 3
                #seg 0
                seg0_fb_input = torch.concat((features[:, 0, :, 0:feature_dim].to(device), act0_init), dim=2).to(device)
                out_seg0, _, _ = model(seg0_fb_input, 0, 0, 0) # last is task id
                #seg1
                seg1_fb_input = torch.concat((features[:, 1, :, 0:feature_dim].to(device), act1_init), dim=2).to(device)
                out_seg1, _, _ = model(seg1_fb_input, 0, 0, 1) # last is task id
                # current model
                X_batch = torch.concat((features[:, 2, :, 0:feature_dim].to(device), act2_init), dim=2).to(device)
                Y_batch = targets[:, 2, :, 0:target_dim].to(device)
            
            # current predictions     
            outputs, _, _ = model(X_batch, 0, 0, task_id)   

            # forward model precictions
            with torch.no_grad():
                if task_id==0:
                    fm_feedback_train_seg0 = forward_model_seg0(torch.concat((s0_pose_mod1[-X_batch.shape[0]:, :, :], s1_pose_mod1[-X_batch.shape[0]:, :, :], outputs), dim=2))
            
                elif task_id==1:
                    fm_feedback_train_seg0 = forward_model_seg0(torch.concat((s0_pose_mod1[-X_batch.shape[0]:, :, :], s1_pose_mod1[-X_batch.shape[0]:, :, :], out_seg0), dim=2))
                    fm_feedback_train_seg1 = forward_model_seg1(torch.concat((s0_pose_mod2[-X_batch.shape[0]:, :, :], s1_pose_mod2[-X_batch.shape[0]:, :, :], out_seg0, outputs), dim=2))
            
                elif task_id==2:
                    fm_feedback_train_seg0 = forward_model_seg0(torch.concat((s0_pose_mod1[-X_batch.shape[0]:, :, :], s1_pose_mod1[-X_batch.shape[0]:, :, :], out_seg0), dim=2))
                    fm_feedback_train_seg1 = forward_model_seg1(torch.concat((s0_pose_mod2[-X_batch.shape[0]:, :, :], s1_pose_mod2[-X_batch.shape[0]:, :, :], out_seg0, out_seg1), dim=2))
                    fm_feedback_train_seg2 = forward_model_seg2(torch.concat((s0_pose_mod3[-X_batch.shape[0]:, :, :], s1_pose_mod3[-X_batch.shape[0]:, :, :], out_seg0, out_seg1, outputs), dim=2))


            outputs_np = outputs.cpu().detach().numpy()
            Y_np = Y_batch.cpu().detach().numpy()

            error_output = np.mean((outputs_np-Y_np) ** 2, axis=2)
            error_output_arr.append(error_output)
            
            # Record forward model predictions
            if task_id == 0:
                forward_mod_pred.append(fm_feedback_train_seg0.detach().cpu().numpy())
                forward_mod_gt.append(features[:, 0, :, 0:6].detach().cpu().numpy())
            elif task_id == 1:
                forward_mod_pred.append(fm_feedback_train_seg1.detach().cpu().numpy())
                forward_mod_gt.append(features[:, 1, :, 0:6].detach().cpu().numpy())
            elif task_id == 2:
                forward_mod_pred.append(fm_feedback_train_seg2.detach().cpu().numpy())
                forward_mod_gt.append(features[:, 2, :, 0:6].detach().cpu().numpy())

            predictions.append(outputs_np)
            ground_truth.append(Y_np)

            if task_id == 0:
                # actuation
                act0_init = outputs.detach()
                # position and orientation
                s1_pose_mod1 = s0_pose_mod1.detach()
                s0_pose_mod1 = fm_feedback_train_seg0.detach()
            elif task_id == 1:
                # actuation
                act0_init = out_seg0.detach()
                act1_init = outputs.detach()
                # position and orientation
                #mod 1
                s1_pose_mod1 = s0_pose_mod1.detach()
                s0_pose_mod1 = fm_feedback_train_seg0.detach()
                ##mod 2
                s1_pose_mod2 = s0_pose_mod2.detach()
                s0_pose_mod2 = fm_feedback_train_seg1.detach()

            elif task_id == 2:
                # actuation
                act0_init = out_seg0.detach()
                act1_init = out_seg1.detach()
                act2_init = outputs.detach()
                # position and orientation
                #mod 1
                s1_pose_mod1 = s0_pose_mod1.detach()
                s0_pose_mod1 = fm_feedback_train_seg0.detach()
                ## mod 2
                s1_pose_mod2 = s0_pose_mod2.detach()
                s0_pose_mod2 = fm_feedback_train_seg1.detach()
                # mod 3
                s1_pose_mod3 = s0_pose_mod3.detach()
                s0_pose_mod3 = fm_feedback_train_seg2.detach()

    # Error calculation based on the saved predictions and ground truth    
    ## convert to array
    predictions_array = np.concatenate(predictions)
    ground_truth_array = np.concatenate(ground_truth)  
    
    fm_pred_array = np.concatenate(forward_mod_pred, axis=0)
    fm_gt_array = np.concatenate(forward_mod_gt, axis=0)


    # # Final step error --> actuation space
    final_pred_act = predictions_array[:, -1, :]
    final_true_act = ground_truth_array[:, -1, :]

    final_pred_pos = fm_pred_array[:, -1, 0:3] 
    final_true_pos = fm_gt_array[:, -1, 0:3] 

    final_pred_ori = fm_pred_array[:, -1, 3:]
    final_true_ori = fm_gt_array[:, -1, 3:]

    # original posiiton error
    final_pred_pos_inv = task_space_inv_transform(final_pred_pos, task_id=task_id)
    final_true_pos_inv = task_space_inv_transform(final_true_pos, task_id=task_id)
    final_error_ts = np.mean(np.linalg.norm((final_pred_pos - final_true_pos), axis=1)) ## task space prediction error
    print("Task space error for the last time step:", final_error_ts*100, "cm")



    return final_pred_act, final_true_act, fm_pred_array[:, -1, :], fm_gt_array[:, -1, :], final_pred_pos_inv, final_true_pos_inv