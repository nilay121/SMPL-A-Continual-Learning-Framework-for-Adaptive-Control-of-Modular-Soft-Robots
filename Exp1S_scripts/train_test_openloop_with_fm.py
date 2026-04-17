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
from utils import CustomDatasetForDataLoader
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_data, load_data_merged
from utils import act_plot, loss_plot, load_forward_model, task_space_inv_transform

warnings.filterwarnings("ignore")

def train(model, train_stream, val_stream, task_id, noise_var, out_dim, act_feedback_dim, args):
    
    train_loss_collector = []
    val_loss_collector = []

    # params
    feature_dim = 6
    target_dim = 2
    device = args.device

    # model init
    model.addColumn(out_dim=out_dim)
    model.to(device)

    train_loader = DataLoader(train_stream, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_stream, batch_size=args.batch_size, shuffle=False)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Early stopping setting
    best_val_loss = float('inf')
    patience_counter = 0

    # ## adding random noise to the position values of the next time step (i.e S2)
    # # scheduler patience and factor can be provided via args.lr_patience and args.lr_factor
    # lr_patience = getattr(args, 'lr_patience', getattr(args, 'patience', 10))
    # lr_factor = getattr(args, 'lr_factor', 0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True)
    # # track current lr to notify when it changes
    # current_lr = optimizer.param_groups[0]['lr']

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        flag = True

        for features, targets in train_loader:

            X_batch, Y_batch = features[:, :, :, 0:feature_dim].to(device), targets[:, :, :, :].to(device)

            # Add the feedback from the forward model
            if flag:
                act_init = torch.zeros((X_batch.shape[0], args.time_step, act_feedback_dim)).to(device)
                flag = False
    
            if task_id==0:  
                X_batch = torch.concat((features[:, 0, :, 0:feature_dim].to(device), act_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                Y_batch = targets[:, 0, :, 0:target_dim].to(device)

            elif task_id==1:
                X_batch = torch.concat((features[:, 1, :, 0:feature_dim].to(device), act_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                Y_batch = torch.concat((mod0_target_act, mod1_target_act), dim=2).to(device)

            elif task_id==2:     
                X_batch = torch.concat((features[:, 2, :, 0:feature_dim].to(device), act_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                mod2_target_act = targets[:, 2, :, 0:target_dim].to(device)
                Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act), dim=2).to(device)

            elif task_id==3:
                X_batch = torch.concat((features[:, 3, :, 0:feature_dim].to(device), act_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                mod2_target_act = targets[:, 2, :, 0:target_dim].to(device) 
                mod3_target_act = targets[:, 3, :, 0:target_dim].to(device)               
                Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act, mod3_target_act), dim=2).to(device)

            elif task_id==4:
                X_batch = torch.concat((features[:, 4, :, 0:feature_dim].to(device), act_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                mod2_target_act = targets[:, 2, :, 0:target_dim].to(device) 
                mod3_target_act = targets[:, 3, :, 0:target_dim].to(device)        
                mod4_target_act = targets[:, 4, :, 0:target_dim].to(device)       
                Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act, mod3_target_act, mod4_target_act), dim=2).to(device)
                
            optimizer.zero_grad()
            outputs, _, _ = model(X_batch, 0, 0, task_id)

            # update the act_init for the next batch
            act_init[:outputs.shape[0], :, 0:2*(task_id+1)] = outputs.detach()

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
                    act_init_val = torch.zeros((X_batch.shape[0], args.time_step, act_feedback_dim)).to(device)
                    flag_val = False

                if task_id==0:
                    X_batch = torch.concat((features[:, 0, :, 0:feature_dim].to(device), act_init_val[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                    Y_batch = targets[:, 0, :, 0:target_dim].to(device)
            
                elif task_id==1:
                    # # current model
                    X_batch = torch.concat((features[:, 1, :, 0:feature_dim].to(device), act_init_val[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                    mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                    mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                    Y_batch = torch.concat((mod0_target_act, mod1_target_act), dim=2).to(device)
            
                elif task_id==2:
                    # current model
                    X_batch = torch.concat((features[:, 2, :, 0:feature_dim].to(device), act_init_val[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                    mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                    mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                    mod2_target_act = targets[:, 2, :, 0:target_dim].to(device)
                    Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act), dim=2).to(device)

                elif task_id==3:
                    X_batch = torch.concat((features[:, 3, :, 0:feature_dim].to(device), act_init_val[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                    mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                    mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                    mod2_target_act = targets[:, 2, :, 0:target_dim].to(device) 
                    mod3_target_act = targets[:, 3, :, 0:target_dim].to(device)               
                    Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act, mod3_target_act), dim=2).to(device)

                elif task_id==4:
                    X_batch = torch.concat((features[:, 4, :, 0:feature_dim].to(device), act_init_val[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                    mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                    mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                    mod2_target_act = targets[:, 2, :, 0:target_dim].to(device) 
                    mod3_target_act = targets[:, 3, :, 0:target_dim].to(device)        
                    mod4_target_act = targets[:, 4, :, 0:target_dim].to(device)       
                    Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act, mod3_target_act, mod4_target_act), dim=2).to(device)
              
                # current model predictions
                outputs, _, _ = model(X_batch, 0, 0, task_id)   

                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()

                act_init_val[:outputs.shape[0], :, 0:2*(task_id+1)] = outputs.detach()

        val_loss /= len(val_loader)
        val_loss_collector.append(val_loss)

        # # Adjust the learning rate
        # scheduler.step(val_loss)
        # # if LR was reduced by the scheduler, reset early-stopping patience counter
        # new_lr = optimizer.param_groups[0]['lr']
        # if new_lr < current_lr:
        #     print(f"Learning rate reduced from {current_lr:.6g} to {new_lr:.6g}")
        #     current_lr = new_lr
        #     # reset patience counter so early stopping gives the model time after lr change
        #     patience_counter = 0

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

    # # freeze the current column
    if args.incremental_training == "yes":
        model.freezeAllColumns()
    else:
        print("Vanila Naive Model!!")
    return train_loss_collector, val_loss_collector


def load_forward_model_all(current_task_id, device, args):
    ## N_seg 1 --> current task id 1
    if current_task_id == 1:
        forward_model_currentMod1_seg1 = load_forward_model(n_seg=current_task_id, use_orien=args.use_orien,
                                                current_task_id=current_task_id,
                                                feature_dim=14, 
                                                target_dim=6,
                                                device=device)
        return forward_model_currentMod1_seg1
    
    elif current_task_id == 2:
        forward_model_currentMod2_seg2 = load_forward_model(n_seg=current_task_id, use_orien=args.use_orien,
                                                current_task_id=current_task_id,
                                                feature_dim=16, 
                                                target_dim=6,
                                                device=device)
        return forward_model_currentMod2_seg2
    elif current_task_id == 3: 
        forward_model_currentMod3_seg3 = load_forward_model(n_seg=current_task_id, use_orien=args.use_orien,
                                                current_task_id=current_task_id,
                                                feature_dim=18, 
                                                target_dim=6,
                                                device=device) 
        return forward_model_currentMod3_seg3
    elif current_task_id == 4:
        forward_model_currentMod4_seg4 = load_forward_model(n_seg=current_task_id, use_orien=args.use_orien,
                                                current_task_id=current_task_id,
                                                feature_dim=20, 
                                                target_dim=6,
                                                device=device) 
        return forward_model_currentMod4_seg4
    elif current_task_id == 5:
        forward_model_currentMod5_seg5 = load_forward_model(n_seg=current_task_id, use_orien=args.use_orien,
                                                current_task_id=current_task_id,
                                                feature_dim=22, 
                                                target_dim=6,
                                                device=device)  
        return forward_model_currentMod5_seg5


def test(model, task_id, noise_var, test_stream, act_feedback_dim, args):
    predictions = []
    ground_truth = []
    forward_mod_pred = []
    forward_mod_gt = []
    error_output_arr = []

    feature_dim = 6  
    target_dim = 2 
    device = args.device
    current_robot_length = 0.05 * (task_id+1) ## in meters
   
    test_batch_size = 1#args.batch_size
    test_loader = DataLoader(test_stream, batch_size=test_batch_size, shuffle=False)
    
    model.load_state_dict(torch.load('saved_models_open_loop/model' + '.pt', map_location=device))
    model.to(device)
    model.eval()

    # # load the forward model
    # if args.test_total_segments == 1:
    #     forward_model_currentMod1_seg1 = load_forward_model_all(current_task_id=task_id+1, device=device, args=args)
    #     forward_model_currentMod1_seg1.eval()
    # elif args.test_total_segments == 2:
    #     forward_model_currentMod2_seg2 = load_forward_model_all(current_task_id=task_id+1, device=device, args=args)
    #     forward_model_currentMod2_seg2.eval()
    # elif args.test_total_segments == 3:
    #     forward_model_currentMod3_seg3 = load_forward_model_all(current_task_id=task_id+1, device=device, args=args)
    #     forward_model_currentMod3_seg3.eval()
    # elif args.test_total_segments == 4:
    #     forward_model_currentMod4_seg4 = load_forward_model_all(current_task_id=task_id+1, device=device, args=args)
    #     forward_model_currentMod4_seg4.eval()
    # elif args.test_total_segments == 5:
    #     forward_model_currentMod5_seg5 = load_forward_model_all(current_task_id=task_id+1, device=device, args=args)
    #     forward_model_currentMod5_seg5.eval()

    flag_test = True

    with torch.no_grad():

        for features, targets in test_loader:    
            X_batch, Y_batch = features[:, :, :, 0:feature_dim].to(device), targets[:, :, :, :].to(device) # exclude the actutaions

            # Add the feedback from the forward model
            if flag_test:                    
                # initial actuatons are 0
                act_init = torch.zeros((X_batch.shape[0], args.time_step, act_feedback_dim)).to(device)

                s0_pose = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)
                s1_pose = torch.zeros((X_batch.shape[0], args.time_step, 6)).to(device)
                
                flag_test = False
    
            if task_id==0:
                X_batch = torch.concat((features[:, 0, :, 0:feature_dim].to(device), act_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                Y_batch = targets[:, 0, :, 0:target_dim].to(device)
          
            elif task_id==1:
                # current model
                X_batch = torch.concat((features[:, 1, :, 0:feature_dim].to(device), act_init[-X_batch.shape[0]:, :, :]), dim=2).to(device)
                mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                Y_batch = torch.concat((mod0_target_act, mod1_target_act), dim=2).to(device)
          
            elif task_id==2:
                X_batch = torch.concat((features[:, 2, :, 0:feature_dim].to(device), act_init), dim=2).to(device)
                mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                mod2_target_act = targets[:, 2, :, 0:target_dim].to(device)
                Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act), dim=2).to(device)

            elif task_id==3:
                X_batch = torch.concat((features[:, 3, :, 0:feature_dim].to(device), act_init), dim=2).to(device)
                mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                mod2_target_act = targets[:, 2, :, 0:target_dim].to(device) 
                mod3_target_act = targets[:, 3, :, 0:target_dim].to(device)               
                Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act, mod3_target_act), dim=2).to(device)

            elif task_id==4:
                X_batch = torch.concat((features[:, 4, :, 0:feature_dim].to(device), act_init), dim=2).to(device)
                mod0_target_act = targets[:, 0, :, 0:target_dim].to(device)
                mod1_target_act = targets[:, 1, :, 0:target_dim].to(device)
                mod2_target_act = targets[:, 2, :, 0:target_dim].to(device) 
                mod3_target_act = targets[:, 3, :, 0:target_dim].to(device)        
                mod4_target_act = targets[:, 4, :, 0:target_dim].to(device)       
                Y_batch = torch.concat((mod0_target_act, mod1_target_act, mod2_target_act, mod3_target_act, mod4_target_act), dim=2).to(device)
            
            # current predictions     
            outputs, _, _ = model(X_batch, 0, 0, task_id)   

            # # forward model predictions
            # with torch.no_grad():
            #     act_output = outputs[:, :, 0:2*(task_id+1)] 
            #     if args.test_total_segments == 1:
            #         fm_feedback_train = forward_model_currentMod1_seg1(torch.concat((s0_pose[-X_batch.shape[0]:, :, :], s1_pose[-X_batch.shape[0]:, :, :], act_output), dim=2))
            #     elif args.test_total_segments == 2:
            #         fm_feedback_train = forward_model_currentMod2_seg2(torch.concat((s0_pose[-X_batch.shape[0]:, :, :], s1_pose[-X_batch.shape[0]:, :, :], act_output), dim=2))
            #     elif args.test_total_segments == 3:
            #         fm_feedback_train = forward_model_currentMod3_seg3(torch.concat((s0_pose[-X_batch.shape[0]:, :, :], s1_pose[-X_batch.shape[0]:, :, :], act_output), dim=2))
            #     elif args.test_total_segments == 4:
            #         fm_feedback_train = forward_model_currentMod4_seg4(torch.concat((s0_pose[-X_batch.shape[0]:, :, :], s1_pose[-X_batch.shape[0]:, :, :], act_output), dim=2))
            #     elif args.test_total_segments == 5:
            #         fm_feedback_train = forward_model_currentMod5_seg5(torch.concat((s0_pose[-X_batch.shape[0]:, :, :], s1_pose[-X_batch.shape[0]:, :, :], act_output), dim=2))
           
            outputs_np = outputs.cpu().detach().numpy()
            Y_np = Y_batch.cpu().detach().numpy()

            error_output = np.mean((outputs_np-Y_np) ** 2, axis=2)
            error_output_arr.append(error_output)
            
            # # Record forward model predictions
            if task_id == 0:
                # forward_mod_pred.append(fm_feedback_train.detach().cpu().numpy())
                forward_mod_gt.append(features[:, 0, :, 0:6].detach().cpu().numpy())
            elif task_id == 1:
                # forward_mod_pred.append(fm_feedback_train.detach().cpu().numpy())
                forward_mod_gt.append(features[:, 1, :, 0:6].detach().cpu().numpy())
            elif task_id == 2:
                # forward_mod_pred.append(fm_feedback_train.detach().cpu().numpy())
                forward_mod_gt.append(features[:, 2, :, 0:6].detach().cpu().numpy())
            elif task_id == 3:
                # forward_mod_pred.append(fm_feedback_train.detach().cpu().numpy())
                forward_mod_gt.append(features[:, 3, :, 0:6].detach().cpu().numpy())
            elif task_id == 4 :
                # forward_mod_pred.append(fm_feedback_train.detach().cpu().numpy())
                forward_mod_gt.append(features[:, 4, :, 0:6].detach().cpu().numpy())

            predictions.append(outputs_np)
            ground_truth.append(Y_np)

            ## Update the feedback actuations for the inverse model
            act_init[:outputs.shape[0], :, 0:2*(task_id+1)] = outputs.detach()
            # ## Update the feedback pose for the forward model
            # s1_pose = s0_pose.detach()
            # s0_pose = fm_feedback_train.detach()

    # Error calculation based on the saved predictions and ground truth    
    ## convert to array
    predictions_array = np.concatenate(predictions)
    ground_truth_array = np.concatenate(ground_truth)  
    
    # fm_pred_array = np.concatenate(forward_mod_pred, axis=0)
    fm_gt_array = np.concatenate(forward_mod_gt, axis=0)


    # # Final step error --> actuation space
    final_pred_act = predictions_array[:, -1, :]
    final_true_act = ground_truth_array[:, -1, :]

    # final_pred_pos = fm_pred_array[:, -1, 0:3] 
    final_true_pos = fm_gt_array[:, -1, 0:3] 

    # final_pred_ori = fm_pred_array[:, -1, 3:]
    final_true_ori = fm_gt_array[:, -1, 3:]

    # Percentage L2 error for actuation space
    max_l2_er_act = 2 * np.sqrt(final_pred_act.shape[1])
    normalized_l2_er_act = (np.linalg.norm(final_pred_act - final_true_act, axis=1) / max_l2_er_act)*100
    print("% L2 error for actuation is ", np.mean(normalized_l2_er_act))

    # original position error --> in mm
    # final_pred_pos_inv = task_space_inv_transform(final_pred_pos, current_seg=task_id+1, args=args)
    final_true_pos_inv = task_space_inv_transform(final_true_pos, current_seg=task_id+1, args=args)
    # final_error_ts = np.mean(np.linalg.norm((final_pred_pos_inv - final_true_pos_inv), axis=1)) ## task space prediction error
    # print("Error in position for the last time step:", final_error_ts*1000, "mm")

    ## Relative task space error for each module with respect to the length of the module

    # relative_error_percent = (np.mean(final_error_ts) / current_robot_length) * 100
    # print("Relative error w.r.t robot length:", relative_error_percent, "%")

    return final_pred_act, final_true_act, final_true_pos_inv, final_true_ori