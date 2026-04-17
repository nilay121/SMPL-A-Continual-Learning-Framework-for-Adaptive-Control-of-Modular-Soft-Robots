import pickle
import joblib
import torch
import numpy as np
from forward_model_scripts.model import LSTMModel_FM
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class CustomDatasetForDataLoader(Dataset):
    def __init__(self, data, targets):
        # convet labels to 1 hot
        self.data = data
        self.targets = targets
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.targets[idx]

def save_dictionary(data, filename):
    with open(f'{filename}', 'wb') as fp:
        pickle.dump(data, fp)

def load_dictionary(filename):
    with open(filename, 'rb') as fp:
        interpolation_steps = pickle.load(fp)
    return interpolation_steps

def act_plot(args, pred_org, gt_org, task_id, samples=300):
    dummy_zeros_pred = np.zeros((pred_org.shape[0], 10))
    dummy_zeros_gt = np.zeros((gt_org.shape[0], 10))
    
    ## Update the dummy zeros
    dummy_zeros_pred[:, :pred_org.shape[1]] = pred_org
    dummy_zeros_gt[:, :gt_org.shape[1]] = gt_org

    pred = dummy_zeros_pred
    gt = dummy_zeros_gt

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(13,5))

    axes[0, 0].plot(pred[:samples, 0], color='navy')
    axes[0, 0].plot(gt[:samples, 0], color='green', alpha=0.5)
    axes[0, 0].set_xlabel("Samples")
    axes[0, 0].set_ylabel("Actuation")
    axes[0, 0].set_title("Module 1 Acts")
    axes[0, 0].legend(['Pred Data', 'Ground truth'])         

    axes[0, 1].plot(pred[:samples, 1], color='navy')
    axes[0, 1].plot(gt[:samples, 1], color='green', alpha=0.5)
    axes[0, 1].set_xlabel("Samples")
    axes[0, 1].set_ylabel("Actuation")
    axes[0, 1].set_title("Module 2 Acts")
    axes[0, 1].legend(['Pred Data', 'Ground truth'])    

    axes[0, 2].plot(pred[:samples, 2], color='navy')
    axes[0, 2].plot(gt[:samples, 2], color='green', alpha=0.5)
    axes[0, 2].set_xlabel("Samples")
    axes[0, 2].set_ylabel("Actuation")
    axes[0, 2].set_title("Module 3 Acts")
    axes[0, 2].legend(['Pred Data', 'Ground truth'])  

    axes[0, 3].plot(pred[:samples, 3], color='navy')
    axes[0, 3].plot(gt[:samples, 3], color='green', alpha=0.5)
    axes[0, 3].set_xlabel("Samples")
    axes[0, 3].set_ylabel("Actuation")
    axes[0, 3].set_title("Module 4 Acts")
    axes[0, 3].legend(['Pred Data', 'Ground truth'])  

    axes[0, 4].plot(pred[:samples, 4], color='navy')
    axes[0, 4].plot(gt[:samples, 4], color='green', alpha=0.5)
    axes[0, 4].set_xlabel("Samples")
    axes[0, 4].set_ylabel("Actuation")
    axes[0, 4].set_title("Module 5 Acts")
    axes[0, 4].legend(['Pred Data', 'Ground truth'])  


    axes[1, 0].plot(pred[:samples, 5], color='navy')
    axes[1, 0].plot(gt[:samples, 5], color='green', alpha=0.5)
    axes[1, 0].set_xlabel("Samples")
    axes[1, 0].set_ylabel("Actuation")
    axes[1, 0].set_title("Module 6 Acts")
    axes[1, 0].legend(['Pred Data', 'Ground truth'])  

    axes[1, 1].plot(pred[:samples, 6], color='navy')
    axes[1, 1].plot(gt[:samples,6], color='green', alpha=0.5)
    axes[1, 1].set_xlabel("Samples")
    axes[1, 1].set_ylabel("Actuation")
    axes[1, 1].set_title("Module 7 Acts")
    axes[1, 1].legend(['Pred Data', 'Ground truth'])  

    axes[1, 2].plot(pred[:samples, 7], color='navy')
    axes[1, 2].plot(gt[:samples, 7], color='green', alpha=0.5)
    axes[1, 2].set_xlabel("Samples")
    axes[1, 2].set_ylabel("Actuation")
    axes[1, 2].set_title("Module 8 Acts")
    axes[1, 2].legend(['Pred Data', 'Ground truth'])  

    axes[1, 3].plot(pred[:samples, 8], color='navy')
    axes[1, 3].plot(gt[:samples, 8], color='green', alpha=0.5)
    axes[1, 3].set_xlabel("Samples")
    axes[1, 3].set_ylabel("Actuation")
    axes[1, 3].set_title("Module 9 Acts")
    axes[1, 3].legend(['Pred Data', 'Ground truth'])  

    axes[1, 4].plot(pred[:samples, 9], color='navy')
    axes[1, 4].plot(gt[:samples, 9], color='green', alpha=0.5)
    axes[1, 4].set_xlabel("Samples")
    axes[1, 4].set_ylabel("Actuation")
    axes[1, 4].set_title("Module 10 Acts")
    axes[1, 4].legend(['Pred Data', 'Ground truth'])  

    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/actuation_plot_Mod{task_id}_totalSize{args.test_total_segments}_shape_{args.shape_type}.png")

def plot_XY(args, pred, gt, task_id, samples=100):

    plt.figure(figsize=(7, 5))
    plt.scatter(pred[:samples, 0], pred[:samples, 1], color='green', alpha=0.5)
    plt.scatter(gt[:samples, 0], gt[:samples, 1], color='navy', alpha=0.3)
    plt.xlabel("X-Pos")
    plt.ylabel("Y-Pos")
    plt.legend(['Pred Data', 'Ground truth'])         
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/XY_{args.shape_type}_modId{task_id}_totalSize{args.test_total_segments}.png")


def plot_XYZ(args, pred, gt, task_id, samples=100):

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pred[:samples, 0], pred[:samples, 1], pred[:samples, 2], c='green')
    ax.scatter(gt[:samples, 0], gt[:samples, 1], gt[:samples, 2], c='navy', alpha=0.5)
    plt.xlabel("X-Pos")
    plt.ylabel("Y-Pos")
    # plt.zlabel("Z-Pos")
    plt.legend(['Pred Data', 'Ground truth'])  
    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/XYZ_{args.shape_type}_taskId{task_id}.png")

def pos_plot(args, pred, gt, current_seg, samples=300):

    pred = np.array(pred)
    gt = np.array(gt)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13,5))

    axes[0].plot(pred[:samples, 0], color='navy')
    axes[0].plot(gt[:samples, 0], color='green', alpha=0.5)
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("X-coordinate")
    axes[0].legend(['Pred Data', 'Ground truth'])         

    axes[1].plot(pred[:samples, 1], color='navy')
    axes[1].plot(gt[:samples, 1], color='green', alpha=0.5)
    axes[1].set_xlabel("Samples")
    axes[1].set_ylabel("Y-coordinate")
    axes[1].legend(['Pred Data', 'Ground truth'])       

    axes[2].plot(pred[:samples, 2], color='navy')
    axes[2].plot(gt[:samples, 2], color='green', alpha=0.5)
    axes[2].set_xlabel("Samples")
    axes[2].set_ylabel("Z-coordinate")
    axes[2].legend(['Pred Data', 'Ground truth'])   
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/pos_error_TotalSeg_{args.test_total_segments}_shape_{args.shape_type}.png")



def orien_plot(args, pred, gt, current_seg, samples=300):

    pred = np.array(pred)
    gt = np.array(gt)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13,5))

    axes[0].plot(pred[:samples, 0], color='navy')
    axes[0].plot(gt[:samples, 0], color='green', alpha=0.5)
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("X-coordinate")
    axes[0].legend(['Pred Data', 'Ground truth'])         

    axes[1].plot(pred[:samples, 1], color='navy')
    axes[1].plot(gt[:samples, 1], color='green', alpha=0.5)
    axes[1].set_xlabel("Samples")
    axes[1].set_ylabel("Y-coordinate")
    axes[1].legend(['Pred Data', 'Ground truth'])       

    axes[2].plot(pred[:samples, 2], color='navy')
    axes[2].plot(gt[:samples, 2], color='green', alpha=0.5)
    axes[2].set_xlabel("Samples")
    axes[2].set_ylabel("Z-coordinate")
    axes[2].legend(['Pred Data', 'Ground truth'])   
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/orien_error_TotalSeg_{args.test_total_segments}_shape_{args.shape_type}.png")

def act_plot_custom(args, pred, gt, task_id, samples=100):
    # concatenate all the batches
    # pred = np.concatenate(pred)
    # gt = np.concatenate(gt)
    # select the 0th timestep 


    print("pred shape after slicing:", pred.shape, "gt shape after slicing :", gt.shape)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,5))

    axes[0].plot(pred[:samples, 0], color='navy')
    axes[0].plot(gt[:samples, 0], color='green', alpha=0.5)
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("Actuation")
    axes[0].legend(['Pred Data', 'Ground truth'])         

    axes[1].plot(pred[:samples, 1], color='navy')
    axes[1].plot(gt[:samples, 1], color='green', alpha=0.5)
    axes[1].set_xlabel("Samples")
    axes[1].set_ylabel("Actuation")
    axes[1].legend(['Pred Data', 'Ground truth'])       
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/actuation_{args.shape_type}_taskId{task_id}.png")


def loss_plot(args, train_loss, val_loss, task_id):
    plt.figure(figsize=(10,5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(f"results/loss_plot_{args.train_total_segments}_taskId{task_id}.png")


def load_forward_model(n_seg, current_task_id, use_orien, feature_dim, target_dim, device):
    model = LSTMModel_FM(input_size=feature_dim, hidden_size=64, num_layers=2, output_size=target_dim).to(device)
    model.load_state_dict(torch.load('forward_models/FM_seg' + str(current_task_id) +"_Mod"+  str(n_seg) +'.pt', map_location=device))
    return model

def orientationError(actual, pred):
    eps = 1e-7
    theta = np.arccos(np.sum(actual*pred, axis=1)/(np.linalg.norm(actual, \
                                    axis=1)*np.linalg.norm(pred, axis=1) + eps))
    return theta 

def task_space_inv_transform(data, current_seg, args):
    if current_seg == 1:
        scaler_mod1 = joblib.load("scalers_fm/1_mod/scaler_pos_seg1.pkl")
        trans_data = scaler_mod1.inverse_transform(data)
    elif current_seg == 2:
        scaler_mod2 = joblib.load("scalers_fm/2_mod/scaler_pos_seg2.pkl")
        trans_data = scaler_mod2.inverse_transform(data)
    elif current_seg == 3:
        scaler_mod3 = joblib.load("scalers_fm/3_mod/scaler_pos_seg3.pkl")
        trans_data = scaler_mod3.inverse_transform(data)
    elif current_seg == 4:
        scaler_mod4 = joblib.load("scalers_fm/4_mod/scaler_pos_seg4.pkl")
        trans_data = scaler_mod4.inverse_transform(data)
    elif current_seg == 5:
        scaler_mod5 = joblib.load("scalers_fm/5_mod/scaler_pos_seg5.pkl")
        trans_data = scaler_mod5.inverse_transform(data)

    return trans_data

def calculate_module_error(mod_pos, mod_ori, target_pos, target_ori):
    # module pose
    mod_pos = np.array(mod_pos).reshape(-1, 3)
    mod_ori = np.array(mod_ori).reshape(-1, 3)
    # target pose
    target_pos = np.array(target_pos).reshape(-1, 3)
    target_ori = np.array(target_ori).reshape(-1, 3)

    robot_curr_pose = torch.tensor(np.expand_dims(np.concatenate((mod_pos, mod_ori), axis=1), axis=0), dtype=torch.float32)
    target_pose = torch.tensor(np.expand_dims(np.concatenate((target_pos, target_ori), axis=1), axis=0), dtype=torch.float32) # (1, 10, 6)
    error_fb = torch.abs(target_pose - robot_curr_pose)
    return error_fb

# extract and rearrange the position and orientation data based on the predicted actuation from the inverse model
def extract_and_rearrange_sim_pred(pos_list, ori_list, seg_numb, args):
    ### Positions
    if seg_numb == 1:
        pos_list_new = pos_list[25, :, :].T
        ori_list_new = ori_list[0, :, 2, :].T
    elif seg_numb == 2:
        mod2_pos =pos_list[50, :, :].T
        pos_list_new = mod2_pos 
        ## orientation
        mod2_ori = ori_list[1, :, 2, :].T
        ori_list_new = mod2_ori
    elif seg_numb == 3:
        mod3_pos = pos_list[75, :, :].T
        pos_list_new = mod3_pos
        ## orientation
        mod3_ori = ori_list[2, :, 2, :].T
        ori_list_new = mod3_ori
    elif seg_numb == 4:
        mod4_pos = pos_list[100, :, :].T
        pos_list_new = mod4_pos
        ## orientation
        mod4_ori = ori_list[3, :, 2, :].T
        ori_list_new = mod4_ori
    elif seg_numb == 5:
        mod5_pos = pos_list[125, :, :].T
        pos_list_new = mod5_pos
        ## orientation
        mod5_ori = ori_list[4, :, 2, :].T
        ori_list_new = mod5_ori

    print("pos_list shape is ", pos_list_new.shape, "ori_list shape is ", ori_list_new.shape)

    return pos_list_new, ori_list_new


# ## Function to draw a circular shape in the task space
# def reconstruct_modules(num_points, P1, P2, P3):
#     module1 = np.zeros((num_points, 3, P1.shape[0]))
#     module2 = np.zeros((num_points, 3, P1.shape[0]))
#     for i in range(P1.shape[0]):
#         # --------------- End-effector 01 Reconstruction ----------------- #
#         t = np.transpose([0, 0, -1])
#         v = np.transpose(P2[i, :]) - np.transpose(P1[i, :])
#         b = np.cross(t, v) / np.linalg.norm(np.cross(t, v))
#         n = np.cross(b, t) / np.linalg.norm(np.cross(b, t))
#         R = np.dot(v, v) / (2 * np.dot(v, n))
#         c = np.transpose(P1[i, :]) + R * n
#         dotp = np.dot((np.transpose(P1[i, :]) - c), (np.transpose(P2[i, :]) - c))
#         theta = math.acos(dotp / (R ** 2))
#         ang = np.linspace(0, theta, num_points)
#         repmat_factor = mb.repmat(c, 1, len(ang))
#         repmat_factor = np.asarray([repmat_factor]).reshape(len(ang), 3)
#         ang_factor = np.zeros((len(ang), 3))
#         for j in range(len(ang)):
#             ang_factor[j, :] = R * ((-n * math.cos(ang[j])) + (t * math.sin(ang[j])))
#         point1 = repmat_factor + ang_factor  # 100 X 3
#         # print('dimension of the point1: ', point1.shape)
#         module1[:, :, i] = point1

#         # --------------- End-effector 02 Reconstruction----------------- #
#         t = (n * math.sin(theta)) + (t * math.cos(theta))
#         v = np.transpose(P3[i, :]) - np.transpose(P2[i, :])
#         b = np.cross(t, v) / np.linalg.norm(np.cross(t, v))
#         n = np.cross(b, t) / np.linalg.norm(np.cross(b, t))
#         R = np.dot(v, v) / (2 * np.dot(v, n))
#         c = np.transpose(P2[i, :]) + R * n
#         dotp = np.dot((np.transpose(P2[i, :]) - c), (np.transpose(P3[i, :]) - c))
#         theta = math.acos(dotp / (R ** 2))
#         ang = np.linspace(0, theta, num_points)
#         repmat_factor = mb.repmat(c, 1, len(ang))
#         repmat_factor = np.asarray([repmat_factor]).reshape(len(ang), 3)
#         ang_factor = np.zeros((len(ang), 3))
#         for j in range(len(ang)):
#             ang_factor[j, :] = R * ((-n * math.cos(ang[j])) + (t * math.sin(ang[j])))
#         point2 = repmat_factor + ang_factor
#         module2[:, :, i] = point2

#     # # Cell to save reconstructed shape at each ith instant,   robot_equidistant_point x 3 x trial_size
#     print('dimension of reconstructed modules: ', module1.shape)

#     return module1, module2