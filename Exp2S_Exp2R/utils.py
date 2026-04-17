import pickle
import joblib
import torch
import numpy as np
from pyhelpers.store import save_fig
from model import LSTMModel
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
    with open(f'arrays/mask_{filename}.pkl', 'wb') as fp:
        pickle.dump(data, fp)
        print('dictionary saved successfully to file')

def load_dictionary(filename):
    with open(filename, 'rb') as fp:
        interpolation_steps = pickle.load(fp)
    return interpolation_steps

def act_plot(args, pred, gt, task_id, samples=100):
    # pred = pred[:, -1, 0:3] # first time step and end effector position
    # gt = gt[:, -1, 0:3]

    color_map = ["red", "green", "blue"]
    alpha_map = [0.4, 0.7, 0.5]
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15,5))

    axes[0].plot(pred[:samples, 0], color=color_map[task_id], linewidth=1.5)
    axes[0].plot(gt[:samples, 0], color=color_map[task_id], alpha=0.7, linewidth= 2.5, linestyle="--")
    axes[0].set_xlabel("Samples", fontsize=14)
    axes[0].set_ylabel("Normalized Actuation", fontsize=14)
    axes[0].grid(linestyle='--', alpha=0.6)
    axes[0].legend(['Model prediction', 'Ground truth'], fontsize=12)   
    axes[0].yaxis.set_tick_params(labelsize=11)      

    axes[1].plot(pred[:samples, 1], color=color_map[task_id], linewidth=1.5)
    axes[1].plot(gt[:samples, 1], color=color_map[task_id], alpha=0.7, linewidth= 2.5, linestyle="--")
    axes[1].set_xlabel("Samples", fontsize=14)
    axes[1].grid(linestyle='--', alpha=0.6)
    axes[1].yaxis.set_tick_params(labelsize=11)
    # axes[1].set_ylabel("Actuation")
    # axes[1].legend(['Pred Data', 'Ground truth'])    

    axes[2].plot(pred[:samples, 2], color=color_map[task_id], linewidth=1.5)
    axes[2].plot(gt[:samples, 2], color=color_map[task_id], alpha=0.7, linewidth= 2.5, linestyle="--")
    axes[2].set_xlabel("Samples", fontsize=14)
    axes[2].grid(linestyle='--', alpha=0.6)
    axes[2].yaxis.set_tick_params(labelsize=11)
    # axes[2].set_ylabel("Actuation")
    # axes[2].legend(['Pred Data', 'Ground truth'])      

    axes[3].plot(pred[:samples, 3], color=color_map[task_id], linewidth=1.5)
    axes[3].plot(gt[:samples, 3], color=color_map[task_id], alpha=0.7, linewidth= 2.5, linestyle="--")
    axes[3].set_xlabel("Samples", fontsize=14)
    axes[3].grid(linestyle='--', alpha=0.6)
    # axes[3].set_ylabel("Actuation")
    # axes[3].legend(['Pred Data', 'Ground truth'])    
    axes[3].yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    save_fig(f"results/actuation_plot_{args.n_seg}_taskId{task_id}.svg", dpi=300, conv_svg_to_emf=True, verbose=True)
    plt.show()
    plt.savefig(f"results/actuation_plot_{args.n_seg}_taskId{task_id}.png")

def plot_XY(args, pred, gt, task_id, samples=100):
    # pred = pred[:, -1, 0:3] # first time step and end effector position
    # gt = gt[:, -1, 0:3]
    plt.figure(figsize=(7, 5))

    plt.scatter(pred[:samples, 0], pred[:samples, 2], color='green', alpha=0.5)
    plt.scatter(gt[:samples, 0], gt[:samples, 2], color='navy', alpha=0.3)
    plt.xlabel("X-Pos")
    plt.ylabel("Y-Pos")
    plt.legend(['Pred Data', 'Ground truth'])         
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/XY_{args.shape_type}_taskId{task_id}.png")


def plot_XYZ(args, pred, gt, task_id, samples=100):
    # pred = pred[:, -1, 0:3] # first time step and end effector position
    # gt = gt[:, -1, 0:3]

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

def TSTime(args, pred, gt, task_id, samples=100):
    # pred = pred[:, -1, 0:3] # first time step and end effector position
    # gt = gt[:, -1, 0:3]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,5))
    axes[0].plot(pred[:samples, 0], color="green")
    axes[0].plot(gt[:samples, 0], color="navy", linestyle='--')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("End Effector X-axis")
    axes[0].grid()
    axes[0].set_title("Prediction vs Target of End Effector")
    axes[0].legend(["Prediction", "Original"])

    axes[1].plot(pred[:samples, 2], color="green")
    axes[1].plot(gt[:samples, 2], color="navy", linestyle='--')
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("End Effector Y-axis")
    axes[1].grid()

    # axes[2].plot(pred[:, 2], color="green")
    # axes[2].plot(gt[:, 2], color="navy")
    # axes[2].set_xlabel("Time")
    # axes[2].set_ylabel("End Effector Z-axis")
    # axes[2].grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/XYZ_time_{args.shape_type}_taskId{task_id}.png")
    plt.clf() 

def act_plot_custom(args, pred, gt, task_id, samples=100):
    # concatenate all the batches
    # pred = np.concatenate(pred)
    # gt = np.concatenate(gt)
    # select the 0th timestep 
    print("pred shape before slicing:", pred.shape, "gt shape before slicing :", gt.shape)
    pred = pred[:, 0, :]
    gt = gt[:, 0, :]

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
    plt.savefig(f"results/loss_plot_{args.n_seg}_taskId{task_id}.png")


def load_forward_model(n_seg, use_orien, feature_dim, target_dim, device):
    model = LSTMModel(input_size=feature_dim, hidden_size=64, num_layers=2, output_size=target_dim).to(device)
    model.load_state_dict(torch.load('forward_models/model_forward_' + str(n_seg+1) +"_orien_"+  str(use_orien) +'.pt', map_location=device))
    return model

def orientationError(actual, pred):
    eps = 1e-7
    theta = np.arccos(np.sum(actual*pred, axis=1)/(np.linalg.norm(actual, \
                                    axis=1)*np.linalg.norm(pred, axis=1) + eps))
    return theta 

def task_space_inv_transform(data, task_id):
    if task_id == 0: 
        scaler_mod1 = joblib.load("scalers/scaler_pos_mod1.pkl")
        trans_data = scaler_mod1.inverse_transform(data)
    elif task_id == 1:
        scaler_mod2 = joblib.load("scalers/scaler_pos_mod2.pkl")
        trans_data = scaler_mod2.inverse_transform(data)
    elif task_id == 2:  
        scaler_mod3 = joblib.load("scalers/scaler_pos_mod3.pkl")
        trans_data = scaler_mod3.inverse_transform(data)
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
