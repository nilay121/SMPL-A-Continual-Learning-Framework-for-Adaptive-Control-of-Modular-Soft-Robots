import joblib
import pickle
import joblib
import numpy as np
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
        print('dictionary saved successfully to file')

def load_dictionary(filename):
    with open(filename, 'rb') as fp:
        interpolation_steps = pickle.load(fp)
    return interpolation_steps

def task_plot(args, pred, gt, current_seg, samples=100):

    pred = np.array(pred)
    gt = np.array(gt)
    # pred = pred.reshape(-1, 6)
    # gt = gt.reshape(-1, 6)
    # pred = pred[:, -1, :]
    # gt = gt[:, -1, :]

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
    plt.savefig(f"results/pos_error_seg{current_seg}_{args.n_seg}_mod.png")



def task_plot_orien(args, pred, gt, current_seg, samples=100):

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
    plt.savefig(f"results/orien_error_seg{current_seg}_{args.n_seg}_mod.png")


def loss_plot(args, train_loss, val_loss, current_seg):
    plt.figure(figsize=(10,5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(f"results/loss_plot_seg{current_seg}_{args.n_seg}_mod.png")


def task_space_inv_transform(data, current_seg, args):
    if args.n_seg == 1:
        if current_seg == 1:
            scaler_mod1 = joblib.load("../scalers_fm/1_mod/scaler_pos_seg1.pkl")
            trans_data = scaler_mod1.inverse_transform(data)
    elif args.n_seg == 2:
        if current_seg == 1:
            scaler_mod1 = joblib.load("../scalers_fm/2_mod/scaler_pos_seg1.pkl")
            trans_data = scaler_mod1.inverse_transform(data)
        elif current_seg == 2:
            scaler_mod2 = joblib.load("../scalers_fm/2_mod/scaler_pos_seg2.pkl")
            trans_data = scaler_mod2.inverse_transform(data)
    elif args.n_seg == 3:
        if current_seg == 1:
            scaler_mod1 = joblib.load("../scalers_fm/3_mod/scaler_pos_seg1.pkl")
            trans_data = scaler_mod1.inverse_transform(data)
        elif current_seg == 2:
            scaler_mod2 = joblib.load("../scalers_fm/3_mod/scaler_pos_seg2.pkl")
            trans_data = scaler_mod2.inverse_transform(data)
        elif current_seg == 3:
            scaler_mod3 = joblib.load("../scalers_fm/3_mod/scaler_pos_seg3.pkl")
            trans_data = scaler_mod3.inverse_transform(data)
    elif args.n_seg == 4:
        if current_seg == 1:
            scaler_mod1 = joblib.load("../scalers_fm/4_mod/scaler_pos_seg1.pkl")
            trans_data = scaler_mod1.inverse_transform(data)
        elif current_seg == 2:
            scaler_mod2 = joblib.load("../scalers_fm/4_mod/scaler_pos_seg2.pkl")
            trans_data = scaler_mod2.inverse_transform(data)
        elif current_seg == 3:
            scaler_mod3 = joblib.load("../scalers_fm/4_mod/scaler_pos_seg3.pkl")
            trans_data = scaler_mod3.inverse_transform(data)
        elif current_seg == 4:
            scaler_mod4 = joblib.load("../scalers_fm/4_mod/scaler_pos_seg4.pkl")
            trans_data = scaler_mod4.inverse_transform(data)
    elif args.n_seg == 5:
        if current_seg == 1:
            scaler_mod1 = joblib.load("../scalers_fm/5_mod/scaler_pos_seg1.pkl")
            trans_data = scaler_mod1.inverse_transform(data)
        elif current_seg == 2:
            scaler_mod2 = joblib.load("../scalers_fm/5_mod/scaler_pos_seg2.pkl")
            trans_data = scaler_mod2.inverse_transform(data)
        elif current_seg == 3:
            scaler_mod3 = joblib.load("../scalers_fm/5_mod/scaler_pos_seg3.pkl")
            trans_data = scaler_mod3.inverse_transform(data)
        elif current_seg == 4:
            scaler_mod4 = joblib.load("../scalers_fm/5_mod/scaler_pos_seg4.pkl")
            trans_data = scaler_mod4.inverse_transform(data)
        elif current_seg == 5:
            scaler_mod5 = joblib.load("../scalers_fm/5_mod/scaler_pos_seg5.pkl")
            trans_data = scaler_mod5.inverse_transform(data)

    return trans_data