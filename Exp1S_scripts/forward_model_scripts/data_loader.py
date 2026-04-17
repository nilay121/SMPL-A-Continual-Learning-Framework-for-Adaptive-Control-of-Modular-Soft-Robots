import torch
import os
import sys
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import save_dictionary, load_dictionary


def normalize_position_train(pos_list, act_list, seg_numb, args):
    if args.n_seg == 1:
        mod1_pos = pos_list[:, :, 0]

        # mod 1 train --> position
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/1_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        pos_list_new = pos_mod1_train

        # save the normalizer
        save_dictionary(scaler_pos_mod1, "../scalers_fm/1_mod/scaler_pos_seg1.pkl")

    elif args.n_seg == 2:
        mod1_pos = pos_list[:, :, 0]
        mod2_pos = pos_list[:, :, 1]

        ## Position Normalization
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/2_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        # mod 2 train
        scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/2_mod/scaler_pos_seg2.pkl")
        pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
        #(N, feature_dim, module_numb)
        pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train], axis=2)

        # save the position normalizer
        save_dictionary(scaler_pos_mod1, "../scalers_fm/2_mod/scaler_pos_seg1.pkl")
        save_dictionary(scaler_pos_mod2, "../scalers_fm/2_mod/scaler_pos_seg2.pkl")

    elif args.n_seg == 3: ## play with this one
        mod1_pos = pos_list[:, :, 0]
        mod2_pos = pos_list[:, :, 1]
        mod3_pos = pos_list[:, :, 2]

        ## Position Normalization
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/3_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        # mod 2 train
        scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/3_mod/scaler_pos_seg2.pkl")
        pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
        # mod 3 train
        scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/3_mod/scaler_pos_seg3.pkl")
        pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
        #(N, feature_dim, module_numb) this is the possibility to know the setiing
        pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train], axis=2)
               
        # save the position normalizer
        save_dictionary(scaler_pos_mod1, "../scalers_fm/3_mod/scaler_pos_seg1.pkl")
        save_dictionary(scaler_pos_mod2, "../scalers_fm/3_mod/scaler_pos_seg2.pkl")       
        save_dictionary(scaler_pos_mod3, "../scalers_fm/3_mod/scaler_pos_seg3.pkl")  

    elif args.n_seg == 4:
        mod1_pos = pos_list[:, :, 0]
        mod2_pos = pos_list[:, :, 1]
        mod3_pos = pos_list[:, :, 2]
        mod4_pos = pos_list[:, :, 3]

        ## Position Normalization
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/4_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        # mod 2 train
        scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/4_mod/scaler_pos_seg2.pkl")
        pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
        # mod 3 train
        scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/4_mod/scaler_pos_seg3.pkl")
        pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
        # mod 4 train
        scaler_pos_mod4 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/4_mod/scaler_pos_seg4.pkl")
        pos_mod4_train = np.expand_dims(scaler_pos_mod4.fit_transform(mod4_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod4.transform(mod4_pos), axis=2)
        #(N, feature_dim, module_numb)
        pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train, pos_mod4_train], axis=2)

        # save the position normalizer
        save_dictionary(scaler_pos_mod1, "../scalers_fm/4_mod/scaler_pos_seg1.pkl")
        save_dictionary(scaler_pos_mod2, "../scalers_fm/4_mod/scaler_pos_seg2.pkl")       
        save_dictionary(scaler_pos_mod3, "../scalers_fm/4_mod/scaler_pos_seg3.pkl")  
        save_dictionary(scaler_pos_mod4, "../scalers_fm/4_mod/scaler_pos_seg4.pkl")  

    elif args.n_seg == 5:
        mod1_pos = pos_list[:, :, 0]
        mod2_pos = pos_list[:, :, 1]
        mod3_pos = pos_list[:, :, 2]
        mod4_pos = pos_list[:, :, 3]
        mod5_pos = pos_list[:, :, 4]

        ## Position Normalization
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/5_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        # mod 2 train
        scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/5_mod/scaler_pos_seg2.pkl")
        pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
        # mod 3 train
        scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/5_mod/scaler_pos_seg3.pkl")
        pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
        # mod 4 train
        scaler_pos_mod4 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/5_mod/scaler_pos_seg4.pkl")
        pos_mod4_train = np.expand_dims(scaler_pos_mod4.fit_transform(mod4_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod4.transform(mod4_pos), axis=2)
        # mod 5 train
        scaler_pos_mod5 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("../scalers_fm/5_mod/scaler_pos_seg5.pkl")
        pos_mod5_train = np.expand_dims(scaler_pos_mod5.fit_transform(mod5_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod5.transform(mod5_pos), axis=2)
        #(N, feature_dim, module_numb)
        pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train, pos_mod4_train, pos_mod5_train], axis=2)

        # save the normalizer
        save_dictionary(scaler_pos_mod1, "../scalers_fm/5_mod/scaler_pos_seg1.pkl")
        save_dictionary(scaler_pos_mod2, "../scalers_fm/5_mod/scaler_pos_seg2.pkl")       
        save_dictionary(scaler_pos_mod3, "../scalers_fm/5_mod/scaler_pos_seg3.pkl")  
        save_dictionary(scaler_pos_mod4, "../scalers_fm/5_mod/scaler_pos_seg4.pkl")  
        save_dictionary(scaler_pos_mod5, "../scalers_fm/5_mod/scaler_pos_seg5.pkl")  

    return pos_list_new

# Load the data
def load_data(args, feature_dim, target_act_dim, current_seg):
    time_step = args.time_step
    use_orien = args.use_orien
    n_seg = args.n_seg
    print("n_seg is ", n_seg)

    data = np.load( "../../../Data_generation_Initial_mods/new_rev_dataset/" + str(args.mode) + "/" + f"{args.n_seg}_mod/" + "babbling_data_combined" + '.npz')

    pos_list = data['pos_list']
    ori_list = data['ori_list']
    act_list = data['act_list']

    print("pos_list shape:" + str(pos_list.shape))
    print("ori_list shape:" + str(ori_list.shape))
    print("act_list shape:" + str(act_list.shape))

    ### Positions
    if n_seg == 1:
        pos_list_new = np.expand_dims(pos_list[25, :, :].T, axis=2)
        ori_list_new = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
    elif n_seg == 2:
        mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
        mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
        pos_list_new = np.concatenate([mod1_pos, mod2_pos], axis=2) #(N, feature_dim, module_numb)
        ## orientation
        mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
        mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)

        ori_list_new = np.concatenate([mod1_ori, mod2_ori], axis=2) #(N, feature_dim, module_numb)
    elif n_seg == 3:
        mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
        mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
        mod3_pos = np.expand_dims(pos_list[75, :, :].T, axis=2)
        pos_list_new = np.concatenate([mod1_pos, mod2_pos, mod3_pos], axis=2) #(N, feature_dim, module_numb)

        ## orientation
        mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
        mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)
        mod3_ori = np.expand_dims(ori_list[2, :, 2, :].T, axis=2)

        ori_list_new = np.concatenate([mod1_ori, mod2_ori, mod3_ori], axis=2) #(N, feature_dim, module_numb)

    elif n_seg == 4:
        mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
        mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
        mod3_pos = np.expand_dims(pos_list[75, :, :].T, axis=2)
        mod4_pos = np.expand_dims(pos_list[100, :, :].T, axis=2)
        pos_list_new = np.concatenate([mod1_pos, mod2_pos, mod3_pos, mod4_pos], axis=2) #(N, feature_dim, module_numb)

        ## orientation
        mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
        mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)
        mod3_ori = np.expand_dims(ori_list[2, :, 2, :].T, axis=2)
        mod4_ori = np.expand_dims(ori_list[3, :, 2, :].T, axis=2)

        ori_list_new = np.concatenate([mod1_ori, mod2_ori, mod3_ori, mod4_ori], axis=2) #(N, feature_dim, module_numb)

    elif n_seg == 5:
        mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
        mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
        mod3_pos = np.expand_dims(pos_list[75, :, :].T, axis=2)
        mod4_pos = np.expand_dims(pos_list[100, :, :].T, axis=2)
        mod5_pos = np.expand_dims(pos_list[125, :, :].T, axis=2)
        pos_list_new = np.concatenate([mod1_pos, mod2_pos, mod3_pos, mod4_pos, mod5_pos], axis=2) #(N, feature_dim, module_numb)

        ## orientation
        mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
        mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)
        mod3_ori = np.expand_dims(ori_list[2, :, 2, :].T, axis=2)
        mod4_ori = np.expand_dims(ori_list[3, :, 2, :].T, axis=2)
        mod5_ori = np.expand_dims(ori_list[4, :, 2, :].T, axis=2)

        ori_list_new = np.concatenate([mod1_ori, mod2_ori, mod3_ori, mod4_ori, mod5_ori], axis=2) #(N, feature_dim, module_numb)
    
    # ## Normalize the position for each module
    pos_list_new = normalize_position_train(pos_list=pos_list_new, act_list=act_list, seg_numb=n_seg, args=args)

    # # clip in-place to strictly enforce scaler feature_range and avoid small FP overflow
    np.clip(pos_list_new, -1.0, 1.0, out=pos_list_new)

    print("Data has been normalized !!", float(pos_list_new.max()), float(pos_list_new.min()))

    pos_list = pos_list_new
    ori_list = ori_list_new

    X = np.zeros((len(act_list) - time_step - 1, time_step, 12), dtype=np.float32)
    Y = np.zeros((len(act_list) - time_step - 1, time_step, target_act_dim), dtype=np.float32)

    if use_orien=="true":
        for i in range(len(act_list) - time_step - 1):
            for j in range(time_step):
                for k in range(current_seg):
                    # INPUT: current state + action

                    ## includes 14 state values
                    if current_seg == 1:
                        X[i, j, 0:2] = act_list[i + j + 1, k]  # a_1
                    elif current_seg == 2:
                        if k == 0:
                            X[i, j, 0:2] = act_list[i + j + 1, k]
                        elif k == 1:
                            X[i, j, 2:4] = act_list[i + j + 1, k]
                    elif current_seg ==3:
                        if k == 0:
                            X[i, j, 0:2] = act_list[i + j + 1, k]
                        elif k == 1:
                            X[i, j, 2:4] = act_list[i + j + 1, k]
                        elif k == 2:
                            X[i, j, 4:6] = act_list[i + j + 1, k]
                    elif current_seg ==4:
                        if k == 0:
                            X[i, j, 0:2] = act_list[i + j + 1, k]
                        elif k == 1:
                            X[i, j, 2:4] = act_list[i + j + 1, k]
                        elif k == 2:
                            X[i, j, 4:6] = act_list[i + j + 1, k]
                        elif k == 3:
                            X[i, j, 6:8] = act_list[i + j + 1, k]
                    elif current_seg ==5:
                        if k == 0:
                            X[i, j, 0:2] = act_list[i + j + 1, k]
                        elif k == 1:
                            X[i, j, 2:4] = act_list[i + j + 1, k]
                        elif k == 2:
                            X[i, j, 4:6] = act_list[i + j + 1, k]
                        elif k == 3:
                            X[i, j, 6:8] = act_list[i + j + 1, k]
                        elif k == 4:
                            X[i, j, 8:10] = act_list[i + j + 1, k]

                # OUTPUT: next next state
                # t=2
                Y[i, j, 0:3] = pos_list[i + j + 2, :, current_seg-1]      # t+1
                Y[i, j, 3:6] = ori_list[i + j + 2, :, current_seg-1] # t+1

    else:
        print("Not executed!!")

    # PyTorch tensor
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return X, Y



def load_data_merged(args, feature_dim, target_act_dim, current_seg):
    if args.mode == "train" or args.mode == "test":
        # train data --> babbling plus some other standard data
        X_bab, Y_bab = load_data(args, feature_dim, target_act_dim, current_seg=current_seg)

        X = X_bab
        Y = Y_bab
        
        if args.mode == "train":
            indices = torch.randperm(X.shape[0])
            X = X[indices]
            Y = Y[indices]
    
    elif args.mode == "test_custom":
        print("Not there!!")

    return X, Y

