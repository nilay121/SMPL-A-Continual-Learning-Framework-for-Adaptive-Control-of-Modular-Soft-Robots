import torch
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from utils import save_dictionary, load_dictionary

def extract_and_rearrange(numpy_file, seg_numb, args):
    pos_list = numpy_file["pos_list"]
    act_list = numpy_file["act_list"]
    ori_list = numpy_file["ori_list"]

    ### Positions
    if seg_numb == 1:
        pos_list_new = np.expand_dims(pos_list[25, :, :].T, axis=2)
        ori_list_new = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
    elif seg_numb == 2:
        mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
        mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
        pos_list_new = np.concatenate([mod1_pos, mod2_pos], axis=2) #(N, feature_dim, module_numb)

        ## orientation
        mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
        mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)

        ori_list_new = np.concatenate([mod1_ori, mod2_ori], axis=2) #(N, feature_dim, module_numb)
    elif seg_numb == 3:
        mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
        mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
        mod3_pos = np.expand_dims(pos_list[75, :, :].T, axis=2)
        pos_list_new = np.concatenate([mod1_pos, mod2_pos, mod3_pos], axis=2) #(N, feature_dim, module_numb)

        ## orientation
        mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
        mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)
        mod3_ori = np.expand_dims(ori_list[2, :, 2, :].T, axis=2)

        ori_list_new = np.concatenate([mod1_ori, mod2_ori, mod3_ori], axis=2) #(N, feature_dim, module_numb)

    elif seg_numb == 4:
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

    elif seg_numb == 5:
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
    
    ## Normalize the position for each module
    pos_list_new = normalize_position_train(pos_list=pos_list_new, seg_numb=seg_numb, args=args)
    # # clip in-place to strictly enforce scaler feature_range and avoid small FP overflow
    np.clip(pos_list_new, -1.0, 1.0, out=pos_list_new)

    print("Data has been normalized !!", float(pos_list_new.max()), float(pos_list_new.min()))

    return act_list, pos_list_new, ori_list_new


def normalize_position_train(pos_list, seg_numb, args):
    if seg_numb == 1:
        mod1_pos = pos_list[:, :, 0]
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/1_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        pos_list_new = pos_mod1_train
        # save the normalizer
        save_dictionary(scaler_pos_mod1, "scalers/1_mod/scaler_pos_seg1.pkl")

    elif seg_numb == 2:
        mod1_pos = pos_list[:, :, 0]
        mod2_pos = pos_list[:, :, 1]
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/2_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        # mod 2 train
        scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/2_mod/scaler_pos_seg2.pkl")
        pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
        #(N, feature_dim, module_numb)
        pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train], axis=2)
        # save the normalizer
        save_dictionary(scaler_pos_mod1, "scalers/2_mod/scaler_pos_seg1.pkl")
        save_dictionary(scaler_pos_mod2, "scalers/2_mod/scaler_pos_seg2.pkl")

    elif seg_numb == 3:
        mod1_pos = pos_list[:, :, 0]
        mod2_pos = pos_list[:, :, 1]
        mod3_pos = pos_list[:, :, 2]
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/3_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        # mod 2 train
        scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/3_mod/scaler_pos_seg2.pkl")
        pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
        # mod 3 train
        scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/3_mod/scaler_pos_seg3.pkl")
        pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
        #(N, feature_dim, module_numb)
        pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train], axis=2)
        # save the normalizer
        save_dictionary(scaler_pos_mod1, "scalers/3_mod/scaler_pos_seg1.pkl")
        save_dictionary(scaler_pos_mod2, "scalers/3_mod/scaler_pos_seg2.pkl")       
        save_dictionary(scaler_pos_mod3, "scalers/3_mod/scaler_pos_seg3.pkl")  


    elif seg_numb == 4:
        mod1_pos = pos_list[:, :, 0]
        mod2_pos = pos_list[:, :, 1]
        mod3_pos = pos_list[:, :, 2]
        mod4_pos = pos_list[:, :, 3]
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/4_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        # mod 2 train
        scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/4_mod/scaler_pos_seg2.pkl")
        pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
        # mod 3 train
        scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/4_mod/scaler_pos_seg3.pkl")
        pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
        # mod 4 train
        scaler_pos_mod4 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/4_mod/scaler_pos_seg4.pkl")
        pos_mod4_train = np.expand_dims(scaler_pos_mod4.fit_transform(mod4_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod4.transform(mod4_pos), axis=2)
        #(N, feature_dim, module_numb)
        pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train, pos_mod4_train], axis=2)

        # save the normalizer
        save_dictionary(scaler_pos_mod1, "scalers/4_mod/scaler_pos_seg1.pkl")
        save_dictionary(scaler_pos_mod2, "scalers/4_mod/scaler_pos_seg2.pkl")       
        save_dictionary(scaler_pos_mod3, "scalers/4_mod/scaler_pos_seg3.pkl")  
        save_dictionary(scaler_pos_mod4, "scalers/4_mod/scaler_pos_seg4.pkl")  

    elif seg_numb == 5:
        mod1_pos = pos_list[:, :, 0]
        mod2_pos = pos_list[:, :, 1]
        mod3_pos = pos_list[:, :, 2]
        mod4_pos = pos_list[:, :, 3]
        mod5_pos = pos_list[:, :, 4]
        # mod 1 train
        scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg1.pkl")
        pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
        # mod 2 train
        scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg2.pkl")
        pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
        # mod 3 train
        scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg3.pkl")
        pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
        # mod 4 train
        scaler_pos_mod4 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg4.pkl")
        pos_mod4_train = np.expand_dims(scaler_pos_mod4.fit_transform(mod4_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod4.transform(mod4_pos), axis=2)
        # mod 5 train
        scaler_pos_mod5 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg5.pkl")
        pos_mod5_train = np.expand_dims(scaler_pos_mod5.fit_transform(mod5_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod5.transform(mod5_pos), axis=2)
        #(N, feature_dim, module_numb)
        pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train, pos_mod4_train, pos_mod5_train], axis=2)
        
        # save the normalizer
        save_dictionary(scaler_pos_mod1, "scalers/5_mod/scaler_pos_seg1.pkl")
        save_dictionary(scaler_pos_mod2, "scalers/5_mod/scaler_pos_seg2.pkl")       
        save_dictionary(scaler_pos_mod3, "scalers/5_mod/scaler_pos_seg3.pkl")  
        save_dictionary(scaler_pos_mod4, "scalers/5_mod/scaler_pos_seg4.pkl")  
        save_dictionary(scaler_pos_mod5, "scalers/5_mod/scaler_pos_seg5.pkl")  

    return pos_list_new


# Load the data
def load_data(args, feature_dim, target_act_dim, mode, seg_numb):
    time_step = args.time_step
    use_orien = args.use_orien
    # n_seg = args.n_seg # we keep n_seg here fixed so that we can have the data for all the segments when used with 3 modules
    if args.shape_type == "babbling" and args.mode == "train":
        data = np.load("../../Data_generation_Initial_mods/new_rev_dataset/" + str(mode) + "/" + f"{seg_numb}_mod/" + str(args.shape_type)+"_data_combined" + '.npz')
    else:
        data = np.load("../../Data_generation_Initial_mods/new_rev_dataset/" + str(mode) + "/" + f"{seg_numb}_mod/" + str(args.shape_type)+"_data" + '.npz')

    act_list, pos_list, ori_list = extract_and_rearrange(numpy_file=data, seg_numb=seg_numb, args=args)

    ## Pre-processing & Normalization
    X = np.zeros((len(act_list) - time_step - 1, seg_numb, time_step, feature_dim+12), dtype=np.float32)
    Y = np.zeros((len(act_list) - time_step - 1, seg_numb, time_step, target_act_dim), dtype=np.float32)
    
    if use_orien == "true":
        for k in range(seg_numb):
            for i in range(len(act_list) - time_step - 1):
                for j in range(time_step):
                    """
                    The lag of +2 is used such the model takes as input the the expected target position and orintation,
                    the previous actutaion and predicts the next actutaion for the desired target position and orientation.
                    
                    s2, a0 -------> a1
                    s3, a1 -------> a2
                    ....
                    ....
                    ....
                    """
                    # s2, a0 informations

                    X[i, k, j, 0:3] = pos_list[i + j + 2, :, k] # [ee number, xyz, ctrl_step]
                    X[i, k, j, 3:6] = ori_list[i + j + 2, :, k]  # [ee number, xyz, ctrl_step]
                    # X[i, k, j, 6:10] = act_list[i + j, k]  # a_0  #[ctrl_step, seg_number] --> gives two actutations at time step "t"
                    
                    # S0 information
                    X[i, k, j, 6:9] = pos_list[i + j, :, k]
                    X[i, k, j, 9:12] = ori_list[i + j, :, k]

                    # S1 information
                    X[i, k, j, 12:15] = pos_list[i + j + 1, :, k]
                    X[i, k, j, 15:18] = ori_list[i + j + 1, :, k]
                    
                    # Target information
                    Y[i, k, j, 0:2] = act_list[i + j + 1, k]             
    else:
        raise NotImplementedError("Requires orientation data to be true")

    # PyTorch tensor
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y


def load_data_merged(args, feature_dim, target_act_dim, mode, seg_numb):
    if args.mode == "train" or args.mode == "test":

        X_bab, Y_bab = load_data(args, feature_dim, target_act_dim, mode, seg_numb)

        X = X_bab
        Y = Y_bab

        if args.mode == "train":
            indices = torch.randperm(X.shape[0])
            X = X[indices]
            Y = Y[indices]

    # elif args.mode == "test_custom":
    #     if args.shape_type == "circle":
    #         X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_circle_t100")
    #     elif args.shape_type == "rect":
    #         X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_rect_t100")
    #     elif args.shape_type == "spiral":
    #         X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_spiral_t150")
    #     elif args.shape_type == "test_babb":
    #         X, Y = load_data(args, feature_dim, target_act_dim, filepath="babbling_test_samples_1k_freq_5.0")

    return X.detach().numpy(), Y.detach().numpy()




















# import torch
# import joblib
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler 
# from utils import save_dictionary, load_dictionary

# def extract_and_rearrange(numpy_file, seg_numb, args):
#     pos_list = numpy_file["pos_list"]
#     act_list = numpy_file["act_list"]
#     ori_list = numpy_file["ori_list"]

#     ### Positions
#     if seg_numb == 1:
#         pos_list_new = np.expand_dims(pos_list[25, :, :].T, axis=2)
#         ori_list_new = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
#     elif seg_numb == 2:
#         mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
#         mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
#         pos_list_new = np.concatenate([mod1_pos, mod2_pos], axis=2) #(N, feature_dim, module_numb)

#         ## orientation
#         mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
#         mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)

#         ori_list_new = np.concatenate([mod1_ori, mod2_ori], axis=2) #(N, feature_dim, module_numb)
#     elif seg_numb == 3:
#         mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
#         mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
#         mod3_pos = np.expand_dims(pos_list[75, :, :].T, axis=2)
#         pos_list_new = np.concatenate([mod1_pos, mod2_pos, mod3_pos], axis=2) #(N, feature_dim, module_numb)

#         ## orientation
#         mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
#         mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)
#         mod3_ori = np.expand_dims(ori_list[2, :, 2, :].T, axis=2)

#         ori_list_new = np.concatenate([mod1_ori, mod2_ori, mod3_ori], axis=2) #(N, feature_dim, module_numb)

#     elif seg_numb == 4:
#         mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
#         mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
#         mod3_pos = np.expand_dims(pos_list[75, :, :].T, axis=2)
#         mod4_pos = np.expand_dims(pos_list[100, :, :].T, axis=2)
#         pos_list_new = np.concatenate([mod1_pos, mod2_pos, mod3_pos, mod4_pos], axis=2) #(N, feature_dim, module_numb)

#         ## orientation
#         mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
#         mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)
#         mod3_ori = np.expand_dims(ori_list[2, :, 2, :].T, axis=2)
#         mod4_ori = np.expand_dims(ori_list[3, :, 2, :].T, axis=2)

#         ori_list_new = np.concatenate([mod1_ori, mod2_ori, mod3_ori, mod4_ori], axis=2) #(N, feature_dim, module_numb)

#     elif seg_numb == 5:
#         mod1_pos = np.expand_dims(pos_list[25, :, :].T, axis=2)
#         mod2_pos = np.expand_dims(pos_list[50, :, :].T, axis=2)
#         mod3_pos = np.expand_dims(pos_list[75, :, :].T, axis=2)
#         mod4_pos = np.expand_dims(pos_list[100, :, :].T, axis=2)
#         mod5_pos = np.expand_dims(pos_list[125, :, :].T, axis=2)
#         pos_list_new = np.concatenate([mod1_pos, mod2_pos, mod3_pos, mod4_pos, mod5_pos], axis=2) #(N, feature_dim, module_numb)

#         ## orientation
#         mod1_ori = np.expand_dims(ori_list[0, :, 2, :].T, axis=2)
#         mod2_ori = np.expand_dims(ori_list[1, :, 2, :].T, axis=2)
#         mod3_ori = np.expand_dims(ori_list[2, :, 2, :].T, axis=2)
#         mod4_ori = np.expand_dims(ori_list[3, :, 2, :].T, axis=2)
#         mod5_ori = np.expand_dims(ori_list[4, :, 2, :].T, axis=2)

#         ori_list_new = np.concatenate([mod1_ori, mod2_ori, mod3_ori, mod4_ori, mod5_ori], axis=2) #(N, feature_dim, module_numb)
    
#     ## Normalize the position for each module
#     pos_list_new = normalize_position_train(pos_list=pos_list_new, seg_numb=seg_numb, args=args)
#     # # clip in-place to strictly enforce scaler feature_range and avoid small FP overflow
#     np.clip(pos_list_new, -1.0, 1.0, out=pos_list_new)

#     print("Data has been normalized !!", float(pos_list_new.max()), float(pos_list_new.min()))

#     return act_list, pos_list_new, ori_list_new


# def normalize_position_train(pos_list, seg_numb, args):
#     if seg_numb == 1:
#         mod1_pos = pos_list[:, :, 0]
#         # mod 1 train
#         scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/1_mod/scaler_pos_seg1.pkl")
#         pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
#         pos_list_new = pos_mod1_train
#         # save the normalizer
#         save_dictionary(scaler_pos_mod1, "scalers/1_mod/scaler_pos_seg1.pkl")

#     elif seg_numb == 2:
#         mod1_pos = pos_list[:, :, 0]
#         mod2_pos = pos_list[:, :, 1]
#         # mod 1 train
#         scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/2_mod/scaler_pos_seg1.pkl")
#         pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
#         # mod 2 train
#         scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/2_mod/scaler_pos_seg2.pkl")
#         pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
#         #(N, feature_dim, module_numb)
#         pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train], axis=2)
#         # save the normalizer
#         save_dictionary(scaler_pos_mod1, "scalers/2_mod/scaler_pos_seg1.pkl")
#         save_dictionary(scaler_pos_mod2, "scalers/2_mod/scaler_pos_seg2.pkl")

#     elif seg_numb == 3:
#         mod1_pos = pos_list[:, :, 0]
#         mod2_pos = pos_list[:, :, 1]
#         mod3_pos = pos_list[:, :, 2]
#         # mod 1 train
#         scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/3_mod/scaler_pos_seg1.pkl")
#         pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
#         # mod 2 train
#         scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/3_mod/scaler_pos_seg2.pkl")
#         pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
#         # mod 3 train
#         scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/3_mod/scaler_pos_seg3.pkl")
#         pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
#         #(N, feature_dim, module_numb)
#         pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train], axis=2)
#         # save the normalizer
#         save_dictionary(scaler_pos_mod1, "scalers/3_mod/scaler_pos_seg1.pkl")
#         save_dictionary(scaler_pos_mod2, "scalers/3_mod/scaler_pos_seg2.pkl")       
#         save_dictionary(scaler_pos_mod3, "scalers/3_mod/scaler_pos_seg3.pkl")  


#     elif seg_numb == 4:
#         mod1_pos = pos_list[:, :, 0]
#         mod2_pos = pos_list[:, :, 1]
#         mod3_pos = pos_list[:, :, 2]
#         mod4_pos = pos_list[:, :, 3]
#         # mod 1 train
#         scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/4_mod/scaler_pos_seg1.pkl")
#         pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
#         # mod 2 train
#         scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/4_mod/scaler_pos_seg2.pkl")
#         pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
#         # mod 3 train
#         scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/4_mod/scaler_pos_seg3.pkl")
#         pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
#         # mod 4 train
#         scaler_pos_mod4 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/4_mod/scaler_pos_seg4.pkl")
#         pos_mod4_train = np.expand_dims(scaler_pos_mod4.fit_transform(mod4_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod4.transform(mod4_pos), axis=2)
#         #(N, feature_dim, module_numb)
#         pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train, pos_mod4_train], axis=2)

#         # save the normalizer
#         save_dictionary(scaler_pos_mod1, "scalers/4_mod/scaler_pos_seg1.pkl")
#         save_dictionary(scaler_pos_mod2, "scalers/4_mod/scaler_pos_seg2.pkl")       
#         save_dictionary(scaler_pos_mod3, "scalers/4_mod/scaler_pos_seg3.pkl")  
#         save_dictionary(scaler_pos_mod4, "scalers/4_mod/scaler_pos_seg4.pkl")  

#     elif seg_numb == 5:
#         mod1_pos = pos_list[:, :, 0]
#         mod2_pos = pos_list[:, :, 1]
#         mod3_pos = pos_list[:, :, 2]
#         mod4_pos = pos_list[:, :, 3]
#         mod5_pos = pos_list[:, :, 4]
#         # mod 1 train
#         scaler_pos_mod1 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg1.pkl")
#         pos_mod1_train = np.expand_dims(scaler_pos_mod1.fit_transform(mod1_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod1.transform(mod1_pos), axis=2)
#         # mod 2 train
#         scaler_pos_mod2 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg2.pkl")
#         pos_mod2_train = np.expand_dims(scaler_pos_mod2.fit_transform(mod2_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod2.transform(mod2_pos), axis=2)
#         # mod 3 train
#         scaler_pos_mod3 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg3.pkl")
#         pos_mod3_train = np.expand_dims(scaler_pos_mod3.fit_transform(mod3_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod3.transform(mod3_pos), axis=2)
#         # mod 4 train
#         scaler_pos_mod4 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg4.pkl")
#         pos_mod4_train = np.expand_dims(scaler_pos_mod4.fit_transform(mod4_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod4.transform(mod4_pos), axis=2)
#         # mod 5 train
#         scaler_pos_mod5 = MinMaxScaler(feature_range=(-1, 1)) if args.mode == "train" else joblib.load("scalers/5_mod/scaler_pos_seg5.pkl")
#         pos_mod5_train = np.expand_dims(scaler_pos_mod5.fit_transform(mod5_pos), axis=2) if args.mode == "train" else np.expand_dims(scaler_pos_mod5.transform(mod5_pos), axis=2)
#         #(N, feature_dim, module_numb)
#         pos_list_new = np.concatenate([pos_mod1_train, pos_mod2_train, pos_mod3_train, pos_mod4_train, pos_mod5_train], axis=2)
        
#         # save the normalizer
#         save_dictionary(scaler_pos_mod1, "scalers/5_mod/scaler_pos_seg1.pkl")
#         save_dictionary(scaler_pos_mod2, "scalers/5_mod/scaler_pos_seg2.pkl")       
#         save_dictionary(scaler_pos_mod3, "scalers/5_mod/scaler_pos_seg3.pkl")  
#         save_dictionary(scaler_pos_mod4, "scalers/5_mod/scaler_pos_seg4.pkl")  
#         save_dictionary(scaler_pos_mod5, "scalers/5_mod/scaler_pos_seg5.pkl")  

#     return pos_list_new


# # Load the data
# def load_data(args, feature_dim, target_act_dim, mode, seg_numb):
#     time_step = args.time_step
#     use_orien = args.use_orien
#     # n_seg = args.n_seg # we keep n_seg here fixed so that we can have the data for all the segments when used with 3 modules
#     if args.shape_type == "babbling" and args.mode == "train":
#         data = np.load("../../Data_generation_Initial_mods/new_rev_dataset/" + str(mode) + "/" + f"{seg_numb}_mod/" + str(args.shape_type)+"_data_combined" + '.npz')
#     else:
#         data = np.load("../../Data_generation_Initial_mods/new_rev_dataset/" + str(mode) + "/" + f"{seg_numb}_mod/" + str(args.shape_type)+"_data" + '.npz')

#     act_list, pos_list, ori_list = extract_and_rearrange(numpy_file=data, seg_numb=seg_numb, args=args)

#     ## Pre-processing & Normalization
#     X = np.zeros((len(act_list) - time_step - 1, seg_numb, time_step, feature_dim+12), dtype=np.float32)
#     Y = np.zeros((len(act_list) - time_step - 1, seg_numb, time_step, target_act_dim), dtype=np.float32)
    
#     if use_orien == "true":
#         for k in range(seg_numb):
#             for i in range(len(act_list) - time_step - 1):
#                 for j in range(time_step):
#                     """
#                     The lag of +2 is used such the model takes as input the the expected target position and orintation,
#                     the previous actutaion and predicts the next actutaion for the desired target position and orientation.
                    
#                     s2, a0 -------> a1
#                     s3, a1 -------> a2
#                     ....
#                     ....
#                     ....
#                     """
#                     # s2, a0 informations

#                     X[i, k, j, 0:3] = pos_list[i + j + 2, :, k] # [ee number, xyz, ctrl_step]
#                     X[i, k, j, 3:6] = ori_list[i + j + 2, :, k]  # [ee number, xyz, ctrl_step]
#                     # X[i, k, j, 6:10] = act_list[i + j, k]  # a_0  #[ctrl_step, seg_number] --> gives two actutations at time step "t"
                    
#                     # S0 information
#                     X[i, k, j, 6:9] = pos_list[i + j, :, k]
#                     X[i, k, j, 9:12] = ori_list[i + j, :, k]

#                     # S1 information
#                     X[i, k, j, 12:15] = pos_list[i + j + 1, :, k]
#                     X[i, k, j, 15:18] = ori_list[i + j + 1, :, k]
                    
#                     # Target information
#                     Y[i, k, j, 0:2] = act_list[i + j + 1, k]             
#     else:
#         raise NotImplementedError("Requires orientation data to be true")

#     # PyTorch tensor
#     X = torch.tensor(X, dtype=torch.float32)
#     Y = torch.tensor(Y, dtype=torch.float32)

#     return X, Y


# def load_data_merged(args, feature_dim, target_act_dim, mode, seg_numb):
#     if args.mode == "train" or args.mode == "test":

#         X_bab, Y_bab = load_data(args, feature_dim, target_act_dim, mode, seg_numb)

#         X = X_bab
#         Y = Y_bab

#         if args.mode == "train":
#             indices = torch.randperm(X.shape[0])
#             X = X[indices]
#             Y = Y[indices]

#     # elif args.mode == "test_custom":
#     #     if args.shape_type == "circle":
#     #         X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_circle_t100")
#     #     elif args.shape_type == "rect":
#     #         X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_rect_t100")
#     #     elif args.shape_type == "spiral":
#     #         X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_spiral_t150")
#     #     elif args.shape_type == "test_babb":
#     #         X, Y = load_data(args, feature_dim, target_act_dim, filepath="babbling_test_samples_1k_freq_5.0")

#     return X.detach().numpy(), Y.detach().numpy()
