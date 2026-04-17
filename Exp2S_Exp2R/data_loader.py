import torch
import numpy as np


# Load the data
def load_data(args, feature_dim, target_act_dim,filepath):
    time_step = args.time_step
    use_orien = args.use_orien
    n_seg = args.n_seg # we keep n_seg here fixed so that we can have the data for all the segments when used with 3 modules
    data = np.load( "processed_data/" + str(filepath) + '.npz')

    pos_list = data['pos_list']
    ori_list = data['ori_list']
    act_list = data['act_list']

    print("pos_list shape:" + str(pos_list.shape))
    print("ori_list shape:" + str(ori_list.shape))
    print("act_list shape:" + str(act_list.shape))


    X = np.zeros((len(act_list) - time_step - 1, n_seg, time_step, feature_dim+12), dtype=np.float32)
    Y = np.zeros((len(act_list) - time_step - 1, n_seg, time_step, target_act_dim), dtype=np.float32)
    
    if use_orien == "true":
        for k in range(n_seg):
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
                    Y[i, k, j, 0:4] = act_list[i + j + 1, k] 
                    
                    
    else:
        raise NotImplementedError("Requires orientation data to be true")

    # PyTorch tensor
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y


def load_data_merged(args, feature_dim, target_act_dim):
    if args.mode == "train" or args.mode == "test":

        # train data --> babbling plus some other standard data
        X_bab, Y_bab = load_data(args, feature_dim, target_act_dim, filepath="processed_comb_babb_data_shapes_combined")

        X = X_bab
        Y = Y_bab

        if args.mode == "train":
            indices = torch.randperm(X.shape[0])
            X = X[indices]
            Y = Y[indices]

    elif args.mode == "test_custom":
        if args.shape_type == "circle":
            X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_circle")
        elif args.shape_type == "rect":
            X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_rect")
        elif args.shape_type == "spiral":
            X, Y = load_data(args, feature_dim, target_act_dim, filepath="processed_spiral")
        elif args.shape_type == "test_babb":
            X, Y = load_data(args, feature_dim, target_act_dim, filepath="babbling_test_samples")

    return X.detach().numpy(), Y.detach().numpy()
