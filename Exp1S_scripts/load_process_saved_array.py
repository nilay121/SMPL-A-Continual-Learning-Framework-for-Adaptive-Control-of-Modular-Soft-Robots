"""
load and process the saved array to display the error in the trajectory tracking experiment for the 
different trajectories. 
Structure : 
- Predicted Trajectory [babbling, circle, rect, spiral]
    - Module length with the different modules (1, 2, 3, 4, 5)
    - For each module we have the actuation, position and orientation error
- For ground truth load the original data from the babbling experiment.....

"""
import numpy as np
import pickle
from utils import extract_and_rearrange_sim_pred, orientationError

def load_predicted_array(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data


def error_calculation_after_all_training():
    # shape_type = ["babbling", "circle", "rect", "spiral"]
    shape_type = ["spiral"]

    for module_length in range(1, 6):
        for shape in shape_type:
            current_robot_length = 0.05 * module_length
            print(f"Processing error for **{shape}** with module length **{module_length}**")
            ## ----------------- load the predicted and gt data -----------------
            pred_act_file_path = f"saved_predicted_data_after_all_training/pred_oloop_act_shape_{shape}_totalSegSize{module_length}.npy"
            pred_pos_file_path = f"saved_predicted_data_after_all_training/pred_oloop_pos_shape_{shape}_totalSegSize{module_length}.npy"
            pred_ori_file_path = f"saved_predicted_data_after_all_training/pred_oloop_ori_shape_{shape}_totalSegSize{module_length}.npy"
            gt_act_file_path = f"saved_predicted_data_after_all_training/gt_oloop_act_shape_{shape}_totalSegSize{module_length}.npy"
            gt_pos_file_path = f"saved_predicted_data_after_all_training/gt_oloop_pos_shape_{shape}_totalSegSize{module_length}.npy"
            gt_ori_file_path = f"saved_predicted_data_after_all_training/gt_oloop_ori_shape_{shape}_totalSegSize{module_length}.npy"
           
            predicted_act = load_predicted_array(pred_act_file_path)
            predicted_pos = load_predicted_array(pred_pos_file_path)
            predicted_ori = load_predicted_array(pred_ori_file_path)
            ground_truth_act = load_predicted_array(gt_act_file_path)
            ground_truth_pos = load_predicted_array(gt_pos_file_path)
            ground_truth_ori = load_predicted_array(gt_ori_file_path)

            ## ----------------- load the ground truth data -----------------
            # data_path = "../../Data_generation_Initial_mods/new_rev_dataset/test/" + str(module_length) + "_mod/" + str(shape) + "_data" + '.npz'
            # act_list = np.load(data_path, allow_pickle=True)['act_list']
            # pos_list = np.load(data_path, allow_pickle=True)['pos_list']
            # ori_list = np.load(data_path, allow_pickle=True)['ori_list']

            # pos_list, ori_list = extract_and_rearrange_sim_pred(pos_list, ori_list, module_length, None)
            ## ----------------- calculate the error -----------------
            # original position error --> in mm
            final_error_ts = np.mean(np.linalg.norm((predicted_pos - ground_truth_pos), axis=1)) ## task space prediction error
            print("Error in position for the last time step:", final_error_ts*1000, "mm")

            ## Relative task space error for each module with respect to the length of the module
            relative_error_percent = (np.mean(final_error_ts) / current_robot_length) * 100
            print("Relative error w.r.t robot length: ", relative_error_percent, "%")

            ## Orientation error in degree
            final_ori_error = np.mean(orientationError(ground_truth_ori, predicted_ori))*180/3.14159
            print("End effector orientation error in degrees is : ", final_ori_error)

# calculate error on all the previous segments that it has been trained on
def incremental_test_error_calculation(shape="babbling"):

    for parent_module_length in range(1, 6): # used only to calculate till whcih module length the model has been trained
        for child_module_length in range(1, parent_module_length+1):
            current_robot_length = 0.05 * child_module_length
            print(f"Processing error for current module length **{child_module_length}** with incremental training till **{parent_module_length}** module")
            ## ----------------- load the predicted and gt data -----------------
            pred_act_file_path = f"saved_predicted_data_increment_training_run2/pred_oloop_act_shape_{shape}_totalSegSize{parent_module_length}_currentSeg{child_module_length}.npy"
            pred_pos_file_path = f"saved_predicted_data_increment_training_run2/pred_oloop_pos_shape_{shape}_totalSegSize{parent_module_length}_currentSeg{child_module_length}.npy"
            pred_ori_file_path = f"saved_predicted_data_increment_training_run2/pred_oloop_ori_shape_{shape}_totalSegSize{parent_module_length}_currentSeg{child_module_length}.npy"
            
            gt_act_file_path = f"saved_predicted_data_after_all_training/gt_oloop_act_shape_{shape}_totalSegSize{child_module_length}.npy"
            gt_pos_file_path = f"saved_predicted_data_after_all_training/gt_oloop_pos_shape_{shape}_totalSegSize{child_module_length}.npy"
            gt_ori_file_path = f"saved_predicted_data_after_all_training/gt_oloop_ori_shape_{shape}_totalSegSize{child_module_length}.npy"
           
            predicted_act = load_predicted_array(pred_act_file_path)
            predicted_pos = load_predicted_array(pred_pos_file_path)
            predicted_ori = load_predicted_array(pred_ori_file_path)
            ground_truth_act = load_predicted_array(gt_act_file_path)
            ground_truth_pos = load_predicted_array(gt_pos_file_path)
            ground_truth_ori = load_predicted_array(gt_ori_file_path)

            ## ----------------- load the ground truth data -----------------
            # data_path = "../../Data_generation_Initial_mods/new_rev_dataset/test/" + str(module_length) + "_mod/" + str(shape) + "_data" + '.npz'
            # act_list = np.load(data_path, allow_pickle=True)['act_list']
            # pos_list = np.load(data_path, allow_pickle=True)['pos_list']
            # ori_list = np.load(data_path, allow_pickle=True)['ori_list']

            # pos_list, ori_list = extract_and_rearrange_sim_pred(pos_list, ori_list, module_length, None)
            ## ----------------- calculate the error -----------------
            # original position error --> in mm
            final_error_ts = np.mean(np.linalg.norm((predicted_pos - ground_truth_pos), axis=1)) ## task space prediction error
            print("Error in position for the last time step:", final_error_ts*1000, "mm")

            ## Relative task space error for each module with respect to the length of the module
            relative_error_percent = (np.mean(final_error_ts) / current_robot_length) * 100
            print("Relative error w.r.t robot length: ", relative_error_percent, "%")

            # ## Orientation error in degree
            # final_ori_error = np.mean(orientationError(ground_truth_ori, predicted_ori))*180/3.14159
            # print("End effector orientation error in degrees is : ", final_ori_error)


if __name__ == "__main__":
    incremental_test_error_calculation()
    # error_calculation_after_all_training()
    