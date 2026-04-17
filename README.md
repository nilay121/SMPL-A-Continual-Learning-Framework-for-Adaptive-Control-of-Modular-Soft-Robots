# SMPL-A-Continual-Learning-Framework-for-Adaptive-Control-of-Modular-Soft-Robots
Soft robots have attracted considerable attention in applications such as medical intervention,
rehabilitation, and robotic manipulation due to their inherent compliance, flexibility, and high degrees of
freedom. Modular soft robots (MSRs), composed of multiple interconnected segments, further enhance this
versatility, enabling the execution of complex tasks. However, most existing approaches require controllers
to be re-trained from scratch whenever the number of modules in an MSR changes. In this work, we argue
that an MSR controller should inherently accommodate newly added modules while preserving knowledge
of previously learned configurations. Moreover, for MSRs with fixed morphology, many existing methods
rely on a single centralized model trained across all modules, limiting modular control and making the
system more susceptible to error propagation across modules. To address these challenges, we propose a
continual learning inspired control framework capable of incrementally learning new MSR configurations
while retaining previously acquired knowledge. For fixed-size MSRs, the same framework is employed in a
distributed manner to learn module-specific dynamics, enabling localized control and improved robustness.
The proposed framework is evaluated through closed-loop trajectory tracking experiments in simulation,
using a tendon-driven robot, and on a real-world three-module pneumatic soft robot. Additionally, we
demonstrate the adaptive capabilities of the approach through a dynamic reaching task, where the controller
selectively activates only the necessary modules to reach a target position, thereby reducing computational
overhead and enabling efficient and precise control.

<img src="https://github.com/nilay121/Continual-learning-for-multimodal-data-fusion-of-a-soft-gripper/blob/main/stapler_signal_train_small.png" height="300px" width="1000px">

## Install the dependencies in a virtual environment

- Create a virtual environment (Python version 3.8.10) 
  
  ```bash
  python3 -m venv Multimodal_cl
  ```

- Activate the virtual environment
  ```bash
  . Multimodal_cl/bin/activate
  
- Install the dependencies

  ```bash
  pip3 install -r requirements.txt
  ```
## Steps to follow
- Put the gripper dataset in the "dataset" folder
- Run the "dataset_vidToImage.py" file to extract train test images from video frame
- Run the "unsupervised_dataset.py" file to generate the unlabeled data for SSL
- Put the pre-trained feature extractors in the "pre_trained_models" folder

## Ros implementation
- Install ROS1 (Noetic Ninjemys distribution) on Ubuntu 20.04.
- Follow the steps provided in the "ros_instruction" file to create the ROS package.
- Copy paste the python scripts for publisher and subscriber nodes alongwith the pre-trained feature extractor and the saved matrices to the dedicated folders.

## Different combinations
- Intra layer feature representation
  ```
  python3 main.py --enable_ilfr True --enable_ssl False --ssl_type None 
  ```

- Semi Supervised learning
  - Unique class case
    ```
    python3 main.py --enable_ilfr True --enable_ssl False --ssl_type unique
    ```
  - Random class case
    ```
    python3 main.py --enable_ilfr True --enable_ssl True --ssl_type None 
    ```

- Intra layer feature representation
  ```
  python3 main.py --enable_ilfr True --enable_ssl True --ssl_type random 
  ```
  
## To cite the paper
  ```bash
@article{kushawaha2024continual,
  title={Continual learning for multimodal data fusion of a soft gripper},
  author={Kushawaha, Nilay and Falotico, Egidio},
  journal={Advanced Robotics Research},
  pages={202500126},
  year={2024},
  publisher={Wiley Online Library}
}
  ```
