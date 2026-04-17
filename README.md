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

## Install the dependencies in a virtual environment

- Create a virtual environment (Python version 3.10.12) or use uv 
  
  ```bash
  python3 -m venv smpl_venv
  ```

- Activate the virtual environment
  ```bash
  . smpl_venv/bin/activate
  
- Install the dependencies

  ```bash
  pip3 install -r smpl_requirements.txt
  ```
## Steps to follow Exp1S experiment
- Generate the motor babbling data using the simulator https://github.com/zixichen007115/23ZCd or custom pyelastica simulator https://docs.cosseratrods.org/en/latest/api/simulator.html and put it in the designated folder.
- Train the forward model on the babbling data using the scripts:
  ```bash
  python3 main.py --mode train --n_seg N --use_orien true
  ```
- Once forward model has been trained, put it in the designated folder, freeze it and train the inverse model:
  ```bash
  python3 main_open_loop.py --mode train --incremental_training yes --incremental_test yes --train_total_segments N
  ```
- After training the inverse model for different MSR configurations, it can be evaluated using:
  ```bash
  python3 main_open_loop.py --mode test --test_total_segments N --shape_type babbling
  ```
## Steps to follow Exp2S and Exp2R experiment
- Generate the motor babbling data using the simulator
- Train the forward model on the babbling data using the scripts:
  ```bash
  python3 main.py --mode train --n_seg N --use_orien true
  ```
- Once forward model has been trained, put it in the designated folder, freeze it and train the inverse model (open loop/closed loop):
  ```bash
  python3 main_closed_loop.py --mode train --n_seg N --use_orien true
  ```
- After training the inverse model, evaluate it for different trajectories:
  ```bash
  python3 main_closed_loop.py --mode test --n_seg N --shape_type babbling
  ```
  
## To cite the paper
  ```bash
  ```
