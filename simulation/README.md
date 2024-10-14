## Installation ##
```bash
conda create -n lcp-loco python=3.8
conda activate lcp-loco
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd ../../rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer
```

## How to use

We have provided pretrained policies for Fourier GR1 and Unitree H1. Please refer to the [link](https://drive.google.com/file/d/17pca2eyYbpB7lqu-CxZUvrAzuORrulkU/view?usp=sharing) for downloading. Then put the `logs` folder into [legged_gym](./legged_gym/) folder. The file structure should be:
```
simulation
├── legged_gym
│   ├── logs
│   ├── legged_gym
│   ├── ...
```


### Training && Playing Policy

``` bash
cd legged_gym/legged_gym/scripts
```

Then:
``` bash
bash run.sh [robot_type] [your_exp_desc] [device] 
# Fourier GR1: bash run.sh gr1 pretrained_exp cuda:0
# Unitree H1: bash run.sh h1 pretrained_exp cuda:0
# Berkeley Humanoid: bash run.sh berkeley pretrained_exp cuda:0
```

For the main training args:
+ `--debug` disables wandb and sets the number of environments to 64, which is useful for debugging;
+ `--fix_action_std` fixes the action std, this is useful for stablizing training;
+ `--resume` indicates whether to resume from the previous experiment;
+ `--resumeid` specifies the exptid to resume from (if resume is set true);

For evaluation, you can run:
``` bash
bash eval.sh [robot_type] [your_exp_desc] # e.g. bash eval.sh gr1 pretrained_exp
```

For the main evaluation args:
+ `--record_video` allows you to record video headlessly, this is useful for sever users;
+ `--checkpoint [int]` specifies the checkpoint to load, this is default set as -1, which is the latest one;
+ `--use_jit` use jit model to play;
+ `--teleop_mode` allows the user to control the robot with the keyboard;

### Save jit model

```bash
bash to_jit.sh [robot_type] [your_exp_desc] # e.g. bash to_jit.sh gr1 pretrained_exp
```

You can specify which checkpoint exactly to save by adding `--checkpoint [int]` to the command, this is default set as -1, which is the latest one.

You can display the jit policy by adding `--use_jit` in the eval script.

### Test Basic Env

You can test the env performance when the action is all 0 by running:
```bash
python ../test/test_env.py --task [task_name] --num_envs [int] # e.g. gr1_walk_phase, h1_walk_phase, berkeley_walk_phase
```

### Sim-to-Sim
You should first save the policy as jit model, then you can perform sim-to-sim directly by running:
```bash
bash mujoco_sim.sh [robot_type] [your_exp_desc] # e.g. bash mujoco_sim.sh gr1 aug31-test
```
This will load the latest jit model and run the sim-to-sim pipeline. You can manually specify the command in [sim2sim.py](./legged_gym/legged_gym/scripts/sim2sim.py).

### Real-world Deployment
You should first save the policy as jit model. And then please refer to the [deployment](./deployment/) folder for real-world deployment instructions.

## Known Issues
The known issues are listed in this section.

### Simulation Frequency
The simulation frequency has a huge impact on the performance of the policy. Most existing codebases for humanoid robots or quadruped robots use a sim frequency of 200Hz. This is enough for simple robots like Berkeley Humanoid robot. For H1 robot, our experiments show that policy trained with 1000Hz sim frequency performs slightly better than that with 200Hz. For GR1 robot, due to the complex structure and dynamics, we suggest setting the sim frequency to 1kHz. Although you can train a reasonable policy in simulation under 200Hz, but it will not work in the real world.

### Training Iterations
The number of training iterations is also important. For Berkeley Humanoid and H1, we use 20000 iterations. For GR1, we use 12000 iterations.

### Sim-to-Sim Performance of Fourier GR1
For successful real-world deployment, system identification is performed and a fitted motor model is used in the training. However, we do not include this in Mujoco for simplicity. Therefore, a slight decrease in performance of sim-to-sim is expected for GR1.
