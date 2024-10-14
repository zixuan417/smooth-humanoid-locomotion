from legged_gym import LEGGED_GYM_ROOT_DIR, envs

import isaacgym
import argparse
from isaacgym.torch_utils import torch_rand_float

import os
import time
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch

from legged_gym.envs import LEGGED_GYM_ROOT_DIR


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec

class HumanoidEnv:
    def __init__(self, policy_path, robot_type="gr1", device="cuda", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        if robot_type == "gr1":
            model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/GR1T1.xml"
            self.stiffness = np.array([
                251.625, 362.52, 200, 200, 10.98, 0.0,
                251.625, 362.52, 200, 200, 10.98, 0.0,
                362.52, 362.52*2, 362.52*2,
                40, 40, 40, 40,
                40, 40, 40, 40,
            ])
            self.damping = np.array([
                14.72, 10.08, 11, 11, 0.60, 0.1,
                14.72, 10.08, 11, 11, 0.60, 0.1,
                10.08, 10.08, 10.08,
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0,
            ])
            
            self.control_indices = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 20, 21, 22])
            self.num_actions = len(self.control_indices)
            self.num_dofs = 23

            self.default_dof_pos = np.array([
                0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # left leg (6)
                0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # right leg (6)
                0.0, -0.0, 0.0,  # waist (3)
                0.0, 0.2, 0.0, -0.3,
                0.0, -0.2, 0.0, -0.3,
            ])
            
            self.torque_limits = np.array([
                48, 60, 160, 160, 16, 8,
                48, 60, 160, 160, 16, 8,
                82.5, 82.5, 82.5,
                18, 18, 18, 18,
                18, 18, 18, 18,
            ])
            
            self.cycle_time = 0.8
        
        elif robot_type == "h1":
            model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1.xml"
            self.stiffness = np.array([
                200, 200, 200, 200, 40,
                200, 200, 200, 200, 40,
                300,
                40, 40, 40, 40,
                40, 40, 40, 40,
            ])
            self.damping = np.array([
                5, 5, 5, 5, 2,
                5, 5, 5, 5, 2,
                6,
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0,
            ])
            self.control_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) # all dofs
            self.num_actions = len(self.control_indices)
            self.num_dofs = 19
            
            self.default_dof_pos = np.array([
                0.0, 0.0, -0.6, 1.2, -0.6,  # left leg (5)
                0.0, 0.0, -0.6, 1.2, -0.6,  # right leg (5)
                0.0,  # waist (1)
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ])
            
            self.torque_limits = np.array([
                200, 200, 200, 300, 40,
                200, 200, 200, 300, 40,
                200,
                40, 40, 18, 18,
                40, 40, 18, 18,
            ])
            
            self.cycle_time = 0.8
            
        elif robot_type == "berkeley":
            model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/berkeley_humanoid/urdf/robot.xml"
            self.stiffness = np.array([
                10, 10, 15, 15, 1, 1,
                10, 10, 15, 15, 1, 1,
            ])
            self.damping = np.array([
                1.5, 1.5, 1.5, 1.5, 0.1, 0.1,
                1.5, 1.5, 1.5, 1.5, 0.1, 0.1,
            ])
            self.control_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) # all dofs
            self.num_actions = len(self.control_indices)
            self.num_dofs = 12
            self.default_dof_pos = np.array([
                -0.071, 0.103, -0.463, 0.983, -0.350, 0.126,  # left leg (6)
                0.071, -0.103, -0.463, 0.983, -0.350, -0.126,  # right leg (6)
            ])
            self.torque_limits = np.array([
                20, 20, 30, 30, 20, 5,
                20, 20, 30, 30, 20, 5,
            ])
            self.cycle_time = 0.64
        
        elif robot_type == "g1":
            model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof.xml"
            self.stiffness = np.array([
                200, 150, 150, 200, 20, 20,
                200, 150, 150, 200, 20, 20,
                200,
                40, 40, 40, 40,
                40, 40, 40, 40,
            ])
            self.damping = np.array([
                5, 5, 5, 5, 4, 4,
                5, 5, 5, 5, 4, 4,
                5,
                10, 10, 10, 10,
                10, 10, 10, 10,
            ])
            self.control_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # all dofs
            self.num_actions = len(self.control_indices)
            self.num_dofs = 21
            self.default_dof_pos = np.array([
                -0.4, 0.0, 0.0, 0.8, -0.35, 0.0,  # left leg (6)
                -0.4, 0.0, 0.0, 0.8, -0.35, 0.0,  # right leg (6)
                0.0,  # torso (1)
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ])
            self.torque_limits = np.array([
                88, 88, 88, 139, 50, 50,
                88, 88, 88, 139, 50, 50,
                88,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
            self.cycle_time = 0.64

        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        if robot_type == "gr1":
            self.obs_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22])
        else:
            self.obs_indices = np.arange(self.num_dofs)
        
        if self.record_video:
            self.sim_duration = 10.0
        else:
            self.sim_duration = 60.0
        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        if self.record_video:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.5
        
        self.n_priv = 0
        if self.robot_type == "gr1":
            self.n_proprio = 2 + 3 + 3 + 2 + 2*(self.num_dofs-2) + self.num_actions
            self.n_priv_latent = 4 + 1 + 2*(self.num_dofs-2) + 3
        else:
            self.n_proprio = 2 + 3 + 3 + 2 + 2*self.num_dofs + self.num_actions
            self.n_priv_latent = 4 + 1 + self.num_dofs*2 + 3
            
        self.history_len = 10
        self.priv_latent = np.zeros(self.n_priv_latent, dtype=np.float32)
        
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.ang_vel_scale = 0.25
        
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
            
        self.commands = np.zeros(3)
        
        self.set_commands()
        
        print("Loading jit for policy: ", policy_path)
        self.policy_path = policy_path
        self.policy_jit = torch.jit.load(policy_path, map_location=self.device)
        
        self.last_time = time.time()
    
    def set_commands(self):
        self.commands[0] = 0.0
        self.commands[1] = 0.0
        self.commands[2] = 0.0
        
    def extract_data(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        self.dof_vel = torch.from_numpy(dof_vel).float().unsqueeze(0).to(self.device)
        return (dof_pos, dof_vel, quat, ang_vel)
        
    def run(self):
        
        if self.record_video:
            import imageio
            video_name = f"{self.robot_type}_{''.join(os.path.basename(self.policy_path).split('.')[:-1])}.mp4"
            path = f"../../logs/mujoco_videos/"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=50)
        
        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            dof_pos, dof_vel, quat, ang_vel = self.extract_data()
            
            if i % self.sim_decimation == 0:
                phase = (i // self.sim_decimation) * self.control_dt / self.cycle_time
                sin_pos = [np.sin(2 * np.pi * phase)]
                cos_pos = [np.cos(2 * np.pi * phase)]
                rpy = quatToEuler(quat)
                if self.robot_type == "g1":
                    rpy[1] -= 0.08
                obs_prop = np.concatenate([
                    sin_pos, cos_pos,
                    self.commands,
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos)[self.obs_indices] * self.dof_pos_scale,
                    dof_vel[self.obs_indices] * self.dof_vel_scale,
                    self.last_action,
                ])
                
                assert obs_prop.shape[0] == self.n_proprio, f"Expected {self.n_proprio} but got {obs_prop.shape[0]}"
                obs_hist = np.array(self.proprio_history_buf).flatten()
                self.proprio_history_buf.append(obs_prop)
                
                obs_buf = np.concatenate([obs_prop, self.priv_latent, obs_hist])
                
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    raw_action = self.policy_jit(obs_tensor).cpu().numpy().squeeze()
                
                self.last_action = raw_action.copy()
                raw_action = np.clip(raw_action, -10., 10.)
                scaled_actions = raw_action * self.action_scale
                
                step_actions = np.zeros(self.num_dofs)
                step_actions[self.control_indices] = scaled_actions
                
                pd_target = step_actions + self.default_dof_pos
                
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                if self.record_video:
                    img = self.viewer.read_pixels()
                    mp4_writer.append_data(img)
                else:
                    self.viewer.render()
                
        
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            
            self.data.ctrl = torque
            
            mujoco.mj_step(self.model, self.data)
        
        self.viewer.close()
        if self.record_video:
            mp4_writer.close()
                 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--robot', type=str, default="gr1") # options: gr1, h1, berkeley, g1
    parser.add_argument('--exptid', type=str, required=True)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--record_video', action='store_true')
    args = parser.parse_args()
    
    policy_pth = f"../../logs/{args.robot}_walk_phase/{args.exptid}/traced/"
    assert os.path.exists(policy_pth), f"Policy path {policy_pth} does not exist!"
    
    model_name, checkpoint = get_load_path(policy_pth, checkpoint=args.checkpoint)
    jit_policy_pth = os.path.join(policy_pth, model_name)
    print(f"Loading model from: {jit_policy_pth}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env = HumanoidEnv(policy_path=jit_policy_pth, robot_type=args.robot, device=device, record_video=args.record_video)
    
    env.run()
        
        