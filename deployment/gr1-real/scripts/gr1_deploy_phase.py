import sys, os, time, argparse
from collections import deque

sys.path.append("../")

import numpy as np
import torch

from gr1_robot import GR1Robot

class GR1Policy:
    def __init__(self, policy_file, freq, log, p_scale, d_scale, joystick=False) -> None:
        self.control_index = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 20, 21, 22])
        self.obs_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22])
        
        self.if_log = log
        self.p_scale = p_scale
        self.d_scale = d_scale
        
        self.freq = freq
        self.action_scale = 0.5
        
        self.stand_timestep = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)
        
        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05
        self.scales_lin_vel = 0.5
        
        self.num_actions = len(self.control_index)
        
        self.num_dof = 23

        self.n_proprio = 2 + 3 + 3 + 2 + 2*(self.num_dof-2) + self.num_actions
        self.n_priv_latent = 4 + 1 + (self.num_dof-2)*2 + 3
        self.history_len = 10
        
        self.last_action = np.zeros(self.num_actions)
        self.proprio_history_buf = deque(maxlen=self.history_len)
        
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
            
        self.commands = np.zeros(3)

        print("Loading jit...")
        self.policy = torch.jit.load(policy_file, map_location=self.device)
        
        self.timestep = 0
        
        self.use_joystick = joystick
        self.robot = GR1Robot(p_scale=self.p_scale, d_scale=self.d_scale, joystick=self.use_joystick)
        
        self.default_dof_pos = np.array([
            0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # left leg (6)
            0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # right leg (6)
            0.0, -0.0, 0.0,  # waist (3)
            0.0, 0.2, 0.0, -0.3,
            0.0, -0.2, 0.0, -0.3,
        ])
        
        self.default_dof_pos_stand = np.array([
            0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # left leg (6)
            0.0, 0.0, -0.4, 0.8, -0.4, 0.0,  # right leg (6)
            0.0, -0.0, 0.0,  # waist (3)
            0.0, 0.2, 0.0, -0.3,
            0.0, -0.2, 0.0, -0.3,
        ])
        
        self.dt = 1.0 / self.freq
        
        self.self_check()
    
    def self_check(self):
        init_motor_angle = np.array(self.robot.GetTrueMotorAngles())
        print("Init motor angles: ", init_motor_angle)
        if np.isclose(np.sum(init_motor_angle), 0):
            print("No data received, exiting")
            exit()
        for _ in range(10):
            self.step(dry_run=True)
        self.timestep = 0
        self.init_motor_angle = init_motor_angle
        
    def step(self, dry_run=False):
        step_start_time = time.time()
        
        obs = self.get_observation()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            raw_actions = self.policy(obs_tensor).numpy().squeeze()
            
        self.last_action = raw_actions.copy()
        raw_actions = np.clip(raw_actions, -10, 10)
        scaled_actions = raw_actions * self.action_scale
        
        step_actions = np.zeros(self.num_dof)
        step_actions[self.control_index] = scaled_actions
        
        pd_target = step_actions + self.default_dof_pos

        if not dry_run:
            duration = time.time() - step_start_time
            self.robot.Step(pd_target)
        
        self.post_step_callback()
        
        
    def post_step_callback(self):
        if self.use_joystick:
            self.process_remote()
        self.timestep += 1
    
    def  _get_phase(self):
        cycle_time = 0.8
        phase = self.timestep * self.dt / cycle_time
        return phase
    
    def get_observation(self):
        self.robot.ReceiveObservation()
        rpy = self.robot.GetBaseRPY()
        rp = rpy[:2]
        if not self.use_joystick:
            self.commands[0] = 0.0
            self.commands[1] = 0.0
            self.commands[2] = 0.0
        phase = self._get_phase()

        sin_pos = [np.sin(2 * np.pi * phase)]
        cos_pos = [np.cos(2 * np.pi * phase)]
        
        obs_prop = np.concatenate([
            sin_pos,
            cos_pos,
            self.commands,
            self.robot.GetAngVel() * self.scales_ang_vel,
            rp,
            (self.robot.GetTrueMotorAngles() - self.default_dof_pos)[self.obs_index], #19
            self.robot.GetMotorVel()[self.obs_index] * self.scales_dof_vel, #19
            self.last_action, #19
        ])
        
        assert obs_prop.shape[0] == self.n_proprio
        obs_hist = np.array(self.proprio_history_buf).flatten()

        priv_latent = np.zeros(self.n_priv_latent)
        
        self.proprio_history_buf.append(obs_prop)
        
        return np.concatenate((obs_prop, priv_latent, obs_hist))
    
    def stand(self, wait=True):
        self.stand_timestep = 0
        self.robot._set_stand_pd()
        
        desired_motor_angle = self.default_dof_pos_stand.copy()

        traj_len = 1000
        
        for t in range(traj_len):
            current_motor_angle = np.array(self.robot.GetTrueMotorAngles())

            blend_ratio = np.minimum(t / 450, 1)
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle
            self.robot.Step(action)
            # warm up network
            obs = self.get_observation()

            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            raw_actions = self.policy(obs_tensor).detach().cpu().numpy().squeeze()

            time.sleep(0.02)
            
            self.stand_timestep += 1

        time.sleep(0.01)
        
    def process_remote(self):
        left_joy, right_joy, btns = self.robot.GetRemoteStates()
        lx, ly = left_joy
        rx, ry = right_joy
        
        self.commands[0] = abs(ly)*0.5 if abs(ly) > 0.1 else 0
        self.commands[1] = -lx*0.2 if abs(lx) > 0.1 else 0
        self.commands[2] = -rx*0.2 if abs(rx) > 0.1 else 0
        
        # termination
        r, p, _ = self.robot.GetBaseRPY()
        
        if btns[0] == 1 or abs(p) > 1.0 or abs(r) > 1.0:
            print("Killed by remote!")
            self.robot.shut_robot()
    
    def shut(self):
        self.robot.shut_robot()
        
    def prepare_rl(self):
        self.robot._set_rl_pd()
    

if __name__ == "__main__":
    FREQ = 50
    STEP_TIME = 1 / FREQ
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--use_joystick", action="store_true")
    
    parser.add_argument("--model_name", type=str, default="gr1-deploy")
    
    parser.add_argument("--p_scale", type=float, default=1.0)
    parser.add_argument("--d_scale", type=float, default=1.0)
    args = parser.parse_args()
    
    args.no_waist = True
    
    model_file = "../deploy_models/" + args.model_name + ".pt"
    
    hardware_pipeline = GR1Policy(policy_file=model_file, freq=FREQ, log=args.log, p_scale=args.p_scale, d_scale=args.d_scale, joystick=args.use_joystick)
    hardware_pipeline.stand()
    
    hardware_pipeline.prepare_rl()
    
    start = time.time()
    i = 0
    print("here")
    
    while True:
        start = time.time()
        hardware_pipeline.step()
        duration = time.time() - start
        print("duration: ", duration)
        if duration < STEP_TIME:
            time.sleep(STEP_TIME - duration)
        
        i += 1
            
