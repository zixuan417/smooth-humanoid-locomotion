


from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, POSE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.humanoid import Humanoid
# from .humanoid_config import HumanoidCfg
from .gr1_walk_phase_config import GR1WalkPhaseCfg
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.gym_utils.math import *
from legged_gym.gym_utils.motor_delay_fft import MotorDelay_130, MotorDelay_80


class GR1WalkPhase(Humanoid):
    def __init__(self, cfg: GR1WalkPhaseCfg, sim_params, physics_engine, sim_device, headless):
        self.control_index = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        self.cfg = cfg
        self.use_motor_model = self.cfg.env.use_motor_model
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        phase = self._get_phase()
        self.compute_ref_state()
        
    def compute_ref_state(self):
        phase = self._get_phase()
        _sin_pos_l = torch.sin(2 * torch.pi * phase)
        _sin_pos_r = torch.sin(2 * torch.pi * phase + torch.pi)
        sin_pos_l = _sin_pos_l.clone()
        sin_pos_r = _sin_pos_r.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        ratio_l = torch.clamp(torch.abs(sin_pos_l) - self.cfg.rewards.double_support_threshold, min=0, max=1) / (1 - self.cfg.rewards.double_support_threshold) * torch.sign(sin_pos_l)
        self.ref_dof_pos[:, 2] = ratio_l * scale_1
        self.ref_dof_pos[:, 3] = -ratio_l * scale_2
        self.ref_dof_pos[:, 4] = ratio_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r > 0] = 0
        ratio_r = torch.clamp(torch.abs(sin_pos_r) - self.cfg.rewards.double_support_threshold, min=0, max=1) / (1 - self.cfg.rewards.double_support_threshold) * torch.sign(sin_pos_r)
        self.ref_dof_pos[:, 8] = ratio_r * scale_1
        self.ref_dof_pos[:, 9] = -ratio_r * scale_2
        self.ref_dof_pos[:, 10] = ratio_r * scale_1
        # Double support phase
        indices = (torch.abs(sin_pos_l) < self.cfg.rewards.double_support_threshold) & (torch.abs(sin_pos_r) < self.cfg.rewards.double_support_threshold)
        self.ref_dof_pos[indices] = 0
        self.ref_dof_pos += self.default_dof_pos_all

        self.ref_action = 2 * self.ref_dof_pos
    
    def _init_buffers(self):
        super()._init_buffers()
        if self.use_motor_model:
            self.motordelay0 = MotorDelay_80(self.num_envs, 1, device=self.device)
            self.motordelay6 = MotorDelay_80(self.num_envs, 1, device=self.device)
            
            self.motordelay2 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay3 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay8 = MotorDelay_130(self.num_envs, 1, device=self.device)
            self.motordelay9 = MotorDelay_130(self.num_envs, 1, device=self.device)
            
            self.fric_para_0 = torch.zeros(self.num_envs, 5,
                                       dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_1 = torch.zeros(self.num_envs, 5,
                                        dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_2 = torch.zeros(self.num_envs, 5,
                                        dtype=torch.float, device=self.device, requires_grad=False)
            self.fric_para_3 = torch.zeros(self.num_envs, 5,
                                        dtype=torch.float, device=self.device, requires_grad=False)
            self.fric = torch.zeros(self.num_envs, self.num_dofs,
                                    dtype=torch.float, device=self.device, requires_grad=False)

    def _get_body_indices(self):
        upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                                 requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                  torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                                requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
    
    def _reset_buffers_extra(self, env_ids):
        if self.use_motor_model:
            self._reset_fric_para(env_ids)
            self.motordelay0.reset(env_ids)
            self.motordelay6.reset(env_ids)
            self.motordelay2.reset(env_ids)
            self.motordelay3.reset(env_ids)
            self.motordelay8.reset(env_ids)
            self.motordelay9.reset(env_ids)

    def _reset_fric_para(self, env_ids):
        self.fric_para_0[env_ids, 0] = torch_rand_float(3.7, 6.6, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 1] = torch_rand_float(3.3, 5.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 2] = torch_rand_float(-5.0, -3.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 3] = torch_rand_float(0.7, 0.9, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_0[env_ids, 4] = torch_rand_float(0.7, 0.9, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_1[env_ids, 0] = torch_rand_float(1.2, 2.75, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 1] = torch_rand_float(1.0, 1.55, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 2] = torch_rand_float(-1.55, -1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 3] = torch_rand_float(0.4, 0.65, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_1[env_ids, 4] = torch_rand_float(0.4, 0.65, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_2[env_ids, 0] = torch_rand_float(1.9, 3.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 1] = torch_rand_float(1.15, 2.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 2] = torch_rand_float(-2.0, -1.3, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 3] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_2[env_ids, 4] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)

        self.fric_para_3[env_ids, 0] = torch_rand_float(0.25, 1.25, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 1] = torch_rand_float(0.2, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 2] = torch_rand_float(-1.0, -0.2, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 3] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)
        self.fric_para_3[env_ids, 4] = torch_rand_float(0.14, 0.18, (len(env_ids), 1), device=self.device).squeeze(1)
    
    def _compute_torques(self, actions):
        #pd controller
        actions_scaled_raw = actions * self.cfg.control.action_scale
        actions_scaled = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        actions_scaled[:, self.control_index] = actions_scaled_raw
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:
                torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.use_motor_model:
            torques[:,[0]] = self.motordelay0(torques[:,[0]] )
            torques[:,[6]] = self.motordelay6(torques[:,[6]] )
            torques[:,[2]] = self.motordelay2(torques[:,[2]] )
            torques[:,[3]] = self.motordelay3(torques[:,[3]] )
            torques[:,[8]] = self.motordelay8(torques[:,[8]] )
            torques[:,[9]] = self.motordelay9(torques[:,[9]] )
            
            self.friction_0_6()
            self.friction_1_7()
            self.friction_2_3_8_9()
            
            torques -= self.fric
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    
    def friction_0_6(self):
        flag_0 = (self.dof_vel[:, 0] <= 0.002) & (self.dof_vel[:, 0] >= -0.002)
        flag_1 = ((self.dof_vel[:, 0] > 0.002) & (self.dof_vel[:, 0] <= 0.16))
        flag_2 = (self.dof_vel[:, 0] > 0.16)
        flag_3 = ((self.dof_vel[:, 0] < -0.002) & (self.dof_vel[:, 0] >= -0.16))
        flag_4 = (self.dof_vel[:, 0] < -0.16)

        self.fric[:, 0] = self.fric_para_0[:, 0] / 0.002 * self.dof_vel[:, 0] * flag_0 + \
                          ((self.fric_para_0[:, 1] - self.fric_para_0[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 0] - 0.002) + self.fric_para_0[:, 0]) * flag_1 + \
                          (self.fric_para_0[:, 1] + self.fric_para_0[:, 3] * (self.dof_vel[:, 0] - 0.16)) * flag_2 + \
                          ((self.fric_para_0[:, 2] + self.fric_para_0[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 0] + 0.002) - self.fric_para_0[:, 0]) * flag_3 + \
                          (self.fric_para_0[:, 2] + self.fric_para_0[:, 4] * (self.dof_vel[:, 0] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 6] <= 0.002) & (self.dof_vel[:, 6] >= -0.002)
        flag_1 = ((self.dof_vel[:, 6] > 0.002) & (self.dof_vel[:, 6] <= 0.16))
        flag_2 = (self.dof_vel[:, 6] > 0.16)
        flag_3 = ((self.dof_vel[:, 6] < -0.002) & (self.dof_vel[:, 6] >= -0.16))
        flag_4 = (self.dof_vel[:, 6] < -0.16)

        self.fric[:, 6] = self.fric_para_0[:, 0] / 0.002 * self.dof_vel[:, 6] * flag_0 + \
                          ((self.fric_para_0[:, 1] - self.fric_para_0[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 6] - 0.002) + self.fric_para_0[:, 0]) * flag_1 + \
                          (self.fric_para_0[:, 1] + self.fric_para_0[:, 3] * (self.dof_vel[:, 6] - 0.16)) * flag_2 + \
                          ((self.fric_para_0[:, 2] + self.fric_para_0[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 6] + 0.002) - self.fric_para_0[:, 0]) * flag_3 + \
                          (self.fric_para_0[:, 2] + self.fric_para_0[:, 4] * (self.dof_vel[:, 6] + 0.16)) * flag_4

    def friction_1_7(self):
        flag_0 = (self.dof_vel[:, 1] <= 0.002) & (self.dof_vel[:, 1] >= -0.002)
        flag_1 = ((self.dof_vel[:, 1] > 0.002) & (self.dof_vel[:, 1] <= 0.16))
        flag_2 = (self.dof_vel[:, 1] > 0.16)
        flag_3 = ((self.dof_vel[:, 1] < -0.002) & (self.dof_vel[:, 1] >= -0.16))
        flag_4 = (self.dof_vel[:, 1] < -0.16)

        self.fric[:, 1] = self.fric_para_1[:, 0] / 0.002 * self.dof_vel[:, 1] * flag_0 + \
                          ((self.fric_para_1[:, 1] - self.fric_para_1[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 1] - 0.002) + self.fric_para_1[:, 0]) * flag_1 + \
                          (self.fric_para_1[:, 1] + self.fric_para_1[:, 3] * (self.dof_vel[:, 1] - 0.16)) * flag_2 + \
                          ((self.fric_para_1[:, 2] + self.fric_para_1[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 1] + 0.002) - self.fric_para_1[:, 0]) * flag_3 + \
                          (self.fric_para_1[:, 2] + self.fric_para_1[:, 4] * (self.dof_vel[:, 1] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 7] <= 0.002) & (self.dof_vel[:, 7] >= -0.002)
        flag_1 = ((self.dof_vel[:, 7] > 0.002) & (self.dof_vel[:, 7] <= 0.16))
        flag_2 = (self.dof_vel[:, 7] > 0.16)
        flag_3 = ((self.dof_vel[:, 7] < -0.002) & (self.dof_vel[:, 7] >= -0.16))
        flag_4 = (self.dof_vel[:, 7] < -0.16)

        self.fric[:, 7] = self.fric_para_1[:, 0] / 0.002 * self.dof_vel[:, 7] * flag_0 + \
                          ((self.fric_para_1[:, 1] - self.fric_para_1[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 7] - 0.002) + self.fric_para_1[:, 0]) * flag_1 + \
                          (self.fric_para_1[:, 1] + self.fric_para_1[:, 3] * (self.dof_vel[:, 7] - 0.16)) * flag_2 + \
                          ((self.fric_para_1[:, 2] + self.fric_para_1[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 7] + 0.002) - self.fric_para_1[:, 0]) * flag_3 + \
                          (self.fric_para_1[:, 2] + self.fric_para_1[:, 4] * (self.dof_vel[:, 7] + 0.16)) * flag_4

    def friction_2_3_8_9(self):
        flag_0 = (self.dof_vel[:, 2] <= 0.002) & (self.dof_vel[:, 2] >= -0.002)
        flag_1 = ((self.dof_vel[:, 2] > 0.002) & (self.dof_vel[:, 2] <= 0.16))
        flag_2 = (self.dof_vel[:, 2] > 0.16)
        flag_3 = ((self.dof_vel[:, 2] < -0.002) & (self.dof_vel[:, 2] >= -0.16))
        flag_4 = (self.dof_vel[:, 2] < -0.16)

        self.fric[:, 2] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 2] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 2] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 2] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 2] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 2] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 3] <= 0.002) & (self.dof_vel[:, 3] >= -0.002)
        flag_1 = ((self.dof_vel[:, 3] > 0.002) & (self.dof_vel[:, 3] <= 0.16))
        flag_2 = (self.dof_vel[:, 3] > 0.16)
        flag_3 = ((self.dof_vel[:, 3] < -0.002) & (self.dof_vel[:, 3] >= -0.16))
        flag_4 = (self.dof_vel[:, 3] < -0.16)

        self.fric[:, 3] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 3] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 3] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 3] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 3] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 3] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 8] <= 0.002) & (self.dof_vel[:, 8] >= -0.002)
        flag_1 = ((self.dof_vel[:, 8] > 0.002) & (self.dof_vel[:, 8] <= 0.16))
        flag_2 = (self.dof_vel[:, 8] > 0.16)
        flag_3 = ((self.dof_vel[:, 8] < -0.002) & (self.dof_vel[:, 8] >= -0.16))
        flag_4 = (self.dof_vel[:, 8] < -0.16)

        self.fric[:, 8] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 8] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 8] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 8] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 8] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 8] + 0.16)) * flag_4

        flag_0 = (self.dof_vel[:, 9] <= 0.002) & (self.dof_vel[:, 9] >= -0.002)
        flag_1 = ((self.dof_vel[:, 9] > 0.002) & (self.dof_vel[:, 9] <= 0.16))
        flag_2 = (self.dof_vel[:, 9] > 0.16)
        flag_3 = ((self.dof_vel[:, 9] < -0.002) & (self.dof_vel[:, 9] >= -0.16))
        flag_4 = (self.dof_vel[:, 9] < -0.16)

        self.fric[:, 9] = self.fric_para_2[:, 0] / 0.002 * self.dof_vel[:, 9] * flag_0 + \
                          ((self.fric_para_2[:, 1] - self.fric_para_2[:, 0]) / (0.16 - 0.002) * (
                                  self.dof_vel[:, 9] - 0.002) + self.fric_para_2[:, 0]) * flag_1 + \
                          (self.fric_para_2[:, 1] + self.fric_para_2[:, 3] * (self.dof_vel[:, 9] - 0.16)) * flag_2 + \
                          ((self.fric_para_2[:, 2] + self.fric_para_2[:, 0]) / (-0.16 + 0.002) * (
                                  self.dof_vel[:, 9] + 0.002) - self.fric_para_2[:, 0]) * flag_3 + \
                          (self.fric_para_2[:, 2] + self.fric_para_2[:, 4] * (self.dof_vel[:, 9] + 0.16)) * flag_4
        
    # ======================================================================================================================
    # Reward functions
    # ======================================================================================================================
