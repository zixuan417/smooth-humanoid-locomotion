


from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, POSE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.humanoid import Humanoid
# from .humanoid_config import HumanoidCfg
from .g1_walk_phase_config import G1WalkPhaseCfg
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.gym_utils.math import *


class G1WalkPhase(Humanoid):
    def __init__(self, cfg: G1WalkPhaseCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.use_estimator = (self.cfg.env.n_priv != 0)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        phase = self._get_phase()
        self.compute_ref_state()
    
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
        self.ref_dof_pos[:, 0] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r > 0] = 0
        self.ref_dof_pos[:, 6] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = -sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        # Double support phase
        indices = (torch.abs(sin_pos_l) < self.cfg.rewards.double_support_threshold) & (torch.abs(sin_pos_r) < self.cfg.rewards.double_support_threshold)
        self.ref_dof_pos[indices] = 0
        self.ref_dof_pos += self.default_dof_pos_all

        self.ref_action = 2 * self.ref_dof_pos
        
    
    # ======================================================================================================================
    # Reward functions
    # ======================================================================================================================
