import os
import time

import numpy
import torch
from robot_rcs.rl.rl_actor_critic_mlp import ActorCriticMLP

from robot_rcs_gr.sdk import ControlGroup, RobotClient


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


class RLWalker:
    def __init__(self, freq=400):
        self.client = RobotClient(freq)
        time.sleep(1.0)
        self.set_gains()
        self.client.set_enable(True)

        self.command = torch.zeros((1, 3))
        self.actor: ActorCriticMLP | None = None
        self.last_action = None
        self.config = {
            "num_joints": 22,
            "num_actor_obs": 39,
            "num_critic_obs": 39,
            "num_actions": 10,
        }
        self.index_joint_controlled = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        assert (
            len(self.index_joint_controlled) == self.config["num_actions"]
        ), "index_joint_controlled length is not equal to num_actions"

        # fmt: off
        self.action_max = torch.tensor([[
            0.79, 0.7, 0.7, 1.92, 0.52,  # left leg (5), no ankle roll, more simple state_estimator
            0.09, 0.7, 0.7, 1.92, 0.52,  # left leg (5), no ankle roll, more simple state_estimator
        ]]) + 60 / 180 * torch.pi / 3
        self.action_min = torch.tensor([[
            -0.09, -0.7, -1.75, -0.09, -1.05,  # left leg (5), no ankle roll, more simple state_estimator
            -0.79, -0.7, -1.75, -0.09, -1.05,  # left leg (5), no ankle roll, more simple state_estimator
        ]]) - 60 / 180 * torch.pi / 3
        self.joint_default_position = torch.tensor([[
            0.0, 0.0, -0.2618, 0.5236, -0.2618, 0.0,  # left leg (6)
            0.0, 0.0, -0.2618, 0.5236, -0.2618, 0.0,  # right leg (6)
            0.0, 0.0, 0.0,  # waist (3)
            0.0, 0.0, 0.0,  # head (3)
            0.0, 0.2, 0.0, -0.3, 0.0, 0.0, 0.0,  # left arm (7)
            0.0, -0.2, 0.0, -0.3, 0.0, 0.0, 0.0,  # right arm (7)
        ]], dtype=torch.float32)
        # fmt: on
        self.gravity_vector = torch.tensor([[0.0, 0.0, -1.0]])

    def set_gains(self):
        # fmt: off
        kp = numpy.array([
            # left leg
            0.583, 0.284, 0.583, 0.583, 0.283, 0.283,
            # right leg
            0.583, 0.284, 0.583, 0.583, 0.283, 0.283,
            # waist
            0.25, 0.25, 0.25,
            # head
            0.005, 0.005, 0.005,
            # left arm
            0.2, 0.2, 0.2, 0.2, 0.2, 0.005, 0.005,
            # right arm
            0.2, 0.2, 0.2, 0.2, 0.2, 0.005, 0.005,
        ])
        kd = numpy.array([
            # left leg
            0.017, 0.013, 0.273, 0.273, 0.005, 0.005,
            # right leg
            0.017, 0.013, 0.273, 0.273, 0.005, 0.005,
            # waist
            0.14, 0.14, 0.14,
            # head
            0.005, 0.005, 0.005,
            # left arm
            0.02, 0.02, 0.02, 0.02, 0.02, 0.005, 0.005,
            # right arm
            0.02, 0.02, 0.02, 0.02, 0.02, 0.005, 0.005,
        ])
        # fmt: on
        self.client.set_gains(kp, kd)

    def step(self, act=False):
        # get states
        imu_quat = self.client.states["imu"]["quat"].copy()
        imu_angular_velocity = self.client.states["imu"]["angular_velocity"].copy()
        joint_measured_position = self.client.states["joint"]["position"].copy()
        joint_measured_velocity = self.client.states["joint"]["velocity"].copy()

        # parse to torch
        imu_quat_tensor = torch.tensor(imu_quat, dtype=torch.float32).unsqueeze(0)
        imu_angular_velocity_tensor = torch.tensor(imu_angular_velocity, dtype=torch.float32).unsqueeze(0)
        joint_measured_position_tensor = torch.tensor(joint_measured_position, dtype=torch.float32).unsqueeze(0)
        joint_measured_velocity_tensor = torch.tensor(joint_measured_velocity, dtype=torch.float32).unsqueeze(0)

        imu_angular_velocity_tensor = imu_angular_velocity_tensor / 180.0 * torch.pi  # unit : rad/s
        joint_measured_position_tensor = joint_measured_position_tensor / 180.0 * torch.pi  # unit : rad
        joint_measured_velocity_tensor = joint_measured_velocity_tensor / 180.0 * torch.pi  # unit : rad/s
        joint_offset_position_tensor = joint_measured_position_tensor - self.joint_default_position

        # load actor
        if self.actor is None:
            model_file_path = os.path.dirname(os.path.abspath(__file__)) + "/data/walk_model.pt"
            print("algorithm_rl_walk model_file_path = ", model_file_path)

            model = torch.load(model_file_path, map_location=torch.device("cpu"))
            model_actor_dict = model["model_state_dict"]

            self.actor = ActorCriticMLP(
                num_actor_obs=self.config["num_actor_obs"],
                num_critic_obs=self.config["num_critic_obs"],
                num_actions=self.config["num_actions"],
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
            )

            self.actor.load_state_dict(model_actor_dict)

        # when first run, last_action is set to measured joint position
        if self.last_action is None:
            self.last_action = torch.zeros((1, self.config["num_actions"]), dtype=torch.float32)
            self.last_action[0, :] = joint_offset_position_tensor[0, self.index_joint_controlled]

        # command
        self.command = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        # project gravity
        project_gravity_tensor = quat_rotate_inverse(imu_quat_tensor, self.gravity_vector)

        # joint position and velocity
        joint_controlled_position_tensor = torch.zeros((1, self.config["num_actions"]), dtype=torch.float32)
        joint_controlled_velocity_tensor = torch.zeros((1, self.config["num_actions"]), dtype=torch.float32)

        joint_controlled_position_tensor[0, :] = joint_offset_position_tensor[0, self.index_joint_controlled]
        joint_controlled_velocity_tensor[0, :] = joint_measured_velocity_tensor[0, self.index_joint_controlled]
        # for i, index in enumerate(self.index_joint_controlled):
        #     joint_controlled_position_tensor[0, i] = joint_offset_position_tensor[
        #         0, index
        #     ]
        #     joint_controlled_velocity_tensor[0, i] = joint_measured_velocity_tensor[
        #         0, index
        #     ]

        # actor-critic
        observation = torch.cat(
            (
                imu_angular_velocity_tensor,
                project_gravity_tensor,
                self.command,
                joint_controlled_position_tensor,
                joint_controlled_velocity_tensor,
                self.last_action,
            ),
            dim=1,
        )

        action = self.actor(observation)

        # clip action
        action = torch.clamp(action, self.action_min, self.action_max)

        # record action
        self.last_action = action

        joint_target_position = torch.zeros((1, self.config["num_joints"]), dtype=torch.float32)

        joint_target_position[0, self.index_joint_controlled] = action[0, :]
        joint_target_position += self.joint_default_position

        # parse to numpy
        joint_target_position = joint_target_position.detach().numpy()
        joint_target_position = joint_target_position / numpy.pi * 180.0
        joint_target_position = joint_target_position[0]

        # print("algorithm_rl_walk joint_target_position = \n", joint_target_position)
        if act:
            self.client.move_joints(ControlGroup.ALL, joint_target_position, 0.0)
        return joint_target_position


if __name__ == "__main__":
    walker = RLWalker(400)
    while True:
        start = time.perf_counter()
        joint_target_position = walker.step(act=True)
        print(f"Execution Time: {time.perf_counter() - start:.4f} s")
        time.sleep(1 / 50)
