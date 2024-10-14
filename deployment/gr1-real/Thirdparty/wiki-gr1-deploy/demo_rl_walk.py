import os
import time

import numpy
import torch
from robot_rcs_gr.sdk import ControlGroup, RobotClient


def quat_rotate_inverse(q, v):
    """
    Calculate the rotation of a vector by a quaternion

    Input:
    - q: quaternion, shape = (batch_size, 4)
    - v: vector, shape = (batch_size, 3)
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
            q_vec
            * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
            * 2.0
    )
    return a - b + c


def calculate_gait_phase(step_count, dt, gait_cycle=0.50, theta_offset_left=0.35, theta_offset_right=0.85,
                         rand_init_phases=1):
    """
    Calculate gait phase for left and right leg

    Input:
    - time: current time of running the algorithm, in second
    - gait_cycle: gait cycle, in second
    - theta_left: initial phase for left leg, in ratio
    - theta_right: initial phase for right leg, in ratio
    """
    gait_clock = step_count * dt
    gait_clock_left_leg = gait_clock \
                          + (rand_init_phases == 1) * (theta_offset_left * gait_cycle) \
                          + (rand_init_phases == -1) * (theta_offset_right * gait_cycle)
    gait_clock_right_leg = gait_clock \
                           + (rand_init_phases == 1) * (theta_offset_right * gait_cycle) \
                           + (rand_init_phases == -1) * (theta_offset_left * gait_cycle)

    gait_phase_left_leg = gait_clock_left_leg / gait_cycle - torch.floor(gait_clock_left_leg / gait_cycle)
    gait_phase_right_leg = gait_clock_right_leg / gait_cycle - torch.floor(gait_clock_right_leg / gait_cycle)

    return gait_phase_left_leg, gait_phase_right_leg


class LowPassFilter:
    """
    Low pass filter class
    """

    def __init__(self, cutoff_freq, damping_ratio, dt, dim):
        self.cutoff_freq = cutoff_freq
        self.damping_ratio = damping_ratio
        self.dt = dt
        self.dim = dim
        # self.omega = 2 * torch.pi * cutoff_freq
        self.alpha = 2 * numpy.pi * cutoff_freq * damping_ratio
        self.beta = numpy.tan(self.alpha * dt / 2)
        self.gamma = self.beta / (1 + self.beta)
        self.prev_output = torch.zeros((dim,))

    def filter(self, input):
        output = torch.zeros((self.dim,))
        for i in range(self.dim):
            output[i] = (self.gamma * input[i] + (1 - self.gamma) * self.prev_output[i])
        self.prev_output = output
        return output


class RLWalker:
    """
    Reinforcement Learning Walker
    """

    def __init__(self, comm_freq=400, step_freq=100):
        """
        Initialize the RL Walker

        Input:
        - comm_freq: communication frequency, in Hz
        - step_freq: step frequency, in Hz
        """

        # setup RobotClient
        self.client = RobotClient(comm_freq)
        time.sleep(1.0)

        self.set_gains()
        self.client.set_enable(True)

        # reset step count
        self.step_count = 0
        self.step_dt = 1.0 / step_freq
        self.gait_cycle = 0.70
        self.theta_offset_left = 0.35
        self.theta_offset_right = 0.85
        self.phase_ratio_left = 0.35
        self.phase_ratio_right = 0.35

        # prepare parameters
        self.filter_for_all = \
            LowPassFilter(cutoff_freq=30, damping_ratio=0.707, dt=self.step_dt, dim=3)
        self.filter_for_base_lin_vel = \
            LowPassFilter(cutoff_freq=10, damping_ratio=0.707, dt=self.step_dt, dim=3)
        self.filter_for_base_ang_vel = \
            LowPassFilter(cutoff_freq=10, damping_ratio=0.707, dt=self.step_dt, dim=3)
        self.filter_for_dof_pos = \
            LowPassFilter(cutoff_freq=10, damping_ratio=0.707, dt=self.step_dt, dim=23)
        self.filter_for_dof_vel = \
            LowPassFilter(cutoff_freq=30, damping_ratio=0.707, dt=self.step_dt, dim=23)

        self.obs_scale_lin_vel = 2.00
        self.obs_scale_ang_vel = 0.25
        self.obs_scale_command = torch.tensor([[self.obs_scale_lin_vel,
                                                self.obs_scale_lin_vel,
                                                self.obs_scale_lin_vel]], dtype=torch.float32)
        self.obs_scale_gravity = 1.00
        self.obs_scale_dof_pos = 1.00
        self.obs_scale_dof_vel = 0.05
        self.obs_scale_height = 5.00
        self.act_scale_action = 0.50

        self.base_linear_velocity_x_avg_factor = 0.02
        self.base_linear_velocity_y_avg_factor = 0.01
        self.base_angular_velocity_yaw_avg_factor = 0.01

        self.actor = None
        self.encoder = None
        self.config = {
            # actor
            "num_joints": 6 + 6 + 3 + 3 + 7 + 7,
            "num_actor_obs": 321,
            "num_actions": 6 + 6 + 3 + 3 + 7 + 7 - 3 - 3 - 3,
            "num_history_measured_joint_position_offset": 5,
            "num_history_measured_joint_velocity": 5,
            # encoder
            "num_encoder_obs": 3600,
        }
        self.index_joint_controlled = [
            0, 1, 2, 3, 4, 5,  # left leg
            6, 7, 8, 9, 10, 11,  # right leg
            12, 13, 14,  # waist
            # 15, 16, 17,  # head
            18, 19, 20, 21,
            # 22, 23, 24,  # left arm
            25, 26, 27, 28,
            # 29, 30, 31,  # right arm
        ]
        assert (
                len(self.index_joint_controlled) == self.config["num_actions"]
        ), "index_joint_controlled length is not equal to num_actions"

        self.encoder_observation = \
            torch.zeros((1, self.config["num_encoder_obs"]), dtype=torch.float32)
        self.actor_observation = \
            torch.zeros((1, self.config["num_actor_obs"]), dtype=torch.float32)

        self.last_action_tensor = \
            torch.zeros((1, self.config["num_actions"]), dtype=torch.float32)
        self.gait_phase_sin_tensor = \
            torch.zeros((1, 2), dtype=torch.float32)
        self.gait_phase_cos_tensor = \
            torch.zeros((1, 2), dtype=torch.float32)
        self.gait_phase_ratio_tensor = \
            torch.zeros((1, 2), dtype=torch.float32)
        self.average_base_lin_vel_x_tensor = \
            torch.zeros((1, 1), dtype=torch.float32)
        self.average_base_lin_vel_y_tensor = \
            torch.zeros((1, 1), dtype=torch.float32)
        self.average_base_ang_vel_yaw_tensor = \
            torch.zeros((1, 1), dtype=torch.float32)
        self.history_measured_joint_position_offset_tensor = \
            torch.zeros((1,
                         self.config["num_history_measured_joint_position_offset"]
                         * self.config["num_actions"]),
                        dtype=torch.float32)
        self.history_measured_joint_velocity_tensor = \
            torch.zeros((1,
                         self.config["num_history_measured_joint_velocity"]
                         * self.config["num_actions"]),
                        dtype=torch.float32)

        # fmt: off
        self.base_height_target = 0.85  # unit : m
        self.action_max = torch.tensor([[
            100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100,
            100, 100, 100,
            100, 100, 100, 100,
            100, 100, 100, 100,
        ]])
        self.action_min = torch.tensor([[
            -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100,
            -100, -100, -100,
            -100, -100, -100, -100,
            -100, -100, -100, -100,
        ]])
        self.joint_default_position = torch.tensor([[
            0.0, 0.0, -0.5236, 1.0472, -0.5236, 0.0,  # left leg (6)
            0.0, 0.0, -0.5236, 1.0472, -0.5236, 0.0,  # right leg (6)
            0.0, 0.0, 0.0,  # waist (3)
            0.0, 0.0, 0.0,  # head (3)
            0.0, 0.3, 0.0, -0.3, 0.0, 0.0, 0.0,  # left arm (7)
            0.0, -0.3, 0.0, -0.3, 0.0, 0.0, 0.0,  # right arm (7)
        ]], dtype=torch.float32)
        # fmt: on
        self.gravity_vector = torch.tensor([[0.0, 0.0, -1.0]])

    def set_gains(self):
        """
        Set gains for the robot
        """

        # fmt: off
        # stiffness = {
        #     'hip_roll': 251.625, 'hip_yaw': 362.5214, 'hip_pitch': 200,
        #     'knee_pitch': 200,
        #     'ankle_pitch': 10.9805, 'ankle_roll': 0.1,  # 'ankleRoll': 0.0,
        #     'waist_yaw': 362.5214, 'waist_pitch': 362.5214, 'waist_roll': 362.5214,
        #
        #     'shoulder_pitch': 92.85, 'shoulder_roll': 92.85, 'shoulder_yaw': 112.06,
        #     'elbow_pitch': 112.06
        # }  # [N*m/rad]
        # damping = {
        #     'hip_roll': 14.72, 'hip_yaw': 10.0833, 'hip_pitch': 11,
        #     'knee_pitch': 11,
        #     'ankle_pitch': 0.5991, 'ankle_roll': 0.1,
        #     'waist_yaw': 10.0833, 'waist_pitch': 10.0833, 'waist_roll': 10.0833,
        #     'shoulder_pitch': 2.575, 'shoulder_roll': 2.575, 'shoulder_yaw': 3.1,
        #     'elbow_pitch': 3.1
        # }
        kp = numpy.array([
            0.9971, 1.0228, 1.0606, 1.0606, 0.5091, 0.5091,  # left leg
            0.9971, 1.0228, 1.0606, 1.0606, 0.5091, 0.5091,  # right leg
            1.0228, 1.0228, 1.0228,  # waist
            0.1000, 0.1000, 0.1000,  # head
            1.0016, 1.0016, 1.0041, 1.0041, 1.0041, 0.1000, 0.1000,  # left arm
            1.0016, 1.0016, 1.0041, 1.0041, 1.0041, 0.1000, 0.1000,  # right arm
        ])
        kd = numpy.array([
            0.0445, 0.0299, 0.2634, 0.2634, 0.0042, 0.0042,  # left leg
            0.0445, 0.0299, 0.2634, 0.2634, 0.0042, 0.0042,  # right leg
            0.0299, 0.0299, 0.0299,  # waist
            0.0050, 0.0050, 0.0050,  # head
            0.0037, 0.0037, 0.0039, 0.0039, 0.0039, 0.0050, 0.0050,  # left arm
            0.0037, 0.0037, 0.0039, 0.0039, 0.0039, 0.0050, 0.0050,  # right arm
        ])
        # fmt: on
        self.client.set_gains(kp, kd)

    def step(self, act=False):
        """
        Step function for the RL Walker

        Input:
        - act: whether to actuate the robot or not
        """

        # get states
        imu_quat = self.client.states["imu"]["quat"].copy()
        imu_euler_angle = self.client.states["imu"]["euler_angle"].copy()
        imu_angular_velocity = self.client.states["imu"]["angular_velocity"].copy()
        joint_measured_position = self.client.states["joint"]["position"].copy()
        joint_measured_velocity = self.client.states["joint"]["velocity"].copy()

        # parse to torch
        imu_quat_tensor = \
            torch.tensor(imu_quat, dtype=torch.float32).unsqueeze(0)
        imu_euler_angle_tensor = \
            torch.tensor(imu_euler_angle, dtype=torch.float32).unsqueeze(0)
        imu_angular_velocity_tensor = \
            torch.tensor(imu_angular_velocity, dtype=torch.float32).unsqueeze(0)
        joint_measured_position_tensor = \
            torch.tensor(joint_measured_position, dtype=torch.float32).unsqueeze(0)
        joint_measured_velocity_tensor = \
            torch.tensor(joint_measured_velocity, dtype=torch.float32).unsqueeze(0)

        imu_euler_angle_tensor = \
            imu_euler_angle_tensor / 180.0 * torch.pi  # unit : rad
        imu_angular_velocity_tensor = \
            imu_angular_velocity_tensor / 180.0 * torch.pi  # unit : rad/s
        joint_measured_position_tensor = \
            joint_measured_position_tensor / 180.0 * torch.pi  # unit : rad
        joint_measured_velocity_tensor = \
            joint_measured_velocity_tensor / 180.0 * torch.pi  # unit : rad/s
        joint_offset_position_tensor = \
            joint_measured_position_tensor - self.joint_default_position  # unit : rad

        # load actor
        if self.actor is None:
            actor_model_file_path = \
                os.path.dirname(os.path.abspath(__file__)) + "/data/student/policy_5700_feet_high.pt"
            print("algorithm_rl_walk actor_model_file_path = ", actor_model_file_path)

            self.actor = torch.jit.load(actor_model_file_path, map_location=torch.device("cpu"))

        # load encoder
        if self.encoder is None:
            encoder_model_file_path = \
                os.path.dirname(os.path.abspath(__file__)) + "/data/student/encoder_1100_5700_feet_high.pt"

            self.encoder = torch.jit.load(encoder_model_file_path, map_location=torch.device("cpu"))

        # when first run, last_action is set to measured joint position
        if self.last_action_tensor is None:
            self.last_action_tensor = torch.zeros(
                (1, self.config["num_actions"]), dtype=torch.float32
            )
            self.last_action_tensor[0, :] = joint_offset_position_tensor[
                0, self.index_joint_controlled
            ]

        # base linear velocity
        base_lin_vel_tensor = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float32
        ).unsqueeze(0)

        # base angular velocity
        base_ang_vel_tensor = imu_angular_velocity_tensor

        # base project gravity
        base_project_gravity_tensor = quat_rotate_inverse(
            imu_quat_tensor,
            self.gravity_vector
        )

        # commands
        commands_tensor = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float32
        )

        # joint position, joint velocity, action
        measured_joint_position_offset_tensor = torch.zeros(
            (1, self.config["num_actions"]), dtype=torch.float32
        )
        measured_joint_velocity_tensor = torch.zeros(
            (1, self.config["num_actions"]), dtype=torch.float32
        )

        measured_joint_position_offset_tensor[0, :] = joint_offset_position_tensor[
            0, self.index_joint_controlled
        ]
        measured_joint_velocity_tensor[0, :] = joint_measured_velocity_tensor[
            0, self.index_joint_controlled
        ]

        measured_joint_velocity_tensor = self.filter_for_dof_vel.filter(
            measured_joint_velocity_tensor[0, :]
        ).unsqueeze(0)

        # gait phase sin, cos, ratio
        gait_phase_left_leg, gait_phase_right_leg = calculate_gait_phase(
            step_count=self.step_count,
            dt=self.step_dt,
            gait_cycle=self.gait_cycle,
            theta_offset_left=self.theta_offset_left,
            theta_offset_right=self.theta_offset_right,
        )

        self.gait_phase_sin_tensor[0, 0] = torch.sin(
            2 * numpy.pi * gait_phase_left_leg
        )
        self.gait_phase_sin_tensor[0, 1] = torch.sin(
            2 * numpy.pi * gait_phase_right_leg
        )
        self.gait_phase_cos_tensor[0, 0] = torch.cos(
            2 * numpy.pi * gait_phase_left_leg
        )
        self.gait_phase_cos_tensor[0, 1] = torch.cos(
            2 * numpy.pi * gait_phase_right_leg
        )
        self.gait_phase_ratio_tensor[0, 0] = self.phase_ratio_left
        self.gait_phase_ratio_tensor[0, 1] = self.phase_ratio_right

        # base height offset
        base_height_offset_tensor = torch.tensor(
            [[0.0]], dtype=torch.float32
        )

        # --------------------------------------------------------------
        # encoder
        # encoder obs
        self.encoder_observation = torch.cat(
            (
                self.encoder_observation[:, self.config["num_actions"] * 3 + 3:],
                measured_joint_position_offset_tensor * self.obs_scale_dof_pos,
                measured_joint_velocity_tensor * self.obs_scale_dof_vel,
                self.last_action_tensor,
                base_project_gravity_tensor * self.obs_scale_gravity,
            ),
            dim=1,
        )

        # encoder nn
        encode_tensor = self.encoder(self.encoder_observation)

        # encoder -> measured
        base_lin_vel_tensor[0, 0] = encode_tensor[0, 0]
        base_lin_vel_tensor[0, 1] = encode_tensor[0, 1]
        base_ang_vel_tensor[0, 2] = encode_tensor[0, 2]
        base_height_offset_tensor[0, 0] = encode_tensor[0, 3]
        # --------------------------------------------------------------

        # average base linear (x, y) and angular velocity (yaw)
        self.average_base_lin_vel_x_tensor = \
            self.base_linear_velocity_x_avg_factor * base_lin_vel_tensor[0, 0] \
            + (1 - self.base_linear_velocity_x_avg_factor) * self.average_base_lin_vel_x_tensor
        self.average_base_lin_vel_y_tensor = \
            self.base_linear_velocity_y_avg_factor * base_lin_vel_tensor[0, 1] \
            + (1 - self.base_linear_velocity_y_avg_factor) * self.average_base_lin_vel_y_tensor
        self.average_base_ang_vel_yaw_tensor = \
            self.base_angular_velocity_yaw_avg_factor * base_ang_vel_tensor[0, 2] \
            + (1 - self.base_angular_velocity_yaw_avg_factor) * self.average_base_ang_vel_yaw_tensor

        # --------------------------------------------------------------
        # actor
        # actor obs
        self.actor_observation = torch.cat(
            (
                base_lin_vel_tensor * self.obs_scale_lin_vel,
                base_ang_vel_tensor * self.obs_scale_ang_vel,
                base_project_gravity_tensor * self.obs_scale_gravity,
                commands_tensor * self.obs_scale_command,
                measured_joint_position_offset_tensor * self.obs_scale_dof_pos,
                measured_joint_velocity_tensor * self.obs_scale_dof_vel,
                self.last_action_tensor,
                self.gait_phase_sin_tensor,
                self.gait_phase_cos_tensor,
                self.gait_phase_ratio_tensor,
                base_height_offset_tensor * self.obs_scale_height,
                self.average_base_lin_vel_x_tensor * self.obs_scale_lin_vel,
                self.average_base_lin_vel_y_tensor * self.obs_scale_lin_vel,
                self.average_base_ang_vel_yaw_tensor * self.obs_scale_ang_vel,
                self.history_measured_joint_position_offset_tensor * self.obs_scale_dof_pos,
                self.history_measured_joint_velocity_tensor * self.obs_scale_dof_vel,
            ),
            dim=1,
        )

        # actor nn
        action_tensor = self.actor(self.actor_observation)

        # clip action
        action_tensor = torch.clamp(action_tensor, self.action_min, self.action_max)
        # --------------------------------------------------------------

        # update history
        self.history_measured_joint_position_offset_tensor = torch.cat(
            (
                self.history_measured_joint_position_offset_tensor[:, self.config["num_actions"]:],
                measured_joint_position_offset_tensor,
            ),
            dim=1,
        )

        self.history_measured_joint_velocity_tensor = torch.cat(
            (
                self.history_measured_joint_velocity_tensor[:, self.config["num_actions"]:],
                measured_joint_velocity_tensor,
            ),
            dim=1,
        )

        # record action
        self.last_action_tensor = action_tensor

        joint_target_position = torch.zeros(
            (1, self.config["num_joints"]), dtype=torch.float32
        )

        joint_target_position[0, self.index_joint_controlled] = action_tensor[0, :] * self.act_scale_action
        joint_target_position += self.joint_default_position

        # parse to np
        joint_target_position = joint_target_position.detach().numpy()
        joint_target_position = joint_target_position / numpy.pi * 180.0
        joint_target_position = joint_target_position[0]

        # print("algorithm_rl_walk joint_target_position = \n", joint_target_position)
        if act:
            self.client.move_joints(ControlGroup.ALL, joint_target_position, 0.0)

        # update step_count
        self.step_count += 1

        return joint_target_position


if __name__ == "__main__":
    walker = RLWalker(comm_freq=400, step_freq=100)

    while True:
        start = time.perf_counter()
        joint_target_position = walker.step(act=True)
        print(f"Execution Time: {time.perf_counter() - start:.4f} s")

        # sleep to keep the loop frequency
        time_for_sleep = walker.step_dt - (time.perf_counter() - start)
        time.sleep(time_for_sleep)
