import os
import time

import numpy
import torch
from robot_rcs_gr.sdk import ControlGroup, RobotClient
from robot_rcs_gr.robot.gr1_nohla.fi_robot_gr1_nohla_algorithm import RobotGR1NohlaAlgorithmStandControlModel


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

        self.step_dt = 1.0 / step_freq

        self.num_of_joints = 6 + 6 + 3 + 3 + 7 + 7
        self.algorithm = RobotGR1NohlaAlgorithmStandControlModel()

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

        # parse to algorithm robot
        joint_measured_position_algorithm = numpy.zeros(self.algorithm.num_of_joints)
        joint_measured_velocity_algorithm = numpy.zeros(self.algorithm.num_of_joints)

        for i in range(self.algorithm.num_of_joints):
            index = self.algorithm.index_of_joints_real_robot[i]
            joint_measured_position_algorithm[i] = joint_measured_position[index] * numpy.pi / 180.0
            joint_measured_velocity_algorithm[i] = joint_measured_velocity[index] * numpy.pi / 180.0

        # algorithm
        _, _, target_joint_position_algorithm = \
            self.algorithm.run(
                joint_measured_position_urdf=joint_measured_position_algorithm,
                joint_measured_velocity_urdf=joint_measured_velocity_algorithm,
            )

        # parse to real robot
        target_joint_position = numpy.zeros(self.num_of_joints)
        for i in range(self.algorithm.num_of_joints):
            index = self.algorithm.index_of_joints_real_robot[i]
            target_joint_position[index] = target_joint_position_algorithm[i] / numpy.pi * 180.0

        # parse to np
        joint_target_position = target_joint_position

        print("algorithm_rl_walk joint_target_position = \n", joint_target_position)

        if act:
            self.client.move_joints(ControlGroup.ALL, joint_target_position, 0.0)

        return joint_target_position


if __name__ == "__main__":
    walker = RLWalker(comm_freq=400, step_freq=100)

    while True:
        start = time.perf_counter()
        joint_target_position = walker.step(act=False)
        print(f"Execution Time: {time.perf_counter() - start:.4f} s")

        # sleep to keep the loop frequency
        time_for_sleep = walker.step_dt - (time.perf_counter() - start)
        time.sleep(time_for_sleep)
