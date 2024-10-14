import time

import numpy
import typer
from ischedule import run_loop, schedule

from robot_rcs_gr.robot.gr1_nohla.fi_robot_gr1_nohla_algorithm_rl import RobotGR1NohlaAlgorithmRLWalkControlModel
from robot_rcs_gr.sdk import ControlGroup, RobotClient


class DemoNohlaRLWalk:
    """
    Reinforcement Learning Walker
    """

    def __init__(self, comm_freq: int = 400, step_freq: int = 100, act: bool = False):
        """
        Initialize the RL Walker

        Input:
        - comm_freq: communication frequency, in Hz
        - step_freq: step frequency, in Hz
        """

        # setup RobotClient
        self.client = RobotClient(comm_freq)
        self.act = act
        time.sleep(1.0)

        self.set_gains()
        self.client.set_enable(True)

        # default position
        # fmt: off
        self.default_position = numpy.array(
            [
                0.0, 0.0, -0.5236, 1.0472, -0.5236, 0.0,  # left leg (6)
                0.0, 0.0, -0.5236, 1.0472, -0.5236, 0.0,  # right leg (6)
                0.0, 0.0, 0.0,  # waist (3)
                0.0, 0.0, 0.0,  # head (3)
                0.0, 0.3, 0.0, -0.3, 0.0, 0.0, 0.0,  # left arm (7)
                0.0, -0.3, 0.0, -0.3, 0.0, 0.0, 0.0,  # right arm (7)
            ]
        )  # unit : rad
        # fmt: on

        # algorithm
        algorithm_control_period = 1.0 / step_freq
        self.algorithm_nohla_rl_walk_control_model = RobotGR1NohlaAlgorithmRLWalkControlModel(
            dt=algorithm_control_period, decimation=int((1 / 100) / algorithm_control_period), warmup_period=1.0
        )

    def set_gains(self):
        """
        Set gains for the robot
        """

        # fmt: off
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

    def step(self):
        """
        Step function for the RL Walker

        Input:
        - act: whether to actuate the robot or not
        """

        # get states
        imu_quat = self.client.states["imu"]["quat"].copy()
        imu_angular_velocity = self.client.states["imu"]["angular_velocity"].copy()  # unit : deg/s
        joint_measured_position_urdf = self.client.states["joint"]["position"].copy()  # unit : deg
        joint_measured_velocity_urdf = self.client.states["joint"]["velocity"].copy()  # unit : deg/s

        # prepare input
        commands = numpy.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        )
        base_measured_quat_to_world = imu_quat
        base_measured_rpy_vel_to_self = numpy.deg2rad(imu_angular_velocity)

        joint_measured_position_nohla_urdf = numpy.zeros(self.algorithm_nohla_rl_walk_control_model.num_of_joints)
        joint_measured_velocity_nohla_urdf = numpy.zeros(self.algorithm_nohla_rl_walk_control_model.num_of_joints)

        # joint: real robot urdf -> algorithm urdf
        for i in range(self.algorithm_nohla_rl_walk_control_model.num_of_joints):
            index = self.algorithm_nohla_rl_walk_control_model.index_of_joints_real_robot[i]
            joint_measured_position_nohla_urdf[i] = numpy.deg2rad(joint_measured_position_urdf[index])
            joint_measured_velocity_nohla_urdf[i] = numpy.deg2rad(joint_measured_velocity_urdf[index])

        # TODO 2024-07-09: use default target position -> urdf joint target position
        init_output = self.algorithm_nohla_rl_walk_control_model.joint_default_position

        # TODO 2024-02-26: Use the joystick button R2 to switch between walking and standing
        self.algorithm_nohla_rl_walk_control_model.gait_phase_ratio = (
            self.algorithm_nohla_rl_walk_control_model.gait_phase_ratio_walk
        )

        # run algorithm
        joint_target_position_nohla_urdf = self.algorithm_nohla_rl_walk_control_model.run(
            init_output=init_output,
            commands=commands,
            base_measured_quat_to_world=base_measured_quat_to_world,
            base_measured_rpy_vel_to_self=base_measured_rpy_vel_to_self,
            joint_measured_position_urdf=joint_measured_position_nohla_urdf,
            joint_measured_velocity_urdf=joint_measured_velocity_nohla_urdf,
        )  # unit : rad

        joint_target_position_nohla_urdf = numpy.rad2deg(joint_target_position_nohla_urdf)  # unit : deg

        joint_target_position_urdf = numpy.rad2deg(self.default_position.copy())  # unit : deg

        for i in range(self.algorithm_nohla_rl_walk_control_model.num_of_joints):
            index = self.algorithm_nohla_rl_walk_control_model.index_of_joints_real_robot[i]
            joint_target_position_urdf[index] = joint_target_position_nohla_urdf[i]

        # print("algorithm_rl_walk joint_target_position = \n", joint_target_position)
        if self.act:
            self.client.move_joints(ControlGroup.ALL, joint_target_position_urdf, 0.0)


def main(comm_freq: int = 400, step_freq: int = 100, act: bool = False):
    walker = DemoNohlaRLWalk(comm_freq=comm_freq, step_freq=step_freq, act=act)

    # start the scheduler
    schedule(walker.step, interval=1 / step_freq)

    # run the scheduler
    run_loop()


if __name__ == "__main__":
    typer.run(main)
