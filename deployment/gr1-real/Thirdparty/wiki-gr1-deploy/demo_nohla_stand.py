import time

import numpy
import typer
from ischedule import run_loop, schedule

from robot_rcs_gr.robot.gr1_nohla.fi_robot_gr1_nohla_algorithm import RobotGR1NohlaAlgorithmStandControlModel
from robot_rcs_gr.sdk import ControlGroup, RobotClient


class DemoNohlaStand:
    """
    Reinforcement Learning Walker
    """

    def __init__(self, comm_freq=400, step_freq=100, act=False):
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

        # algorithm
        algorithm_control_period = 1.0 / step_freq
        self.algorithm_nohla_stand_control_model = RobotGR1NohlaAlgorithmStandControlModel(dt=algorithm_control_period)

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
        joint_measured_position = self.client.states["joint"]["position"].copy()
        joint_measured_velocity = self.client.states["joint"]["velocity"].copy()

        # prepare input
        joint_measured_position_nohla = numpy.zeros(self.algorithm_nohla_stand_control_model.num_of_joints)
        joint_measured_velocity_nohla = numpy.zeros(self.algorithm_nohla_stand_control_model.num_of_joints)

        for i in range(self.algorithm_nohla_stand_control_model.num_of_joints):
            index = self.algorithm_nohla_stand_control_model.index_of_joints_real_robot[i]
            joint_measured_position_nohla[i] = numpy.deg2rad(joint_measured_position[index])
            joint_measured_velocity_nohla[i] = numpy.deg2rad(joint_measured_velocity[index])

        # run algorithm
        _, _, joint_target_position_nohla = self.algorithm_nohla_stand_control_model.run(
            joint_measured_position_urdf=joint_measured_position_nohla,
            joint_measured_velocity_urdf=joint_measured_velocity_nohla,
        )  # [rad]

        joint_target_position_nohla = numpy.rad2deg(joint_target_position_nohla)
        joint_target_position = numpy.zeros_like(joint_measured_position)

        for i in range(self.algorithm_nohla_stand_control_model.num_of_joints):
            index = self.algorithm_nohla_stand_control_model.index_of_joints_real_robot[i]
            joint_target_position[index] = joint_target_position_nohla[i]

        # print("algorithm_rl_walk joint_target_position = \n", joint_target_position)
        if self.act:
            self.client.move_joints(ControlGroup.ALL, joint_target_position, 0.0)

        return joint_target_position


def main(comm_freq: int = 400, step_freq: int = 100, act: bool = False):
    walker = DemoNohlaStand(comm_freq=comm_freq, step_freq=step_freq, act=act)

    # start the scheduler
    schedule(walker.step, interval=1 / step_freq)

    # run the scheduler
    run_loop()


if __name__ == "__main__":
    typer.run(main)
