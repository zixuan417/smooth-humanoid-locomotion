import sys
import time
import argparse

import numpy as np

sys.path.append("../")
from utils.joystick import Joystick

from robot_rcs_gr.control_system.fi_control_system_gr import ControlSystemGR as ControlSystem

np.set_printoptions(precision=4)

class GR1Robot:
    def __init__(self, num_motors=23, p_scale=0., d_scale=0., est_freq=50, joystick=False) -> None:
        self.p_scale = p_scale
        self.d_scale = d_scale
        
        # Robot state variables
        self._num_motors = num_motors
        
        self._init_complete = False
        self._base_position = np.zeros((3,))
        self._base_orientation = None
        self._last_position_update_time = time.time()
        
        self._raw_state = None
        self._last_raw_state = None
        
        self._motor_angles = np.zeros(self._num_motors)
        self._motor_velocities = np.zeros(self._num_motors)
        
        self._joint_states = None
        self._last_reset_time = time.time()
        
        if joystick:
            self.joystick = Joystick()
        
        self.control_dict = {}
        
        self.control_mode = [
                # left leg
                4, 4, 4, 4, 4, 4,
                # right leg
                4, 4, 4, 4, 4, 4,
                # waist
                4, 4, 4,
                # head
                4, 4, 4,
                # left arm
                4, 4, 4, 4, 4, 4, 4,
                # right arm
                4, 4, 4, 4, 4, 4, 4,
            ]
        
        self.target_pos = np.zeros(23 + 9)
        
        self.control_dof_indices = [0, 1, 2, 3, 4, 5, 
                                    6, 7, 8, 9, 10, 11, 
                                    12, 13, 14,
                                    18, 19, 20, 21,
                                    25, 26, 27, 28]
        
        self.control_system = ControlSystem()
        self.control_system.developer_mode(servo_on=True)
        
        self.joint_pd_control_kp = np.array([
            0.997, 1.023, 1.061, 1.061, 0.508, 0.508,
            0.997, 1.023, 1.061, 1.061, 0.508, 0.508,
            1.023, 1.023*5, 1.023*5,
            0.556, 0.556, 0.556,
            0.556, 0.556, 0.556, 0.556, 0, 0, 0,
            0.556, 0.556, 0.556, 0.556, 0, 0, 0,
        ])
            
        self.joint_pd_control_kd = np.array([
            0.044, 0.03, 0.263, 0.263, 0.004, 0.004,
            0.044, 0.03, 0.263, 0.263, 0.004, 0.004,
            0.03, 0.03, 0.03,
            0.03, 0.03, 0.03,
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
            0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
        ])
        
        self.joint_limit_upper = np.array([
            0.79, 0.7, 0.7, 1.92, 0.52, 0.44,
            0.09, 0.7, 0.7, 1.92, 0.52, 0.44,
            1.05, 1.22, 0.7,
            1.92, 3.27, 2.97, 2.27,
            1.92, 0.57, 2.97, 2.27,
        ])
        
        self.joint_limit_lower = np.array([
            -0.09, -0.7, -1.75, -0.09, -1.05, -0.44,
            -0.79, -0.7, -1.75, -0.09, -1.05, -0.44,
            -1.05, -0.52, -0.7,
            -2.79, -0.57, -2.97, -2.27,
            -2.79, -3.27, -2.97, -2.27,
        ])
        
        self.cur_kp = self.joint_pd_control_kp.copy() * self.p_scale
        self.cur_kd = self.joint_pd_control_kd.copy() * self.d_scale

        # Initiate interface for robot state and actions
        time.sleep(0.1)
        print("Testing observation ...")
        for _ in range(100):
            self.ReceiveObservation()
            time.sleep(0.01)
        print("Observation test passed!")
        
        print("GR1 Robot initialized!")
    
    def ReceiveObservation(self):
        state_dict = self.control_system.robot_control_loop_get_state()
        self._motor_angles = state_dict["joint_position"][self.control_dof_indices] * np.pi / 180.
        self._motor_velocities = state_dict["joint_velocity"][self.control_dof_indices] * np.pi / 180.
        self._base_rpy = state_dict["imu_euler_angle"].copy() * np.pi / 180.
        self._base_quat = state_dict["imu_quat"]
        self._ang_vel = state_dict["imu_angular_velocity"].copy() * np.pi / 180.
        
    def Step(self, pd_targets):
        pd_targets[0] = np.clip(pd_targets[0], -0.09, 0.79)
        pd_targets[1] = np.clip(pd_targets[1], -0.7, 0.7)
        pd_targets[6] = np.clip(pd_targets[6], -0.79, 0.09)
        pd_targets[7] = np.clip(pd_targets[7], -0.7, 0.7)
        self.target_pos[self.control_dof_indices] = pd_targets * 180. / np.pi
        
        self.control_dict.update({
            "control_mode": self.control_mode,
            "kp": self.cur_kp,
            "kd": self.cur_kd,
            "position": self.target_pos,
        })
        
        self.control_system.robot_control_loop_set_control(self.control_dict)
    
    def GetTrueMotorAngles(self):
        return self._motor_angles.copy()
  
    def GetMotorVel(self):
        return self._motor_velocities.copy()
    
    def GetBaseRPY(self):
        return self._base_rpy.copy()
    
    def GetAngVel(self):
        return self._ang_vel.copy()
    
    def GetBaseQuat(self):
        return self._base_quat.copy()
    
    def _set_stand_pd(self):
        self.cur_kp = self.joint_pd_control_kp.copy()
        self.cur_kd = self.joint_pd_control_kd.copy()
    
    def _set_rl_pd(self):
        self.cur_kp = self.joint_pd_control_kp.copy() * self.p_scale
        self.cur_kd = self.joint_pd_control_kd.copy() * self.d_scale

    # =================== Remote Control ===================
    def GetRemoteStates(self):
        data = self.joystick.read()
        if data is None:
            raise ValueError("Joystick not connected!")
        left_joy, right_joy = data["joystick"]
        btns = data["buttons"]
        # A, B, X, Y
        return left_joy, right_joy, btns
    
    def shut_robot(self):
        self.control_system.developer_mode(servo_on=False)
        self.joystick.stop_reading()
        print("Robot shut down!")
