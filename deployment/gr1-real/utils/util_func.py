import os
import yaml
import serial

import numpy as np


def yamlToDict(file_path):
    with open(os.path.join(os.getcwd(), file_path), 'r') as f:
        cfg_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    return cfg_dict


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.atan2(siny_cosp, cosy_cosp)
    
    return eulerVec

def eulerToQuat(euler):
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([x, y, z, w])

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.

    Parameters:
    q1, q2 -- numpy arrays representing quaternions [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([x, y, z, w])

def apply_quaternion_to_vector(q, v):
    """
    Applies the rotation represented by quaternion q to vector v.

    Parameters:
    q -- numpy array representing a quaternion [x, y, z, w]
    v -- numpy array representing a 3D vector [a, b, c]
    """
    q_vector = np.append(v, 0.0)  # Convert vector to quaternion
    q_conjugate = np.array([-q[0], -q[1], -q[2], q[3]])

    qv = quaternion_multiply(q, q_vector)
    qvq = quaternion_multiply(qv, q_conjugate)

    return qvq[:3]  # Return only the vector part


def configure_serial(port, baudrate):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = int(baudrate)
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    return ser


def quat_rotate_inverse_np(q, v):
    # Extract the scalar part (q_w) and vector part (q_vec) from the quaternion
    q_w = q[-1]
    q_vec = q[:3]
    
    # Compute part 'a': Scaled vector based on quaternion's scalar component
    a = v * (2.0 * q_w ** 2 - 1.0)

    # Compute part 'b': Cross product of q_vec and v, scaled by quaternion's scalar component
    b = np.cross(q_vec, v) * q_w * 2.0
    
    # Compute part 'c': Dot product of q_vec and v, scaled by q_vec
    dot_product = np.dot(q_vec, v)
    c = q_vec * (dot_product * 2.0)

    return a - b + c

