import numpy as np

def cal_1307e ( new_kp, new_kd):
    gw = 21 * 7 /360
    old_kd = new_kd /(gw * 0.255 * 7 * 180/np.pi )
    old_kp = new_kp /(old_kd * 7 * 0.255 * 7  * 180/np.pi)
    return old_kp, old_kd


def cal_601750 ( new_kp, new_kd):
    gw = 10 * 50 /360
    old_kd = new_kd /(gw * 0.0846 * 50 * 180/np.pi )
    old_kp = new_kp /(old_kd * 50 * 0.0826 * 50  * 180/np.pi)
    return old_kp, old_kd

def cal_802030 ( new_kp, new_kd):
    gw = 21 * 30 / 360
    old_kd = new_kd /(gw * 0.11 * 30 * 180/np.pi )
    old_kp = new_kp /(old_kd * 30 * 0.11 * 30  * 180/np.pi)
    return old_kp, old_kd

def cal_802028 ( new_kp, new_kd):
    gw = 21 * 28.8 / 360
    old_kd = new_kd /(gw * 0.07 * 28.8 * 180/np.pi )
    old_kp = new_kp /(old_kd * 28.8 * 0.07 * 28.8  * 180/np.pi)
    return old_kp, old_kd

def cal_361480 ( new_kp, new_kd):
    gw = 10 * 80 / 360
    old_kd = new_kd /(gw * 0.067 * 80 * 180/np.pi )
    old_kp = new_kp /(old_kd * 80 * 0.067 * 80  * 180/np.pi)
    return old_kp, old_kd

def cal_3611100 ( new_kp, new_kd):
    gw = 10 * 100 / 360
    old_kd = new_kd /(gw * 0.05 * 100 * 180/np.pi )
    old_kp = new_kp /(old_kd * 100 * 0.05 * 100  * 180/np.pi)
    return old_kp, old_kd


### use ankle pitch sim pd is ok

def cal_36b36e ( new_kp, new_kd):
    
    gw = 10 * 36 / 360
    old_kd = new_kd /(gw * 0.0688 * 36 * 180/np.pi )
    old_kp = new_kp /(old_kd * 36 * 0.0688 * 36  * 180/np.pi)
    return old_kp, old_kd


if __name__ == "__main__":
    sim_stiffness_dict = {'hip_roll': 200, 'hip_yaw': 200, 'hip_pitch': 200, 'knee_pitch': 200, 'ankle_pitch': 40, 'ankle_roll': 40,
                 'waist_yaw': 300, 'waist_pitch': 300, 'waist_roll': 300,
                 'shoulder_pitch': 40, 'shoulder_roll': 40, 'shoulder_yaw': 40, 'elbow_pitch': 40}
    
    sim_damping_dict = {'hip_roll': 5, 'hip_yaw': 5, 'hip_pitch': 5, 'knee_pitch': 5, 'ankle_pitch': 2, 'ankle_roll': 2,
                    'waist_yaw': 6, 'waist_pitch': 6, 'waist_roll': 6,
                    'shoulder_pitch': 2, 'shoulder_roll': 2, 'shoulder_yaw': 2, 'elbow_pitch': 2}
    
    hip_roll_kp, hip_roll_kd = cal_802030(sim_stiffness_dict['hip_roll'], sim_damping_dict['hip_roll'])
    hip_yaw_kp, hip_yaw_kd = cal_601750(sim_stiffness_dict['hip_yaw'], sim_damping_dict['hip_yaw'])
    hip_pitch_kp, hip_pitch_kd = cal_1307e(sim_stiffness_dict['hip_pitch'], sim_damping_dict['hip_pitch'])
    knee_pitch_kp, knee_pitch_kd = cal_1307e(sim_stiffness_dict['knee_pitch'], sim_damping_dict['knee_pitch'])
    ankle_pitch_kp, ankle_pitch_kd = cal_36b36e(sim_stiffness_dict['ankle_pitch'], sim_damping_dict['ankle_pitch'])
    ankle_roll_kp, ankle_roll_kd = cal_36b36e(sim_stiffness_dict['ankle_roll'], sim_damping_dict['ankle_roll'])
    
    waist_yaw_kp, waist_yaw_kd = cal_601750(sim_stiffness_dict['waist_yaw'], sim_damping_dict['waist_yaw'])
    waist_pitch_kp, waist_pitch_kd = cal_601750(sim_stiffness_dict['waist_pitch'], sim_damping_dict['waist_pitch'])
    waist_roll_kp, waist_roll_kd = cal_601750(sim_stiffness_dict['waist_roll'], sim_damping_dict['waist_roll'])
    
    shoulder_pitch_kp, shoulder_pitch_kd = cal_361480(sim_stiffness_dict['shoulder_pitch'], sim_damping_dict['shoulder_pitch'])
    shoulder_roll_kp, shoulder_roll_kd = cal_361480(sim_stiffness_dict['shoulder_roll'], sim_damping_dict['shoulder_roll'])
    shoulder_yaw_kp, shoulder_yaw_kd = cal_3611100(sim_stiffness_dict['shoulder_yaw'], sim_damping_dict['shoulder_yaw'])
    elbow_pitch_kp, elbow_pitch_kd = cal_3611100(sim_stiffness_dict['elbow_pitch'], sim_damping_dict['elbow_pitch'])
    
    joint_pd_control_kp = [hip_roll_kp, hip_yaw_kp, hip_pitch_kp, knee_pitch_kp, ankle_pitch_kp, ankle_roll_kp,
                           hip_roll_kp, hip_yaw_kp, hip_pitch_kp, knee_pitch_kp, ankle_pitch_kp, ankle_roll_kp,
                           waist_yaw_kp, waist_pitch_kp, waist_roll_kp,
                           shoulder_pitch_kp, shoulder_roll_kp, shoulder_yaw_kp, elbow_pitch_kp,
                           shoulder_pitch_kp, shoulder_roll_kp, shoulder_yaw_kp, elbow_pitch_kp]
    joint_pd_control_kd = [hip_roll_kd, hip_yaw_kd, hip_pitch_kd, knee_pitch_kd, ankle_pitch_kd, ankle_roll_kd,
                            hip_roll_kd, hip_yaw_kd, hip_pitch_kd, knee_pitch_kd, ankle_pitch_kd, ankle_roll_kd,
                            waist_yaw_kd, waist_pitch_kd, waist_roll_kd,
                            shoulder_pitch_kd, shoulder_roll_kd, shoulder_yaw_kd, elbow_pitch_kd,
                            shoulder_pitch_kd, shoulder_roll_kd, shoulder_yaw_kd, elbow_pitch_kd]
    
    np.set_printoptions(precision=3)
    
    print("kp: ", np.array(joint_pd_control_kp))
    print("kd: ", np.array(joint_pd_control_kd))