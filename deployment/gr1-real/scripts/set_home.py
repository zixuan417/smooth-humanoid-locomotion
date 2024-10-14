import time

from robot_rcs_gr.control_system.fi_control_system_gr import ControlSystemGR as ControlSystem

def main():
    ControlSystem().developer_mode(servo_on=False)
    info_dict = ControlSystem().get_info()
    print(info_dict)
    
    algorithm_set_home()
    
def algorithm_set_home():
    from robot_rcs.robot.fi_robot_base_task import RobotBaseTask
    
    ControlSystem().robot_control_set_task_command(task_command=RobotBaseTask.TASK_SET_HOME)
    
    time.sleep(10)
    
if __name__ == "__main__":
    main()
