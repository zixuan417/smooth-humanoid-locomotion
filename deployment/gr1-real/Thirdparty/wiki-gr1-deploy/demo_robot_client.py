import threading
import time

import msgpack_numpy as m
import numpy as np
import zenoh
from rich.console import Console
from rich.pretty import pprint
from rich.prompt import Confirm, Prompt
from rich.table import Table

from robot_rcs_gr.sdk import ControlGroup, RobotClient

m.patch()

zenoh.init_logger()
console = Console()
log = console.log
print = console.print

FREQUENCY = 150


def task_enable(client: RobotClient):
    client.set_enable(True)


def task_disable(client: RobotClient):
    client.set_enable(False)


def task_set_home(client: RobotClient):
    client.set_home()


def task_reboot(client: RobotClient):
    client.reboot()


def task_move_left_arm_to_default(client: RobotClient):
    client.set_enable(True)
    time.sleep(0.5)
    client.move_joints(
        ControlGroup.LEFT_ARM,
        client.default_group_positions[ControlGroup.LEFT_ARM],
        2.0,
        blocking=False,
    )


def task_move_to_default(client: RobotClient):
    client.set_enable(True)
    time.sleep(0.5)
    client.move_joints(
        ControlGroup.UPPER,
        client.default_group_positions[ControlGroup.UPPER],
        2.0,
        blocking=False,
    )


def task_abort(client: RobotClient):
    client.abort()


def task_print_states(client: RobotClient):
    table = Table("Type", "Data", title="Current :robot: state")
    for sensor_type, sensor_data in client.states.items():
        for sensor_name, sensor_reading in sensor_data.items():
            print(sensor_type + "/" + sensor_name, sensor_reading.tolist())
            table.add_row(
                sensor_type + "/" + sensor_name,
                str(np.round(sensor_reading, 3)),
            )
    print(table)


def task_list_frames(client: RobotClient):
    frames = client.list_frames()
    print(frames)


def task_get_transform(client: RobotClient):
    q = client.joint_positions.copy()
    q[-3] += 20.0
    # q = None
    transform = client.get_transform("base", "l_wrist_roll", q=q)
    print(f"Transform from base to l_wrist_roll: {transform} in configuration q={q}")


def record(client: RobotClient):
    traj = []
    client.set_enable(False)

    time.sleep(1)

    reply = Prompt.ask("Move to start position and press enter")
    if reply == "":
        client.update_pos()
        time.sleep(0.1)
        client.set_enable(True)
        time.sleep(1)
        for sensor_type, sensor_data in client.states.items():
            for sensor_name, sensor_reading in sensor_data.items():
                if sensor_type == "joint":
                    print(sensor_type + "/" + sensor_name, sensor_reading.tolist())
    else:
        return
    time.sleep(0.5)
    reply = Confirm.ask("Start recording?")

    if not reply:
        return

    # client.update_pos()
    client.set_enable(False)
    time.sleep(1)
    event = threading.Event()

    def inner_task():
        while not event.is_set():
            client.loop_manager.start()
            traj.append(client.joint_positions.copy())
            client.loop_manager.end()
            client.loop_manager.sleep()

    thread = threading.Thread(target=inner_task)
    thread.daemon = True
    thread.start()

    reply = Prompt.ask("Press enter to stop recording")
    if reply == "":
        event.set()
        thread.join()

        client.update_pos()
        time.sleep(0.1)
        client.set_enable(True)
        np.save("record.npy", traj)
        return traj


def task_record(client: RobotClient):
    traj = record(client)
    pprint(traj)


def play(recorded_traj: list[np.ndarray], client: RobotClient):
    client.set_enable(True)
    time.sleep(1)

    first = recorded_traj[0]
    client.move_joints(ControlGroup.ALL, first, 2.0, blocking=True)
    for pos in recorded_traj[1:]:
        client.move_joints(ControlGroup.ALL, pos, duration=0.0)
        time.sleep(1 / FREQUENCY)
    time.sleep(1)
    client.set_enable(False)


def task_play(client: RobotClient):
    rec = np.load("record.npy", allow_pickle=True)
    play(rec, client)


def task_set_gains(client: RobotClient):
    kp = np.array([0.1] * 32)
    kd = np.array([0.01] * 32)
    new_gains = client.set_gains(kp, kd)
    print(new_gains)


def task_exit(client: RobotClient):
    import sys

    client.close()
    sys.exit(0)


if __name__ == "__main__":
    client = RobotClient(FREQUENCY)
    time.sleep(0.5)
    while True:
        task = Prompt.ask(
            "What do you want the :robot: to do?",
            choices=[
                "enable",
                "disable",
                "set_home",
                "set_gains",
                "reboot",
                "print_states",
                "move_to_default",
                "record",
                "play",
                "abort",
                "list_frames",
                "get_transform",
                "exit",
            ],
        )
        if task == "enable":
            task_enable(client)
        elif task == "disable":
            task_disable(client)
        elif task == "set_home":
            task_set_home(client)
        elif task == "set_gains":
            task_set_gains(client)
        elif task == "reboot":
            task_reboot(client)
        elif task == "move_to_default":
            task_move_to_default(client)
        elif task == "abort":
            task_abort(client)
        elif task == "print_states":
            task_print_states(client)
        elif task == "record":
            task_record(client)
        elif task == "play":
            task_play(client)
        elif task == "exit":
            task_exit(client)
        elif task == "list_frames":
            task_list_frames(client)
        elif task == "get_transform":
            task_get_transform(client)

        time.sleep(0.5)

    # client.spin()
    # time.sleep(1)
    # client.set_enable(False)
