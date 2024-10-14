import pygame
import time
from multiprocessing import Process, Queue
from collections import deque


class JoystickProcess(Process):
    def __init__(self, buffer):
        super().__init__()
        
        self.buffer = buffer
        
    def run(self) -> None:
        pygame.init()
        pygame.joystick.init()
        pygame.joystick.Joystick(0).init()
        
        self.num_buttons = pygame.joystick.Joystick(0).get_numbuttons()
        self.num_axes = pygame.joystick.Joystick(0).get_numaxes()
        self.num_hats = pygame.joystick.Joystick(0).get_numhats()
        
        while True:
            data = {
                "joystick": self.get_joystick(),
                "buttons": self.get_btns()
            }
            self.buffer.put(data)
    
    def _get_axis(self, axis):
        return pygame.joystick.Joystick(0).get_axis(axis)

    def _get_button(self, button):
        return pygame.joystick.Joystick(0).get_button(button)

    def _get_hat(self, hat):
        return pygame.joystick.Joystick(0).get_hat(hat)
    
    def get_joystick(self):
        l_x = self._get_axis(0) # right positive
        l_y = self._get_axis(1) # down positive
        r_x = self._get_axis(2) # right positive
        r_y = self._get_axis(3) # down positive
        
        # change to right positive, up positive
        left_stick = [l_x, -l_y]
        right_stick = [r_x, -r_y]
        
        return (left_stick, right_stick)
    
    def get_btns(self):
        A_button = self._get_button(0)
        B_button = self._get_button(1)
        X_button = self._get_button(2)
        Y_button = self._get_button(3)
        
        return [A_button, B_button, X_button, Y_button]


class Joystick:
    def __init__(self):
        self.last_data = {
            "joystick": ([0, 0], [0, 0]),
            "buttons": [0, 0, 0, 0]
        }
        self.joystick_buffer = Queue(maxsize=100)
        
        self.joystick_process = JoystickProcess(buffer=self.joystick_buffer)
        self.joystick_process.daemon = True
        self.joystick_process.start()
        
    def read(self):
        try:
            data = self.joystick_buffer.get_nowait()
            self.last_data = data.copy()
        except:
            data = self.last_data
        
        return data


if __name__ == "__main__":
    # axis
    # 0: left stick x
    # 1: left stick y
    # 2: right stick x
    # 3: right stick y

    joy = Joystick()
    for _ in range(1000):
        print(joy.read())
        print("================================")
        time.sleep(0.02)
    exit()