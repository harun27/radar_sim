# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:40:00 2024

@author: harun
"""

from time import time
import keyboard

from .model import Model
from .view import View

class Presenter:
    def __init__(self):
        self.model = Model()
        self.view = View(self.model.trans_pos)
        self.update_time = 0.1 # seconds until the plot is reloaded and new steps are calculated
        self.last = time()
        self.running = True
        self.step = False
        self.quit = False
        keyboard.on_press(self.__on_key_event)
        
    def __on_key_event(self, e):
        if e.name == 'c':
            self.running = True
            print("continue")
        if e.name == 'p':
            self.running = False
            print("pause")
        if e.name == 'q':
            print("quit")
            self.view.close()
            self.quit = True
            keyboard.unhook_all()
        if e.name == 's':
            self.step = True
            print("step")
    
    def run(self):
        
        if (((time() - self.last) > self.update_time) and self.running) or self.step:
            self.model.step()
            self.view.step(self.model.estimations, self.model.ground_truth)
            self.last = time()
            self.step = False
        else:
            pass
        
        if self.quit:
            return False
        else:
            return True