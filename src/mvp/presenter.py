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
        self.last = time()
        self.running = True
        self.step = False
        self.quit = False
        self.verbose = True
        self.model = Model()
        self.view = View(self.model.trans_pos, self.model.dT, self.verbose)
        self.iter_num = 0
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
            self.quit = True
            keyboard.unhook_all()
        if e.name == 's':
            self.step = True
            print("step")
    
    def run(self):
        
        if (((time() - self.last) > self.model.dT) and self.running) or self.step:
            self.iter_num += 1
            self.model.step(self.verbose)
            self.view.step(self.model.targets, self.model.estimations, self.model.ground_truth, self.iter_num, self.verbose, y_raw=self.model.raw_radar, x_raw=self.model.raw_radar_range, idx_raw=self.model.max_i, kf_targets=self.model.kf_targets)
            self.last = time()
            self.step = False
        else:
            self.view.draw()
        
        if self.quit:
            self.view.close()
            return False
        else:
            return True