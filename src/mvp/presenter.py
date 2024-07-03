# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:40:00 2024

@author: harun
"""

from time import time

from .model import Model
from .view import View

class Presenter:
    def __init__(self):
        self.model = Model()
        self.view = View(self.model.trans_pos)
        self.update_time = 0.5 # second
        self.last = time()
    
    def run(self):
        if (time() - self.last) > self.update_time:
            self.model.step()
            self.view.step(self.model.estimations, self.model.ground_truth)
            self.last = time()
        else:
            pass
        
        