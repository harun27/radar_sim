# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:39:41 2024

@author: harun
"""

import numpy as np

from radarsim.target import Target
from radarsim.transceiver import Transceiver
from radarsim.tracker import Tracker


class Model:
    def __init__(self):
        self.__init_nodes()
        self.tracker = Tracker()
        self.__dT = 0.1 # 1s
        
    ##########
    ## Getters
    ##########
    @property
    def dT(self):
        return self.__dT
    
    @property
    def estimations(self):
        return self.__estimations
    
    @property
    def ground_truth(self):
        return self.__ground_truth
    
    @property
    def trans_pos(self):
        return self.__trans_pos
        
    def __init_nodes(self):
        # Init the transceivers and targets
        tar1_pos = [-1, -1, 0]
        tar2_pos = [-5, 5, 0]
        tar3_pos = [3, -3, 0]
        
        tar1 = Target(tar1_pos, 'constant')
        tar2 = Target(tar2_pos, 'linear', speed=0.5, direction=[1, -1, 0])
        tar3 = Target(tar3_pos, 'circular', speed=1, diameter=2)
        
        trans1_pos = [0, -10, 0] 
        trans2_pos = [10, 10, 0]
        trans3_pos = [-10, 10, 0]
        
        trans1 = Transceiver(trans1_pos)
        trans2 = Transceiver(trans2_pos)
        trans3 = Transceiver(trans3_pos)
        
        self.__trans_pos = np.hstack((trans1.position, trans2.position, trans3.position))
        self.__ground_truth = np.hstack((tar1.position, tar2.position, tar3.position))
    
    def __move_all_targets(self, dT):
        for i, target in enumerate(Target.all):
            target.next_pos(dT)
            self.__ground_truth[:, i] = target.position.flatten()
            
    def __measure_all_transceivers(self):
        H = None
        for i, transceiver in enumerate(Transceiver.all):
            if i == 0:
                H = transceiver.measure_targets(Target.all)
            else:
                H = np.hstack((H, transceiver.measure_targets(Target.all)))
        return H
        
    def step(self):
        self.__move_all_targets(self.dT)
        H = self.__measure_all_transceivers()
        self.__estimations = self.tracker.localize(H, Transceiver.B, self.trans_pos)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        