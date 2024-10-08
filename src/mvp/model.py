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
        self.__dT = 0.1 # s
        
        self.raw_radar = self.raw_radar_range = self.max_i = None
        self.__estimations = None
        self.__targets = None
        self.__tracks = {}
        
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
    def targets(self):
        return self.__targets
    
    @property
    def tracks(self):
        return self.__tracks
    
    @property
    def ground_truth(self):
        return self.__ground_truth
    
    @property
    def trans_pos(self):
        return self.__trans_pos
        
    def __init_nodes(self):
        # Init the Transceiver and Targets
        
        ## Targets
        tar1_pos = [-1, -1, 0]
        tar2_pos = [-5, 5, 0]
        tar3_pos = [3, -3, 0]
        
        rc1 = 1 # reflection coefficient
        rc2 = 1
        rc3 = 1
        
        Target(tar1_pos, 'constant', reflection_coeff=rc1)
        Target(tar2_pos, 'linear', reflection_coeff=rc2, speed=.5, direction=[1, -1, 0])
        Target(tar3_pos, 'circular', reflection_coeff=rc3, speed=1, diameter=2)
        
        
        ## Transceiver
        trans1_pos = [0, -10, 0] 
        trans2_pos = [10, 10, 0]
        trans3_pos = [-10, 10, 0]
        
        noise = .05
        
        Transceiver(trans1_pos, noise)
        Transceiver(trans2_pos, noise)
        Transceiver(trans3_pos, noise)
        
        
        
        self.__trans_pos = np.hstack([trans.position for trans in Transceiver.all])
        self.__ground_truth = np.hstack([tar.position for tar in Target.all])
    
    def __move_all_targets(self, dT):
        for i, target in enumerate(Target.all):
            target.next_pos(dT)
            self.__ground_truth[:, i] = target.position.flatten()
            
    def __measure_wall_transceivers(self):
        H = None
        for i, transceiver in enumerate(Transceiver.all):
            if i == 0:
                H = transceiver.measure_targets(Target.all)
            else:
                H = np.hstack((H, transceiver.measure_targets(Target.all)))
        return H
        
    def step(self, verbose=False):
        self.__move_all_targets(self.dT)
        H = self.__measure_wall_transceivers()
        if verbose:
            self.__targets, self.__estimations, self.raw_radar, self.raw_radar_range, self.max_i, self.__tracks = self.tracker.track(H, Transceiver.BW, self.trans_pos, self.dT, verbose)
        else:
            self.__tracks = self.tracker.track(H, Transceiver.BW, self.trans_pos, self.dT)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        