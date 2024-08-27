# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:36:59 2024

@author: harun
"""

import numpy as np
import scipy

from .node import Node

class Transceiver(Node):
    all = []
    f_start = 110e9
    f_stop = 170e9
    BW = abs(f_stop - f_start)
    # BW: Bandwidth of the frequency modulation / sweep
    T = 300e-9
    # T: Periodendauer des Frequenzsweeps (definiert die maximale Reichweite). 
    
    ##############
    ## Constructor
    ##############
    def __init__(self, pos, noise):
        super().__init__(pos)
        
        num_freq_points = int(Transceiver.BW * Transceiver.T)
        f = np.linspace(Transceiver.f_start, Transceiver.f_stop, num=num_freq_points)
        w = 2 * np.pi * f
        self.__k = (w / scipy.constants.c).reshape(-1, 1)
        self.noise = noise
        
        Transceiver.all.append(self)
    
    ##########
    ## Methods
    ##########
    def __calc_distance(self, pos):
        return np.sqrt(np.sum(np.power(self.position - pos, 2))) + np.random.normal(loc=0, scale=self.noise)
        
    
    def measure_targets(self, targets):
        # This function returns the received echo of the transceiver
        # This is modeled with a dampening with 'amplitude' and a time delay 
        # calculated with the distance to the target
        H = np.zeros((len(self.__k), 1), dtype=np.complex128)
        
        for i, target in enumerate(targets):
            dist = self.__calc_distance(target.position)
            H = np.add(H, target.refl_coeff * np.exp(-1j * 2 * self.__k * dist))
        return np.real(H).reshape(-1, 1)
    
    def remove(self):
        Transceiver.all.remove(self)
        
    def remove_all():
        Transceiver.all = []