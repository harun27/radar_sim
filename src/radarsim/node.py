# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:36:47 2024

@author: harun
"""

import numpy as np

class Node:
    def __init__(self, pos):
        self.position = pos
        
    @property
    def position(self):
        return self.__position
    
    @position.setter
    def position(self, pos):
        if type(pos) == list:
            assert len(pos) == 3, "The dimension of the position must be 3"
            self.__position = np.array(pos, dtype=np.float64).reshape(3, 1)
        elif type(pos) == np.ndarray:
            assert pos.size == 3, "The dimension of the position must be 3"
            self.__position = np.array(pos, dtype=np.float64).reshape(3, 1)
        else:
            raise AssertionError("The position should be a numpy array or a list")
        