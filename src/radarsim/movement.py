# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:37:06 2024

@author: harun
"""
import numpy as np

class Movement:
    movement_types_list = ['circular', 'linear', 'constant']
    movement_parameter_list = ['speed', 'diameter', 'direction']
    
    ##############
    ## Constructor
    ##############
    def __init__(self, movement_type, speed=0, diameter=0, direction=None):
        self.__movement_type = movement_type
        self.__speed = speed
        self.__diameter = diameter
        self.__direction = direction
        self.__movement_vector = np.zeros((3, 1), dtype=np.float64)
        self.__T_circle = 0
        self.__angular_speed = 0
        self.__offset = np.array([], dtype=np.float64)
        self.__phi = 0
        
        self.__set_movement_type()
        
    ##########
    ## Getters
    ##########
    @property
    def movement_type(self):
        return self.__movement_type
    
    @property
    def speed(self):
        return self.__speed
    
    @property
    def diameter(self):
        return self.__diameter
    
    @property
    def direction(self):
        return self.__direction
    
    @property
    def movement_vector(self):
        return self.__movement_vector
    
    @property
    def T_circle(self):
        return self.__T_circle
    
    @property
    def angular_speed(self):
        return self.__angular_speed
    
    @property
    def offset(self):
        return self.__offset
    
    @property
    def phi(self):
        return self.__phi
    
    ##########
    ## Methods
    ##########
    def __set_movement_type(self):
        assert self.movement_type in self.movement_types_list, "You can only specify one of the given movement types"
        
        if self.movement_type == 'constant':
            return
        else: # here we will have a movement, not a constant
            assert self.speed != 0, "You must specify a speed"
            if self.movement_type == 'linear':
                if type(self.direction) == list:
                    assert len(self.direction) == 3, "The direction vector must have 3 elements"
                    self.__direction = np.array(self.direction, dtype=np.float64)
                elif type(self.direction) == np.ndarray:
                    assert self.direction.size == 3, "The direction vector must have 3 elements"
                    self.__direction = np.array(self.direction, dtype=np.float64)
                else:
                    raise AssertionError("Specify a direction vector as a list or a numpy array")
                self.__movement_vector = self.direction.reshape(3, 1) / np.sum(np.abs(self.direction)) * self.speed
            elif self.movement_type == 'circular':
                assert self.diameter != 0, "You must specify a diameter for circular motion"
                s_circle = np.pi * self.diameter
                self.__T_circle = s_circle / self.speed
                self.__angular_speed = 2 * self.speed / self.diameter
            else:
                raise ValueError("Control the movement type")
    
    def __pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        z = 0
        return np.array([x, y, z], dtype=np.float64).reshape(3, 1)
    
    def next_pos(self, current_pos, dT):
        """This calculates the next position, given the current position.

        Parameters
        ----------
        current_pos : np.ndarray
            The current position.
        dT : float
            The time difference between the current and next position.
            In other words: the time difference between the 2 measurements.

        Returns
        -------
        next_pos : np.ndarray
            The next position.

        """
        if self.movement_type == 'constant':
            return current_pos
        elif self.movement_type == 'linear':
            return current_pos + dT * self.movement_vector
        elif self.movement_type == 'circular':
            if self.offset.size == 0:
                self.__offset = current_pos + np.array([-self.diameter/2, 0 , 0], dtype=np.float64).reshape(3, 1)
            self.__phi += self.angular_speed * dT
            pos = self.__pol2cart(self.diameter/2, self.phi)
            return pos + self.offset
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        