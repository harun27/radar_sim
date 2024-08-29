# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:36:54 2024

@author: harun
"""

from .node import Node
from .movement import Movement

class Target(Node):
    
    all = []
    
    def __init__(self, pos, movement_type, reflection_coeff=1, **movement_params):
        """This initializes the Target.
        
        Parameters
        ----------
        pos : np.array(3, 1)
            The start position.
        movement_type : STRING
            Possible options are: 'circular', 'linear', 'constant'. (see Movement.movement_types_list)
        **kwargs : DICT
            This contains parameters for the movement type. (see see Movement.movement_parameter_list)
            3 parameters can be set depending on the movement type: `speed`, `direction` and `diameter`.
            For more info, look at the Movement class.
            Example:
                # a linear motion only in 2-D starting from [0,0] making a linear motion to the direciton [1, 2]
                tar = Target(np.array([0, 0, 0]), 'linear', speed=1.2, direction=np.array([1, 2, 0]))

        Returns
        -------
        None.

        """
        super().__init__(pos)
        self.__movement = Movement(movement_type, **movement_params)
        
        assert reflection_coeff > 0, "The reflecion coefficient must be greater than 0"
        self.__refl_coeff = reflection_coeff
        self.__target_id = len(Target.all)
        
        Target.all.append(self)
        
    def remove(self):
        Target.all.remove(self)
        
    @property
    def refl_coeff(self):
        return self.__refl_coeff
    
    @property
    def target_id(self):
        return self.__target_id
    
    @property
    def movement(self):
        return self.__movement
    
        
    def next_pos(self, dT):
        self.position = self.__movement.next_pos(self.position, dT)
        return self.position
    
    def remove_all():
        Target.all = []