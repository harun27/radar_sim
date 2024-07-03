# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:00:33 2024

@author: harun
"""

import unittest
import numpy as np

from radarsim.target import Target
from radarsim.transceiver import Transceiver
from radarsim.tracker import Tracker

class Tests(unittest.TestCase):
    def test_target(self):
        pos = np.array([0, 0, 0]).reshape(3, 1)
        direction = np.array([1, 2, 0])
        tar = Target(pos, 'linear', speed=1.2, direction=direction)
        self.assertTrue(np.array_equal(tar.position, pos))
        pos2 = [[1], [1], [1]]
        tar.position = pos2
        self.assertTrue(np.array_equal(tar.position, pos2))
        
        Target.remove_all()
        
    def test_target_movement_linear(self):
        pos = np.array([1, 1, 0.5]).reshape(3, 1)
        direction = np.array([1, 2, 3]).reshape(3, 1)
        speed = 3
        dT = 0.4
        tar = Target(pos, 'linear', speed=speed, direction=direction)

        next_pos_test = pos + direction / np.sum(np.abs(direction)) * speed * dT
        tar.next_pos(dT)
        self.assertTrue(np.array_equal(tar.position, next_pos_test))

        next_pos_test += direction / direction.sum() * speed * dT
        tar.next_pos(dT)
        self.assertTrue(np.array_equal(tar.position, next_pos_test))
        
        Target.remove_all()
        
    def test_target_movement_constant(self):
        pos = np.array([1, 1, 0.5]).reshape(3, 1)
        direction = np.array([1, 2, 3]).reshape(3, 1)
        speed = 3
        dT = 0.4
        next_pos_test = pos
        tar = Target(pos, 'constant')
        tar.next_pos(dT)
        self.assertTrue(np.array_equal(tar.position, next_pos_test))
        
        Target.remove_all()
        
    def test_target_movement_circular(self):
        """
        This is only a test of the 2 dimensions of the circular movement.
        Improvement: expand the functionality to 3 dimensions
        """
        pos = np.array([1, 3, 0]).reshape(3, 1)
        diam = 2
        speed = 7
        T_circle = np.pi * diam / speed
        dT = T_circle / 4
        tar = Target(pos, 'circular', speed=speed, diameter=diam)
        
        next_pos_test_1 = pos
        next_pos_test_2 = pos + np.array([-diam/2, +diam/2, 0]).reshape(3, 1)
        next_pos_test_3 = pos + np.array([-diam, 0, 0]).reshape(3, 1)
        next_pos_test_4 = pos + np.array([-diam/2, -diam/2, 0]).reshape(3, 1)
        next_pos_test_5 = pos
        
        self.assertTrue(np.allclose(tar.position, next_pos_test_1, rtol=1e-10))
        tar.next_pos(dT)
        self.assertTrue(np.allclose(tar.position, next_pos_test_2, rtol=1e-10))
        tar.next_pos(dT)
        self.assertTrue(np.allclose(tar.position, next_pos_test_3, rtol=1e-10))
        tar.next_pos(dT)
        self.assertTrue(np.allclose(tar.position, next_pos_test_4, rtol=1e-10))
        tar.next_pos(dT)
        self.assertTrue(np.allclose(tar.position, next_pos_test_5, rtol=1e-10))
        
        Target.remove_all()
    
    def test_transceiver_with_target(self):
        pos = np.array([1, 1, 0.5]).reshape(3, 1)
        direction = np.array([1, 2, 3]).reshape(3, 1)
        speed = 3
        dT = 0.4
        tar1 = Target(pos, 'constant')
        tar2 = Target(pos-2, 'linear', speed=speed, direction=direction)
        
        pos_trans1 = [-10, 0, 0]
        pos_trans2 = [0, -10, 0]
        pos_trans3 = [10, 10, 0]
        trans1 = Transceiver(pos_trans1)
        trans2 = Transceiver(pos_trans2)
        trans3 = Transceiver(pos_trans3)
        H = None
        
        for i, transceiver in enumerate(Transceiver.all):
            if i == 0:
                H = transceiver.measure_targets(Target.all)
            else:
                H = np.hstack((H, transceiver.measure_targets(Target.all)))
        
        self.assertEqual(H.shape[1], len(Transceiver.all))
        
        Target.remove_all()
        Transceiver.remove_all()
        
    def test_tracker_localization_single(self):
        pos_trans1 = [-10, 0, 0]
        pos_trans2 = [0, -10, 0]
        pos_trans3 = [10, 10, 0]
        trans1 = Transceiver(pos_trans1)
        trans2 = Transceiver(pos_trans2)
        trans3 = Transceiver(pos_trans3)
        
        pos_tar = np.array([1, 1, 0]).reshape(3, 1)
        tar = Target(pos_tar, 'constant')
        
        for i, transceiver in enumerate(Transceiver.all):
            if i == 0:
                H = transceiver.measure_targets(Target.all)
            else:
                H = np.hstack((H, transceiver.measure_targets(Target.all)))
        
        tracker = Tracker()
        
        trans_pos = np.hstack((trans1.position, trans2.position, trans3.position))
        estimation = tracker.localize(H, Transceiver.B, trans_pos)
        self.assertTrue(np.allclose(estimation, pos_tar, rtol=0.5))
        
        Target.remove_all()
        Transceiver.remove_all()
        
    def test_tracker_localization_multiple(self):
        pos_trans1 = [-10, 0, 0]
        pos_trans2 = [0, -10, 0]
        pos_trans3 = [10, 10, 0]
        trans1 = Transceiver(pos_trans1)
        trans2 = Transceiver(pos_trans2)
        trans3 = Transceiver(pos_trans3)
        
        pos_tar1 = np.array([1, 1, 0]).reshape(3, 1)
        pos_tar2 = np.array([-2, -3, 0]).reshape(3, 1)
        tar1 = Target(pos_tar1, 'constant')
        tar2 = Target(pos_tar2, 'constant')
        
        for i, transceiver in enumerate(Transceiver.all):
            if i == 0:
                H = transceiver.measure_targets(Target.all)
            else:
                H = np.hstack((H, transceiver.measure_targets(Target.all)))
        
        tracker = Tracker()
        
        trans_pos = np.hstack((trans1.position, trans2.position, trans3.position))
        estimation = tracker.localize(H, Transceiver.B, trans_pos)
        
        self.assertTrue(np.allclose(estimation, np.hstack((pos_tar1, pos_tar2)), rtol=0.5) or np.allclose(estimation, np.hstack((pos_tar2, pos_tar1)), rtol=0.5))
        
        Target.remove_all()
        Transceiver.remove_all()
        
    
if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    