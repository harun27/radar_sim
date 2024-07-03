# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:39:53 2024

@author: harun
"""

import matplotlib.pyplot as plt

class View:
    def __init__(self, trans_pos):
        ## Initialize the map plot
        num_trans = trans_pos.shape[1]
        self.map_fig, self.map_ax = plt.subplots() # ("Targets and Transceivers")
        self.map_ax.scatter(*zip(*trans_pos[:2, :].T), marker='x', color='b', label='Transceivers')
        self.map_gt = self.map_ax.scatter([], [], marker='o', color='k', label='Targets', s=100)
        self.map_estim = self.map_ax.scatter([], [], marker='o', color='c', label='Estimation', s=30, edgecolors='k')
        
        self.__text_offset = 0.5
        for i in range(num_trans):
            text = "Tr" + str(i)
            plt.text(*trans_pos[:2, i].T + self.__text_offset, text)
        
        plt.legend()
        plt.grid(True)
        plt.xlabel('x-position [m]')
        plt.ylabel('y-position [m]')
        plt.show()
        
    
    def step(self, estimations, ground_truth):
        # Redrawing map plot
        #[text.set_position(tars[num].get_pos()[:2] + text_offset) for num, text in enumerate(targ_texts)]
        self.map_gt.set_offsets(ground_truth[:2, :].T)
        self.map_estim.set_offsets(estimations[:2, :].T)
        self.map_fig.canvas.draw_idle()
        plt.pause(0.05)