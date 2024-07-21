# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:39:53 2024

@author: harun
"""

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

class View:
    def __init__(self, trans_pos, verbose=False):
        # Some plots may need to reinitialize in the step() funciton, if done, this can be set to True
        self.reinit = False
        
        ## Initialize the map plot
        self.num_trans = trans_pos.shape[1]
        self.map_fig, self.map_ax = plt.subplots() # ("Targets and Transceivers")
        self.map_ax.scatter(*zip(*trans_pos[:2, :].T), marker='x', color='b', label='Transceivers')
        self.map_gt = self.map_ax.scatter([], [], marker='o', color='k', label='Targets', s=100)
        self.map_estim = self.map_ax.scatter([], [], marker='o', color='c', label='Estimation', s=30, edgecolors='k')
        
        self.__text_offset = 0.5
        for i in range(self.num_trans):
            text = "Tr" + str(i)
            self.map_ax.text(*trans_pos[:2, i].T + self.__text_offset, text)
        
        self.map_ax.legend()
        self.map_ax.grid(True)
        self.map_ax.set_xlabel('x-position [m]')
        self.map_ax.set_ylabel('y-position [m]')
        
        if verbose:
            self.raw_fig, self.raw_ax = plt.subplots(self.num_trans, 1)
            self.raw_plot = []
            self.raw_max = []
            self.raw_text = []
            for ax in self.raw_ax:
                self.raw_plot.append(ax.plot([], [], color='b', label='Raw measurement')[0])
                self.raw_max.append(ax.scatter([], [], color='k', label='Maxima'))
                self.raw_text.append(ax.text(0, 0, ""))
                ax.grid(True)
                ax.legend()
                ax.set_xlabel('x-position [m]')
                ax.set_ylabel('H [dB]')
        
    
    def step(self, estimations, ground_truth, verbose=False, **kwargs):
        # Redrawing map plot
        self.map_gt.set_offsets(ground_truth[:2, :].T)
        self.map_estim.set_offsets(estimations[:2, :].T)
        self.map_fig.canvas.draw_idle()
        
        # Redrawing raw data plot
        if verbose:
            for i in range(self.num_trans):
                if not self.reinit:
                    self.raw_ax[i].set_xlim(0, kwargs['x_raw'][-1])
                    self.raw_ax[i].set_ylim(-200, 100) # max(kwargs['y_raw'][:, i]))
                self.raw_text[i].set_text("Maxima: " + str(kwargs['x_raw'][kwargs['idx_raw'][i]]))
                self.raw_plot[i].set_data(kwargs['x_raw'], kwargs['y_raw'][:, i])
                self.raw_max[i].set_offsets(np.array([kwargs['x_raw'][kwargs['idx_raw'][i]], kwargs['y_raw'][kwargs['idx_raw'][i], i]]).T)
            self.raw_fig.canvas.draw_idle()
        
        self.reinit = True
        self.draw()
    
    def draw(self):
        figs = list(map(plt.figure, plt.get_fignums()))
        for fig in figs:
            fig.canvas.flush_events()
        
    def close(self):
        print("Closing Figures")
        plt.close('all')
        print("Closed Figures")
        