# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:39:53 2024

@author: harun
"""

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class DynamicEllipsePlot:
    def __init__(self, ax):
        self.ax = ax
        self.ellipse = None

    def plot_covariance_ellipse(self, mean, cov, conf_perc=0.95, facecolor='none', **kwargs):
        """
        Plots the covariance ellipse of a Gaussian distribution.
        
        Parameters:
        mean : array_like, shape (2,)
            The mean (center) of the Gaussian distribution.
        cov : array_like, shape (2, 2)
            The 2x2 covariance matrix.
        conf_perc : float
            The percentage of the confidence interval
        facecolor : str
            The fill color of the ellipse.
        **kwargs : additional keyword arguments
            Passed to matplotlib.patches.Ellipse.
        """
        # Remove the previous ellipse if it exists
        if self.ellipse:
            self.ellipse.remove()

        # Eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues and eigenvectors by descending eigenvalue
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        # Calculate the angle of rotation for the ellipse
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        
        # Width and height of the ellipse are 2 * n_std * sqrt(eigenvalue)
        conf_factor = np.sqrt(-2 * np.log(1 - conf_perc))
        width, height = 2 * conf_factor * np.sqrt(eigvals)
        # The factor of 2 is necessary because the Ellipse object in matplotlib expects the full width and height
        
        # Create the ellipse patch
        self.ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)
        
        # Add the ellipse to the plot
        self.ax.add_patch(self.ellipse)

class View:
    def __init__(self, trans_pos, dT, verbose=False):
        self.dT = dT
        
        # Some plots may need to reinitialize in the step() funciton, if done, this can be set to True
        self.reinit = False
        
        # Initialize the map plot
        self.num_trans = trans_pos.shape[1]
        self.map_fig, self.map_ax = plt.subplots() # ("Targets and Transceivers")
        self.map_ax.set_title("Iteration " + str(0) + "\nTime: " + str(0) + "s")
        
        self.map_ax.scatter(*zip(*trans_pos[:2, :].T), marker='x', color='b', label='Transceivers')
        self.__text_offset = 0.5
        for i in range(self.num_trans):
            text = "Tr" + str(i)
            self.map_ax.text(*trans_pos[:2, i].T + self.__text_offset, text)
        
        self.map_gt = self.map_ax.scatter([], [], marker='o', color='k', label='GT Targets', s=100, alpha=0.3)
        
        if verbose:
            self.map_estim = self.map_ax.scatter([], [], marker='o', label='Clusters', s=30, edgecolors='k', alpha=1)
            self.map_cluscen = self.map_ax.scatter([], [], marker='x', label='Cluster Center', color='k')
            self.map_kf = self.map_ax.scatter([], [], marker='x', label='KF Estimation', color='g')
            
            # Initalizing raw data plot
            self.raw_fig, self.raw_ax = plt.subplots(self.num_trans, 1)
            self.raw_plot = []
            self.raw_max = []
            self.raw_text = []
        
        
        self.map_ax.legend()
        self.map_ax.grid(True)
        self.map_ax.set_xlabel('x-position [m]')
        self.map_ax.set_ylabel('y-position [m]')
        
        self.ellipse_plot = DynamicEllipsePlot(self.map_ax)
        
        
    
    def step(self, targets, estimations, ground_truth, iter_num, verbose=False, **kwargs):
        # Redrawing map plot
        self.map_gt.set_offsets(ground_truth[:2, :].T)
        self.map_ax.set_title("Iteration " + str(iter_num) + "\nTime: " + str(iter_num*self.dT) + "s")
        
        # Redrawing verbose plots such as the raw data plot or error plot
        if verbose:
            ## Initializing verbose plots
            if not self.reinit:
                # These instructions should only be called one time for initialization purposes
                # After it is done, self.reinit is set to true, meaning that the reinitialization is done
                
                ### Initializing map plot
                self.map_estim.set_cmap('tab20')
                # note: maximum 20 clusters can be used because of the used colormap
                
                ### Initializing Raw data plot
                for ax in self.raw_ax:
                    self.raw_plot.append(ax.plot([], [], color='b', label='Raw measurement')[0])
                    self.raw_max.append(ax.scatter([], [], color='k', label='Maxima'))
                    self.raw_text.append(ax.text(0, 0, ""))
                    ax.grid(True)
                    ax.legend()
                    ax.set_xlabel('x-position [m]')
                    ax.set_ylabel('H [dB]')
                    ax.set_xlim(0, kwargs['x_raw'][-1])
                    ax.set_ylim(-200, 100) # max(kwargs['y_raw'][:, i]))
                
                self.reinit = True
            
            ## Redrawing verbose things on the map plot
            if estimations.size != 0:
                self.map_estim.set_offsets(estimations[:2, :].T)
                self.map_estim.set_array(estimations[4, :])
                self.map_cluscen.set_offsets(targets.T)
                
            kf_targets = kwargs['kf_targets']
            for i, tar in enumerate(kf_targets):
                mean = tar[0][:2]
                C = tar[1][:2, :2]
                self.map_kf.set_offsets(mean)
                self.ellipse_plot.plot_covariance_ellipse(mean, C, facecolor='b', alpha=0.1)
            
            ## Redrawing Raw plot
            for i in range(self.num_trans):                        
                self.raw_text[i].set_text("Maxima: " + str(kwargs['x_raw'][kwargs['idx_raw'][i]]))
                self.raw_plot[i].set_data(kwargs['x_raw'], kwargs['y_raw'][:, i])
                self.raw_max[i].set_offsets(np.array([kwargs['x_raw'][kwargs['idx_raw'][i]], kwargs['y_raw'][kwargs['idx_raw'][i], i]]).T)
                
            self.raw_fig.canvas.draw_idle()
        
        self.map_fig.canvas.draw_idle()
        self.draw()
        
        
    
    def draw(self):
        ## This function redraws the figures. This is either used directly from the self.step() method
        ## or the presenter to make the plots responsive if paused
        figs = list(map(plt.figure, plt.get_fignums()))
        for fig in figs:
            fig.canvas.flush_events()
        
    def close(self):
        ## Closing all figures
        print("Closing Figures")
        plt.close('all')
        print("Closed Figures")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        