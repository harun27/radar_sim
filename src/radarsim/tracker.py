# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:36:15 2024

@author: harun
"""

import numpy as np
import scipy
from itertools import product

class Tracker:
    def __init__(self):
        pass
    
    ##########
    ## Getters
    ##########
    @property
    def R_range(self):
        return self.__R_range
    
    ##########
    ## Methods
    ##########
    def __nearestPowerOf2(self, N):
        # Calculate log2 of N
        a = int(np.log2(N))
     
        # If 2^a is equal to N, return N
        if np.power(2, a) == N:
            return N
         
        # Return 2^(a + 1)
        return np.power(2, a+1)
    
    
    def __preprocessing(self, H, B):
        """Process the measurements
        
        Parameters
        ----------
        H : np.ndarray
            Measurements of all the Transceivers.
        B : float
            Bandwidth of the Transmitters.

        Returns
        -------
        None.

        """
        
        medium = 1 # set to 1 for wave propagation in vacuum
        len_meas = H.shape[0]
        zero_pad = 1 # set to 1 for no zero padding
        len_fft = self.__nearestPowerOf2(len_meas*zero_pad)
        
        # Calculate the Range axis
        R_range = scipy.constants.c / (np.sqrt(medium) * 2*B) * np.linspace(-len_meas/2, len_meas/2, len_fft)
        self.__R_range = R_range[int(len_fft/2):]
        
        # Apply window function
        H = np.hanning(len_meas).reshape(-1, 1) * H
        
        # Apply fft
        H = np.fft.fft(H, n=len_fft, axis=0)
        
        # We simulate here to sample the complex signal, so we take the abs. So we also take only half of the signal since it is even
        H = np.abs(H[:int(len_fft/2), :])
        
        # return it in the dB 
        return 20*np.log10(H)
    
    
    def __multilateration(self, dists, trans_pos, num_transceivers):
        # Setting the A-matrix
        A = np.array([]).reshape(0, 3)
        for i in range (num_transceivers-1):
            A = np.vstack((A, trans_pos[:, -1] - trans_pos[:, i]))
        A = 2*A
        # removing the z-coordinate since all transceivers are in the same z-level for now to plot it
        # otherwise we get a mathematical error. The z-component is then set to 0 again
        A = A[:, :-1]
        z = 0
        
        # here all the coordinates of the possible targets are put in the first 3 columns (x, y, z). The 4th column is used to store the cost function value
        possible_targets = np.array([], dtype=np.float64, ndmin=2).reshape(4, 0)
        
        possible_combinations = list(product(*dists))
        for poss_comb in possible_combinations:
            b = np.array([]).reshape(0, 1)
            for tr in range(num_transceivers-1):
                b = np.append(b, (poss_comb[tr]**2 - poss_comb[-1]**2) - np.sum(np.power(trans_pos[:, tr], 2)) + np.sum(np.power(trans_pos[:, -1], 2)))
            
            coordinate = np.dot(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b).reshape(-1, 1)
                
            # Cost function. Look how far away the predicted and measured ranges are with L2-Loss
            # r: range measurement.
            # f: predicted ranges. predicted coordinate minus transceiver pos
            r = poss_comb
            f = np.sqrt(np.sum(np.power(coordinate - trans_pos[:2, :], 2), axis=0))
            cost_function = np.dot((r-f).T, (r-f))
            
            # reappend the z-coordinate as 0 and append the cost function
            coordinate = np.append(coordinate, [z, cost_function]).reshape(4, 1)
            possible_targets = np.hstack((possible_targets, coordinate))
            
        return possible_targets
        
    
    def localize(self, H, B, trans_pos):
        
        H_db = self.__preprocessing(H, B)
        
        num_transceivers = H.shape[1]
        min_height = 0 # dB
        prominence = 100
        width = None
        # with this function, ``find_peaks()`` we get also some other information, which we filter out by only taking the 0th element of the list
        max_i = [scipy.signal.find_peaks(H_db[:, num], height=min_height, prominence=prominence, width=width)[0] for num in range(num_transceivers)]
        dists = [self.R_range[max_i[num]] for num in range(len(max_i))]
        
        # Find all the possible targets with Multilateration
        possible_targets = self.__multilateration(dists, trans_pos, num_transceivers)
        
        # Set the threshold to filter out the targets. this shall be adaptive later
        threshold = 1e-2
        possible_targets = possible_targets[:, possible_targets[3, :] < threshold]
        return possible_targets[:3, :]
        
    
    def track(self):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    