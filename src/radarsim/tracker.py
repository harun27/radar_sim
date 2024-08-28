# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:36:15 2024

@author: harun
"""

import numpy as np
import scipy
from itertools import product
from sklearn.cluster import DBSCAN

from .filters import KF, EKF, IMMEstimator, Q_kinematic

class Track:
    status_type = {'tentative': 0, 'terminated': 1, 'confirmed': 2}
    all = []
    
    def __init__(self, position):
        self.status = Track.status_type['tentative']
        self.__pos = np.array(position).reshape(3, 1)
        self.filters = []
        self.IMM = None
        Track.all.append(self)
    
    @property
    def pos(self):
        return self.__pos
        
    @property
    def state(self):
        return self.__state
    
    @property
    def x(self):
        return self.IMM.x
    
    @property
    def P(self):
        return self.IMM.P
    
    def new_position(self, new_pos, dT):
        self.__pos = np.hstack((self.pos, new_pos.reshape(3, 1)))
        
        if self.status == Track.status_type['tentative'] and not self.filters:
            self.status = Track.status_type['confirmed']
            v = [0, 0, 0] # (self.pos[:, -1] - self.pos[:, -2]) / dT
            # a = [0, 0, 0]
            w = 1 # this is arbitrarily set to 1 (this shouldn't be 0 so we don't divide by 0)
            self.__state = np.append(np.dstack((new_pos, v)).flatten(), w)
            self.init_filter(dT)
        else:
            self.IMM.predict()
            self.IMM.update(new_pos)
            
    def predict(self):
        if self.IMM != None:
            self.IMM.predict()
        else:
            print("Filter is not initialized yet")
            return
            
    def init_filter(self, dT):
        ## Variables used by all the filters
        dim_x = 7
        dim_z = 3
        
        H = np.zeros((3, 7))
        H[0, 0] = H[1, 2] = H[2, 4] = 1
        
        P_factor = 10
        R_factor = .1
        
        # Constant velocity filter
        lin_filter = KF(dim_x=dim_x, dim_z=dim_z)
        
        lin_filter.x = self.state
        
        lin_filter.H = H
        
        F_lin = np.kron(np.eye(2), np.array([[1, dT, dT**2/2], [0, 1, dT], [0, 0, 1]]))
        F_lin = np.vstack((F_lin, np.zeros((1, F_lin.shape[1]))))
        F_lin = np.hstack((F_lin, np.zeros((F_lin.shape[0], 1))))
        lin_filter.F = F_lin
        
        lin_filter.R *= R_factor
        lin_filter.P *= P_factor
        
        var_lin = 1e-0
        var_circ = 0
        lin_filter.Q = self.process_noise(3, dT, 2, var_lin, var_circ)
        
        self.filters.append(lin_filter)
        
        # Constant Turn Filter 1
        turn_filter = EKF(dim_x=dim_x, dim_z=dim_z, dT=dT, state_trans_func=self.__circular_prediction, state_trans_jacob_func=self.__circular_jacobian)
        
        turn_filter.x = self.state
        turn_filter.H = H
        
        var_lin = 1e-0
        var_circ = 2e-3
        turn_filter.Q = self.process_noise(3, dT, 2, var_lin, var_circ)
        
        turn_filter.R *= R_factor
        turn_filter.P *= P_factor
        
        self.filters.append(turn_filter)
        
        # Constant Turn Filter 2
        turn_filter2 = EKF(dim_x=dim_x, dim_z=dim_z, dT=dT, state_trans_func=self.__circular_prediction, state_trans_jacob_func=self.__circular_jacobian)
        
        turn_filter2.x = self.state
        turn_filter2.H = H
        
        var_lin = 1e-1
        var_circ = 1e-5
        turn_filter.Q = self.process_noise(3, dT, 2, var_lin, var_circ)
        
        
        turn_filter2.R *= R_factor
        turn_filter2.P *= P_factor
        
        self.filters.append(turn_filter2)
        
        # Initializing the IMM Estimator
        mu = [1/2, 1/4, 1/4]
        trans = np.array([[0.95, 0.05, 0], [0.2, 0.6, 0.2], [0, 0.2, 0.8]])
        self.IMM = IMMEstimator(self.filters, mu, trans)
        
    def process_noise(self, dim, dT, block_size, var_lin, var_circ):
        Q = Q_kinematic(dim=dim, dt=dT, var=var_lin, block_size=block_size)
        Q = np.vstack((Q, np.zeros((1, Q.shape[1]))))
        Q = np.hstack((Q, np.zeros((Q.shape[0], 1))))
        Q[-1, -1] = var_circ * dT
        
        return Q
    
    
    def __circular_prediction(self, state, dT):
        w = state[-1]
        F = np.array([[1,   np.sin(w*dT)/w,         0,      -(1 - np.cos(w*dT))/w,      0,  0,  0], 
                      [0,   np.cos(w*dT),           0,      -np.sin(w*dT),              0,  0,  0],
                      [0,   (1-np.cos(w*dT))/w,     1,      np.sin(w*dT)/w,             0,  0,  0],
                      [0,   np.sin(w*dT),           0,      np.cos(w*dT),               0,  0,  0],
                      [0,   0,                      0,      0,                          1,  dT, 0],
                      [0,   0,                      0,      0,                          0,  1,  0],
                      [0,   0,                      0,      0,                          0,  0,  1]])
        
        # F = np.array([[1,   np.sin(w*dT)/w,         ,        0,      -(1 - np.cos(w*dT))/w,      ,  0,   0,  0,         0], 
        #               [0,   np.cos(w*dT),           ,        0,      -np.sin(w*dT),              ,  0,   0,  0,         0],
        #               [0,   -w*np.sin(w*dT),        ],
        #               [0,   (1-np.cos(w*dT))/w,     ,        1,      np.sin(w*dT)/w,             ,  0,   0,  0,         0],
        #               [0,   np.sin(w*dT),           ,        0,      np.cos(w*dT),               ,  0,   0,  0,         0],
        #               [0,   w*np.cos(w*dT),         ],
        #               [0,   0,                      ,        0,      0,                          0,  1,  dT, dT**2/2,   0],
        #               [0,   0,                      ,        0,      0,                          0,  0,  1,  dT,        0],
        #               [0,   0,                      0,       0,      0,                          0,  0,  0,  1,         0],
        #               [0,   0,                      ,        0,      0,                          0,  0,  0,  0,         1]])
        
        return np.dot(F, state)
        
        
    
    def __circular_jacobian(self, state, dT):
        w = state[-1]
        F = np.array([[1,   np.sin(w*dT)/w,         0,      -(1 - np.cos(w*dT))/w,      0,  0,  0], 
                      [0,   np.cos(w*dT),           0,      -np.sin(w*dT),              0,  0,  0],
                      [0,   (1-np.cos(w*dT))/w,     1,      np.sin(w*dT)/w,             0,  0,  0],
                      [0,   np.sin(w*dT),           0,      np.cos(w*dT),               0,  0,  0],
                      [0,   0,                      0,      0,                          1,  dT, 0],
                      [0,   0,                      0,      0,                          0,  1,  0],
                      [0,   0,                      0,      0,                          0,  0,  1]])
        
        return F
    
    
        
        
    def remove(self):
        Track.all.remove(self)
        
    def remove_all():
        Track.all = []
    

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
    
    
    def __preprocessing(self, H, BW):
        """Process the measurements
        
        Parameters
        ----------
        H : np.ndarray
            Measurements of all the Transceivers.
        BW : float
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
        R_range = scipy.constants.c / (np.sqrt(medium) * 2*BW) * np.linspace(-len_meas/2, len_meas/2, len_fft)
        self.__R_range = R_range[int(len_fft/2):]
        
        # Apply window function
        H = np.hanning(len_meas).reshape(-1, 1) * H
        
        # Apply fft
        H = np.fft.fft(H, n=len_fft, axis=0)
        
        # We simulate here to sample the complex signal, so we take the abs. So we also take only half of the signal since it is even
        H = np.abs(H[:int(len_fft/2), :])
        
        # return it in the dB 
        return 20*np.log10(np.abs(H))
    
    
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
        
    
    def localize(self, H, BW, trans_pos, verbose=False):
        
        H_db = self.__preprocessing(H, BW)
        
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
        if verbose:
            return possible_targets, H_db, max_i
        else:
            return possible_targets
    
    
    def __clustering(self, locs):
        """Cluster the locations after localization
        

        Returns
        -------
        None.

        """
        # Parameters for DBSCAN (These parameters may be changed for real data):
        eps = 0.3 # This is the radius in which points of the cluster should be searched
        minpts = 1 # This is the minimum number of points a cluster can consist of. Noise points will be clustered, but will be filtered away in data association
        clusters = DBSCAN(eps=eps, min_samples=minpts).fit(locs[:3, :].T)
        return np.vstack((locs, clusters.labels_))
    
    def __filtering(self, targets, dT):
        kf_targets = []
        
        if len(Track.all) == 0:
            if targets.size != 0:
                Track(targets[:3, 0])
        else:
            for i, track in enumerate(Track.all):
                    if targets.size != 0:
                        track.new_position(targets[:3, i], dT)
                    elif track.status == Track.status_type['confirmed']:
                        track.predict()
                    else:
                        continue
                        
                        
                    x = track.x
                    # x = np.hstack((track.x[:-1:3], track.x[1::3], track.x[2::3], track.x[-1]))
                    x = np.hstack((track.x[:-1:2], track.x[1::2], track.x[-1]))
                    
                    
                    P = track.P
                    # P = np.vstack((P[:-1:3, :], P[1::3, :], P[2::3, :], P[-1, :]))
                    # P = np.hstack((P[:, :-1:3], P[:, 1::3], P[:, 2::3], P[:, -1].reshape(-1, 1)))
                    P = np.vstack((P[:-1:2, :], P[1::2, :], P[-1, :]))
                    P = np.hstack((P[:, :-1:2], P[:, 1::2], P[:, -1].reshape(-1, 1)))
                    
                    kf_targets.append((x, P))
                
                
        return kf_targets
        
        
    def __association(self, clustered_locs, dT):
        # Reducing the clusters into one single point at the center
        targets = np.empty((4, 0))
        num_clusters = np.max(clustered_locs[4, :]) + 1
        cluster_list = np.arange(0, num_clusters)
        for c in cluster_list:
            tmp = np.mean(clustered_locs[:4, clustered_locs[4, :]==c], axis=1).reshape(4, 1)
            targets = np.hstack((targets, tmp))
        
        
        
        kf_targets = self.__filtering(targets, dT)
        
        #row, col = scipy.optimize.linear_sum_assignment(cost_matrix)
        #cost_matrix[row, col]
        
        return targets, kf_targets
        
    
    def track(self, H, BW, trans_pos, dT, verbose=False):
        if verbose: 
            locations, H_db, max_i = self.localize(H, BW, trans_pos, verbose)
        else:
            locations = self.localize(H, BW, trans_pos)
        
        if (locations.size) == 0:
            kf_targets = self.__filtering(np.empty(0), dT)
            return np.empty(0), np.empty(0), H_db, self.R_range, max_i, kf_targets
        
        clustered_locations = self.__clustering(locations)
        targets, kf_targets = self.__association(clustered_locations, dT)
    
        if verbose:
            return targets, clustered_locations, H_db, self.R_range, max_i, kf_targets
        else:
            return targets
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    