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
    all = [] # list of all tracks
    
    def __init__(self, position, score, dT):
        self.status = Track.status_type['tentative']
        self.__pos = np.array(position)
        self.dT = dT
        
        self.filters = []
        self.IMM = None
        self.init_filter()
        
        poss_tids = set(np.arange(len(Track.all) + 1))
        all_tids = [track.tid for track in Track.all]
        self.__tid = poss_tids.difference(all_tids).pop()
        self.__score = score
        self.__max_score = score
        Track.all.append(self)
    
    @property
    def score(self):
        return self.__score
    
    @score.setter
    def score(self, s):
        self.__score = s
        if s > self.max_score:
            self.__max_score = s
    
    @property
    def max_score(self):
        return self.__max_score
    
    @property
    def tid(self):
        return self.__tid
    
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
    
    @property
    def S(self):
        return self.IMM.S
    
    def mahalanobis(self, targets):
        """Calculate the mahalanobis distances to all targets

        Parameters
        ----------
        targets : np.array(): shape: (num_dim, num_targets)
            This matrix contains the coordinate of every target detected.

        Returns
        -------
        mah : np.array(): shape: (num_targets,)
            This is the vector that contains the mahalanobis distances to all
            given targets.

        """
        mah = np.array([])
        for tar in targets.T:
            mah = np.append(mah, self.IMM.mahalanobis(tar[:3]))
        
        return mah
    
    def update(self, new_pos):
        self.IMM.update(new_pos)
        
    def predict(self):
        self.IMM.predict()
            
    def init_filter(self):
        v = [0, 0, 0] # (self.pos[:, -1] - self.pos[:, -2]) / self.dT
        w = 1 # this is arbitrarily set to 1 (this shouldn't be 0 so we don't divide by 0)
        self.__state = np.append(np.dstack((self.pos, v)).flatten(), w)
        
        #### Variables used by all the filters
        
        dim_x = 7
        dim_z = 3
        
        # Measurement Matrix
        H = np.zeros((3, 7))
        H[0, 0] = H[1, 2] = H[2, 4] = 1
        
        R_factor = .05
        
        # inital covariance matrix P
        v_max = 10
        w_max = 10
        P = np.kron(np.eye(3), np.array([[R_factor, 0], [0, v_max**2/3]]))
        P = np.vstack((P, np.zeros((1, P.shape[1]))))
        P = np.hstack((P, np.zeros((P.shape[0], 1))))
        P[-1, -1] = w_max**2 / 3
        
        #### Constant velocity filter
        
        lin_filter = KF(dim_x=dim_x, dim_z=dim_z)
        
        lin_filter.x = self.state
        lin_filter.H = H
        
        F_lin = np.kron(np.eye(3), np.array([[1, self.dT], [0, 1]]))
        F_lin = np.vstack((F_lin, np.zeros((1, F_lin.shape[1]))))
        F_lin = np.hstack((F_lin, np.zeros((F_lin.shape[0], 1))))
        lin_filter.F = F_lin
        
        lin_filter.R *= R_factor
        lin_filter.P = P
        
        var_lin = 1e-3
        var_circ = 0
        lin_filter.Q = self.process_noise(3, 2, var_lin, var_circ)
        
        self.filters.append(lin_filter)
        
        #### Constant Turn Filter 1
        
        turn_filter = EKF(dim_x=dim_x, dim_z=dim_z, state_trans_func=self.__circular_prediction, state_trans_jacob_func=self.__circular_jacobian)
        
        turn_filter.x = self.state
        turn_filter.H = H
        
        var_lin = 1e-2
        var_circ = 2e-3
        turn_filter.Q = self.process_noise(3, 2, var_lin, var_circ)
        
        turn_filter.R *= R_factor
        turn_filter.P = P
        
        self.filters.append(turn_filter)
        
        #### Constant Turn Filter 2
        
        turn_filter2 = EKF(dim_x=dim_x, dim_z=dim_z, state_trans_func=self.__circular_prediction, state_trans_jacob_func=self.__circular_jacobian)
        
        turn_filter2.x = self.state
        turn_filter2.H = H
        
        var_lin = 1e-4
        var_circ = 1e-5
        turn_filter2.Q = self.process_noise(3, 2, var_lin, var_circ)
        
        turn_filter2.R *= R_factor
        turn_filter2.P = P
        
        self.filters.append(turn_filter2)
        
        #### Initializing the IMM Estimator
        
        mu = [1/2, 1/4, 1/4]
        trans = np.array([[0.95, 0.05, 0], [0.2, 0.6, 0.2], [0, 0.2, 0.8]])
        
        self.IMM = IMMEstimator(self.filters, mu, trans)
    
    def process_noise(self, dim, block_size, var_lin, var_circ):
        Q = Q_kinematic(dim=dim, dt=self.dT, var=var_lin, block_size=block_size)
        Q = np.vstack((Q, np.zeros((1, Q.shape[1]))))
        Q = np.hstack((Q, np.zeros((Q.shape[0], 1))))
        Q[-1, -1] = var_circ * self.dT
        
        return Q
    
    def __circular_prediction(self, state):
        dT = self.dT
        w = state[-1]
        F = np.array([[1,   np.sin(w*dT)/w,         0,      -(1 - np.cos(w*dT))/w,      0,  0,      0], 
                      [0,   np.cos(w*dT),           0,      -np.sin(w*dT),              0,  0,      0],
                      [0,   (1-np.cos(w*dT))/w,     1,      np.sin(w*dT)/w,             0,  0,      0],
                      [0,   np.sin(w*dT),           0,      np.cos(w*dT),               0,  0,      0],
                      [0,   0,                      0,      0,                          1,  dT,     0],
                      [0,   0,                      0,      0,                          0,  1,      0],
                      [0,   0,                      0,      0,                          0,  0,      1]])
        
        return np.dot(F, state)
    
    def __circular_jacobian(self, state):
        dT = self.dT
        w = state[-1]
        vx = state[1]
        vy = state[3]
        F = np.array([[1,   np.sin(w*dT)/w,         0,      -(1 - np.cos(w*dT))/w,      0,  0,      (vx/w**2)*(w*dT*np.cos(w*dT) - np.sin(w*dT)) - (vy/w**2)*(w*dT*np.sin(w*dT)-(1-np.cos(w*dT)))],
                      [0,   np.cos(w*dT),           0,      -np.sin(w*dT),              0,  0,      -dT*(vx*np.sin(w*dT)+vy*np.cos(w*dT))],
                      [0,   (1-np.cos(w*dT))/w,     1,      np.sin(w*dT)/w,             0,  0,      (vy/w**2)*(w*dT*np.cos(w*dT) - np.sin(w*dT)) + (vx/w**2)*(w*dT*np.sin(w*dT)-(1-np.cos(w*dT)))],
                      [0,   np.sin(w*dT),           0,      np.cos(w*dT),               0,  0,      dT*(vx*np.cos(w*dT)-vy*np.sin(w*dT))],
                      [0,   0,                      0,      0,                          1,  dT,     0],
                      [0,   0,                      0,      0,                          0,  1,      0],
                      [0,   0,                      0,      0,                          0,  0,      1]])
        
        return F
    
    def remove(self):
        Track.all.remove(self)
        del self
        
    def clean():
        tracks = Track.all
        for t in tracks:
            if t.status == Track.status_type['terminated']:
                t.remove()
                del t
    

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
        
    
    def _localize(self, H, BW, trans_pos, verbose=False):
        
        H_db = self.__preprocessing(H, BW)
        
        num_transceivers = H.shape[1]
        prominence = 70
        # with this function, ``find_peaks()`` we get also some other information, which we filter out by only taking the 0th element of the list
        max_i = [scipy.signal.find_peaks(H_db[:, num], prominence=prominence)[0] for num in range(num_transceivers)]
        dists = [self.R_range[max_i[num]] for num in range(len(max_i))]
        
        # Find all the possible targets with Multilateration
        possible_targets = self.__multilateration(dists, trans_pos, num_transceivers)
        
        # Set the threshold to filter out the targets. this shall be adaptive later
        threshold = 3e-3*num_transceivers
        possible_targets = possible_targets[:, possible_targets[3, :] < threshold]
        if verbose:
            return possible_targets, H_db, max_i
        else:
            return possible_targets
    
    
    def _clustering(self, locs):
        """Cluster the locations after localization
        

        Returns
        -------
        None.

        """
        if locs.size != 0:
            # Parameters for DBSCAN (These parameters may be changed for real data):
            eps = 0.3 # This is the radius in which points of the cluster should be searched
            minpts = 1 # This is the minimum number of points a cluster can consist of. Noise points will be clustered, but will be filtered away in data association
            clusters = DBSCAN(eps=eps, min_samples=minpts).fit(locs[:3, :].T)
            return np.vstack((locs, clusters.labels_))
        else:
            return np.empty(0)
        
    
    def _association(self, clustered_locs, dT):
        tracks = {}
        num_tracks = len(Track.all)
        
        # Parameters for Track Management
        # Gating:
        C = 4 
        # This parameter defines the "probability that the (true) measurement will fall in the gate"
        # with C = 4, it is 73.9% and with C = 9 it is 97.1% 
        # Track confirmation threshold
        P_con = 20
        # Track deletion threshold
        P_del = 20
        # The value that is subtracted if the track is not measured
        no_meas_diff = 5
        
        # Parameters for Scoring Function
        beta_NT = 1e-3 # density of new targets
        P_D = 0.8 # probability of detection
        P_FA = 1e-2 # probability of false alert
        M = 3 # num of dimensions
        V_C = 20*20 # volume of the clutter
        beta_FT = P_FA / V_C # false target density. density of false alarms per unit volume
        LLR_0 = np.log(P_D * beta_NT/beta_FT)
        
        if clustered_locs.size != 0:
            # Reducing the clusters into one single point at the center by taking the mean position of the cluster
            targets = np.empty((4, 0))
            num_clusters = np.max(clustered_locs[4, :]) + 1
            cluster_list = np.arange(0, num_clusters)
            for c in cluster_list:
                tmp = np.mean(clustered_locs[:4, clustered_locs[4, :]==c], axis=1).reshape(4, 1)
                targets = np.hstack((targets, tmp))
            
            
            
            
            num_targets = targets.shape[1]
            # Initialize Tracks
            # If there are no tracks at all. All targets become tentative tracks
            # Otherwise we update each track with the measured targets
            if num_tracks == 0:
                for tar in targets.T:
                    Track(tar[:3], LLR_0, dT)
                return targets, np.empty(0)
            else:
                for track in Track.all:
                    track.predict()
            
            # Assignment matrix
            Assignment = np.zeros((num_tracks, num_targets)) 
            for i, track in enumerate(Track.all):
                # Mahalanobis Distances
                Assignment[i] = track.mahalanobis(targets)**2
                # Gating
                Assignment[i, Assignment[i] > C] = np.inf
                # Scoring:
                S = np.sqrt(np.linalg.det(track.S))
                factor = np.log((P_D * V_C) / ((2*np.pi)**(M/2) * beta_FT * S))
                Assignment[i] = Assignment[i] * (-1/2) + factor
            
            
            if Assignment.size != 0:
                # if a track does not have a measurement that can be assigned to it, 
                # so every measurement is outside of the gates, the row will be full of np.inf
                # in that case, we have to remove that row to solve the assignment problem
                # those tracks that do not have any measurement, are reduced by score
                # later. for now i will calculate an array of detected and not-detected targets
                # where False means that the object was not detected
                no_det = np.array([not np.isinf(Assignment[i]).all() for i in range(num_tracks)])
                Assignment = Assignment[no_det, :]
                # we also need to convert the -inf to a real value because in some
                # cases the hungarian method is not solvable with -inf
                Assignment[Assignment==-np.inf] = -no_meas_diff
                
                # Hungarian / Munkres Algorithm
                row, col = scipy.optimize.linear_sum_assignment(Assignment, maximize=True)
                
                # here i expand the list of no-detections if the number of tracks is
                # more than the number of targets
                if num_tracks > len(row):
                    num_poss_tracks = len(Assignment)

                    unassigned_tracks = np.arange(num_poss_tracks)
                    unassigned_tracks = list(set(unassigned_tracks).difference(row))
                    no_det[np.where(no_det)[0][unassigned_tracks]] = False
                    
                    rows = np.zeros(num_tracks)
                    rows[no_det] = row
                    row = np.astype(rows, int)
                    
                    cols = np.zeros(num_tracks)
                    cols[no_det] = col
                    col = np.astype(cols, int)
            else:
                # There are no detections
                no_det = np.full(num_tracks, False)
            
        
        else:
            # This is executed if there are no targets detected at all
            no_det = np.zeros(num_tracks)
            no_det.fill(False)
            targets = clustered_locs
            for track in Track.all:
                track.predict()
            
        
        # Update the tracks according to the new Assignmentand do Track management
        for i, track in enumerate(Track.all):
            # Score update
            if no_det[i] == False:
                # when the track wasn't assigned a measurement
                track.score -= no_meas_diff
            else:
                track.score += Assignment[row[i], col[i]]
                track.update(targets[:3, col[i]])
                
            # Track Management
            if track.max_score - track.score >= P_del:
                track.status = Track.status_type['terminated']
            elif track.score >= P_con:
                track.status = Track.status_type['confirmed']
            
            # Update
            if track.status == Track.status_type['confirmed']:
                x = track.x
                # x = np.hstack((track.x[:-1:3], track.x[1::3], track.x[2::3], track.x[-1]))
                x = np.hstack((track.x[:-1:2], track.x[1::2], track.x[-1]))
                
                P = track.P
                # P = np.vstack((P[:-1:3, :], P[1::3, :], P[2::3, :], P[-1, :]))
                # P = np.hstack((P[:, :-1:3], P[:, 1::3], P[:, 2::3], P[:, -1].reshape(-1, 1)))
                P = np.vstack((P[:-1:2, :], P[1::2, :], P[-1, :]))
                P = np.hstack((P[:, :-1:2], P[:, 1::2], P[:, -1].reshape(-1, 1)))
            
                tracks[track.tid] = (x, P)
        
        Track.clean()
        
        if clustered_locs.size != 0:
            # Initialize new tracks
            # get the unassigned measurements
            unassigned_meas = np.arange(num_targets)
            # Filter the the not assigned measurements
            unassigned_meas = set(unassigned_meas).difference(col)
            # only those measurements that were outside of gates are used to initialize new tracks
            for i in unassigned_meas:
                if (Assignment[row, i] == -5).all():
                    # Initialize the new tracks
                    Track(targets[:3, i], LLR_0, dT)
        
        
        return targets, tracks
        
    
    def track(self, H, BW, trans_pos, dT, verbose=False):
        if verbose: 
            locations, H_db, max_i = self._localize(H, BW, trans_pos, verbose)
        else:
            locations = self._localize(H, BW, trans_pos)
        
        clustered_locations = self._clustering(locations)
        targets, tracks = self._association(clustered_locations, dT)
    
        if verbose:
            return targets, clustered_locations, H_db, self.R_range, max_i, tracks
        else:
            return tracks
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    