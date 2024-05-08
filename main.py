# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:52:11 2023

@author: Harun Kumru
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from time import time, sleep
from itertools import product

class Movement:
    def __init__(self, x0, y0, z0, T):
        self.x = x0
        self.y = y0
        self.z = z0
        self.T = T
    
    def constant(self):
        while True:
            yield (self.x, self.y, self.z)
            
    def linear_movement(self, x_vel, y_vel, z_vel):
        while True:
            #self.x += x_vel * self.T
            #self.y += y_vel * self.T
            #self.z += z_vel * self.T
            # ich mache das jetzt nicht mit den geschwindigkeiten, da die simulation zu lange dauert
            self.x += x_vel
            self.y += y_vel
            self.z += z_vel 
            yield (self.x, self.y, self.z)
            
    def circular_movement(self, rad, T_circle):
        # T_circle is the time-period for one circle
        # rad: radius of the circle
        
        #steps = T_circle / self.T
        # ich mache das jetzt nicht mit den geschwindigkeiten, da die simulation zu lange dauert
        steps = T_circle
        dphi = 2*np.pi / steps
        phi = 0
        while True:
            self.x, self.y = pol2cart(rad, phi)
            phi += dphi
            yield (self.x, self.y, self.z)
            

class Transceiver:
    def __init__(self, x_pos, y_pos, z_pos, k):
        self.x = x_pos
        self.y = y_pos
        self.z = z_pos
        self.len_output = len(k)
        self.k = k.reshape(len(k), 1)
        
    def get_pos(self):
        return [self.x, self.y, self.z]
    
    def calc_distance(self, x_target, y_target, z_target):
        dist = np.sqrt((self.x - x_target)**2 + (self.y - y_target)**2 + (self.z - z_target)**2)
        return dist
    
    def receive(self, targets):
        # This function returns the received echo of the transceiver
        # This is modeled with a dampening with 'amplitude' and a time delay 
        # calculated with the distance to the target
        H = np.zeros((len(self.k), 1), dtype=np.complex128)
        
        for i, target in enumerate(targets):
            dist = self.calc_distance(target.x, target.y, target.z)
            H = np.add(H, target.reflection * np.exp(-1j * 2 * self.k * dist))
        return np.real(H)
    
class Target:
    def __init__(self, x_pos, y_pos, z_pos, reflection=1):
        self.x = x_pos
        self.y = y_pos
        self.z = z_pos
        self.reflection = reflection
    
    def new_pos(self, x_pos, y_pos, z_pos):
        self.x = x_pos
        self.y = y_pos
        self.z = z_pos
    
    def shift_pos(self, x_shift, y_shift, z_shift):
        self.new_pos(self.x + x_shift, self.y + y_shift, self.z + z_shift)
        
    def get_pos(self):
        return np.array([self.x, self.y, self.z])

def nearestPowerOf2(N):
    # Calculate log2 of N
    a = int(np.log2(N))
 
    # If 2^a is equal to N, return N
    if np.power(2, a) == N:
        return N
     
    # Return 2^(a + 1)
    return np.power(2, a+1)

def send_receive_all(len_fft, targets, transceivers):
    H = np.array([]).reshape(len_fft, -1)
    for i in range(len(transceivers)):
        H = np.hstack((H, np.fft.fft(np.hanning(transceivers[i].len_output).reshape(-1, 1) * transceivers[i].receive(targets), n=len_fft, axis=0)))
    
    H_abs = np.abs(H[:int(len_fft/2), :])
    
    return H_abs

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def init():
    # Verbose Parameters
    circle_show = True
    
    
    # Constants and Variables
    c_0 = scipy.constants.c
    f_start = 110e9
    f_stop = 170e9
    B = abs(f_stop - f_start)
    T = 300e-9 
    # T: Periodendauer des Frequenzsweeps (definiert die maximale Reichweite). 
    # Wir gehen von einer durchgehenden aussendung aus, ohne Pause 
    # zwischen den einzelnen Frequenzsweeps
    num_freq_points = int(B * T)
    zero_pad = 1
    len_fft = nearestPowerOf2(num_freq_points*zero_pad)
    R_range = c_0 / (np.sqrt(1) * 2*B) * np.linspace(-num_freq_points/2, num_freq_points/2, len_fft)
    R_range = R_range[int(len_fft/2):]
    
    f = np.linspace(f_start, f_stop, num=num_freq_points)
    w = 2 * np.pi * f
    k = w / c_0
    
    # Setting up the Targets and Transceivers
    tr_pos = np.array([[0, -10, 0], [10, 10, 0], [-10, 10, 0]], dtype=np.float64)
    num_transceivers = tr_pos.shape[0]
    transceivers = []
    tr_text  = []
    for num in range(num_transceivers):
        tr_text.append("Tr" + str(num))
        transceivers.append(Transceiver(*tr_pos[num, :], k))
        
    # This is the matrix for multilateration to solve it analytically via least-squares algorithm
    A = np.array([]).reshape(0, 3)
    for i in range (num_transceivers-1):
        A = np.vstack((A, tr_pos[-1] - tr_pos[i]))
    A = 2*A
    # removing the z-coordinate since the transceivers are all in the same level. otherwise we get a mathematical error. The z-component is then set to 0 again
    A = A[:, :-1]
    
    
    tar_pos = np.array([[0, 0, 0]], dtype=np.float64, ndmin=2)
    num_targets = tar_pos.shape[0]
    t_text = []
    tars = []
    for num in range(num_targets):
        t_text.append("T" + str(num))
        tars.append(Target(*tar_pos[num, :]))
    
    # Drawing Targets and Transceivers
    fig_elements = plt.figure("Targets and Transceivers")
    fig_elements.canvas.manager.window.raise_()
    plots_trans = plt.scatter(*zip(*tr_pos[:, :2]), marker='x', color='b', label='Transceivers')
    plots_targ = plt.scatter(*zip(*tar_pos[:, :2]), marker='o', color='r', label='Targets')
    targ_texts = []
    text_offset = 0.5
    for (i, text) in enumerate(tr_text):
        plt.text(*tr_pos[i, :2]+text_offset, text)
    for (i, text) in enumerate(t_text):
        targ_texts.append(plt.text(*tar_pos[i, :2]+text_offset, text))
    plt.legend()
    plt.grid(True)
    plt.xlabel('x-position [m]')
    plt.ylabel('y-position [m]')
    
    # Sending and Receiving Signals
    H_abs_db = 20*np.log10(send_receive_all(len_fft, tars, transceivers))
    
    # Setting up the plots
    fig_plots = plt.figure("Plots")
    fig_plots.canvas.manager.window.raise_()
    ax = []
    plots = []
    for i in range(num_transceivers):
        ax.append(plt.subplot(num_transceivers, 1, i+1))
        plots.append(ax[i].plot(R_range, H_abs_db[:, i])[0])
        plt.xlabel('Distance [m]')
        plt.ylabel('|H(w)| [dB]')
        plt.grid(True)
        #dist.append(H_abs[i].argmax());
        #print("Distance is: " + str(R_range[dist[i]]) + " m")
    
    # Define Movements
    movements = []
    # movements.append(Movement(*tars[0].get_pos(), T).circular_movement(5, 30))
    # movements.append(Movement(*tars[1].get_pos(), T).constant())
    movements.append(Movement(*tars[0].get_pos(), T).linear_movement(0.02, -0.01, 0))
    
    # Plot Circles
    if (circle_show):
        circle_fig, circle_ax = plt.subplots()
        circle_ax.set_xlim((-20, 20))
        circle_ax.set_ylim((-20, 20))
        colors = ['b', 'r', 'g', 'k', 'y', 'm']
    
    plt.show()
    # Main loop
    D = 60 # duration of simulation in seconds
    wait_time = 1.0
    
    print("Press s to start simulation ...")
    import keyboard
    
    while True:
        if keyboard.is_pressed("s"):
            print("You pressed s")
            break
    
    print("Starting the simulation")
    start_time = time()
    time_diff = 0
    
    while(D > time_diff):
        time_diff = time() - start_time
        
        
        # Movement:
        [tars[i].new_pos(*next(movements[i])) for i in range(len(movements))]
            
        # New Measurement
        H_abs_db = 20*np.log10(send_receive_all(len_fft, tars, transceivers))
        
        # Redrawing the db plots
        [plot.set_ydata(H_abs_db[:, num]) for num, plot in enumerate(plots)]
        fig_plots.canvas.draw()
        fig_plots.canvas.flush_events()
        
        
        # Tracking
        
        # Multilateration: R = anzahl radare und T = Anzahl Ziele
        # Wir haben T^R mögliche Positionen für Ziele und insgesamt
        # T*R verschiedene TOA
        min_height = 0 # dB
        prominence = 100
        width = None
        max_i = [scipy.signal.find_peaks(H_abs_db[:, num], height=min_height, prominence=prominence, width=width)[0] for num in range(num_transceivers)]
        dists = [R_range[max_i[num]] for num in range(len(max_i))]
        # with this function, ``find_peaks()`` we get also some other information, which we filter out by only taking the 0th element of the list
        
        # Multilateration calculation analytically with least squares
        possible_combinations = list(product(*dists))
        possible_targets = np.array([]).reshape(0, 3)
        for poss_comb in possible_combinations:
            b = np.array([])
            for tr in range(num_transceivers-1):
                b = np.append(b, (poss_comb[tr]**2 - poss_comb[-1]**2) - np.sum(np.power(tr_pos[tr], 2)) + np.sum(np.power(tr_pos[-1], 2)))
            
            coordinate = np.dot(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
            possible_targets = np.vstack((possible_targets, np.append(coordinate, 0)))
        
        
        # Drawing the circles
        if (circle_show):
            circle_ax.cla()
            circle_ax.set_xlim((-20, 20))
            circle_ax.set_ylim((-20, 20))
            for i_tr in range(num_transceivers):
                for i_tar in range(len(dists[i_tr])):
                    circle = plt.Circle((tr_pos[i_tr][0], tr_pos[i_tr][1]), dists[i_tr][i_tar], fill=False, color=colors[i_tar])
                    circle_ax.add_patch(circle)
            
            circle_fig.canvas.draw()
            circle_fig.canvas.flush_events()
        
        
        
        # Redrawing Targets
        plots_targ.set_offsets([tars[i].get_pos()[:2] for i in range(len(tars))])
        [text.set_position(tars[num].get_pos()[:2] + text_offset) for num, text in enumerate(targ_texts)]
         
        fig_elements.canvas.draw()
        fig_elements.canvas.flush_events()
        
        # Wait for a time before updating
        sleep(wait_time)
        
        
if __name__ == "__main__":
    print("Initializing ...")
    init()
    print("End")











































































































