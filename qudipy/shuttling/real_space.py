"""
Real space pulse shuttilng evolution module

@author: Kewei
"""

import os, sys
sys.path.append(os.path.dirname(os.getcwd())[:-6])

import qudipy as qd
import numpy as np
import qudipy.potential as pot
import qudipy.qutils as qt
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy import sparse
from scipy.sparse import diags
from scipy.linalg import expm
import matplotlib.pyplot as plt
import timeit
import csv


def time_evolution(pot_interp, shut_pulse, show_animation = True, save_data = False, 
        update_ani_frames = 100, save_data_points = 500):
    '''
    Displays the animation of time evolution of the wave functions or save adiabaticity evolution to csv file.

    Parameters
    ----------

    pot_interp: PotentialInterpolator object
        A 1D potential interpolator object for the quantum dot system.

    shut_pulse: ControlPulse object
        A shuttling pulse that that tunnels an electron through a quantum dot system.

    show_animation: boolean
        whether the animation is displayed

    save_data: boolean
        whether the adiabaticity data is saved

    update_ani_frames: int
        how often the animation updates its frame

    save_data_points: int
        how many total points are saved

    Returns
    -------
    None.

    '''

    # Find the initial potential
    init_pulse = shut_pulse([0])[0]
    init_pot = pot_interp(init_pulse)

    # Initialize the constants class with the Si/SiO2 material system 
    consts = pot_interp.constants

    # First define the x-coordinates
    X = pot_interp.x_coords

    # Create a GridParameters object of the initial potential
    gparams = pot.GridParameters(X, potential=init_pot)

    # Find the initial ground state wavefunction
    __, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
    psi_x = e_vecs[:,0]


    # time step
    dt = 5E-16
    # indices of grid points
    I = [(idx-gparams.nx/2) for idx in range(gparams.nx)]   
    # vector of momentum grid
    P = np.asarray([2 * consts.pi * consts.hbar * i / (gparams.nx*gparams.dx) for i in I])

    # exponents present in evolution
    exp_K = np.exp(-1j*dt/2*np.multiply(P,P)/(2*consts.me*consts.hbar))
    exp_KK = np.multiply(exp_K,exp_K)

    # Start the evolution
    # Calculate the runtime
    start = timeit.default_timer()

    # Get the list of pulses with time steps of dt
    p_length = shut_pulse.length
    t_pts = np.linspace(0,p_length, round(p_length*1E-12/dt))
    int_p = shut_pulse(t_pts)

    # Convert to momentum space
    psi_p = fftshift(fft(psi_x))
    psi_p = np.multiply(exp_K,psi_p)

    # Calculate interpolated potentials at each time step
    potential_L = pot_interp(int_p)

    # Initialize the plot if show animation is true
    if show_animation:
        # absolute square of the wave function
        prob = [abs(x)**2 for x in psi_x]
        # Define the limits of the plots
        ymax = 2 * max(prob)
        potmax = max(init_pot) + 1E-21
        potmin = min(init_pot) - 3E-21

        # Plot evolution
        plt.ion()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.tight_layout(pad=2)
        fig.suptitle('Time Evolution of 1D potential and Electron Shuttling', y = 1)

        line1, = ax1.plot(X, init_pot,color='r')
        line2, = ax2.plot(X, prob)

    # Initialize the arrays where data is store if save_data is true
    if save_data:
        t_selected = np.zeros(save_data_points)
        adiabaticity = np.zeros(save_data_points)
        # Calculate how often we save the data to get a resolution of 500 points
        checkpoint_counter = len(t_pts)//save_data_points
        
    with open('data.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Loop through each time step
        for t_idx in range(len(t_pts)):
            potential = potential_L[t_idx]

            # diagonal matrix of potential energy in position space
            exp_P = np.exp(-1j*dt/consts.hbar * potential)

            # Start the split operator method
            psi_x = ifft(ifftshift(psi_p))
            if show_animation and t_idx % update_ani_frames  == 0:
                prob = [abs(x)**2 for x in psi_x]
                line1.set_data(X, potential)
                line2.set_data(X, prob)

                # ax1.autoscale_view(True,True,True)
                # ax1.relim()
                ax1.set_xlabel("x(m)")
                ax1.set_ylabel("Potential")
                ax1.set_xlim(min(X), max(X))
                ax1.set_ylim(potmin,potmax)

                ax2.set_xlabel("x(m)")
                ax2.set_ylabel("Probability")
                # ax2.set_xlim(-2e-7, 2e-7)
                ax2.set_ylim(-5e6, ymax)

                plt.draw()
                plt.pause(1e-15)
            if save_data and t_idx%checkpoint_counter  == 0:
                # find the ground state under this pulse
                __, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
                ground_psi = e_vecs[:,0]
                # calculate the inner product between the ground state and the current state
                inner = abs(qd.qutils.math.inner_prod(gparams, psi_x, ground_psi))**2
                # print(t_pts[t_idx], inner)
                writer.writerow([p_length, t_pts[t_idx], inner])
                t_selected[t_idx//checkpoint_counter] = t_pts[t_idx]
                adiabaticity[t_idx//checkpoint_counter] = inner

            psi_x = np.multiply(exp_P,psi_x)
            psi_p = fftshift(fft(psi_x))     
            if t_idx != len(t_pts)-1:
                psi_p = np.multiply(exp_KK,psi_p)
            else:
                psi_p = np.multiply(exp_K,psi_p)
                psi_x = ifft(ifftshift(psi_p))

        # Pring the runtime
        stop = timeit.default_timer()
        print('Time: ', stop - start) 