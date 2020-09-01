"""
Real space pulse evolution module

@author: Kewei
"""

import qudipy as qd
import numpy as np
import qudipy.potential as pot
import qudipy.qutils as qt
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
import timeit
import csv


def RSP_time_evolution_1D(pot_interp, ctrl_pulse, dt=5E-16, 
                   show_animation=True, save_data=False, 
                   update_ani_frames=2000, save_data_points=500):
    '''
    Perform a time evolution of a 1D real space Hamiltonian (i.e. one that has
    the form H = K + V(x)) according to an arbitrary control pulse. Simulation
    is done using the split-operator method.
    
    Parameters
    ----------
    pot_interp: PotentialInterpolator object
        A 1D potential interpolator object for the quantum dot system.
    ctrl_pulse: ControlPulse object
        An arbitrary control pulse.
    
    Keyword Arguments
    -----------------
    dt : float
        Specify the time step in [s] between simulation frames. Lower values
        produce more accurate simulations at the expense of longer runtime.
        The default is 5E-16 s.
    show_animation : boolean
        Specifies whether the animation is displayed. The default is True.
    save_data : boolean, optional
        Specifies whether the adiabaticity data is saved. The default is True.
    update_ani_frames : int, optional
        How many simulated times steps between new animation frames. The 
        default is 2000.
    save_data_points : int, optional
        How many total data points are saved during the simulation. The 
        default is 500.
        
    Returns
    -------
    None.

    '''

    # Get the material system Constants object from the potential interpolator
    consts = pot_interp.constants
    
    # First define the x-coordinates
    X = pot_interp.x_coords

    # Find the initial potential
    init_pulse = ctrl_pulse([0])[0]
    init_pot = pot_interp(init_pulse)

    # Create a GridParameters object of the initial potential
    gparams = pot.GridParameters(X, potential=init_pot)

    # Find the initial ground state wavefunction
    __, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
    psi_x = e_vecs[:,0]

    # indices of grid points
    I = [(idx-gparams.nx/2) for idx in range(gparams.nx)]   
    # Define the momentum coordinates
    P = np.asarray([2 * consts.pi * consts.hbar * i / (gparams.nx*gparams.dx)
                    for i in I])

    # Calculate kinetic energy operators used in split-operator method
    exp_K = np.exp(-1j*dt/2*np.multiply(P,P)/(2*consts.me*consts.hbar))
    exp_KK = np.multiply(exp_K,exp_K)

    # Start the evolution
    # Calculate the runtime
    start = timeit.default_timer()

    # Get the list of pulses with time steps of dt
    p_length = ctrl_pulse.length
    t_pts = np.linspace(0, p_length, round(p_length/dt))
    int_p = ctrl_pulse(t_pts)

    # Convert the initial state to momentum space and evolve
    psi_p = fftshift(fft(psi_x))
    psi_p = np.multiply(exp_K,psi_p)

    # Calculate interpolated potentials at each time step
    potential_L = pot_interp(int_p)

    # Initialize the plot if show animation is true
    if show_animation:
        # Find wavefunction probability
        prob = np.multiply(psi_x.conj(), psi_x)
        
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

        # Initialize progress bar
        bar = qd.utils.TerminalProgressBar(
            'Running RSPTE1D simulation of length {:.2E} secs'.format(p_length))

        # Loop through each time step
        for t_idx in range(len(t_pts)):
            potential = potential_L[t_idx]

            # diagonal matrix of potential energy in position space
            exp_P = np.exp(-1j*dt/consts.hbar * potential)

            # Start the split operator method
            psi_x = ifft(ifftshift(psi_p))
            
            # Update progress bar periodically
            if t_idx % 1000 == 0:
                bar.update(t_idx/len(t_pts))
            
            # Show animation periodically
            if show_animation and t_idx % update_ani_frames == 0:
                # Get wavefunction probability
                prob = np.multiply(psi_x.conj(), psi_x)
                
                # Update figure data
                line1.set_data(X, potential)
                line2.set_data(X, prob)

                ax1.set_xlabel("x(m)")
                ax1.set_ylabel("Potential")
                ax1.set_xlim(min(X), max(X))
                ax1.set_ylim(potmin,potmax)

                ax2.set_xlabel("x(m)")
                ax2.set_ylabel("Probability")
                ax2.set_ylim(-5e6, ymax)

                plt.draw()
                plt.pause(1e-15)
                
            # Save data periodically
            if save_data and t_idx%checkpoint_counter == 0:
                # Find the current ground state of the potential
                __, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams,
                                                             n_sols=1)
                ground_psi = e_vecs[:,0]
                
                # Calculate fidelity of simulated wavefunction w.r.t. ground 
                # state
                inner = abs(qd.qutils.math.inner_prod(gparams, psi_x,
                                                      ground_psi))**2

                # Save fidelity data to csv file
                writer.writerow([p_length, t_pts[t_idx], inner])
                t_selected[t_idx//checkpoint_counter] = t_pts[t_idx]
                adiabaticity[t_idx//checkpoint_counter] = inner

            # Update wavefunction using split-operator method
            psi_x = np.multiply(exp_P,psi_x)
            psi_p = fftshift(fft(psi_x))     
            if t_idx != len(t_pts)-1:
                psi_p = np.multiply(exp_KK,psi_p)
            else:
                psi_p = np.multiply(exp_K,psi_p)
                psi_x = ifft(ifftshift(psi_p))

        # Final update for progress bar
        bar.update(1)
        
        # Print the runtime
        stop = timeit.default_timer()
        print('Simulation complete. Elapsed time is {.3E} seconds.'
              .format(stop - start))
        
        