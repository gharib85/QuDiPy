import os, sys
sys.path.append("/Users/keweizhou/Google_Drive/Research/20summer/Waterloo/QuDiPy")
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


def generate_shut_pulse(min_v, max_v, pulse_length, pot_interp, plot_pulse = False):
    '''
    Generate a shuttling pulse that would make the electron tunnel through three dots.

    Parameters
    ----------
    min_v: float
        initial voltage of the second and third dot

    max_v: float
        initial voltage of the first dot

    pulse_length: float
        the total length of the pulse in picoseconds

    pot_interp: PotentialInterpolator object
        potential interpolator object calculated from next_nano files

    plot_pulse: boolean

    Returns
    -------
    shut_pulse: ControlPulse object

    '''

    # Define the pulse with 9 points

    pt1 = [max_v, min_v, min_v]
    # pot_interp.plot(pt1, plot_type='1D', show_wf=True)

    vv = pot_interp.find_resonant_tc(pt1, 1)
    most_vv = vv - 0.035 * (max_v - min_v)
    pt2 = pt1.copy()
    pt2[1] = most_vv

    pt3 = pt2.copy()
    pt3[1] = vv

    pt4 = pt3.copy()
    pt4[0] = most_vv

    pt5 = pt4.copy()
    pt5[0] = min_v

    vv = pot_interp.find_resonant_tc(pt5, 2)
    most_vv = vv - 0.035 * (max_v - min_v)
    pt6 = pt5.copy()
    pt6[2] = most_vv

    pt7 = pt6.copy()
    pt7[2] = vv

    pt8 = pt7.copy()
    pt8[1] = most_vv

    pt9 = pt8.copy()
    pt9[1] = min_v

    shuttle_pulse = np.array([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9])

    shut_pulse = qd.circuit.ControlPulse('shuttle_test', 'experimental', 
                                        pulse_length=pulse_length)
    shut_pulse.add_control_variable('V2',shuttle_pulse[:,0])
    shut_pulse.add_control_variable('V3',shuttle_pulse[:,1])
    shut_pulse.add_control_variable('V4',shuttle_pulse[:,2])
    ctrl_time_L = 10.0 * np.array([0, 1/20, 1/4, 1/2-1/20, 1/2, 1/2+1/20, 3/4, 19/20, 1])
    shut_pulse.add_control_variable('time',ctrl_time_L)

    if plot_pulse == True:
        shut_pulse.plot()

    return shut_pulse

def time_evolution(loaded_data, pot_interp, shut_pulse, animation = True, adiabaticity_data = False, 
        animation_res = 100, adiabaticity_res = 500):
    '''
    Displays the animation of time evolution of the wave functions or save adiabaticity evolution to csv file.

    Parameters
    ----------
    loaded_data: dictionary
        data loaded from nextnano files

    pot_interp: PotentialInterpolator object
        potential interpolator calculated from loaded data

    shut_pulse: ControlPulse object
        a shuttling pulse that would make the electron tunnel through three dots

    animation: boolean
        whether the animation is displayed

    adiabaticity_data: boolean
        whether the adiabaticity data is saved

    animation_res: int
        how often the animation updates its frame

    adiabaticity_res: int
        how many total points are saved

    Returns
    -------
    None.

    '''

    # Find the initial potential
    init_pulse = shut_pulse([0])[0]
    init_pot = pot_interp(init_pulse)

    # Initialize the constants class with the Si/SiO2 material system 
    consts = qd.Constants("Si/SiO2")

    # First define the x-coordinates
    X = loaded_data['coords'].x

    # Create a GridParameters object of the initial potential
    gparams = pot.GridParameters(X, potential=init_pot)

    # Find the initial ground state wavefunction
    __, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
    psi_x = e_vecs[:,0]
    prob = [abs(x)**2 for x in psi_x]

    # Define the limits of the plots
    ymax = 2 * max(prob)
    potmax = max(init_pot) + 1E-21
    potmin = min(init_pot) - 3E-21

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

    # Get the list of pulses
    # 10ps with time steps of dt
    t_pts = np.linspace(0,10, round(10E-12/dt))
    int_p = shut_pulse(t_pts)

    # Convert to momentum space
    psi_p = fftshift(fft(psi_x))
    psi_p = np.multiply(exp_K,psi_p)

    # Calculate interpolated potentials at each time step
    potential_L = pot_interp(int_p)

    if animation:
        # Plot evolution
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        plt.tight_layout(pad=2)
        fig.suptitle('Time Evolution of 1D potential and Electron Shuttling', y = 1)

        line1, = ax1.plot(X, init_pot)
        line2, = ax2.plot(X, prob)

        # Loop through each time step
        for t_idx in range(len(t_pts)):
            potential = potential_L[t_idx]

            # diagonal matrix of potential energy in position space
            exp_P = np.exp(-1j*dt/consts.hbar * potential)

            # Start the split operator method
            psi_x = ifft(ifftshift(psi_p))
            if t_idx%animation_res  == 0:
                prob = [abs(x)**2 for x in psi_x]
                line1.set_data(X, potential)
                line2.set_data(X, prob)

                # ax1.autoscale_view(True,True,True)
                # ax1.relim()
                ax1.set_xlabel("x(m)")
                ax1.set_ylabel("Potential")
                ax1.set_xlim(-2e-7, 2e-7)
                ax1.set_ylim(potmin,potmax)

                ax2.set_xlabel("x(m)")
                ax2.set_ylabel("Probability")
                ax2.set_xlim(-2e-7, 2e-7)
                ax2.set_ylim(-5e6, ymax)

                plt.draw()
                plt.pause(1e-15)
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
        
    if adiabaticity_data:
        t_selected = []
        adiabaticity = []
        p_length = shut_pulse.length
        # Calculate how often we save the data to get a resolution of 500 points
        checkpoint_counter = len(t_pts)//adiabaticity_res

        with open('adiabaticity.csv', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Loop through each time step
            for t_idx in range(len(t_pts)):
                potential = potential_L[t_idx]

                gparams = pot.GridParameters(X, potential=potential)
                # find the ground state under this pulse
                __, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
                ground_psi = e_vecs[:,0]

                # diagonal matrix of potential energy in position space
                exp_P = np.exp(-1j*dt/consts.hbar * potential)

                # Start the split operator method
                psi_x = ifft(ifftshift(psi_p))

                if t_idx%checkpoint_counter  == 0:
                    inner = abs(qd.qutils.math.inner_prod(gparams, psi_x, ground_psi))**2
                    # print(t_pts[t_idx], inner)
                    writer.writerow([p_length, t_pts[t_idx], inner])
                    t_selected.append(t_pts[t_idx])
                    adiabaticity.append(inner)
                psi_x = np.multiply(exp_P,psi_x)
                psi_p = fftshift(fft(psi_x))     
                if t_idx != len(t_pts)-1:
                    psi_p = np.multiply(exp_KK,psi_p)
                else:
                    psi_p = np.multiply(exp_K,psi_p)
                    psi_x = ifft(ifftshift(psi_p))

        # Pring the runtime
        stop = timeit.default_timer()
        print('Pulse length: ', p_length, '\n Time: ', stop - start) 