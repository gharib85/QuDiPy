"""
Real space pulse evolution module

@author: Kewei
"""

import qudipy as qd
import numpy as np
import pandas as pd
import qudipy.potential as pot
import qudipy.qutils as qt
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
import timeit, datetime, time
import os
from tqdm import tqdm

def RSP_time_evolution_1D(pot_interp, ctrl_pulse, dt=5E-16, 
                   show_animation=True, save_data=False, 
                   update_ani_frames=2000, save_points=500,
                   save_dir=None, save_name=None, save_overwrite=False):
    '''
    Perform a time evolution of a 1D real space Hamiltonian (i.e. one that has
    the form H = K + V(x)) according to an arbitrary control pulse. Simulation
    is done using the split-operator method.
    
    Parameters
    ----------
    pot_interp: PotentialInterpolator object
        A 1D potential interpolator object for the quantum dot system.
    ctrl_pulse: ControlPulse object or iterable of ControlPulse objects
        An arbitrary control pulse or iterable of ControlPulse objects to
        sweep over.
    
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
    save_points : int, optional
        How many total data points are saved during the simulation. The 
        default is 500.
    save_dir : string, optional
        Path to location of where to save data (if applicable). The default is
        the current working directory.
    save_name : string, optional
        Filename for the saved data (if applicable). If name is already used,
        the data in the original file will be overwritten. The default is the
        current time in YYYY-MM-DD_HH-MM-SS.csv.
    save_overwrite : bool, optional
        If True and a file of the same name exists in the save directory, 
        overwrite it. If False, then the file to be overwritten will be moved
        to save_name+'_ow' and the current simulation will be written to 
        save_name. The default is False.
        
    Returns
    -------
    None.

    '''

    # Get the material system Constants object from the potential interpolator
    consts = pot_interp.constants
    
    # Get the x-coordinates
    X = pot_interp.x_coords
    gparams = pot.GridParameters(X)
    
    # indices of grid points
    I = [(idx-gparams.nx/2) for idx in range(gparams.nx)]   
    # Define the momentum coordinates
    P = np.asarray([2 * consts.pi * consts.hbar * i / (gparams.nx*gparams.dx)
                    for i in I])

    # Calculate kinetic energy operators used in split-operator method
    exp_K = np.exp(-1j*dt/2*np.multiply(P,P)/(2*consts.me*consts.hbar))
    exp_KK = np.multiply(exp_K,exp_K)

    # Check if control pulse is iterable (i.e. multiple were supplied). If
    # only one was given, then wrap it in a list.
    try:
        iter(ctrl_pulse)
    except TypeError:
        ctrl_pulse = [ctrl_pulse]
        
    # Calculate the runtime of overall simulation
    start_overall = timeit.default_timer()    
    
    # Initialize numpy arrays for saving data
    if save_data:
        save_t = np.zeros(len(ctrl_pulse)*save_points)
        save_fid0 = np.zeros(len(ctrl_pulse)*save_points)
        save_pulse_len = np.zeros(len(ctrl_pulse)*save_points)
        
        # Intialize index for saving data
        save_idx = 0
        
        
    #*********OUTER PULSE LOOP**********#
    # Iterate over all the control pulses
    #***********************************#
    for pulse_idx, curr_pulse in enumerate(ctrl_pulse):        
        print(f'Running RSPTE1D simulation ({pulse_idx+1}/'+
              f'{len(ctrl_pulse)}) '+ 
              f'with control pulse: {curr_pulse.name}.')
    
        # Calculate the runtime
        start_individual = timeit.default_timer()
    
        # Find the initial potential
        init_pulse = curr_pulse([0])[0]
        init_pot = pot_interp(init_pulse)

        # Update GridParameters with the initial potential
        gparams.update_potential(init_pot)

        # Find the initial ground state wavefunction
        __, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
        psi_x = e_vecs[:,0]

        # Get array of time points to sweep over
        p_length = curr_pulse.length
        t_pts = np.linspace(0, p_length, round(p_length/dt))

        # Convert the initial state to momentum space and evolve
        psi_p = fftshift(fft(psi_x))
        psi_p = np.multiply(exp_K,psi_p)

        # Initialize the plot if show animation is true
        if show_animation:
            # Find wavefunction probability
            prob = np.real(np.multiply(psi_x.conj(), psi_x))
    
            # Set up figure for plotting evolution
            fig, ax_pot = plt.subplots()
            # plt.tight_layout(pad=2)
            fig.suptitle('Real space time evolution in 1D', y=1)
            
            # Set up potential axis
            color = 'tab:blue'
            ax_pot.set_ylabel('1D potential [J]', color=color)
            ax_pot.tick_params(axis='y', labelcolor=color)
            
            line_pot, = ax_pot.plot(X, init_pot, color=color)
    
            # Make a twin axis for the wavefunction probability
            ax_wf = ax_pot.twinx()
            
            color = 'tab:red'
            ax_wf.set_ylabel('Simulated state probability', color=color)
            ax_wf.tick_params(axis='y', labelcolor=color)
    
            line_wf, = ax_wf.plot(X, prob, color=color)

        # Inner loop save data initial stuff
        if save_data:
            # Add pulse length data to save data array
            start_ind = 0 + save_points*pulse_idx
            end_ind = start_ind + save_points
            save_pulse_len[start_ind:end_ind] = np.ones(save_points)*p_length
            
            # Find time indicies to save data at
            save_t_idx = np.round(np.linspace(0, len(t_pts)-1, 
                                                      save_points))
                        
        #********INNER PULSE LOOP********#
        # Iterate over all the time points
        #********************************$
        time.sleep(0.5) # Needed for tqdm to work properly
        outer_pot_idx = 0
        for t_idx in tqdm(range(len(t_pts))):
            
            # Get chunks of the interpolated potential along the time axis to
            # save overhead from repeated calls to pot_interp()
            chunk_amt = 200
            if np.mod(t_idx,chunk_amt) == 0:
                # Reset potential indexing counter
                pot_idx = 0
                
                # Get current t_idx for the start of current potential chunk
                start_t_idx = outer_pot_idx*chunk_amt
                
                # Increment counter for which batch of t_idx in the pulse to
                # start interpolating at next time
                outer_pot_idx += 1
                
                # Get the end t_idx for the end of current potential chunk
                end_t_idx = outer_pot_idx*chunk_amt - 1;
                if end_t_idx > len(t_pts):
                    end_t_idx = len(t_pts)
                
                # Get the pulse chunk in time then get the potential chunk 
                # in time
                pulse_chunk = curr_pulse(t_pts[start_t_idx:end_t_idx+1])
                pot_chunk = pot_interp(pulse_chunk)
                
            # Get current potential and increment counter
            curr_potential = pot_chunk[pot_idx]
            pot_idx += 1

            # diagonal matrix of potential energy in position space
            exp_P = np.exp(-1j*dt/consts.hbar * curr_potential)

            # Update wavefunction using split-operator method
            psi_x = ifft(ifftshift(psi_p))
            psi_x = np.multiply(exp_P,psi_x)
            psi_p = fftshift(fft(psi_x))     
            if t_idx != len(t_pts)-1:
                psi_p = np.multiply(exp_KK,psi_p)
            else:
                psi_p = np.multiply(exp_K,psi_p)
                psi_x = ifft(ifftshift(psi_p))
            
            # Show animation periodically
            if show_animation and (t_idx % update_ani_frames == 0 or
                                   t_idx == len(t_pts)-1):
                # Get wavefunction probability
                prob = np.real(np.multiply(psi_x.conj(), psi_x))
                                
                # Update figure data
                line_pot.set_data(X, curr_potential)
                line_wf.set_data(X, prob)

                ax_pot.set_ylim(min(curr_potential),max(curr_potential))

                plt.draw()
                plt.pause(1E-15)
            
            # Save data periodically
            if save_data and (t_idx in save_t_idx):
                # Find the current ground state of the potential
                gparams.update_potential(curr_potential)
                __, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams,
                                                             n_sols=1)
                ground_psi = e_vecs[:,0]
                
                # Save current time index
                save_t[save_idx] = t_pts[t_idx]
                
                # Calculate fidelity of simulated wavefunction w.r.t. ground 
                # state
                inner = abs(qd.qutils.math.inner_prod(gparams, ground_psi,
                                                      psi_x))**2
        
                # Save fidelity w.r.t. ground state
                save_fid0[save_idx] = inner
                
                # Update save index
                save_idx += 1
                        
        #******END INNER PULSE LOOP******#
        # Iterate over all the time points
        #********************************#
        
        time.sleep(0.5) # Needed for tqdm to work properly

        # Print the runtime here, but only if there are multiple control 
        # pulses to sweep over.
        stop_individual = timeit.default_timer()
        if len(ctrl_pulse) != 1:
            individual_time = stop_individual - start_individual
            print('Current simulation complete. Elapsed time is '+
                  f'{individual_time:.3E} seconds.')
        
        # Close figure animation if necessary
        if show_animation:
            plt.close(fig)
                
    #*******END OUTER PULSE LOOP********#
    # Iterate over all the control pulses
    #***********************************#
    
    # Print the total runtime
    stop_overall = timeit.default_timer()
    overall_time = stop_overall - start_overall
    print('\nAll simulations complete! Elapsed time is '+
          f'{overall_time:.3E} seconds.')
    
    # Write the data to .csv file if specified
    if save_data:
        df = pd.DataFrame({
            'Pulse length': save_pulse_len,
            'Time': save_t,
            'Fidelity-0': save_fid0
        })
        
        # If name not specified, use default
        if save_name is None:
            now = datetime.datetime.now()
            save_name = now.strftime("%Y-%m-%d_%H-%M-%S")
        # Add csv extension if not there
        if save_name[-4:] != '.csv':
            save_name += '.csv'
            
        # If save directory not specified, use default
        if save_dir is None:
            save_dir = os.getcwd()
            
        f_path = save_dir + '/' + save_name
        # If a file of the same name exists and we don't want to overwrite it,
        # then copy the old file to a new place before overwriting it. Before
        # writing overwrite file though we check for previous overwrite files
        # and avoid overwriting those too.
        if os.path.exists(f_path) and not save_overwrite:
            ow_idx = 0
            while os.path.exists(f_path[:-4] + f'_ow{ow_idx}.csv'):
                ow_idx += 1
            os.rename(f_path, f_path[:-4] + f'_ow{ow_idx}.csv')
            
        df.to_csv(f_path, index=False)
        
        