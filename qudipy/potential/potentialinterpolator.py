"""
PotentialInterpolator class

@author: simba
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import fminbound

import qudipy as qd
import qudipy.utils as utils
from qudipy.qutils.solvers import solve_schrodinger_eq
from qudipy.qutils.math import inner_prod
from qudipy.potential import GridParameters

class PotentialInterpolator:
    
    def __init__(self, ctrl_vals, ctrl_names, interp_data, single_dim_idx, 
                 constants=qd.Constants()):
        '''
        Initialize the class which, at its core, is basically a wrapper for 
        the scipy RegularGridInterpolator class.

        Parameters
        ----------
        ctrl_vals : 2D list
            List of grid vectors for each control variable (gate voltages AND
            x and y coords).
        interp_data : nd array
            Array of 2D potential (or electric field) data which is to be
            interpolated. Number of dimensions should be num_ctrls + 2 (for x
            and y coords)
        single_dim_idx : list of ints
            List of all control indices which only have a singleton dimension.
        constants : Constants object, optional
            Constants object containing material parameter details.
            The default is a Constants object assuming air as the material
            system.

        Returns
        -------
        None.

        '''
        # Build interpolator object (spline is not supported...)
        self.interp_obj = RegularGridInterpolator(tuple(ctrl_vals),
                                                  interp_data)
        
        # Track how many of the originally inputted grid vectors had only a 
        # single dimension
        self.single_dims = single_dim_idx
        self.n_voltage_ctrls = len(ctrl_vals)-2
        
        # Extract grid vectors
        self.gate_values = ctrl_vals[:-2]
        self.x_coords = ctrl_vals[-1]
        self.y_coords = ctrl_vals[-2]
        
        # Get min/max values for each voltage control
        self.min_max_vals = []
        for idx in range(self.n_voltage_ctrls):
            curr_unique_volts = set(self.gate_values[idx])
            
            self.min_max_vals.append([min(curr_unique_volts),
                                      max(curr_unique_volts)])
        # Store constants
        self.constants = constants
        
        # Store control names
        self.ctrl_names = ctrl_names
    
    def __call__(self, volt_vec_input):
        '''
        Call method for class

        Parameters
        ----------
        volt_vec : 1D float array
            Array of voltage vectors at which we wish to find the interpolated
            2D potential.

        Returns
        -------
        result : 2D float array
            Interpolated 2D potential at the supplied voltage vector.

        '''
        
        # If it's not a list (i.e. like a numpy array, convert to list)
        if not isinstance(volt_vec_input,list):
            volt_vec = list(volt_vec_input)
        # We append stuff to the voltage vector so need to make a copy to make
        # sure we don't change the vector used to actually call the function.
        else:
            volt_vec = volt_vec_input.copy()

        
        # First check if the singleton dimensions were included in volt_vec
        # and remove if so
        # if True ==> they were NOT included
        if len(volt_vec) != self.n_voltage_ctrls:
            # Double check that the inputted voltage vector has at least the
            # original amount of voltages inputted (including the single dim
            # voltages)
            exp_num = self.n_voltage_ctrls + len(self.single_dims)
            if len(volt_vec) != exp_num:
                raise ValueError('Input voltage vector does not have' +
                                 ' correct number of elements.\n' + 
                                 f"Expected {exp_num} or {exp_num-2} number" +
                                 f" of elements, got {len(volt_vec)} instead.")
            else:
                volt_vec = [volt_vec[idx] for idx in range(len(volt_vec)) if
                            idx not in self.single_dims]
        
        # Check if values are out of min/max range
        for idx in range(self.n_voltage_ctrls):
            if (volt_vec[idx] >= self.min_max_vals[idx][0] and
                volt_vec[idx] <= self.min_max_vals[idx][1]):
                pass
            else:
                raise ValueError('Input voltage vector values are out of' +
                                 ' range of grid vectors.')
            
        # Add the x and y coordinates so we interpolate the whole 2D potenial
        volt_vec.extend([self.y_coords, self.x_coords])
        
        # Now build up meshgrid of points we want to query the interpolant
        # object at
        points = np.meshgrid(*volt_vec)
        # Flatten it so the interpolator is happy
        flat = np.array([m.flatten() for m in points])
        # Do the interpolation
        out_array = self.interp_obj(flat.T)
        # Reshape back to our original shape
        result = np.squeeze(out_array.reshape(*points[0].shape))
        
        return result
        
    
    def plot(self, volt_vec, plot_type='2D', y_slice=0, x_slice=None,
             show_wf=False, wf_n=0):
        '''
        Method for plotting the potential landscape at an arbitrary voltage 
        configuration.

        Parameters
        ----------
        volt_vec : 1D float array
            Array of voltage vectors at which we wish to find the interpolated
            2D potential.
        plot_type : string, optional
            Type of plot to show of the potential. Accepted arguments are:
            - '2D': show a 2D plot of the potential landscape. 
            - '1D': show 1D slices of the potential landscape along x- and y-axes.
            The default is '2D'.
        y_slice : float, optional
            Location in [m] along y-axis of where to take a 1D slice of the 
            potential along the x-axis to plot. Only applies if plot_type='1D'.
            The default is 0.
        x_slice : float, optional
            Location in [m] along x-axis of where to take a 1D slice of the 
            potential along the y-axis to plot. Only applies if plot_type='1D'. 
            If not specified, then only a plot along the x-axis will be shown.
            The default is None.
        show_wf : bool, optional
            If True, the wavefunction probability will be overlaid on the 
            potential for 1D plots, and for 2D plots, there will be a subplot 
            added showing the wavefunction probability.
            The deafult is False.
        wf_n : int, optional
            Will plot the nth energy wavefunction. Indexing starts at 0
            which indicates the ground state wavefunction.
            The default is 0.

        Returns
        -------
        None.

        '''

        # Helper function to help add wavefunction overlay to potential plots.
        def _add_wf_overlay(ax, int_pot, slice_idx, slice_axis, wf_n=0):
            '''
            Adds the wavefunction probability to an already created axes 
            object plot of the potential.

            Parameters
            ----------
            ax : axis object
                Axis object for the desired axis where you want to add the 
                overlay of the wavefunction.
            int_pot : 2D array
                The 2D interpolated potential we are currently plotting.
            slice_idx : int
                The index indicating where to take a 1D slice of the 2D
                wavefunction.
            slice_axis : string, optional
                String indicating which axis we are taking a slice of the 2D 
                wavefunction. Changes how we take a slice of the wavefunction
                meshgrid.
            wf_n : int, optional
                Will plot the nth energy wavefunction. Indexing starts at 0
                which indicates the ground state wavefunction.
                The default is 0.
            
            Returns
            -------
            None.

            '''
            # Update color for old axis
            color = 'tab:blue'
            ax.set_ylabel('1D potential [J]', color=color)
            ax.tick_params(axis='y', labelcolor=color)
            
            # Make new axis for overlay
            ax_wf = ax.twinx()
            
            # Set color of new axis to be red
            color = 'tab:red'
            ax_wf.set_ylabel(f'State {wf_n} probability', color=color)
            ax_wf.tick_params(axis='y', labelcolor=color)
            
            # Find the nth wavefunction probabilty
            gparams = GridParameters(self.x_coords, y=self.y_coords, 
                                     potential=int_pot)
            _, state = solve_schrodinger_eq(self.constants, gparams, 
                                            n_sols=(wf_n+1))
            
            # Get correct slice of wavefunction
            if slice_axis == 'y':
                gparams_1D = GridParameters(self.x_coords)
                state = np.squeeze(state[slice_idx,:,wf_n])
            elif slice_axis == 'x':
                gparams_1D = GridParameters(self.y_coords)
                state = np.squeeze(state[:,slice_idx,wf_n])
                
            # Renormalize wf and find probability
            state = state/np.sqrt(inner_prod(gparams_1D, state, state))
            state_prob = np.real(np.multiply(state, state.conj()))
               
            if slice_axis == 'y':
                ax_wf.plot(self.x_coords/1E-9, state_prob, color=color)
            elif slice_axis == 'x':
                ax_wf.plot(self.y_coords/1E-9, state_prob, color=color)

        # Get the potential        
        int_pot = self(volt_vec)
        
        # Do a 1D plot
        if plot_type.upper() == '1D':
            # Get the y-axis slice index
            y_idx, y_val = utils.find_nearest(self.y_coords, y_slice)
            
            # If x-axis slice isn't sepcified, just show x-axis plot.
            if x_slice is None:
                fig, ax = plt.subplots(figsize=(8,8))
                ax.plot(self.x_coords/1E-9, int_pot[y_idx,:].T)
                ax.set(xlabel='x-coords [nm]', ylabel='1D potential [J]',
                   title=f'Potential along x-axis at y={y_val/1E-9:.2f} nm')
                ax.grid()
                
                # Overlay wavefunction if desired
                if show_wf:
                    _add_wf_overlay(ax, int_pot, wf_n=wf_n,
                                    slice_idx=y_idx, slice_axis='y')
                    
                    # otherwise the right y-label is slightly clipped
                    fig.tight_layout()

            # If x-axis slice is specified, show both x- and y-axes plots
            else:
                # Get the x-axis slice index
                x_idx, x_val = utils.find_nearest(self.x_coords, x_slice)
                
                fig = plt.figure(figsize=(12,8))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                
                # potential along x-axis at y-axis slice
                ax1.plot(self.x_coords/1E-9, int_pot[y_idx,:].T)
                ax1.set(xlabel='x-coords [nm]', ylabel='1D potential [J]',
                       title=f'Potential along x-axis at y={y_val/1E-9:.2f} nm')
                ax1.grid()
                                
                # potential along y-axis at x-axis slice
                ax2.plot(self.y_coords/1E-9, int_pot[:,x_idx])
                ax2.set(xlabel='y-coords [nm]', ylabel='1D potential [J]',
                       title=f'Potential along y-axis at x={x_val/1E-9:.2f} nm')
                ax2.grid()
                
                fig.tight_layout()
                
                # Overlay wavefunction if desired
                if show_wf:
                    
                    _add_wf_overlay(ax1, int_pot, wf_n=wf_n,
                                    slice_idx=y_idx, slice_axis='y')
                    
                    _add_wf_overlay(ax2, int_pot, wf_n=wf_n,
                                    slice_idx=x_idx, slice_axis='x')
                    
                    fig.tight_layout(pad=3.0)
        # Do a 2D plot
        elif plot_type.upper() == '2D':
            
            if not show_wf:
                fig, ax = plt.subplots(figsize=(8,8))
                ax.imshow(int_pot, interpolation='bilinear', cmap='viridis',
                               origin='lower', extent=[self.x_coords.min()/1E-9, 
                               self.x_coords.max()/1E-9, self.y_coords.min()/1E-9,
                               self.y_coords.max()/1E-9]
                               )
                ax.set(xlabel='x-coords [nm]',ylabel='y-coords [nm]',
                       title='2D potential')
            # Overlay wavefunction if desired
            else:
                fig, ax = plt.subplots(1,2,figsize=(12,8))
                
                # Plot potential in first subplot
                ax[0].imshow(int_pot, interpolation='bilinear', cmap='viridis',
                               origin='lower', extent=[self.x_coords.min()/1E-9, 
                               self.x_coords.max()/1E-9, self.y_coords.min()/1E-9,
                               self.y_coords.max()/1E-9]
                               )
                ax[0].set(xlabel='x-coords [nm]',ylabel='y-coords [nm]',
                            title='2D potential')
                
                # Find the wavefunction probability
                gparams = GridParameters(self.x_coords, y=self.y_coords, 
                                             potential=int_pot)
                _, state = solve_schrodinger_eq(self.constants, gparams,
                                                n_sols=(wf_n+1))
                state = np.squeeze(state[:,:,wf_n])
                
                state_prob = np.real(np.multiply(state, state.conj()))
                       
                ax[1].imshow(state_prob, interpolation='bilinear', cmap='viridis',
                               origin='lower', extent=[self.x_coords.min()/1E-9, 
                               self.x_coords.max()/1E-9, self.y_coords.min()/1E-9,
                               self.y_coords.max()/1E-9]
                               )
                ax[1].set(xlabel='x-coords [nm]',ylabel='y-coords [nm]',
                            title=f'State {wf_n} probability')
                          
        # Raise an error if we don't recognize the plot_type
        else:
            raise ValueError(f'Error with specified plot_type {plot_type}. ' +
                             'Either ''1D'' or ''2D'' allowed.')
            
        plt.show()
                    
    
    def find_resonant_tc(self, volt_vec, swept_ctrl, bnds=None, peak_threshold=1E5,
                         slice_axis='y', slice_val=0):

        # If swept_ctrl is an integer, then no need to find the corresponding
        # index. If swept_ctrl is a string, we need to check that it is one of
        # the control names and then find the corresponding index.
        if not isinstance(swept_ctrl,int):
            
            raise_error_flag = 0
            try:
                # Get corresponding control index
                ctrl_idx = self.ctrl_names.index(swept_ctrl)
            # Want to raise custom error message.
            except ValueError:
                raise_error_flag = 1
                
            if raise_error_flag:
                raise ValueError(f'Supplied swept_ctrl name {swept_ctrl} is '+
                                 'invalid.\nUseable control names are:\n'+
                                 f'{self.ctrl_names}')
                            
            # If user omits singleton dimensions control from volt_vec, then
            # we need to correct the ctrl_idx
            if len(volt_vec) == self.n_voltage_ctrls + len(self.single_dims):
                pass
            else:
                amt_to_sub = 0
                for idx in self.single_dims:
                    if idx < ctrl_idx:
                        amt_to_sub += 1
                
                ctrl_idx -= amt_to_sub     
            
            # Swap out swept_ctrl for the index now instead
            swept_ctrl = ctrl_idx
        # If it is an int, check that it's not out of range w.r.t. volt_vec
        elif swept_ctrl >= len(volt_vec):
            raise ValueError(f'Supplied swept_ctrl index {swept_ctrl} is invalid.\n'+
                             f'Voltage vector only has {len(volt_vec)} elements.')
          
        # If bnds are not supplied, then set as min max of available voltage
        # ranges
        if bnds is None:
            bnds = [self.min_max_vals[swept_ctrl][0], 
                    self.min_max_vals[swept_ctrl][1]]
              
        # Right now... We are assuming a linear chain of quantum dots where
        # the axis of the linear chain is centered at y=0.
        gparams = GridParameters(self.x_coords, self.y_coords)
        
        if slice_axis=='y':
            # Need for renormalizing 1D wfs
            gparams_1D = GridParameters(self.x_coords)
            slice_idx = utils.find_nearest(self.y_coords, slice_val)[0]
        elif slice_axis=='x':
            gparams_1D = GridParameters(self.y_coords)
            slice_idx = utils.find_nearest(self.x_coords, slice_val)[0]
        else:
            raise ValueError(f'Inputted slice_axis {slice_axis} is not '+
                                 'recognized. Supported values are ''x'' and ''y''.')
        
        # Helper function to take in a voltage vector and return the peaks
        def _find_peaks(curr_val):
            curr_volt_vec = volt_vec.copy()
            curr_volt_vec[swept_ctrl] = curr_val
            curr_pot = self(curr_volt_vec)
            
            gparams.update_potential(curr_pot)
        
            # Find wavefunction and probability distribution
            _, state = solve_schrodinger_eq(self.constants, gparams)
            # Get 1D wavefunction and renormalize
            if slice_axis=='y':
                state = np.squeeze(state[slice_idx,:])
            elif slice_axis=='x':
                state = np.squeeze(state[:,slice_idx])
            state = state/np.sqrt(inner_prod(gparams_1D, state, state))
            # np.real shouldn't be needed, but numerical imprecision causes a warning
            state_prob = np.real(np.multiply(state, state.conj()))
            
            # Find wavefunction peak
            curr_peaks, curr_props = find_peaks(state_prob,
                                           height=peak_threshold,
                                           width=3)
            
            return curr_peaks, curr_props
        
        # Check bnds window to see if a solution even exists.
        # We do this by tuning to the min/max bounds and seeing if the peak
        # changes position from dot to dot. If it does, then a solution exists.
        # If not, then better bounds need to be chosen.  
                
        # Find wavefunction peak at min boundary
        min_peaks, min_props = _find_peaks(bnds[0])
        
        # Find wavefunction peak at max boundary
        max_peaks, max_props = _find_peaks(bnds[1])
        
        # CHECK THE BOUNDARIES TO SEE IF VALID
        # If min wavefunction has two peak locations, then the boundary point
        # is actually a close guess. We need to make sure the largest peak
        # of the two peaks does not move w.r.t to the max boundary though.
        # Otherwise, we won't pass through the resonant tunnel coupling point.
        if len(min_peaks) == 2 and len(max_peaks) == 1:
            # Find the taller peak of the two.
            tall_pk_idx = min_props['peak_heights'].argmax()
            # If that peak location is the same as the location for the max
            # boundary peak, then we won't find a resonant tc point.
            if (abs(max_peaks - min_peaks[tall_pk_idx]) < np.mean(
                    [min_props['widths'][tall_pk_idx], max_props['widths']])):
                
                print('Invalid bounds to search for resonant tunnel coupling '+
                  'point.\nThe min bound shows two peaks but the wavefunction '+
                  'is more localized in\nthe same location as at the maximum '+
                  'boundary point.\nNo control value in the given bounds will '+
                  'be able to find the resonant tunnel coupling.\nPlease LOWER '+
                  'the MINIMUM bounds and try again.\n')
                
                return None
        # Same as above but vice versa for the max boundary
        elif len(min_peaks) == 1 and len(max_peaks) == 2:
            # Find the taller peak of the two.
            tall_pk_idx = max_props['peak_heights'].argmax()
            # If that peak location is the same as the location for the max
            # boundary peak, then we won't find a resonant tc point.
            if (abs(max_peaks[tall_pk_idx] - min_peaks) < np.mean(
                    [min_props['widths'], max_props['widths'][tall_pk_idx]])):
                
                print('Invalid bounds to search for resonant tunnel coupling '+
                  'point.\nThe max bound shows two peaks but the wavefunction '+
                  'is more localized in\nthe same location as at the minimum '+
                  'boundary point.\nNo control value in the given bounds will '+
                  'be able to find the resonant tunnel coupling.\nPlease INCREASE '+
                  'the MAXIMUM bounds and try again.\n')   
                
                return None
        # If both bounds have only 1 peak, then do this check.
        elif len(min_peaks) == 1 and len(max_peaks) == 1:
            # If the wavefunction location doesn't change more than the width of
            # the wavefunction between the min and max bounds, then the 
            # wavefunction is staying in the same location.
            if (abs(max_peaks - min_peaks) < np.mean(
                  [min_props['widths'], max_props['widths']])):
            
                print('Invalid bounds to search for resonant tunnel coupling '+
                      'point.\nThe min and max bounds showed the wavefunction '+
                      'staying in the same potential minima.\nPlease change '+
                      'the bounds so that the min and max boundaries have the '+
                      'wavefunction\nlocalized in different potential minima.\n')
            
                return None
        # If both boundaries have two peaks, then need to make sure the
        # tallest peak for both boundary wavefunction changes.
        elif len(min_peaks) == 2 and len(max_peaks) == 2:
            # Get both tallest peaks
            tall_pk_min_idx = min_props['peak_heights'].argmax()
            tall_pk_max_idx = max_props['peak_heights'].argmax()
            # Check that tallest peak moves location between boundary points
            if (abs(max_peaks[tall_pk_max_idx] - min_peaks[tall_pk_min_idx]) <
                np.mean([min_props['widths'][tall_pk_min_idx],
                         max_props['widths'][tall_pk_max_idx]])):
                
                print('Invalid bounds to search for resonant tunnel coupling '+
                      'point.\nThe min and max bounds showed the wavefunction '+
                      'staying in the same potential minima.\nPlease change '+
                      'the bounds so that the min and max boundaries have the '+
                      'wavefunction\nlocalized in different potential minima.\n')
                
                return None
        # Catch all other conditions.
        else:
            print('Invalid bounds to search for resonant tunnel coupling '+
                  f'point.\n{len(min_peaks)} peaks found at lower boundary.\n'+
                  f'{len(max_peaks)} peaks found at higher boundary.\n'+
                  'Provide new boundaries or change the input control value array.\n')
            
            return None
                
        # If the boundary window is very large (>1 mV), we will try to narrow 
        # it down to help make fminbound work otherwise it can miss the 
        # resonance point due to the way we define the find_peak_difference
        # function.
        # Need these for while loop. Get tallest peak location index for 
        # min and max boundary points.
        tall_min_peak_idx = min_peaks[min_props['peak_heights'].argmax()]
        tall_max_peak_idx = max_peaks[max_props['peak_heights'].argmax()]
        while np.diff(bnds) > 1E-3:
            # Curr midpoint
            bnd_mid = np.mean(bnds)
            
            # Get peaks at midpoint
            mid_peaks, mid_props = _find_peaks(bnd_mid)
            
            # Get tallest peak location index
            tall_mid_peak_idx = mid_peaks[mid_props['peak_heights'].argmax()]
            
            # See if tallest peak is closer to min or max boundary max peak
            # and then replace closer boundary with the midpoint.
            if (abs(tall_min_peak_idx - tall_mid_peak_idx) <
                abs(tall_max_peak_idx - tall_mid_peak_idx)):
                tall_min_peak_idx = tall_mid_peak_idx
                bnds[0] = bnd_mid
            else:
                tall_max_peak_idx = tall_mid_peak_idx
                bnds[1] = bnd_mid
            
        # Helper function for fminbound search
        def find_peak_difference(curr_val):
            
            # Find wavefunction peak at curr_val
            peaks, props = _find_peaks(curr_val)
            
            # If there are two peaks, then return the peak difference, otherwise
            # return a large value to the minimzer.
            if len(peaks) == 2:
                pk_diff = abs(np.diff(props['peak_heights']))
            else:
                pk_diff = 1E8
                        
            return pk_diff
        
        # Search for the point where the peak heights are equal. Since the dots 
        # are assumed to be similarly sized, this would be close to the 0
        # detuning point.        
        volt_tolerance = 1E-6 # muV
        res = fminbound(find_peak_difference, bnds[0], bnds[1], 
                        xtol=volt_tolerance)
        
        # Round value to the order of magnitude given by the tolerance.
        res = np.round(res,int(np.ceil(abs(np.log10(volt_tolerance)))))
                
        return float(res)
    
    