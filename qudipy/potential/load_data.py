import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.interpolate import RegularGridInterpolator, interp2d
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.optimize import fminbound

import qudipy as qd
import qudipy.utils as utils
from qudipy.qutils.solvers import solve_schrodinger_eq
from qudipy.potential import GridParameters


class PotentialInterpolator:
    
    def __init__(self, ctrl_vals, interp_data, single_dim_idx):
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
        result : 2D array
            Interpolated 2D potential at the supplied voltage vector.

        '''
        # We append stuff to the voltage vector so need to make a copy to make
        # sure we don't change the vector used to actually call the function.
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
    
    def plot(self, volt_vec, plot_type='2D', y_slice=0, x_slice=None):
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

        Returns
        -------
        None.

        '''

        # Get the potential        
        int_pot = self(volt_vec)
        
        # Do a 1D plot
        if plot_type == '1D':
            # Get the y-axis slice index
            y_idx, y_val = utils.find_nearest(self.y_coords, y_slice)
            
            # If x-axis slice isn't sepcified, just show x-axis plot.
            if x_slice is None:
                fig, ax = plt.subplots(figsize=(8,8))
                ax.plot(self.x_coords/1E-9, int_pot[y_idx,:].T)
                ax.set(xlabel='x-coords [nm]', ylabel='potential [J]',
                   title=f'Potential along x-axis at y={y_val/1E-9:.2f} nm')
                ax.grid()

            # If x-axis slice is specified, show both x- and y-axes plots
            else:
                # Get the x-axis slice index
                x_idx, x_val = utils.find_nearest(self.x_coords, x_slice)
                
                f = plt.figure(figsize=(12,8))
                ax1 = f.add_subplot(121)
                ax2 = f.add_subplot(122)
                
                # potential along x-axis at y-axis slice
                ax1.plot(self.x_coords/1E-9, int_pot[y_idx,:].T)
                ax1.set(xlabel='x-coords [nm]', ylabel='potential [J]',
                       title=f'Potential along x-axis at y={y_val/1E-9:.2f} nm')
                ax1.grid()
                
                # potential along y-axis at x-axis slice
                ax2.plot(self.y_coords/1E-9, int_pot[:,x_idx])
                ax2.set(xlabel='y-coords [nm]', ylabel='potential [J]',
                       title=f'Potential along y-axis at x={x_val/1E-9:.2f} nm')
                ax2.grid()
        # Do a 2D plot
        elif plot_type == '2D':
            
            fig, ax = plt.subplots(figsize=(8,8))
            ax.imshow(int_pot, interpolation='bilinear', cmap='viridis',
                           origin='lower', extent=[self.x_coords.min()/1E-9, 
                           self.x_coords.max()/1E-9, self.y_coords.min()/1E-9,
                           self.y_coords.max()/1E-9]
                           )
            ax.set(xlabel='x-coords [nm]',ylabel='y-coords [nm]')
        # Raise an error
        else:
            raise ValueError(f'Error with specified plot_type {plot_type}. ' +
                             'Either ''1D'' or ''2D'' allowed.')
                    
    
    def find_resonant_tc(self, volt_vec, swept_ctrl, bnds, peak_threshold=1000):

        
        # We will need some Constants
        cnst = qd.Constants("Si/SiO2")

        # Right now... We are assuming a linear chain of quantum dots where
        # the axis of the linear chain is centered at y=0.
        gparams = GridParameters(self.x_coords)
        y_idx = utils.find_nearest(self.y_coords, 0)[0]
        
        # Check bnds window to see if a solution even exists.
        # We do this by tuning to the min/max bounds and seeing if the peak
        # changes position from dot to dot. If it does, then a solution exists.
        # If not, then better bounds need to be chosen.  
        
        # Get min volt vec/1D potential
        min_volt_vec = volt_vec.copy()
        min_volt_vec[swept_ctrl] = bnds[0]
        min_pot = self(min_volt_vec)
        gparams.update_potential(np.squeeze(min_pot[y_idx,:]))
        
        # Find wavefunction and probability distribution
        _, state = solve_schrodinger_eq(cnst, gparams)
        state = np.squeeze(state)
        # np.real shouldn't be needed, but numerical imprecision causes a warning
        state_prob = np.real(np.multiply(state, state.conj()))
        
        # Find wavefunction peak
        min_peaks, min_props = find_peaks(state_prob,
                                       height=peak_threshold,
                                       width=3)
        
        # Get max volt vec/1D potential
        max_volt_vec = volt_vec.copy()
        max_volt_vec[swept_ctrl] = bnds[1]
        max_pot = self(max_volt_vec)
        gparams.update_potential(np.squeeze(max_pot[y_idx,:]))
        
        # Find wavefunction and probability distribution
        _, state = solve_schrodinger_eq(cnst, gparams)
        state = np.squeeze(state)
        # np.real shouldn't be needed, but numerical imprecision causes a warning
        state_prob = np.real(np.multiply(state, state.conj()))
        
        # Find wavefunction peak
        max_peaks, max_props = find_peaks(state_prob,
                                       height=peak_threshold,
                                       width=3)
                
        # If the wavefunction location doesn't change more than the width of
        # the wavefunction between the min and max bounds, then the 
        # wavefunction is staying in the same location.
        if abs(max_peaks - min_peaks) < np.mean(
                [min_props['widths'], max_props['widths']]):
            print('Invalid bounds to search for resonant tunnel coupling '+
                  'point.\nThe min and max bounds showed the wavefunction '+
                  'staying in the same potential minima.\nPlease increase '+
                  'the bounds so that the min and max boundaries have the '+
                  'wavefunction\nlocalized in different potential minima.')
            
            return None
        else:
            pass
        
        # Helper function for fminbounds search
        def find_peak_difference(curr_val):
            curr_volt_vec = volt_vec.copy()
            curr_volt_vec[swept_ctrl] = curr_val
            curr_pot = self(curr_volt_vec)
            
            gparams.update_potential(curr_pot)
            
            e_ens, e_vecs = solve_schrodinger_eq(cnst, gparams)
            y_idx = utils.find_nearest(self.y_coords, 0)[0]
            state_1D = np.squeeze(e_vecs[y_idx,:,0])
            gparams_1D = GridParameters(gparams.x)
            state_1D_prob = np.multiply(state_1D, state_1D.conj())
            # state_1D_prob = state_1D_prob/np.sqrt(inner_prod(gparams_1D,state_1D_prob,state_1D_prob))
            peaks, _ = find_peaks(state_1D_prob,height=peak_threshold)
            
            if len(peaks) == 2:
                pk_diff = abs(np.diff(state_1D_prob[peaks]))
            else:
                pk_diff = 1E4
                        
            return pk_diff
        
        volt_tolerance = 1E-6 # muV
        if bnds is not None:
            res = fminbound(find_peak_difference, bnds[0], bnds[1], 
                            xtol=volt_tolerance)
        else:
            res = fminbound(find_peak_difference, self.min_max_vals[swept_ctrl][0],
                            self.min_max_vals[swept_ctrl][1], 
                            xtol=volt_tolerance)
        
        # Round value to the order of magnitude given by the tolerance.
        # res = np.round(res,int(np.ceil(abs(np.log10(volt_tolerance)))))
        # print('Result:',res)
        
        # Plot final result
        # curr_volt_vec = volt_vec.copy()
        # curr_volt_vec[swept_ctrl] = res
        # curr_pot = self(curr_volt_vec)
        # gparams.update_potential(curr_pot)
        # e_ens, e_vecs = solve_schrodinger_eq(cnst, gparams)
        # y_idx = utils.find_nearest(self.y_coords, 0)[0]
        # state_1D = np.squeeze(e_vecs[y_idx,:,0])
        # gparams_1D = GridParameters(gparams.x)
        # state_1D_prob = np.multiply(state_1D, state_1D.conj())
        # state_1D_prob = state_1D_prob/np.sqrt(inner_prod(gparams_1D,state_1D_prob,state_1D_prob))
        
        # plt.figure()
        # y_idx = utils.find_nearest(self.y_coords, 0)[0]
        # plt.plot(gparams.x/1E-9, np.real(state_1D_prob))
        
        # self.plot(curr_volt_vec)
        
        # return res
        
def build_interpolator(load_data_dict):
    '''
    This function constructs an interpolator object for either a group of 
    potential or electric field files.

    Parameters
    ----------
    all_data_sep : dict
        Dictionary containing the x and y coordinates for the loaded files,
        the potential data for each loaded file, and the corresponding votlage
        vector for each file.
        Fields = ['coords', 'potentials', 'ctrl_vals']

    Returns
    -------
    interp_obj : Mod_RegularGridInterpolator class
        Interpolant object for the data inputted into the function.

    '''
    
    # Get first set of x and y coordinates
    x_coords = load_data_dict['coords'][0]
    y_coords = load_data_dict['coords'][1]
    
    # Extract all the control values
    all_ctrls = np.asarray(load_data_dict['ctrl_vals'])
    
    # Get total number of ctrls (including singleton dims)
    n_ctrls = len(load_data_dict['ctrl_vals'][0])
        
    # Find which gate voltages have singleton dimension. We need to keep track
    # because the interpolator cannot handle singleton dimensions
    single_dims = []
    n_dims = []
    ctrl_values = []
    for idx in range(n_ctrls):
        n_unique = len(set(all_ctrls[:,idx]))
        if n_unique == 1:
            single_dims.append(idx)
        else:
            n_dims.append(n_unique)
            ctrl_values.append(sorted(list(set(all_ctrls[:,idx]))))
        
    # Now assemble the data to be interpolated
    temp_n_dims = [range(n) for n in n_dims]
    
    # Add the y and x coordinate lengths so we know the expected dimensions of 
    # the total nd array of data to interpolate
    all_data_stacked = np.zeros((np.prod(n_dims),len(y_coords),len(x_coords)))
    n_dims.extend([len(y_coords),len(x_coords)])  

    # Go and stack the potential data together and then reshape it into
    # correct format    
    for idx, curr_gate_idx in enumerate(product(*temp_n_dims)):
        all_data_stacked[idx,:,:] = load_data_dict['potentials'][idx]
    
    all_data_stacked = np.reshape(all_data_stacked,(n_dims))
    
    # Construct the interpolator
    ctrl_values.extend([y_coords,x_coords])
    interp_obj = PotentialInterpolator(ctrl_values, all_data_stacked,
                                             single_dims)
    
    return interp_obj
    
def _load_one_file(fname):
    '''
    This function loads a single file of either potential or electric field
    data and returns the coordinate and 2D data. The data is always upsampled
    via a spline interpolation to the next power of 2.

    Parameters
    ----------
    fname : string
        Name of file to be loaded.

    Returns
    -------
    new_x_coord : float 1D array
        x-coordinate data after loading and interpolation.
    new_y_coord : float 1D array
        y-coordinate data after loading and interpolation.
    new_pot_xy : float 2D array
        Potential or electric field data after loading and interpolation.

    '''

    # Load file
    data = pd.read_csv(fname, header=None).to_numpy()
    
    # Extract items
    x_coord = data[0,1:]
    y_coord = data[1:,0]
    pot_xy = data[1:,1:]
    
    # Do a spline interpolation to increase number of x/y coordinates to the
    # highest power of 2 (for faster ffts if needed)
    new_x_len = 1 if len(x_coord) == 0 else 2**(len(x_coord) - 1).bit_length()
    new_y_len = 1 if len(y_coord) == 0 else 2**(len(y_coord) - 1).bit_length()
    
    new_x_coord = np.linspace(x_coord.min(), x_coord.max(), new_x_len)
    new_y_coord = np.linspace(y_coord.min(), y_coord.max(), new_y_len)
    
    f = interp2d(x_coord, y_coord, pot_xy, kind='cubic')
    new_pot_xy = f(new_x_coord, new_y_coord)
    
    return new_x_coord, new_y_coord, new_pot_xy


def load_potentials(ctrl_vals, ctrl_names, f_type='pot', f_dir=None, 
                    f_pot_units='J', f_dis_units='m'):
    '''
    This function loads many potential files specified by all combinations of 
    the control values given in ctrl_vals and the control names given in
    ctrl_names. The potential files MUST be 2D potential slices (if you have
    3D nextnano simulations, you must preprocess them first). Potential files
    are assumed to follow the syntax: 
    'TYPE_C1NAME_C1VAL_C2NAME_C2VAL_..._CNNAME_CNVAL.txt'
    where TYPE = 'Uxy' or 'Ez'. 
    Refer to tutorial for a more explicit example.

    Parameters
    ----------
    ctrl_vals : list of list of floats
        List of relevant control values for the files to load.  The first list
        index corresponds to the ith control variable and the second list
        index correspond to the ith value for that control variable.
    ctrl_names : list of strings
        List of each ctrl variable name. Must be the same length as ctrl_vals 
        first dimension.
    f_type : string, optional
        Type of file to load (either potential or electric field). Acceptable 
        arguments include ['pot','potential','Uxy','electric','field','Ez'].
        Default is potential. The default is 'pot'
    f_dir : string, optional
        Path to find files specified in f_list. The default is is the current
        working directory.
    f_pot_units : string, optional
        Units of the potential in the files to load. Units from file will be
        converted to J.
        Supported inputs are 'J' and 'eV'.
    f_dis_units : string, optional
        Units of the x and y coordinates in the files to load. Units from file
        will be converted to m. 
        Supported inputs are 'm' and 'nm'. 
    
    Returns
    -------
    all_files : dict
        Dictionary containing the x and y coordinates for the loaded files,
        the potential data for each loaded file, and the corresponding votlage
        vector for each file.
        Fields = ['coords', 'potentials', 'ctrl_vals']

    '''

    # Check inputs
    if len(ctrl_vals) != len(ctrl_names):
        raise ValueError('Incorrect number of control names given, must be ' +
                         'equal to first dimension of ctrl_vals ' +
                         f' {len(ctrl_vals)}.')
    
    # Check if dir was given
    if f_dir is None:
        f_dir = ''

    all_files = {}
    cval_array = []
    pots_array = []
    # Loop through all combinations of gate voltages to load the files
    for idx, curr_cvals in enumerate(product(*ctrl_vals)):
        # Now build up the current file name
        # First figure out type of file to load (electric or potential)
        if f_type in ['pot', 'potential', 'Uxy']:
            f_name = 'Uxy'
        elif f_type in ['field', 'electric', 'Ez']:
            f_name = 'Ez'
            
        for name, val in zip(ctrl_names, curr_cvals):
            f_name = f_name+ '_' + name + '_' + "{:.3f}".format(val)
        
        f_name += '.txt'
        
        # After file name is constructed, load the data from file into a larger
        # list containing information about all the loaded files.
        x, y, pot = _load_one_file(f_dir + f_name)
        
        # Convert units if needed
        if f_pot_units == 'eV':
            # Just need to get electron charge
            constants = qd.Constants('air')
            pot *= constants.e
            
        if f_dis_units == 'nm':
            x *= 1E-9
            y *= 1E-9
        
        if idx == 0:
            # Have coordinates be a namedtuple
            Coordinates = namedtuple('Coordinates',['x','y'])
            all_files['coords'] = Coordinates(x,y)
            
        cval_array.append(list(curr_cvals))
        pots_array.append(pot)
        
    all_files['ctrl_vals'] = cval_array
    all_files['potentials'] = pots_array
    
    
    return all_files

    
if __name__ == "__main__":
    
    # Enter the name of the folder where the potential files are located. 
    # If this argument is not supplied it will assume the current working directory.
    pot_dir = '/Users/simba/Documents/GitHub/Silicon-Modelling/tutorials/QuDiPy tutorial data/Pre-processed potentials/'
    
    # Specify the control voltage names (C#NAME as mentioned above)
    ctrl_names = ['V1','V2','V3','V4','V5']
    
    # Specify the control voltage values you wish to load.
    # The cartesian product of all these supplied voltages will be loaded and MUST exist in the directory.
    V1 = [0.1]
    V2 = [0.2, 0.22, 0.24, 0.26, 0.28]
    V3 = [0.2, 0.22, 0.24, 0.26, 0.28, 0.29]
    V4 = [0.2, 0.22, 0.24, 0.26, 0.28, 0.29]
    V5 = [0.1]
    # Add all voltage values to a list
    ctrl_vals = [V1, V2, V3, V4, V5]    
    
    # Now load the potentials.  
    # load_files returns a dictionary of all the information loaded
    loaded_data = load_potentials(ctrl_vals, ctrl_names, f_type='pot', 
                                  f_dir=pot_dir, f_pot_units="eV", 
                                  f_dis_units="nm")
    
    # Now building the interpolator object is trivial
    pot_interp = build_interpolator(loaded_data)
        
    # RIGHT ANSWER
    # v_vec = [0.28,0.2616,0.27]
    v_vec = [0.28,0.26,0.27]
    # pot_interp.plot(v_vec)
    
    # pot_interp.find_resonant_tc(v_vec,1,[0.261,0.262])
    # pot_interp.find_resonant_tc(v_vec,1,[0.255,0.257])
    
    # pot_interp.plot(v_vec)
    # pot_interp.plot(v_vec, plot_type='2D')
    # pot_interp.plot(v_vec, plot_type='1D')
    pot_interp.plot(v_vec, plot_type='1D',y_slice=10E-9)
    # pot_interp.plot(v_vec, plot_type='1D',y_slice=1E-9,x_slice=2E-9)
    # pot_interp.plot(v_vec, plot_type='1D',x_slice=20E-9)
    
        # find_res_tc TODO
        # Add Constants class as a class variable
        # How to improve sweep if bnd window is too large?
        # How to deal with noise? (only take highest value peak maybe?)
        # Needs to be generalized to deal with 2D peaks
        # Change find_res_tc to have swept_ctrl be index for the ctrl_values
        # How to improve swept_ctrl (make more robust and maybe take in ctrl names as well)
        # Update loading tutorial to use plot methods.
        
        # plot TODO
        # Add functionality to overlay wavefunction?

    



