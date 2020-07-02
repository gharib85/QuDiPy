import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp2d
import os
import re
from itertools import product


def interp(potential, voltages, coord):
    """
    inputs: 
        potential is a n-dimensional array of 
        voltages is a list of gate voltages
        coord is a
    output:
        interpolating function with inputs of gate voltages and coordinates
    """
    x = [float(xi) for xi in coord[0]]
    y = [float(yi) for yi in coord[1]]
    variables = ()
    for v in voltages:
        if len(v) > 1:
            variables = variables + (v,)
    variables = variables+ (x,y)
    interpolating_func = RegularGridInterpolator(variables, potential)
    return interpolating_func

def build_interpolator(all_data_sep):
    
    # Get first set of x and y coordinates
    x_coords = all_data_sep[0][2][0]
    y_coords = all_data_sep[0][2][1]
    
    # Get total number of ctrls (including singleton dims)
    n_ctrls = len(all_data_sep[0][0])
    
    # Assemble the gate voltages
    all_ctrls = np.zeros((len(all_data_sep), n_ctrls))
    for idx in range(len(all_data_sep)):
        all_ctrls[idx,:] = all_data_sep[idx][0]
        
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
    
    # It's a bit convoluted.. But we need to actually reverse the dimensions
    # for the interpolant object due to the way we assembled all the potentials
    # in the load_files function in order to most easily work with constructing
    # the interpolator
    n_dims.reverse()
    ctrl_values = [ctrl_values[idx] for idx in reversed(range(len(ctrl_values)))]
    
    # Add the y and x coordinate lengths so we know the expected dimensions of 
    # the total nd array of data to interpolate
    all_data_stacked = np.zeros((np.prod(n_dims),len(y_coords),len(x_coords)))
    n_dims.extend([len(y_coords),len(x_coords)])  

    # Go and stack the data together and then reshape it into correct format    
    for idx, curr_gate_idx in enumerate(product(*temp_n_dims)):
        all_data_stacked[idx,:,:] = all_data_sep[idx][1]
    
    all_data_stacked = np.reshape(all_data_stacked,(n_dims))
    
    # Construct the interpolator
    ctrl_values.extend([y_coords,x_coords])
    interp_obj = RegularGridInterpolator(tuple(ctrl_values), all_data_stacked)
    
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


def load_files(ctrl_vals, ctrl_names, f_type='pot', f_dir=None):
    '''
    This function loads many potential files specified by all combinations of 
    the control values given in ctrl_vals and the control names given in
    ctrl_names. The potential files MUST be 2D potential slices (if you have
    3D nextnano simulations, you must preprocess them first). Potential files
    are assumed to follow the syntax: 
    'TYPE_C1NAME_C1VAL_C2NAME_C2VAL_..._CNNAME_CNVAL.txt'
    where TPYE = 'Uxy' or 'Ez'. 
    Refer to tutorial for a more explicit example.

    Parameters
    ----------
    ctrl_vals : list of float lists
    ctrl_names : string list
        List of each ctrl variable name. Must be the same length as ctrl_vals 
        first dimension.
    f_type : string
        Type of file to load (either potential or electric field). Acceptable 
        arguments include ['pot','potential','Uxy','electric','field','Ez'].
        Default is potential.
    f_dir : string
        Path to find files specified in f_list.  This is an optional argument.
        The default location to search is the current working directory.
    
    Returns
    -------
    None.

    '''

    # Check inputs
    if len(ctrl_vals) != len(ctrl_names):
        raise ValueError('ctrl_vals and ctrl_names must have the same number\
                         of elements.')
    
    # Check if dir was given
    if f_dir is None:
        f_dir = ''

    all_pots = []
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
        all_pots.append([curr_cvals])
        all_pots[idx].append(pot)
        all_pots[idx].append((x,y))
        
    return all_pots

if __name__ == "__main__":
    
    # # number of gates
    # numberOfGates = 5
    
    # gate voltages
    V1 = [0.1]
    V2 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28, 0.29, 0.30]
    V3 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28, 0.29, 0.30]
    V4 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28, 0.29, 0.30]
    V5 = [0.1]
    ctrl_vals = [V1, V2, V3, V4, V5]
    
    ctrl_names = ['V1','V2','V3','V4','V5']
    
    pot_dir = './Sliced_potentials/'
    
    potentialL = load_files(ctrl_vals, ctrl_names, f_type='pot', f_dir=pot_dir)
    
    pot_interp = build_interpolator(potentialL)
    

    # folder = 'nextnanoSims_Small'
    
    # potentialL = import_folder(folder)
    # voltages = [V1, V2, V3, V4, V5]
    # coord = potentialL[0][2]
    # potentialND = group_2D_potential(potentialL, voltages, coord, -1, "potential")
    # out = interp(potentialND, voltages, coord)
    # print(out([0.2,1.0,1.0]))
    
    # potentialL = import_folder(folder)
    # fieldND = group_2D_potential(potentialL, voltages, coord, -1, "field")
    # out2 = interp(fieldND, voltages, coord)
    # print(out2([0.2,1.0,1.0]))
    
    
    
    




