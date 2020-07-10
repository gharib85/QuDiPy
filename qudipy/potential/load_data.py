import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp2d
from itertools import product

class Mod_RegularGridInterpolator:
    
    def __init__(self, ctrl_vals, interp_data, single_dim_idx):
        '''
        Initialize the class which is basically a wrapper for the scipy
        RegularGridInterpolator class.

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
    
    def __call__(self, volt_vec):
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
        
        # First check if the singleton dimensions were included in volt_vec
        # and remove if so (
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
    interp_obj = Mod_RegularGridInterpolator(ctrl_values, all_data_stacked,
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


def load_potentials(ctrl_vals, ctrl_names, f_type='pot', f_dir=None):
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
    f_type : string
        Type of file to load (either potential or electric field). Acceptable 
        arguments include ['pot','potential','Uxy','electric','field','Ez'].
        Default is potential.
    f_dir : string
        Path to find files specified in f_list.  This is an optional argument.
        The default location to search is the current working directory.
    
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
        
        if idx == 0:
            all_files['coords'] = (x,y)
            
        cval_array.append(list(curr_cvals))
        pots_array.append(pot)
        
    all_files['ctrl_vals'] = cval_array
    all_files['potentials'] = pots_array
    
    
    return all_files

    
    




