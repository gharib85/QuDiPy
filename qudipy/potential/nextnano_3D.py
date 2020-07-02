import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp2d
import os
import re
from itertools import product

class mod_RegularGridInterpolator:
    
    def __init__(self, variables, nd_data):
        

def __is_float(str_data):
    '''
    Checks if a string can be converted to a float or not.

    Parameters
    ----------
    str_data : string

    Returns
    -------
    bool
        Returns True if string is a float, else False.

    '''
    
    try:
        return float(str_data)
    except ValueError:
        return False

def __is_int(str_data):
    '''
    Checks if a string can be converted to an int or not.

    Parameters
    ----------
    str_data : string

    Returns
    -------
    bool
        Returns True if string is an int, else False.

    '''
    try:
        return int(str_data)
    except ValueError:
        return False

def load_file(filename):
    """
    returns a single array ordered by the coordinates for potential.dat
            a tuple of 3 element, x, y, z for coord files
    """
    data = []
    x = []
    y = []
    z = []
    counter = 0
    with open(filename, 'r') as f:
        d = f.readlines()
        # .dat file indicates it's a nextnano file
        if filename[-4:] == '.dat':
            for i in d:
                k = i.rstrip().split(" ")
                data.append(float(k[0]))     
            data = np.array(data, dtype='O')
            return data
        # Coordinate file so we need to load the x,y,z coordinate data
        elif filename[-6:] == '.coord':
            for i in d:
                k = i.rstrip().split(" ")
                print(k)
                if __is_float(i)==False:
                    # append number list if the element is an int but not float
                    try:
                        int(i)
                        if counter == 0:
                            x.append(float(k[0]))
                        elif counter == 1:
                            y.append(float(k[0]))
                        else:
                            z.append(float(k[0]))
                    # ValueError happens when it hits an empty line
                    except ValueError:
                        # print(i)
                        counter+=1
                # counter keeps track of which coord the data belong to
                elif counter == 0:
                    x.append(float(k[0]))
                elif counter == 1:
                    y.append(float(k[0]))
                else:
                    z.append(float(k[0]))
            x = np.array(x, dtype='O')
            y = np.array(y, dtype='O')
            z = np.array(z, dtype='O')
            return x, y, z
        elif filename[-4:] in ['.csv', '.txt']:
            print('CSV')

def reshape_potential(potential, x, y, z, slice, option):
    """
    input:  1d potential array, 
            lists of x, y ,z coordinates
            the z coordinate indicating the slice of x-y plane
    output: a 2d array of the potentials in the x-y plane
    """
    index = np.where(z==slice)[0]
    N = len(x)
    M = len(y)
    Q = len(z)
    pot3DArray = np.reshape(potential,(N,M,Q),order='F')
    if option == "field":
        gradient = np.gradient(pot3DArray,x,y,z)[-1]
        pot2DArray = gradient[:, :, index]
    else:
        pot2DArray = pot3DArray[:, :, index]
    return pot2DArray
    
def parse_voltage(filename):
    """
    input: a string, the filename 
           an int, number of gates
    output: a list of voltages of each gate
    """
    org = re.split("[_/]",filename)
    s = []
    delete = []
    for i in org:
        try:
            if float(i) < 100:
                s.append(float(i))
        except ValueError:
            delete.append(i)
    return s

def import_folder(folder):
    """
    input: a string, name of the folder where nextnano++ files are stored 
    output: a list, where each element is a list of voltages, potentials, and coordinates
    """
    L = []                  # each element in L would be a list of voltages, potentials, and coordinates
    counter = 0             # track which subdirectory 
    for subdir, dirs, files in os.walk(folder):
        if subdir != folder and subdir[-7:] != '/output':
            counter += 1
            voltage = parse_voltage(subdir)
            L.append([voltage])
        for file in files:
            filename = os.path.join(subdir, file)
            # always first .dat then .coord
            if filename[-4:] == '.dat' or filename[-6:] == '.coord':
                L[counter-1].append(load_file(filename))
    return L

def group_2D_potential(potentialL, voltages, coord, slice, option):
    """
    input:  a list, where each element is a list of voltages, potentials, and coordinates
            a list of gate voltages
            a float indicating the x-y plane
    output: an n-dimensial potential file, where n = number of gates + 2
    """
    potentialL_copy = potentialL.copy()
    # loop through each combination of gate voltages
    for i in potentialL_copy:
        if option == "potential":
            # slice an x-y plane of the potentials
            potential2D = reshape_potential(i[1], i[2][0], i[2][1], i[2][2], slice, option)
        elif option == "field":
            potential2D = reshape_potential(i[1], i[2][0], i[2][1], i[2][2], slice, option)
        i[1] = potential2D
        # reverse the list of voltages for sorting purpose
        i[0].reverse()
    potentialL_copy.sort()

    # stack up the potential arrays in the correct order
    potential_elmt = ()
    for i in range(len(potentialL_copy)):
        potential_elmt = potential_elmt + (potentialL_copy[i][1],) 
    potential_overall = np.stack(potential_elmt, axis = 0)

    # get the shape of the potential based on the number of gates and the voltages of each gate
    shape = ()
    for v in voltages:
        if len(v) > 1:
            shape = shape + (len(v),)
    shape = shape+ (len(coord[0]), len(coord[1]))
    
    potential_reshaped = np.reshape(potential_overall,shape)
    return potential_reshaped


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

def build_interpolator(all_data):
    
    
    
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
    # V2 = [0.2, 0.22, 0.24, 0.26, 0.27]
    # V3 = [0.2, 0.22, 0.24, 0.26, 0.27]
    # V4 = [0.2, 0.22, 0.24, 0.26, 0.27]
    V2 = [0.2, 0.22]
    V3 = [0.2, 0.22]
    V4 = [0.2, 0.22]
    V5 = [0.1]
    ctrl_vals = [V1, V2, V3, V4, V5]
    
    ctrl_names = ['V1','V2','V3','V4','V5']
    
    pot_dir = './Sliced_potentials/'
    
    potentialL = load_files(ctrl_vals, ctrl_names, f_type='pot', f_dir=pot_dir)
    

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
    
    
    
    




