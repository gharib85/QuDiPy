"""
General helper utilities

@author: simba
"""

import numpy as np

def find_nearest(array, value):
    '''
    Function to find the closest value to a number in a given array.
    If array contains duplicate values of the nearest number, then first
    instance in the array will be returned.
    
    Parameters
    ----------
    array : ND float array-like object
        An ND array type object of floats.
    value : float
        Value which we want to find the closest number to in the array.

    Returns
    -------
    near_idx : tuple of ints
        Tuple of indices for the nearest value in array.
    near_value : float
        Nearest value in array.

    '''
    
    # Convert to numpy array if not already
    array = np.asarray(array)
    
    # Obtain the indices corresponding to which element in array is closest 
    # to value
    near_idx = np.unravel_index((np.abs(array - value)).argmin(), array.shape)
        
    # Return also the nearest value
    near_value = array[near_idx]
    
    return near_idx, near_value


