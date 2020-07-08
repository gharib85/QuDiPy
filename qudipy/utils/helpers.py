"""
General helper utilities for the codebase
"""

import numpy as np

def find_nearest(array, value):
    '''
    Function to find the closest value to a number in a given array.
    If array contains duplicate values of the nearest number, then first
    instance in the array will be returned.
    
    Parameters
    ----------
    array : 1D float array-like object
        A 1D array type object of floats.
    value : float
        Value which we want to find the closest number to in the array.

    Returns
    -------
    near_idx : int
        Index of nearest value in array.
    near_value : float
        Nearest value in array.

    '''
    
    array = np.asarray(array)
    near_idx = (np.abs(array - value)).argmin()
    
    near_value = array[near_idx]
    
    return near_idx, near_value


