"""
General helper utilities

@author: simba
"""

import numpy as np
import math

class TerminalProgressBar:
    '''
    Class for displaying a progress bar in the terminal.
    
    '''
    
    def __init__(self,title='Running...', char_width=30, symbol='='):
        '''
        Initialize the class

        Keywrod Arguments
        -----------------
        title : string , optional
            Title for the progress bar you want to display. The default is
            'Running...'.
        char_width : int, optional
            Number of ascii characters making up the width of the progress
            bar in the temrinal. The default is 50.
        symbol : string, optional
            Symbol to use as the bar for the progress bar. Must be only a 
            single ascii character. The default is '='.

        Returns
        -------
        None.

        '''
        
        # Store values
        self.name = title
        self.width = char_width
        
        if len(symbol) != 1:
            raise(ValueError,'Keyword argument symbol must have only a single'+
                  ' ascii character')
        else:
            self.symbol = symbol
        
        # Initialize the progress bar
        print(self.name)
        self.update(0)
        
    def update(self, completed_ratio):
        '''
        

        Parameters
        ----------
        completed_ratio : float
            Ratio of total progress for the progress bar. Must be a number 
            between 0 <= x <= 1.

        Returns
        -------
        None.

        '''
        
        # Recreate progress bar string
        # bar_str is the actual progress bar
        bar_str = self.symbol*math.floor(completed_ratio*self.width)
        bar_str = bar_str + ' '*(self.width - len(bar_str))
        bar_str = '|' + bar_str + '|'
            
        # Also display percentage of progress
        prog_str = ' Progress {:06.2f}% |'.format(100*completed_ratio)
        
        # Display progress bar
        if completed_ratio != 1:
            end_str = ''
        else:
            end_str = '\n'
    
        print('\r',end='')
        print(bar_str+prog_str,end=end_str)
        
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


