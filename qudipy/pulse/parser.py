"""
Functions for parsing control pulse files .ctrlp
"""

import pandas as pd
from .controlPulse import ControlPulse

def load_pulse(f_name):
    '''
    

    Parameters
    ----------
    f_name : string
        Full file path to the control pulse file to load.

    Returns
    -------
    ctrl_pulse : ControlPulse object
        Control pulse object containing all information loaded from the file.

    '''
    
    # Count the number of lines which don't the control pulse vs time information
    f = open(f_name, "r")
    line_cnt = 0
    for x in f:
        # Keep track of how many lines to skip for reading in the pulse table
        line_cnt += 1
        # Parse line by line (don't care about ordering except that Control 
        # pulses is the last line)
        if "Control pulses:" in x:
            break
        elif "Name" in x or "name" in x:
            # Removing any leading/ending white space
            pulse_name = x.split(":")[1].strip()
        elif "type:" in x:
            # Remove any leading/ending white space and convert to lowercase
            pulse_type = x.split(":")[1].strip().lower()
        elif "length" in x:
            pulse_length = int(x.split(":")[1].strip())
            
    # Initialize the control pulse object
    ctrl_pulse = ControlPulse(pulse_name, pulse_type, pulse_length)
    
    # Read the rest of the pulse file to get the pulse information
    df = pd.read_csv(f_name, skiprows = line_cnt)
    
    # Loop over each column in the pulse table and add control variables to 
    # control pulse object
    ctrl_var_names = df.columns
    for ctrl_var in ctrl_var_names:
        ctrl_pulse.add_control_variable(ctrl_var, df[ctrl_var])
    
    return ctrl_pulse

