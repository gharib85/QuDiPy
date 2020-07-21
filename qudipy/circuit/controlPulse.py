"""
Class for a control pulse
"""

import numpy as np

class ControlPulse:
    
    def __init__(self, pulse_name, pulse_type, pulse_length=0, ideal_gate=None):
        '''
        Initialize a ControlPulse object.

        Parameters
        ----------
        pulse_name : string
            Name of pulse.
        pulse_type : string
            Specify whether pulse is described with "experimental" or 
            "effective" control variables.
        pulse_length : int, optional
            Total length of pulse in ps. The default is 0.
        ideal_gate : string, optional
            The ideal gate keyword for this control pulse. The default is None.

        Returns
        -------
        None.

        '''
        
        self.pulse_type = pulse_type
        
        self.name = pulse_name
        self.length = pulse_length # Units are ps
        
        self.ctrl_pulses = {
            }
        self.ctrl_names = []
        self.n_ctrls = 0
        
        self.ideal_gate = ideal_gate
        
    def set_pulse_length(self, pulse_length):
        '''
        Change the pulse length parameter

        Parameters
        ----------
        pulse_length : int
            Pulse legnth for control pulse in ps.

        Returns
        -------
        None.

        '''
        
        self.length = pulse_length
        
    def add_control_variable(self, var_name, var_pulse):
        '''
        Adds a control variable to the pulse.
        
        Parameters
        ----------
        var_name : string
            Name of new control variable being added.
        var_pulse : 1D array
            DESCRIPTION.

        Returns
        -------
        None.

        '''
            
        self.ctrl_pulses[var_name] = np.array(var_pulse)
        self.ctrl_names = list(self.ctrl_pulses.keys())
        self.n_ctrls = len(self.ctrl_names)
        
        
        
        
        
        
        
        