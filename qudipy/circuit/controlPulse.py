"""
Class for a control pulse

@author: simba
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
            This is in optional keyword because 
            sometimes you may wish to vary the pulse length across different
            simulations for a given pulse. Pulse length can be set later using
            the set_pulse_length() method.
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
        
        self.ctrl_time = None
        
    def __call__(self, time_pts):
                
        '''
        Call method for class.

        Parameters
        ----------
        time_pts : 1D float array
            Time points where we want to obtain the interpolated pulse values.

        Returns
        -------
        interp_pulse : 2D float array
            The interpolated pulse values at the inputted time points.

        '''
        
        # Check if the interpolators have been constructed. If not, then 
        # make them.
        if not hasattr(self,'ctrl_interps'):
            self._generate_ctrl_interpolators()
        
        # Loop through each control variable and get the interpolated pulse
        # for each time point.
        interp_pulse = np.zeros((len(time_pts),len(self.ctrl_names)))
        for ctrl_idx, ctrl in enumerate(self.ctrl_names):
            interp_pulse[:,ctrl_idx] = self.ctrl_interps[ctrl](time_pts)

        return interp_pulse    
    
    def plot(self, plot_ctrls='all', time_interval='full', n=250):
        '''
        Plot the control pulse. Can plot a subset of the control variables 
        and within some time interval.

        Parameters
        ----------
        plot_ctrls : list of strings, optional
            Specify the name of each control variable pulse you wish to plot
            or plot 'all'. The default is 'all'.
        time_interval : list of floats
            Specify the time interval over which to plot the pulse. Cannot be
            less than 0 or greater than the current pulse length. The default 
            is the full time interval.
        n : int, optional
            Number of points to use in the pulse when plotting. The default is
            250.

        Returns
        -------
        None.

        '''
        
        # Get the time interval and time points to plot
        if time_interval == 'full':
            min_time = 0
            max_time = self.length
        else:
            min_time = time_interval[0]
            max_time = time_interval[1]
            
        t_pts = np.linspace(min_time,max_time,n)
        
        # Get the actual pulse
        pulse = self(t_pts)
        
        # Check the plot_ctrls input
        if not isinstance(plot_ctrls,(list,tuple,set)):
            if plot_ctrls.lower() == 'all':
                plot_ctrls = self.ctrl_names
            # A single control variable was inputted but wasn't wrapped in a
            # list, tuple, or set, so we will wrap it in a list
            elif plot_ctrls in self.ctrl_names:
                plot_ctrls = [plot_ctrls]
            else:
                raise ValueError('Unrecognized input for control variables '+
                                 f'to plot {plot_ctrls}.\nAllowed inputs are '+
                                 'either ''all'' or a list of allowed names:\n'+
                                 f' {self.ctrl_names}')
        # If a list of names was specified, check that all are valid ctrl 
        # variable names
        elif not set(plot_ctrls).issubset(self.ctrl_names):
            raise ValueError('Unrecognized input for control variables '+
                             f'to plot {plot_ctrls}.\nAllowed inputs are '+
                             'either ''all'' or a list of allowed names:\n'+
                             f'{self.ctrl_names}')
           
        # For each pulse specified by ctrl_vars, plot those pulses
        ctrl_idxs = []
        for ctrl in plot_ctrls:
            ctrl_idxs.append(self.ctrl_names.index(ctrl))
            
        # Generate fig
        plt.figure()
        plt.plot(t_pts, pulse[:,ctrl_idxs])   
        lgd_names = [self.ctrl_names[idx] for idx in ctrl_idxs]
        plt.legend(lgd_names,loc='best')
        plt.xlabel('Time')
        plt.show()
        
        
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
        Adds a control variable to the pulse. If the variable name is time,
        then it will store the time points in the ctrl_time variable.
        
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
        
        if var_name.lower() == 'time':
            self.ctrl_time = np.array(var_pulse)
            self.set_pulse_length(var_pulse[-1])
        else:
            self.ctrl_pulses[var_name] = np.array(var_pulse)
            # self.ctrl_pulses.keys() should be in the order items are added 
            # to the dict... But just to make sure we will manually append 
            # our own list.
            self.ctrl_names.append(var_name)
            self.n_ctrls = len(self.ctrl_names)
            
    def _generate_ctrl_interpolators(self):
        '''
        Loop through every control variable and make a 1D time interpolation 
        object.

        Returns
        -------
        None.

        '''
        
        # Initialize an empty dictionary to store all the interpolators
        self.ctrl_interps = {}
        
        # Check if the time array is set, if not, then assume each point in
        # the pulse is linearly spaced in time
        linear_time_step = False
        if self.ctrl_time is None:
            linear_time_step = True
        
        # For each pulse, find the time axis points and then make the 1D
        # interpolator. All ctrl pulses will be linearly interpolated.
        for ctrl in self.ctrl_names:
            curr_pulse = self.ctrl_pulses[ctrl]
            
            if linear_time_step:
                curr_time = np.linspace(0,self.length, len(curr_pulse))
            else:
                curr_time = self.ctrl_time
                
            self.ctrl_interps[ctrl] = interp1d(curr_time, curr_pulse)        
            
        
        
        
        
        
        
        
        
        
        