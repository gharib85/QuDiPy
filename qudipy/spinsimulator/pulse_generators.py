# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:20:47 2020

The script contains functions that generate simple constant pulses 
in spin space.

@author: bkhromet
"""
#For data manipulation
import numpy as np

import math

#Spin simulator module
#import qudipy.spinsimulator.spin_simulator as sps

#Circuit module containing control pulses and ideal circuits
from qudipy.circuit import ControlPulse
#Constants class
from qudipy.utils.constants import Constants

#material system is chosen to be GaAs by default because such parameters as 
#effective mass or dielectric constant do not matter for spin simulations;
#this could be changed later if needed
cst = Constants("GaAs")

def rot(qubits, axis, theta, sys, B_rf, delta_g=0., num_val=300):
    """
    Function that creates a ROTX, ROTY or ROTZ pulse for a given RF field
    
    Parameters:
        qubits: int / iterable of ints
            qubit(s) to be exposed to the rotation pulse
        axis: string
            specifies the axis of rotation: "X", "Y" or "Z"
        theta: float
            specifies the angle of rotation (in degrees)
        sys: SpinSys object
            the system at which the pulse will be applied
        B_rf: float
            constant ESR field magnitude during the pulse
        num_val: int
            number of data points
        delta_g: float
            approximate value of deviation g-factor used to cancel the effect 
            of global RF field on the unspecified qubits
            
    Returns: rotpulse
        ControlPulse object / tuple of such objects corresponding to 
        a given rotation
    """
    #if axis=="X":
           
    if axis=="X" or axis=="Y":
        phis = [0.]*num_val
        if axis=="Y":
            phis = [math.pi/2]*num_val
        if theta<0:
            phis = [(phi + math.pi) for phi in phis]
        bs = [B_rf]*num_val
        
        si_pulse_length = abs((theta*math.pi/180) * 
                    cst.hbar / (2*cst.muB*B_rf))
        
        rotpulse = ControlPulse("ROT{}_{}".format(axis, theta), 
                    "effective", pulse_length= si_pulse_length *1e12)
        rotpulse.add_control_variable("phi", np.array(phis))
        rotpulse.add_control_variable("B_rf", np.array(bs))
        
        # adding deviation g-factors to make unused qubits 
        # effectively "idle" under the pulse
        
        ifint = isinstance(qubits, int)    #track a single qubit
        ifiterable = (isinstance(qubits, (tuple,list, set))
              and math.prod(isinstance(val, int) for val in qubits))
               
        if ifiterable: 
            set_qubits = set(qubits) 
        elif ifint:
            set_qubits = {qubits}
        else:
            raise ValueError("The tracked qubits should be properly specified"  
                             "by an int or an iterable of ints. None of the"  
                                 "qubits has been detuned to idle")
        
        #tuning the target qubit(s) on resonance
        omega = 2 * cst.muB * sys.B_0 / cst.hbar
        Omega = 2 * cst.muB * B_rf / cst.hbar
        
        if sys.B_0 != 0:
            dg0 = 2.0*(2*math.pi*sys.f_rf/omega -1)
            
            for qub in set_qubits:
                rotpulse.add_control_variable("delta_g_{}".format(qub), 
                                             np.array(([dg0] * num_val)))         
        
        N = int(math.log2(sys.rho.shape[0])) 
        idle_qubs = set(range(1, N +1)) - set_qubits
        # calculating the exact value of delta_g, see the write-up for the 
        # derivation

        #number of full rotations on the Bloch sphere for the idling qubit
        nrot = int(math.sqrt((omega * (1+0.5 * delta_g)-2 * math.pi * 
                              sys.f_rf)**2 + Omega ** 2) * 
                               si_pulse_length / (2*math.pi) ) + 1
        
        dg1 = (math.sqrt((2 * math.pi * nrot/si_pulse_length)**2 
                   - Omega**2) - omega +2 * math.pi *sys.f_rf ) * 2 / omega
        
        dg2 =  (-math.sqrt((2 * math.pi * nrot/si_pulse_length)**2
                   - Omega**2) - omega +2 * math.pi *sys.f_rf ) * 2 / omega
        
        #choosing the closest value
        exact_delta_g = dg1 if abs(dg1-delta_g) < abs(dg2-delta_g) else dg2
            
        for qub in idle_qubs:
            rotpulse.add_control_variable("delta_g_{}".format(qub), 
                                          np.array(([exact_delta_g] * num_val)))
                
        return rotpulse
        #del rotpulse
    elif axis=="Z":
        return [rot(qubits, "X", -90, sys, B_rf, delta_g, num_val), 
                rot(qubits,"Y", theta, sys, B_rf, delta_g, num_val), 
                rot(qubits,"X", 90, sys, B_rf, delta_g, num_val)]
    else:
        raise ValueError("Incorrect input of axis, please try again")
        return 0
    
def swap(qubit, J, sys, num_val=300):
    """
    Function that builds a simple SWAP gate
    
    Parameters:
        qubit: int
            number of the left qubit in a pair
        J: float
            exchange between qubit and qubit+1
        sys: SpinSys object
            the system at which the pulse will be applied
        num_val: int
            number of data points
            
    Returns: swappulse
        ControlPulse object corresponding to the SWAP between two 
        neighboring qubits
    """
    Js = [J]*num_val
    swappulse = ControlPulse("SWAP_{}_{}".format(qubit, qubit+1), 
                                "effective", pulse_length = cst.h/(2*J) * 1e12) 
    swappulse.add_control_variable("J_{}".format(qubit),
                                                       np.array(Js))
    #tuning all qubits on resonance
    N = int(math.log2(sys.rho.shape[0]))
    if sys.B_0 != 0:
        omega = 2 * cst.muB * sys.B_0 / cst.hbar
        dg0 = 2.0*(2*math.pi*sys.f_rf/omega -1)
        for qub in range(1,N+1):
            swappulse.add_control_variable("delta_g_{}".format(qub), 
                                         np.array(([dg0] * num_val)))
    
    return swappulse

def rswap(qubit, J, sys, num_val=100):
    """
    Function that builds a simple RSWAP gate
    Parameters:
        qubit: int
            number of the left qubit in a pair
        J: float
            exchange between qubit and qubit+1
        num_val: int
            number of data points
    Returns: swappulse
        ControlPulse object corresponding to the SWAP between
        two neighboring qubits
    """
    Js = [J]*num_val
    rswappulse = ControlPulse("RSWAP_{}_{}".format(qubit, qubit+1), 
                                "effective", pulse_length = cst.h/(4*J) *1e12) 
    rswappulse.add_control_variable("J_{}".format(qubit), 
                                                        np.array(Js))
    #tuning all qubits on resonance
    N = int(math.log2(sys.rho.shape[0])) 
    if sys.B_0 != 0:
        omega = 2 * cst.muB * sys.B_0 / cst.hbar
        dg0 = 2.0*(2*math.pi*sys.f_rf/omega -1)
        for qub in range(1,N+1):
            rswappulse.add_control_variable("delta_g_{}".format(qub), 
                                         np.array(([dg0] * num_val)))
    return rswappulse

