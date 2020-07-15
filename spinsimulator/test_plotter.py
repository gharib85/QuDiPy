# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:45:00 2020

@author: bkhromet
"""

import numpy as np
import spin_simulator as sps
import math

import matplotlib.pyplot as plt

rho0 = np.array([[1,0],[0,0]])

paramdict={"B_0" : 1, "T_1" : 1e-1, "T" : 77, "f_rf"  : 2.799249E9}
    #such values are chosen to achieve resonance

rabi_osc = sps.SpinSys(rho0, sys_param_dict=paramdict)


pulseparams = {"B_x": np.array([1e-3]*10000), "pulse_time": 3E-6}


evol = rabi_osc.evolve({"onlypulse" : pulseparams}, is_purity=True, 
                       track_qubits=1, are_Bloch_vectors=True)


plt.plot(evol["track_time"], evol["sigma_x_1"])

plt.show()
    # Spyder console shows a list of warnings
    