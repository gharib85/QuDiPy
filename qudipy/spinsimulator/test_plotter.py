# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:45:00 2020

@author: bkhromet
"""
#Run the script from QuDiPy folder!


import numpy as np
import qudipy.spinsimulator as sps
import math

import matplotlib.pyplot as plt

rho0 = np.array([[1,0],[0,0]])

paramdict={"B_0" : 1, "T_1" : 1e-5, "T" : 77, "f_rf"  : 2.799249E10}
    #such values are chosen to achieve resonance

rabi_osc = sps.SpinSys(rho0, sys_param_dict=paramdict)

lis = [1e-2]*10000
pulseparams = {"B_x": np.array(lis), "pulse_time": 3e-8}


evol = rabi_osc.evolve({"onlypulse" : pulseparams}, is_purity=True, 
                       track_qubits=1, are_Bloch_vectors=True, track_points_per_pulse=1000)


plt.plot(evol["track_time"], np.real(evol["sigma_y_1"]) )

plt.plot(evol["track_time"], np.real(evol["sigma_z_1"]) )

plt.show()
    # Spyder console shows a list of warnings
    
print(rabi_osc.rho)