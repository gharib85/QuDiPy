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

rho0 = np.kron(np.array([[1,0],[0,0]]) , np.array([[0,0],[0,1]]))

paramdict={"B_0" : 1, "T_2" : 1e-4, "T" : 4, "f_rf"  : 2.799249E10}
    #such values are chosen to achieve resonance

rabi_osc = sps.SpinSys(rho0, sys_param_dict=paramdict)

lisbx = [2e-3] * 1500 #list of Bx

#lisdg1 = [1e-2] * 500 + [0] * 500 + [1e-2] * 500   #delta g 1
#lisdg2 = [0] * 500 + [1e-2] * 500 + [0] * 500      #delta g 2
lisj = [1.6e-21]*1500

pulseparams = {"B_x": np.array(lisbx), 
             #  "delta_g_1": np.array(lisdg1), "delta_g_2": np.array(lisdg2),
               "J_0": np.array(lisj),
               "pulse_time": 1e-7}


evol = rabi_osc.evolve({"onlypulse" : pulseparams}, is_purity=True, 
                       track_qubits=(1,2), are_Bloch_vectors=True, 
                       track_points_per_pulse=1000)


#plt.plot(evol["track_time"], np.real(evol["sigma_y_1"]) )

plt.plot(evol["track_time"], np.real(evol["sigma_z_1"]) )


plt.plot(evol["track_time"], np.real(evol["sigma_z_2"]) )

plt.show()
    # Spyder console shows a list of warnings
    
print(rabi_osc.rho)