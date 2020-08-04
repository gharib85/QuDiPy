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

# rho0 = 1/2*np.array([[1,1],[1,1]])
rho0 = np.array([[1,0],[0,0]])

paramdict={"B_0" : 1, "T_1" : 1E-7, "T" : 77, "f_rf" : 2.7992E10}
    #such values are chosen to achieve resonance

rabi_osc = sps.SpinSys(rho0, sys_param_dict=paramdict)

lis = [1e-2]*10000
# lis = [0]*3
pulseparams = {"B_x": np.array(lis), "pulse_time": 100e-9}


evol = rabi_osc.evolve({"onlypulse" : pulseparams}, is_purity=True, 
                       track_qubits=1, are_Bloch_vectors=True)


plt.plot(evol["track_time"], np.real(evol["sigma_z_1"] ))

plt.show()
    # Spyder console shows a list of warnings
    
print(rabi_osc.rho)

# Known issues...
# 1. delta_t should not be dictated by number of values in the pulse. Should
# either be a fixed constant value OR change with the smallest energy spacing
# of the Hamiltonian during the evolution.
# 2. Should be a flag in the self.linbladian method to easily turn off 
# disapative terms (in general.. things should be easily toggable by a flag)
# 3. PEP 8 needs to be followed.
# 4. Change partial_trace to partial_trace_qubit (or something equivalent) and
# add back general program so both are available.  
# 5. Runge-Kutta is wrong (missing coefficients of 2 in front of K2 and K3)
# 6. Construction of Linbladian missing 1/hbar coefficient of unital term (or
# is that taken care of because you normalize by hbar in construction of matrices)
# 7. Change numpy.dot to be @ as it's more readable
# 8. Why the 1j in the z_sum_omega term? Makes diagonal elements complex for H