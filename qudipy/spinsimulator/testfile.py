# -*- coding: utf-8 -*-
"""
One of the testing files 

A bit messy but still could be helpful

Run from QuDiPy directory!
@author: bkhromet
"""

import numpy as np
import qudipy.spinsimulator as sps
import math

#matrix is random, trace is unity
rho0=np.array([[0.20678022, 0.30122333, 0.20440789, 0.11719614],
       [0.14699382, 0.26453817, 0.34076206, 0.12694217],
       [0.33011879, 0.11725161, 0.36363945, 0.26628559],
       [0.32509702, 0.19180277, 0.31644896, 0.16504217]])


print("rho0\n", rho0)
testpulse={"B_x":[0.01]*100}

paramdict={"B_0":1, "T_1":1e-1}

rabi = sps.SpinSys(rho0, sys_param_dict=paramdict)

deltagdict={"delta_g_1":0.1, "delta_g_2":0.2 }

#print("\n\nHamiltonian\n\n", 
#      rabi.hamiltonian(rabi.cdict[-1], pulse_params=deltagdict),
#      "\n\nLindbladian\n\n",
#      rabi.lindbladian(rabi.rho, rabi.cdict[-1], pulse_params=deltagdict)
#      )

# testing evolution

gpulse={"delta_g_1":(0.001,0.002,0.001), "delta_g_2":(0.002,-0.001,0.003), "pulse_time":3e-12 }

gpulse2={"delta_g_1":(0.001,0.002,0.001), "delta_g_2":(0.002,-0.001,0.0063), "pulse_time":6e-12 }

#evol = rabi.evolve({"noname": gpulse, "anothername": gpulse2, "thirdrname": gpulse}, 
 #                  is_purity=True, track_qubits=(1,), are_Bloch_vectors=True)

#print(np.trace(rabi.rho), "\n evol \n", evol, "\n", gpulse2, gpulse[list(gpulse.keys())[0]])


evol2 = rabi.evolve({"newname":gpulse}, track_qubits=1, are_Bloch_vectors=True)

print(evol2, int(math.log2(rabi.rho.shape[0])) ,
                  rabi.track_subsystem(1, False))



