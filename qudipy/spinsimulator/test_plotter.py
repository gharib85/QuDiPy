# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:45:00 2020

@author: bkhromet
"""
#Run the script from QuDiPy folder!


import numpy as np
import qudipy.spinsimulator as sps
import math

import qudipy.circuit as circ
import matplotlib.pyplot as plt

from qudipy.utils.constants import Constants 
#material system is chosen to be GaAs by default because such parameters as 
#effective mass or dielectric constant do not matter for spin simulations

cst = Constants("GaAs")

rho0 = np.kron(np.array([[1,0],[0,0]], dtype=complex) , 
               np.array([[0,0],[0,1]], dtype=complex))



paramdict={"B_0" : 1, "T_2" : 1e-4, "T" : 4, "f_rf"  : 2.799249E10}
    #such values are chosen to achieve resonance

rabi_osc = sps.SpinSys(rho0, sys_param_dict=paramdict)

lisbx = [2e-3] * 1500 #list of Bx

#lisdg1 = [1e-2] * 500 + [0] * 500 + [1e-2] * 500   #delta g 1
#lisdg2 = [0] * 500 + [1e-2] * 500 + [0] * 500      #delta g 2
lisj = [1.6e-25]*1500  

#testing pulse object
pulse_obj = circ.ControlPulse("Rabi_test_pulse", "effective", 
                              pulse_length=1e5) #(in picoseconds)


pulseparams = {#"B_x": np.array(lisbx), 
              # "delta_g_1": np.array(lisdg1), "delta_g_2": np.array(lisdg2),
               "J_1": np.array(lisj),
             #  "pulse_time": 1e-7
               }



for var in pulseparams:
    pulse_obj.add_control_variable(var, pulseparams[var])

#evol = rabi_osc.evolve(pulse_obj, is_purity=True,  
#                       track_qubits=(1,2), are_Bloch_vectors=True, 
#                       track_points_per_pulse= 1500)

#plt.plot(evol["time"], np.real(evol["sigma_y_1"]) )

#plt.plot(evol["time"], np.real(evol["sigma_z_1"]) ,"m-")

#plt.plot(evol["time"], np.real(evol["sigma_z_2"]))

#plt.plot(evol["time"], np.real(evol["purity"]), "go" )


#plt.show()
    # Spyder console shows a list of warnings
    
#print(evol)

rho3 = np.kron(np.kron(np.array([[1,0],[0,0]], dtype=complex) , 
               np.array([[0,0],[0,1]], dtype=complex)), 
               np.array([[1,0],[0,0]], dtype=complex)
               )
#rho3 = np.kron(np.array([[0,0,0,0],[0,0.5,0,0], [0,0,0.5,0], [0,0,0,0]], dtype=complex) , 
#               np.array([[0,0],[0,1]], dtype=complex)), 
#               np.array([[1,0],[0,0]], dtype=complex)
#               )


param3dict={"B_0" : 1, "T_2" : 1e-4, "T" : 4, "f_rf"  : 2.799249E10}
    #such values are chosen to achieve resonance

lisbx3 = [1e-3] * 1000

lisj31 = [1.6e-25]*500   + [0] * 500
lisj32 = [0] * 500 + [1.6e-25]*500  

pulse3params = {#"B_x":lisbx3,
                "J_1": np.array(lisj31), "J_2":np.array(lisj32)}


exch = sps.SpinSys(rho3, sys_param_dict=param3dict)

exch_pulse_obj = circ.ControlPulse("3_qubit_exchange_pulse", "effective", 
                              pulse_length=2e4) #(in picoseconds)

for var in pulse3params:
    exch_pulse_obj.add_control_variable(var, pulse3params[var])

exch_evol = exch.evolve(exch_pulse_obj, is_fidelity=True, 
                        rho_reference = rho3,
                        track_qubits = (1,2,3), are_Bloch_vectors=True)

plt.plot(exch_evol["time"], np.real(exch_evol["sigma_z_1"]) )

#plt.plot(exch_evol["time"], np.real(exch_evol["sigma_z_2"]) ,"m-")

#plt.plot(exch_evol["time"], np.real(exch_evol["sigma_z_3"]))

plt.plot(exch_evol["time"], np.real(exch_evol["fidelity"]) , "r-")

plt.grid()
plt.rc('grid', linestyle="--", color='gray')
plt.show()


#testing echo pulses


