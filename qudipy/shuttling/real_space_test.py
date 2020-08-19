# import os, sys
# sys.path.append("/Users/keweizhou/Google_Drive/Research/20summer/Waterloo/QuDiPy")
# import qudipy as qd
# import numpy as np
# import qudipy.potential as pot
# import qudipy.qutils as qt
# from numpy.fft import fft, ifft, fftshift, ifftshift
# from scipy import sparse
# from scipy.sparse import diags
# from scipy.linalg import expm
# import matplotlib.pyplot as plt
# import timeit

# ############################################
# ########## Potential Interpolator ##########
# ############################################

# # Load preprocessed potential files from the potentail folder
# pot_dir = '/Users/keweizhou/Google_Drive/Research/20summer/Waterloo/QuDiPy/qudipy/potential/Sliced_potentials/'
    
# # Specify the control voltage names (C#NAME as mentioned above)
# ctrl_names = ['V1','V2','V3','V4','V5']

# # Specify the control voltage values you wish to load.
# # The cartesian product of all these supplied voltages will be loaded and MUST exist in the directory.
# V1 = [0.1]
# V2 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
# V3 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
# V4 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
# V5 = [0.1]
# # Add all voltage values to a list
# ctrl_vals = [V1, V2, V3, V4, V5]    

# # Now load the potentials.  
# # load_files returns a dictionary of all the information loaded
# loaded_data = qd.potential.load_potentials(ctrl_vals, ctrl_names, f_type='pot', 
#                               f_dir=pot_dir, f_pot_units="eV", 
#                               f_dis_units="nm")

# # Now building the interpolator object is trivial
# pot_interp = qd.potential.build_interpolator(loaded_data, 
#                                              constants=qd.Constants("Si/SiO2"),y_slice= 0)


# ######################################
# ########### Shuttling Pulse ##########
# ######################################

# # Build up a pulse
# min_v = 0.2
# max_v = 0.278

# pt1 = [max_v, min_v, min_v]
# # pot_interp.plot(pt1, plot_type='1D', show_wf=True)

# vv = pot_interp.find_resonant_tc(pt1, 1)
# most_vv = vv - 0.035 * (max_v - min_v)
# pt2 = pt1.copy()
# pt2[1] = most_vv
# pot_interp.plot(pt2, show_wf=True)

# pt3 = pt2.copy()
# pt3[1] = vv
# # pot_interp.plot(pt3, plot_type='1D', show_wf=True)

# pt4 = pt3.copy()
# pt4[0] = most_vv
# # pot_interp.plot(pt4, plot_type='1D', show_wf=True)

# pt5 = pt4.copy()
# pt5[0] = min_v
# # pot_interp.plot(pt5, plot_type='1D', show_wf=True)

# vv = pot_interp.find_resonant_tc(pt5, 2)
# most_vv = vv - 0.035 * (max_v - min_v)
# pt6 = pt5.copy()
# pt6[2] = most_vv

# pt7 = pt6.copy()
# pt7[2] = vv

# pt8 = pt7.copy()
# pt8[1] = most_vv

# pt9 = pt8.copy()
# pt9[1] = min_v

# shuttle_pulse = np.array([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9])

# shut_pulse = qd.circuit.ControlPulse('shuttle_test', 'experimental', 
#                                      pulse_length=10)
# shut_pulse.add_control_variable('V2',shuttle_pulse[:,0])
# shut_pulse.add_control_variable('V3',shuttle_pulse[:,1])
# shut_pulse.add_control_variable('V4',shuttle_pulse[:,2])
# ctrl_time_L = 10.0 * np.array([0, 1/20, 1/4, 1/2-1/20, 1/2, 1/2+1/20, 3/4, 19/20, 1])
# shut_pulse.add_control_variable('time',ctrl_time_L)


# shut_pulse.plot()
