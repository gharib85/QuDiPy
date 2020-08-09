import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import qudipy as qd
import numpy as np

# TODO: potentialInterpolator
# Absolutely needs to be generalized to interpolant multiple voltage vectors 
# at one time (HIGH PRIORITY)
# Need to add ability to be a 1D or 2D interpolator (shuttling is slow with 2D)

# TODO: find_res_tc
# Needs to be generalized to deal with 2D peaks (can probably be done by taking
# derivative and removing all points with a small derivative (i.e. = 0) AND
# small regular amplitude in probabilty. From there you could cluster the
# remaining peaks and do peak finding by cluster maxima. Problem then becomes
# how do we find characteristic widths of peaks?
# Add docstring

# TODO: ControlPulse 
# .plot(): should we split it into multiple subplots? For variables with different units
# plotting on same window doesn't make sense.
# Check with add_control_variable that new pulses have same number of points 
# previous pulses
# Do we want default units to be s?
# Change __call__ to return a data frame with ctrl_names as column headers 
# instead of just a 2D numpy array

# TODO: General
# Check out dataclass and Enum class to see if usable somewhere
# Look for places to add filter and map methods (maybe reduce?)
# Move _check stuff into their own methods (single purpose programming)

# Sketch out how we could make an faux exchange library.

pot_dir = '/Users/simba/Documents/GitHub/Silicon-Modelling/tutorials/QuDiPy tutorial data/Pre-processed potentials/'
    
# Specify the control voltage names (C#NAME as mentioned above)
ctrl_names = ['V1','V2','V3','V4','V5']

# Specify the control voltage values you wish to load.
# The cartesian product of all these supplied voltages will be loaded and MUST exist in the directory.
V1 = [0.1]
V2 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
V3 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
V4 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
V5 = [0.1]
# Add all voltage values to a list
ctrl_vals = [V1, V2, V3, V4, V5]    

# Now load the potentials.  
# load_files returns a dictionary of all the information loaded
loaded_data = qd.potential.load_potentials(ctrl_vals, ctrl_names, f_type='pot', 
                              f_dir=pot_dir, f_pot_units="eV", 
                              f_dis_units="nm")

# Now building the interpolator object is trivial
pot_interp = qd.potential.build_interpolator(loaded_data, 
                                             constants=qd.Constants("Si/SiO2"))
    
# RIGHT ANSWER
# v_vec = [0.28,0.2616,0.27]
# v_vec = [0.28,0.26,0.27]
# pot_interp.plot(v_vec)

# v2 = pot_interp.find_resonant_tc(v_vec, 1, [0.2614,0.262])

# v2 = pot_interp.find_resonant_tc(v_vec, 'V3')
# print(v2)
# v2 = pot_interp.find_resonant_tc([0.1,0.28,0.26,0.27,0.1], 'V3', [0.261,0.262])
# print(v2)
# Checking ctrl_sweep inputs
# v2 = pot_interp.find_resonant_tc([0.1,0.28,0.26,0.27,0.1], 'VV3', [0.261,0.262])
# v2 = pot_interp.find_resonant_tc([0.1,0.28,0.26,0.27,0.1], 6, [0.261,0.262])
# v2 = pot_interp.find_resonant_tc([0.28,0.26,0.27], 3, [0.261,0.262])
# Checking ranges
# pot_interp.find_resonant_tc(v_vec,1,[0.258,0.26])
# pot_interp.find_resonant_tc(v_vec,1,[0.263,0.265])
# pot_interp.find_resonant_tc(v_vec,1,[0.2617,0.265])
# pot_interp.find_resonant_tc(v_vec,1,[0.26,0.2615])

# Build up a pulse
min_v = 0.2
max_v = 0.278
pt1 = [max_v, min_v, min_v]
pot_interp.plot(pt1, plot_type='1D', show_wf=True)

vv = pot_interp.find_resonant_tc(pt1, 1)
pt2 = pt1.copy()
pt2[1] = vv
pot_interp.plot(pt2, plot_type='1D', show_wf=True)

pt3 = pt2.copy()
pt3[0] = min_v
pot_interp.plot(pt3, plot_type='1D', show_wf=True)

vv = pot_interp.find_resonant_tc(pt3, 2)
pt4 = pt3.copy()
pt4[2] = vv
pot_interp.plot(pt4, plot_type='1D', show_wf=True)

pt5 = pt4.copy()
pt5[1] = min_v
pot_interp.plot(pt5, plot_type='1D', show_wf=True)

shuttle_pulse = np.array([pt1, pt2, pt3, pt4, pt5])

shut_pulse = qd.circuit.ControlPulse('shuttle_test', 'experimental', 
                                     pulse_length=10)
shut_pulse.add_control_variable('V2',shuttle_pulse[:,0])
shut_pulse.add_control_variable('V3',shuttle_pulse[:,1])
shut_pulse.add_control_variable('V4',shuttle_pulse[:,2])
t_pts = [1,2,3,4,5]
int_p = shut_pulse(t_pts)

# for t_idx in range(len(t_pts)):
#     print(pot_interp(int_p[t_idx,:]))
#     print(pot_interp(int_p[t_idx,:]).shape)


    
