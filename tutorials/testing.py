import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import qudipy as qd


# TODO: find_res_tc
# How to improve sweep if bnd window is too large?
# How to deal with noise? (only take highest value peak maybe?)
# Needs to be generalized to deal with 2D peaks
# Change find_res_tc to have swept_ctrl be index for the ctrl_values
# How to improve swept_ctrl (make more robust and maybe take in ctrl names as well)
# Update loading tutorial to use plot methods.
# Add docstring

# TODO: PotentialInterpolator plot
# Add overlay wavefunction

# TODO: ControlPulse 
# Add stuff to keep track of time control value
# Add method to get control values at arbitrary time

# TODO : tutorial (loading potentials)
# Add that Constants needs to be passed when building interpolator

pot_dir = '/Users/simba/Documents/GitHub/Silicon-Modelling/tutorials/QuDiPy tutorial data/Pre-processed potentials/'
    
# Specify the control voltage names (C#NAME as mentioned above)
ctrl_names = ['V1','V2','V3','V4','V5']

# Specify the control voltage values you wish to load.
# The cartesian product of all these supplied voltages will be loaded and MUST exist in the directory.
V1 = [0.1]
V2 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28, 0.29]
V3 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28, 0.29]
V4 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28, 0.29]
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
v_vec = [0.28,0.26,0.27]
# pot_interp.plot(v_vec)

v2 = pot_interp.find_resonant_tc(v_vec,1,[0.261,0.262], show_wf=True)
pot_interp.plot([0.28,v2,0.27],plot_type='1D')
# pot_interp.find_resonant_tc(v_vec,1,[0.258,0.26])
# pot_interp.find_resonant_tc(v_vec,1,[0.263,0.265])
v2 = pot_interp.find_resonant_tc(v_vec,1,[0.2617,0.265], show_wf=True)
pot_interp.plot([0.28,0.2617,0.27],plot_type='1D', show_wf=True)
pot_interp.plot([0.28,0.265,0.27],plot_type='1D', show_wf=True)
pot_interp.plot([0.28,v2,0.27],plot_type='1D')
# pot_interp.find_resonant_tc(v_vec,1,[0.26,0.2615])

    



