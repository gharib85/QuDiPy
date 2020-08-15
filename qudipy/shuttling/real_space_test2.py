import os, sys
sys.path.append("/Users/keweizhou/Google_Drive/Research/20summer/Waterloo/QuDiPy")
import qudipy as qd
import numpy as np
import qudipy.potential as pot
import qudipy.qutils as qt
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy import sparse
from scipy.sparse import diags
from scipy.linalg import expm
import matplotlib.pyplot as plt
import timeit

############################################
########## Potential Interpolator ##########
############################################

# Load preprocessed potential files from the potentail folder
pot_dir = '/Users/keweizhou/Google_Drive/Research/20summer/Waterloo/QuDiPy/qudipy/potential/Sliced_potentials/'
    
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
                                             constants=qd.Constants("Si/SiO2"),y_slice= 0)


######################################
########### Shuttling Pulse ##########
######################################

# Build up a pulse
min_v = 0.2
max_v = 0.278
pt1 = [max_v, min_v, min_v]
# pot_interp.plot(pt1, plot_type='1D', show_wf=True)

vv = pot_interp.find_resonant_tc(pt1, 1)
pt2 = pt1.copy()
pt2[1] = vv
# pot_interp.plot(pt2, plot_type='1D', show_wf=True)

pt3 = pt2.copy()
pt3[0] = min_v
# pot_interp.plot(pt3, plot_type='1D', show_wf=True)

vv = pot_interp.find_resonant_tc(pt3, 2)
pt4 = pt3.copy()
pt4[2] = vv
# pot_interp.plot(pt4, plot_type='1D', show_wf=True)

pt5 = pt4.copy()
pt5[1] = min_v
# pot_interp.plot(pt5, plot_type='1D', show_wf=True)

shuttle_pulse = np.array([pt1, pt2, pt3, pt4, pt5])

shut_pulse = qd.circuit.ControlPulse('shuttle_test', 'experimental', 
                                     pulse_length=10)
shut_pulse.add_control_variable('V2',shuttle_pulse[:,0])
shut_pulse.add_control_variable('V3',shuttle_pulse[:,1])
shut_pulse.add_control_variable('V4',shuttle_pulse[:,2])


############################################
########## Wavefunction Evolution ##########
############################################

########## Initialize wavefunction ##########

# Find the initial potential
init_pulse = shut_pulse([0])[0]
init_pot = pot_interp(init_pulse)
# Initialize the constants class with the Si/SiO2 material system 
consts = qd.Constants("Si/SiO2")
# First define the x-coordinates
X = loaded_data['coords'].x
# Create a GridParameters object of the initial potential
gparams = pot.GridParameters(X, potential=init_pot)

# Find the initial ground state wavefunction
e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
psi_x = e_vecs[:,0]
prob = [abs(x)**2 for x in psi_x]
ymax = 2* max(prob)              # TODO: delete
# print(psi_x)

########## Constants necessary for the computation ##########

# time step
dt = 5E-16
# indices of grid points
I = [(idx-gparams.nx/2) for idx in range(gparams.nx)]   
# vector of momentum grid
P = np.asarray([2 * consts.pi * consts.hbar * i / (gparams.nx*gparams.dx) for i in I])

# exponents present in evolution
exp_K = np.exp(-1j*dt/2*np.multiply(P,P)/(2*consts.me*consts.hbar))
exp_KK = np.multiply(exp_K,exp_K)


########## Time Evolution ##########

# Calculate the runtime
start = timeit.default_timer()

# Get the list of pulses
# 10ps with time steps of dt
t_pts = np.linspace(0,10, round(10E-12/dt))
int_p = shut_pulse(t_pts)

# Plot evolution
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(X, prob, 'r-')

# Convert to momentum space
psi_p = fftshift(fft(psi_x))
psi_p = np.multiply(exp_K,psi_p)

# Loop through each time step
for t_idx in range(len(t_pts)):
    potential = pot_interp(int_p[t_idx,:])

    # gparams = pot.GridParameters(X, potential=potential)
    # # find the ground state under this pulse
    # e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
    # ground_psi = e_vecs[:,0]

    # diagonal matrix of potential energy in position space
    exp_P = np.exp(-1j*dt/consts.hbar * potential)

    # Start the split operator method
    psi_x = ifft(ifftshift(psi_p))
    if t_idx%100  == 0:
            prob = [abs(x)**2 for x in psi_x]
            plt.plot(X, prob)
            plt.xlim(-1e-7, 1e-7)
            plt.ylim(-5e6, ymax + 5e6)
            plt.draw()
            plt.pause(1e-15)
            plt.clf()
    psi_x = np.multiply(exp_P,psi_x)
    psi_p = fftshift(fft(psi_x))     
    if t_idx != len(t_pts)-1:
        psi_p = np.multiply(exp_KK,psi_p)
    else:
        psi_p = np.multiply(exp_K,psi_p)
        psi_x = ifft(ifftshift(psi_p))

# Pring the runtime
stop = timeit.default_timer()
print('Time: ', stop - start) 
    
