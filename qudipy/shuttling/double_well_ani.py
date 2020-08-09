import os, sys
sys.path.append('/Users/keweizhou/Google_Drive/Research/20summer/Waterloo/QuDiPy/')

import qudipy as qd
import qudipy.potential as pot
import qudipy.qutils as qt
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy import sparse
from scipy.sparse import diags
from scipy.linalg import expm
import matplotlib.pyplot as plt
import timeit

# import matplotlib.animation as animation


def initialize_params():
    """
    initialize the constants class with the Si/SiO2 material system and the GridParameters obejct 
    of the harmonic potential
    """
    # Initialize the constants class with the Si/SiO2 material system 
    consts = qd.Constants("Si/SiO2")

    # First define the x-coordinates
    x = np.linspace(-100,100,256)*1E-9
    # Define harmonic oscillator frequency
    omega = 5E12
    sep = 27.25E-9
    # Now construct the harmonic potential
    harm_pot_L = 1/2*consts.me*omega**2*np.square(x - sep)
    harm_pot_R = 1/2*consts.me*omega**2*np.square(x + sep)
    harm_pot = np.minimum(harm_pot_L, harm_pot_R)
    
    # Create a GridParameters object
    gparams = pot.GridParameters(x, potential=harm_pot)

    return consts, gparams

def initialize_wf(consts, gparams):
    """
    find the initial wavefunction psi and the length of the simulation
    the initial wavefunction is a 1D array of dimension nx_local using 
    the constants class with the Si/SiO2 material system and the GridParameters 
    obejct of the harmonic potential
    """
    # Pass sparams, gparams to the solve_schrodinger_eq qutils method to obtain the eigenvalues and eigenvectors
    e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=5)      # n_sols set to 0 to obtain ground state
    # psi = np.real(e_vecs[:,0])
    print("energy 0: ", e_ens[0])
    print("energy 1: ", e_ens[1])
    print("energy dff:", e_ens[1] - e_ens[0])
    t_time = 1/(2*(e_ens[1] - e_ens[0])/6.626E-34)
    print("tunnel time [s]:", t_time)

    psi = 1/np.sqrt(2)*(e_vecs[:,0] + e_vecs[:,1])
    
    print('Norm psi: ', qd.qutils.math.inner_prod(gparams, psi, psi))

    return psi, t_time

def split_operator(psi_x, dt):
    global consts, gparams, P
    # exponents present in evolution
    exp_K = np.exp(-1j*dt/2*np.multiply(P,P)/(2*consts.me*consts.hbar))
    # exp_KK = np.multiply(exp_K,exp_K)
    exp_P = np.exp(-1j*dt/consts.hbar*gparams.potential)

    psi_p = fftshift(fft(psi_x))
    psi_p = np.multiply(exp_K,psi_p)

    psi_x = ifft(ifftshift(psi_p))     
    psi_x = np.multiply(exp_P,psi_x)
    
    psi_p = fftshift(fft(psi_x))   
    psi_p = np.multiply(exp_K,psi_p)  

    psi_x = ifft(ifftshift(psi_p))

    return psi_x



##############################################
########## Time Evolution Animation ##########
##############################################

# initialize relevant constants and parameters for the calculation
consts, gparams = initialize_params()
# diagonal matrix of potential energy in position space
PE_1D = gparams.potential

dt = 5E-16
# vector of position grid
X = gparams.x                

# indices of grid points
I = [(idx-gparams.nx/2) for idx in range(gparams.nx)]   
# vector of momentum grid
P = np.asarray([2 * consts.pi * consts.hbar * i / (gparams.nx*gparams.dx) for i in I])

# initialize psi(t=0)
psi_x, t_time = initialize_wf(consts, gparams)
prob = [abs(x)**2 for x in psi_x]
ymax = max(prob)

# Calculate the runtime
start = timeit.default_timer()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

line1, = ax.plot(X, prob, 'r-')


nt = int(np.round(t_time/dt))
for step in range(nt):
    psi_x = split_operator(psi_x, dt
    if step%2000  == 0:
        prob = [abs(x)**2 for x in psi_x]
        plt.plot(X, prob)
        plt.xlim(-1e-7, 1e-7)
        plt.ylim(-5e6, ymax + 5e6)
        plt.draw()
        plt.pause(1e-15)
        plt.clf()

stop = timeit.default_timer()
print('Time: ', stop - start)  
