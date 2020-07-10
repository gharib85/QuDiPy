import os, sys
sys.path.append('../../')

import qudipy as qd
import qudipy.potential as pot
import qudipy.qutils as qt
from qudipy.shuttling.parameters import Params 
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy import sparse
from scipy.sparse import diags
from scipy.linalg import expm

import matplotlib.pyplot as plt

def initialize_params():
    """
    initialize the constants class with the Si/SiO2 material system and the GridParameters obejct 
    of the harmonic potential
    """
    # Initialize the constants class with the Si/SiO2 material system 
    consts = qd.Constants("Si/SiO2")

    # First define the x-coordinates
    x = np.linspace(-70,70,301)*1E-9
    # Define harmonic oscillator frequency
    omega = 5E12
    # Now construct the harmonic potential
    harm_pot = 1/2*consts.me*omega**2*np.square(x)
        
    # Create a GridParameters object
    gparams = pot.GridParameters(x, potential=harm_pot)

    return consts, gparams

def initialize_wf(consts, gparams):
    """
    find the initial wavefunction psi, which is a 1D array of dimension nx_local
    using the constants class with the Si/SiO2 material system and the GridParameters obejct 
    of the harmonic potential
    TODO: generalize to 2D wavefunctions
    """
    # Pass sparams, gparams to the solve_schrodinger_eq qutils method to obtain the eigenvalues and eigenvectors
    e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=5)      # n_sols set to 0 to obtain ground state
    # psi = np.real(e_vecs[:,0])
    print("energy: ", e_ens[0])
    psi = e_vecs[:,0]

    return psi

def main():
    # initialize relevant constants and parameters for the calculation
    consts, gparams = initialize_params()
    # diagonal matrix of potential energy in position space
    PE_1D = gparams.potential
    other_params = Params()
    # vector of position grid
    X = gparams.x                
    # number of grid points   TODO: delete
    nx = len(X)

    # spacing between grid points        TODO: delete
    dx = (max(X) - min(X))/(nx-1)   
    # indices of grid points
    I = [(i-nx/2) for i in range(nx)]   
    # vector of momentum grid
    P = [2 * consts.pi * consts.hbar * i / (nx*dx) for i in I]
    print(P)

    # diagonal matrix of kinetic energy in momentum space
    KE_1D = np.asarray([p**2/(2* consts.me) for p in P])

    # exponents present in evolution
    j = complex(0,1)
    exp_K = np.exp(-j * other_params.dt / (2 * consts.hbar) * KE_1D)
    exp_P = np.exp(-j * other_params.dt/consts.hbar  * PE_1D)

    # initialize psi(t=0)
    psi_x = initialize_wf(consts, gparams)
    # print("initial: ", psi_x)
    # print("initial probability is: ", [abs(x)**2 for x in psi_x])
    print("Plotting the initial wavefunction...")
    # plt.plot(X, [abs(x)**2 for x in psi_x])
    plt.plot(X, psi_x)
    plt.show()

    # number of time steps
    nt = 1000
    # iterate through nprint time steps
    for step in range(nt):
        # fourier transform into momentum space, psi(p)
        psi_p = fftshift(fft(psi_x))
        # multiply psi(p) by exp(K/2)
        psi_p = np.multiply(psi_p, exp_K)
        # inverse fourier transform back into position space, psi(x)
        psi_x = ifft(fftshift(psi_p))
        psi_x = np.multiply(psi_x, exp_P)
        psi_p = fftshift(fft(psi_x))
        psi_p = np.multiply(psi_p, exp_K)
        psi_x = ifft(fftshift(psi_p))

    output = psi_x
    # print("output: ", output)
    # print("the resultant probability is: ", [abs(x)**2 for x in output])
    print("Plotting the wavefunction at time ",nt * other_params.dt)
    plt.plot(P, [abs(x)**2 for x in output])
    plt.show() 

    


main()