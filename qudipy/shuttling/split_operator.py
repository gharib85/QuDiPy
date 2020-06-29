import os, sys
sys.path.append('../../')

import qudipy as qd
import qudipy.potential as pot
import qudipy.qutils as qt
import qudipy.shuttling.parameters as params
import numpy as np
from scipy.fft import fft, ifft
from scipy import sparse

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

def initialize_ham(consts, gparams):
    """
    find the kinetic term and the potential term of a 1D Hamiltonian
    using the constants class with the Si/SiO2 material system and the GridParameters obejct 
    of the harmonic potential
    TODO: generalize to 2D wavefunctions
    """
    # Build potential energy hamiltonian term
    PE_1D = sparse.diags(gparams.potential)
    
    # Build the kinetic energy hamiltonian term
    
    # Construct dummy block matrix B
    KE_1D = sparse.eye(gparams.nx)*(-2/(gparams.dx**2))
    # Add the +/-1 off diagonal entries for the 1/dx^2 elements
    KE_1D = KE_1D + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),-1)
    KE_1D = KE_1D + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),1)
    
    # Multiply by unit coefficients
    if consts.units == 'Ry':
        KE_1D = -KE_1D
    else:
        KE_1D = -consts.hbar**2/(2*consts.me)*KE_1D    

    return KE_1D, PE_1D

def initialize_wf(consts, gparams):
    """
    find the initial wavefunction psi, which is a 1D array of dimension nx_local
    using the constants class with the Si/SiO2 material system and the GridParameters obejct 
    of the harmonic potential
    TODO: generalize to 2D wavefunctions
    """
    # Pass sparams, gparams to the solve_schrodinger_eq qutils method to obtain the eigenvalues and eigenvectors
    e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)      # n_sols set to 0 to obtain ground state
    psi = np.real(e_vecs[:,0])

    return psi

def user_observe(t, psi, exp_K, exp_P):
    """
    for every time step, this function is called with inputs being the current time t
    and the current state psi
    """
    # fourier transform into momentum space, psi(p)
    psi_p = fft(psi)
    # multiply psi(p) by exp(K/2)
    exp_K_2 = exp_K/2
    psi_p *= np.exp(exp_K_2)
    # inverse fourier transform back into position space, psi(x)
    psi_x = ifft(psi_p)

    # iterate through nprint
    for i in range(tprint):
        psi_x *= exp(exp_K_2)
        psi_p = fft(psi_x)
        psi_p *= exp(exp_P)
        psi_x = ifft(psi_p)
    
    psi_x *= exp(exp_P)
    psi_p = fft(psi_x)
    psi_p *= exp(exp_K_2)
    psi_x = ifft(psi_p)

    # TODO: calculate observable?

    return t+tprint, psi_x

def main():
    consts, gparams = initialize_params
    KE_1D, PE_1D = initialize_ham(consts, gparams)

    # exponents present in evolution
    exp_K = -j * dt * p**2 / (2*hbar * 2*params.mass)
    exp_P = -j * dt * PE_1D /hbar

    # initialize psi(t=0)
    psi = initialize_wf(consts, gparams)
    print(psi)

    t = 0
    # iterate through nprint time steps
    for j in range(nt/nprint):
        t, psi = user_observe(t, psi, exp_K, exp_P)

    output = psi

main()