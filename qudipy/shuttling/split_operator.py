import os, sys
sys.path.append('../../')

import qudipy as qd
import qudipy.potential as pot
import qudipy.qutils as qt
from qudipy.shuttling.parameters import Params 
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import sparse
from scipy.sparse import diags
from scipy.linalg import expm

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
    # psi = np.real(e_vecs[:,0])
    psi = e_vecs[:,0]

    return psi


def main():
    consts, gparams = initialize_params()
    KE_1D, PE_1D = initialize_ham(consts, gparams)
    KE_1D = KE_1D.toarray()  
    PE_1D = PE_1D.toarray()  
    print("KE_1D: ", type(KE_1D))
    print("PE_1D: ", type(PE_1D))
    other_params = Params()
    X = gparams.x
    # print("X is:", X)
    nx = len(X)
    dx = (max(X) - min(X))/(nx-1)
    I = [(i-nx/2) for i in range(nx)]
    P = [2 * consts.pi * consts.hbar * i / (nx*dx) for i in I]

    # exponents present in evolution
    j = complex(0,1)
    exp_K = expm(-j * other_params.dt / (2 * consts.hbar) * KE_1D)
    exp_P = expm(-j * other_params.dt/consts.hbar  * PE_1D)
    print(type(exp_K))
    print(type(exp_P))


    # initialize psi(t=0)
    # TODO delete
    # note: wf is a list
    psi_x = initialize_wf(consts, gparams)
    print("initial wavefunction is: ", psi_x)

    t = 0
    nt = 1500
    # iterate through nprint time steps
    for step in range(nt):
        # fourier transform into momentum space, psi(p)
        psi_p = fft(psi_x)
        # multiply psi(p) by exp(K/2)
        psi_p = np.matmul(psi_p, exp_K)
        # inverse fourier transform back into position space, psi(x)
        psi_x = ifft(psi_p)
        psi_x = np.matmul(psi_x, exp_P)
        psi_p = fft(psi_x)
        psi_p = np.matmul(psi_p, exp_K)
        psi_x = ifft(psi_p)

    output = psi_x
    print(output)

main()