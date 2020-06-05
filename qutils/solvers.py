#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:58:04 2020

@author: simba
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

def build_1DSE_hamiltonian(sparams, gparams):
    '''
    
    Build a single electron Hamilonian for the 1-dimensional potential 
    specified in the gparams class. The laplacian operator is approximated by
    using a 1D 3-point stencil. The Hamilonian assumes a natural ordering 
    format along the main diagonal.

    Parameters
    ----------
    sparams : SimParameters class
        Contains simulation information
    
    gparams : GridParameters class
        Contains grid and potential information

    Returns
    -------
    ham_1D : sparse 2D array
        1-dimensional Hamtilonian. The diagonal elements are in natural
        ordering format

    '''
    
    # Build potential energy hamiltonian term
    PE_1D = sparse.diags(gparams.VV)
    
    # Build the kinetic energy hamiltonian term
    
    # Construct B matrix
    KE_1D = sparse.eye(gparams.nx)*(-2/(gparams.dx**2))
    # Add the +/-1 off diagonal entries for the 1/dx^2 elements
    KE_1D = KE_1D + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),-1)
    KE_1D = KE_1D + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),1)
    
    # Multiply by unit coefficients
    if sparams.units == 'Rydberg':
        KE_1D = -KE_1D
    else:
        KE_1D = -sparams.hbar**2/(2*sparams.me)*KE_1D    
        
    # Assemble the full Hamiltonian with potential and kinetic terms
    ham_1D = PE_1D + KE_1D
    
    return ham_1D

def build_2DSE_hamiltonian(sparams, gparams):
    '''
    
    Build a single electron Hamilonian for the 2-dimensional potential 
    specified in the gparams class. The laplacian operator is approximated by 
    using a 2D 5-point stencil. The Hamiltonian assumes a natural ordering
    format along the main diagonal.

    Parameters
    ----------
    sparams : SimParameters class
        Contains simulation information
    
    gparams : GridParameters class
        Contains grid and potential information

    Returns
    -------
    ham_2D : sparse 2D array
        2-dimensional Hamtilonian. The diagonal elements are in natural
        ordering format.

    '''
    
    # Build potential energy hamiltonian term
    PE_2D = sparse.diags(gparams.convert_MG_to_NO(gparams.VV))
    
    # Build the kinetic energy hamiltonian term
    
    # Construct B matrix
    B = sparse.eye(gparams.nx)*(-2/(gparams.dx**2) - 2/(gparams.dy**2))
    # Add the +/-1 off diagonal entries for the 1/dx^2 elements
    B = B + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),-1)
    B = B + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),1)
    
    # Now create a block diagonal matrix of Bs
    KE_2D = sparse.kron(sparse.eye(gparams.ny), B)
    # Now set the off diagonal entries for the 1/dy^2 elements
    KE_2D = KE_2D + sparse.kron(sparse.diags(np.ones(gparams.ny-1),-1),
                                sparse.eye(gparams.nx)/gparams.dy**2)
    KE_2D = KE_2D + sparse.kron(sparse.diags(np.ones(gparams.ny-1),1),
                                sparse.eye(gparams.nx)/gparams.dy**2)
    
    # Multiply by appropriate unit coefficients
    if sparams.units == 'Rydberg':
        KE_2D = -KE_2D
    else:
        KE_2D = -sparams.hbar**2/(2*sparams.me)*KE_2D    
        
    # Assemble the full Hamiltonian with potential and kinetic terms
    ham_2D = PE_2D + KE_2D
    
    return ham_2D

def solve_schrodinger_eq(sparams, gparams, n_sols=1):
    '''
    
    Solve the time-independent Schrodinger-Equation H|Y> = E|Y> where H is
    the single-electron 1 (or 2)-dimensional Hamiltonian.

    Parameters
    ----------
    sparams : SimParameters class
        Contains simulation information.
    
    gparams : GridParameters class
        Contains grid and potential information.
        
    n_sols: int
        Number of eigenvectors and eigenenergies to return (default is 1).

    Returns
    -------
    eig_ens : complex 1D array
        Lowest eigenenergies sorted in ascending order.

    eig_vecs : complex 2D array
        Corresponding eigenvectors in natural order format. eig_vecs[:,i] is 
        the eigenvector for eigenvalue eig_ens[i].
        

    '''
    
    # Determine if a 1D or 2D grid and build the respective Hamiltonian
    if gparams.grid_type == '1D':
        hamiltonian = build_1DSE_hamiltonian(sparams, gparams)
    elif gparams.grid_type == '2D':
        hamiltonian = build_2DSE_hamiltonian(sparams, gparams)   
        
    # Solve the schrodinger equation (eigenvalue problem)
    eig_ens, eig_vecs = eigs(hamiltonian.tocsc(), k=n_sols, M=None,
                                           sigma=gparams.VV.min())
    
    # Sort the eigenvalues in ascending order (if not already)
    idx = eig_ens.argsort()   
    eig_ens = eig_ens[idx]
    eig_vecs = eig_vecs[:,idx]
    
    return eig_ens, eig_vecs

if __name__ == "__main__":
    import sys
    sys.path.insert(1, '/Users/simba/Documents/GitHub/Silicon-Modelling')
    import potential as pot
    
    x = np.linspace(-120,120,401)*1E-9
    
    class SimulationParameters():
        pass
    sparams = SimulationParameters()
    sparams.units = 'SI'
    sparams.me = 9.11E-31*0.191
    omega = 5E12
    sparams.hbar = 6.626E-34/2/3.14159
    # print(1/(m*omega/hbar)**(1/2))
    harm_pot = 1/2*sparams.me*omega**2*np.square(x)
    
    gparams = pot.GridParameters(x, potential=harm_pot)
    
    e_ens, e_vecs = solve_schrodinger_eq(sparams, gparams, 4)
        

    