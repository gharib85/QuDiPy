"""
Quantum utility solver functions

@author: simba
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from qudipy.qutils.math import inner_prod

def build_1DSE_hamiltonian(consts, gparams):
    ''' 
    Build a single electron Hamilonian for the 1-dimensional potential 
    specified in the gparams class. The laplacian operator is approximated by
    using a 1D 3-point stencil. The Hamilonian assumes a natural ordering 
    format along the main diagonal.

    Parameters
    ----------
    consts : Constants class
        Contains constants value for material system.
    gparams : GridParameters class
        Contains grid and potential information

    Returns
    -------
    ham_1D : sparse 2D array
        1-dimensional Hamtilonian. The diagonal elements are in natural
        ordering format

    '''
    
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
        
    # Assemble the full Hamiltonian with potential and kinetic terms
    ham_1D = PE_1D + KE_1D
    
    return ham_1D

def build_2DSE_hamiltonian(consts, gparams):
    '''
    Build a single electron Hamilonian for the 2-dimensional potential 
    specified in the gparams class. The laplacian operator is approximated by 
    using a 2D 5-point stencil. The Hamiltonian assumes a natural ordering
    format along the main diagonal.

    Parameters
    ----------
    consts : Constants class
        Contains constants value for material system.
    gparams : GridParameters class
        Contains grid and potential information

    Returns
    -------
    ham_2D : sparse 2D array
        2-dimensional Hamtilonian. The diagonal elements are in natural
        ordering format.

    '''
    
    # Build potential energy hamiltonian term
    PE_2D = sparse.diags(
        np.squeeze(gparams.convert_MG_to_NO(gparams.potential)))
    
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
                                sparse.eye(gparams.nx)/(gparams.dy**2))
    KE_2D = KE_2D + sparse.kron(sparse.diags(np.ones(gparams.ny-1),1),
                                sparse.eye(gparams.nx)/(gparams.dy**2))
    
    # Multiply by appropriate unit coefficients
    if consts.units == 'Ry':
        KE_2D = -KE_2D
    else:
        KE_2D = -consts.hbar**2/(2*consts.me)*KE_2D    
        
    # Assemble the full Hamiltonian with potential and kinetic terms
    ham_2D = PE_2D + KE_2D
    
    return ham_2D

def solve_schrodinger_eq(consts, gparams, n_sols=1):
    '''
    Solve the time-independent Schrodinger-Equation H|Y> = E|Y> where H is
    the single-electron 1 (or 2)-dimensional Hamiltonian.

    Parameters
    ----------
    consts : Constants class
        Contains constants value for material system.
    gparams : GridParameters class
        Contains grid and potential information.   
        
    Keyword Arguments
    -----------------
    n_sols: int, optional
        Number of eigenvectors and eigenenergies to return. The default is 1.

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
        hamiltonian = build_1DSE_hamiltonian(consts, gparams)
    elif gparams.grid_type == '2D':
        hamiltonian = build_2DSE_hamiltonian(consts, gparams)   
        
    # Solve the schrodinger equation (eigenvalue problem)
    eig_ens, eig_vecs = eigs(hamiltonian.tocsc(), k=n_sols, M=None,
                                           sigma=gparams.potential.min())
    
    # Sort the eigenvalues in ascending order (if not already)
    idx = eig_ens.argsort()   
    eig_ens = eig_ens[idx]
    eig_vecs = eig_vecs[:,idx]
    
    # Normalize the wavefunctions and convert to meshgrid format if it's a 2D
    # grid system
    if gparams.grid_type == '2D':
        eig_vecs_mesh = np.zeros((gparams.ny, gparams.nx, n_sols),
                                 dtype=complex)
    for idx in range(n_sols):
        curr_wf = eig_vecs[:,idx]
        
        if gparams.grid_type == '1D':
            norm_val = inner_prod(gparams, curr_wf, curr_wf)
        
            eig_vecs[:,idx] = curr_wf/np.sqrt(norm_val)
        
        if gparams.grid_type == '2D':
            norm_val = inner_prod(gparams, gparams.convert_NO_to_MG(
                curr_wf), gparams.convert_NO_to_MG(curr_wf))
        
            curr_wf = curr_wf/np.sqrt(norm_val)
            
            eig_vecs_mesh[:,:,idx] = gparams.convert_NO_to_MG(curr_wf)
            
    if gparams.grid_type == "2D":
        eig_vecs = eig_vecs_mesh
    
    return eig_ens, eig_vecs


        

    