"""
Quantum utility math functions

@author: simba
"""

import numpy as np

def find_overlap_matrix(gparams, basis_1_wfs, basis_2_wfs):
    '''
    Calculates the overlap matrix S between two sets of basis vectors. The
    size of S is NxM where N = size(basis 1) and M = size(basis 2).

    Parameters
    ----------
    gparams : GridParameters object
        Contains grid information.
    basis_1_wfs : array
        Set of basis states where the first index in the array corresponds to 
        the ith state. The state must be a meshgrid if a 2D grid is used.
    basis_2_wfs : array
        Set of basis states where the first index in the array corresponds to 
        the ith state. The state must be a meshgrid if a 2D grid is used.

    Returns
    -------
    S_matrix : array
        Overlap matrix between basis 1 and basis 2.

    '''
    
    num_basis_1 = basis_1_wfs.shape[0]
    num_basis_2 = basis_2_wfs.shape[0]
    
    S_matrix = np.zeros((num_basis_2, num_basis_1), dtype=complex)
    
    for j_idx in range(num_basis_1):
        wf_1 = basis_1_wfs[j_idx]
        
        for i_idx in range(num_basis_2):
            wf_2 = basis_2_wfs[i_idx]
            
            # S_ij = <b2_i|b1_j>
            S_matrix[i_idx,j_idx] = inner_prod(gparams, wf_2, wf_1)

    return S_matrix

def inner_prod(gparams, wf1, wf2):
    '''
    Evaluates the inner product between two complex wavefunctions.

    Parameters
    ----------
    gparams : GridParameters object
        Contains grid and potential information.
    wf1 : complex array
        'Bra' wavefunction for the inner product. If grid is 2D, then the 
        array should be in meshgrid format.
    wf2 : complex array
        'Ket' wavefunction for the inner product. If grid is 2D, then the 
        array should be in meshgrid format.

    Returns
    -------
    inn_prod : complex float
        The inner product between the two inputted wavefunctions.

    '''
    
    if gparams.grid_type == '1D':
        inn_prod = np.trapz(wf1.conj()*wf2, x=gparams.x)
    elif gparams.grid_type == '2D':
        inn_prod = np.trapz(np.trapz(wf1.conj()*wf2, x=gparams.x, axis=1),
                            gparams.y)
            
    return inn_prod

def partial_trace(rho, dim, sys):
    '''
    This code takes the partial trace of a density matrix.  It is adapted from
    the open-source TrX file written by Toby Cubitt for MATLAB.
    
    TODO: Add code for ket states as well.. I didn't understand how the MATLAB
    code was working.

    Parameters
    ----------
    rho : complex 2D array
        A 2D array describing a density matrix.
    dim : 1D array
        An array of the dimensions of each subsystem for the whole system for
        psi. [2,4,2] corresponds to a system of size 2x4x2 with 3 subsystems.
    sys : 1D array
        An array of the subsystems to trace out. For dim=[2,4,2], sys=[1,3] 
        would trace out the first and third subsystems leaving only a subspace
        with size 4.

    Returns
    -------
    traced_rho : complex 2D array
        The resulting rho after taking the partial trace.
        
    '''
    # Convert inputs to numpy arrays if not already inputted as such.
    rho = np.array(rho)
    sys = np.array(sys)
    dim = np.array(dim)
    
    # Check inputs
    if any(sys > len(dim)-1) or any(sys < 0):
        print(sys > len(dim))
        print(any(sys < 0))
        # Error with sys variable
        raise ValueError("Invalid subsytem in sys.")
    if ((len(dim) == 1 and np.mod(len(rho)/dim,1) != 0) 
        and len(rho) != np.prod(dim)):
        # Error with dim or psi variables
        raise ValueError("Size of rho inconsistent with dim.")
    
    # Get rid of any singleton dimensions in dim
    sys = np.setdiff1d(sys, np.argwhere(dim == 1))
    dim = np.concatenate([dim[idx] for idx in np.argwhere(dim != 1)])
    
    # Number of subsystems
    n = len(dim)
    dim_reversed = dim[::-1]
    subsystems_keep = np.array([idx for idx in np.linspace(0,n-1,n) 
                                if idx not in sys])
    # Dimension of psi to trace out
    dim_trace = dim[sys].prod()
    # Dimension of psi leftover after partial trace
    dim_keep = len(rho)/dim_trace
    
    # Reshape density matrix into tensor with one row and one column index for
    # each subsystem, permute traced subsytem indices to the end, reshape
    # again so that first two indices are row and column multi-indices for
    # kept subsystems and third index is a flatted index for traced subsytems,
    # then sum third index over "diagonal" entries.
    perm = n-1 - np.concatenate([subsystems_keep[::-1], subsystems_keep[::-1]-n,
                               sys, sys-n])
    rho1 = np.reshape(rho, np.concatenate([dim_reversed, dim_reversed]),
                      order='F')
    rho2 = np.transpose(rho1, perm.astype(int))
    rho = np.reshape(rho2, np.array([dim_keep, dim_keep, dim_trace**2],
                                    dtype=np.uint), order='F')
    traced_rho = np.sum(rho[:,:,range(0,dim_trace**2,dim_trace+1)], axis=2)
            
    return traced_rho












