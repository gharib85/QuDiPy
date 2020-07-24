# -*- coding: utf-8 -*-
"""
Constant matrices used to define quantum gates

@author: hromecB
"""

import numpy as np

#Pauli matrices

_X = np.array([[0,1],[1,0]], dtype=complex)
_Y = np.array([[0,-1j],[1j,0]], dtype=complex)
_Z = np.array([[1,0],[0,-1]], dtype=complex)
_I = np.array([[1,0],[0,1]], dtype=complex)
    
#constant matrices and expressions with them
    

def x_matrix(N,k):
    """
    Parameters
    ----------
    N : number of electrons in the system
    k : position of the X matrix

    Returns
    -------
    numpy array
        Matrix X_k of dimensions 2**N x 2**N 

    """
    if k==1:
        return np.kron(_X,np.eye(2**(N-1)))
    else:
        temp=_I
        for m in range(2,N+1):
            if k is m:
                temp=np.kron(temp,_X)
            else:
                temp=np.kron(temp,_I)
        return temp
            
            
def y_matrix(N,k):
    """
    Parameters
    ----------
    N : number of electrons in the system
    k : position of the Y matrix

    Returns
    -------
    numpy array
        Matrix Y_k of dimensions 2**Nx2**N 

    """
    if k==1:
        return np.kron(_Y,np.eye(2**(N-1)))
    else:
        temp=_I
        for m in range(2,N+1):
            if k is m:
                temp=np.kron(temp,_Y)
            else:
                temp=np.kron(temp,_I)
        return temp

          
def z_matrix(N,k):
    """
    Parameters
    ----------
    N : number of electrons in the system
    k : position of the Z matrix

    Returns
    -------
    numpy array
        Matrix Z_k of dimensions 2**Nx2**N 

    """
    if k==1:
        return np.kron(_Z,np.eye(2**(N-1)))
    else:
        temp=_I
        for m in range(2,N+1):
            if k is m:
                temp=np.kron(temp,_Z)
            else:
                temp=np.kron(temp,_I)
        return temp


def sigma_plus_matrix(N,k):
    return x_matrix(N,k)+1j*y_matrix(N,k)


def sigma_minus_matrix(N,k):
    return x_matrix(N,k)-1j*y_matrix(N,k)


def up_matrix(N,k):
    """
    Defines matrix that projects k-th qubit on the state |0>
    
    Parameters
    ----------
    N : number of electrons in the system
    k : number of the projected qubit

    Returns
    -------
    numpy array
        Matrix |0><0|k of dimensions 2**Nx2**N 

    """
    return 0.5*(unit_matrix(N) + z_matrix(N,k))
    
def down_matrix(N,k):
    """
    Defines matrix that projects k-th qubit on the state |1>
    
    Parameters
    ----------
    N : number of electrons in the system
    k : number of the projected qubit

    Returns
    -------
    numpy array
        Matrix |1><1|k of dimensions 2**Nx2**N 

    """
    return 0.5*(unit_matrix(N) - z_matrix(N,k))

def unit_matrix(N):
    """
    Defines unit matrix of dimensions 2**Nx2**N
    
    Parameters
    ----------
    N : number of electrons in the system

    Returns
    -------
    numpy array
       Unit matrix of dimensions 2**Nx2**N
    """
    return np.eye(2**N, 2**N, dtype = complex )


def cnot_matrix(N, ctrl, trgt):
    """
    Defines matrix for CNOT gate 
    
    Parameters
    ----------
    N : number of electrons in the system
    ctrl: control qubit
    trgt: target qubit

    Returns
    -------
    numpy array
        Matrix for CNOT gate

    """
    return up_matrix(N, ctrl) + down_matrix(N, ctrl) @ x_matrix(N, trgt)


def swap_matrix(N, k1, k2):
    """
    Defines SWAP gate matrix for qubits # k1, k2
    
    Parameters
    ----------
    N : number of electrons in the system
    k1, k2: positions of qubits

    Returns
    -------
    numpy array
        Matrix for SWAP gate 

    """
    return cnot_matrix(N, k1, k2) @ cnot_matrix(N, k2, k1) @ cnot_matrix(N, k1, k2)
    


def sigma_product_matrix(N, k1, k2):
    """
    Matrix coupled to the exchange parameter J_{k_1, k_2}
    
    Parameters
    ----------
    N : number of electrons in the system
    k1, k2 : positions of qubits

    Returns
    -------
    numpy array
        \vec{sigma_k1} \cdot \vec{sigma_k2}

    """    
    if k1==k2:
        return np.zeros((2**N, 2**N))
    else:
        return (
            x_matrix(N, k1) @ x_matrix(N, k2) 
            + y_matrix(N, k1) @ y_matrix(N, k2) 
            + z_matrix(N, k1) @ z_matrix(N, k2))


def rswap_matrix(N, k1, k2):
    """
    Defines sqrt(SWAP) gate matrix for qubits # k1, k2
    
    Parameters
    ----------
    N : number of electrons in the system
    k1, k2: positions of qubits

    Returns
    -------
    numpy array
        Matrix for SWAP gate 

    """
    return (1-1j)/4 * sigma_product_matrix(N, k1, k2) + (3+1j)/2 * unit_matrix(N)

