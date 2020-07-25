# -*- coding: utf-8 -*-
"""
Constant matrices used to define quantum gates

@author: hromecB
"""

import numpy as np

#Pauli matrices

PAULI_X = np.array([[0,1],[1,0]], dtype=complex)
PAULI_Y = np.array([[0,-1j],[1j,0]], dtype=complex)
PAULI_Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULI_I = np.array([[1,0],[0,1]], dtype=complex)
    
#constant matrices and expressions with them
    

def x(N,k):
    """
    Creates matrix X_k of dimensions 2**N x 2**N 
    
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
        return np.kron(PAULI_X,np.eye(2**(N-1)))
    else:
        temp=PAULI_I
        for m in range(2,N+1):
            if k is m:
                temp=np.kron(temp,PAULI_X)
            else:
                temp=np.kron(temp,PAULI_I)
        return temp
            
            
def y(N,k):
    """
    Creates matrix Y_k of dimensions 2**N x 2**N 
    
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
        return np.kron(PAULI_Y,np.eye(2**(N-1)))
    else:
        temp=PAULI_I
        for m in range(2,N+1):
            if k is m:
                temp=np.kron(temp,PAULI_Y)
            else:
                temp=np.kron(temp,PAULI_I)
        return temp

          
def z(N,k):
    """
    Parameters
    ----------
    Creates matrix Z_k of dimensions 2**N x 2**N 
    
    N : number of electrons in the system
    k : position of the Z matrix

    Returns
    -------
    numpy array
        Matrix Z_k of dimensions 2**Nx2**N 

    """
    if k==1:
        return np.kron(PAULI_Z,np.eye(2**(N-1)))
    else:
        temp=PAULI_I
        for m in range(2,N+1):
            if k is m:
                temp=np.kron(temp,PAULI_Z)
            else:
                temp=np.kron(temp,PAULI_I)
        return temp


def sigma_plus(N,k):
    return x(N,k)+1j*y(N,k)


def sigma_minus(N,k):
    return x(N,k)-1j*y(N,k)


def e_up(N,k):
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
    return 0.5*(unit(N) + z(N,k))
    
def e_down(N,k):
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
    return 0.5*(unit(N) - z(N,k))

def unit(N):
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


def cnot(N, ctrl, trgt):
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
    return e_up(N, ctrl) + e_down(N, ctrl) @ x(N, trgt)


def swap(N, k1, k2):
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
    return cnot(N, k1, k2) @ cnot(N, k2, k1) @ cnot(N, k1, k2)
    


def sigma_product(N, k1, k2):
    """
    Dot product of two Pauli vectors
    
    Parameters
    ----------
    N : number of electrons in the system
    k1, k2 : positions of qubits

    Returns
    -------
    numpy array
        \vec{sigma_k1} \cdot \vec{sigma_k2}

    """    
    
    return (
            x(N, k1) @ x(N, k2) 
            + y(N, k1) @ y(N, k2) 
            + z(N, k1) @ z(N, k2))


def rswap(N, k1, k2):
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
    return (1-1j)/4 * sigma_product(N, k1, k2) + (3+1j)/2 * unit(N)

