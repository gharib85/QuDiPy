__all__ = [
    'inner_prod','partial_trace', 'project_up', 'project_down', 'matrix_sqrt']

import numpy as np
import math

def inner_prod(gparams, wf1, wf2):
    '''
    Evaluates the inner product between two complex wavefunctions.

    Parameters
    ----------
    gparams : GridParameters class
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
        inn_prod = np.trapz(np.multiply(wf1.conj(),wf2), x=gparams.x)
    elif gparams.grid_type == '2D':
        inn_prod = np.trapz(np.trapz(np.multiply(wf1.conj(),wf2), 
                                     x=gparams.x, axis=1),gparams.y)
            
    return inn_prod

   
def project_up(rho, elem):
    """
    Projects the system density matrix onto the spin-up state of 
    the k^th electron
    
    Parameters
    ----------
    rho : numpy array
        matrix of the dimensions 2**N x 2**N(density matrix in our case)
    elem : int / iterable of ints
        number(s) of the electron(s) whose state(s) is to project the system 
        density matrix on

    Returns
    -------
    complex 2D array
        new density matrix of the dimensions 2**m x 2**m, m = N - dim(elem) 

    """
    dim=rho.shape[0]    #dimension of the matrix
    N = int(math.log2(rho.shape[0]))   
            #number of qubits encoded by density matrix
    
    if isinstance(elem, int):
        
        temp=[]
            #the idea behind bitwise shifts is that every matrix in tensor product
            #doubles the dimension of the resulting matrix, i. e. adds one binary 
            #digit to the number. It follows from the definition of Kronecker
            #product that the most significant bit defines the element of the
            #leftmost matrix in Kronecker product, and vice versa
        for i in range(0, dim):
            if (i >> (N - elem))%2==0:
                var=[]
                for j in range(0, dim):
                    if (j >> (N - elem))%2==0:
                        var.append(rho[i][j])
                temp.append(var)
            else:
                continue
    
        return  np.array(temp)
    
    elif isinstance(elem, (tuple, list, set) ):
        temp = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            # consecutive application of the project_up operations 
        for el in elems_sorted:
            temp = project_up(temp, el)
        return temp
    
    else:
        print("Qubits that are traced out should be defined by a single int \
              number or an iterable of ints. Try again")
            

def project_down(rho, elem):
    """
    Projects the system density matrix onto the spin-down state 
    of the k^th electron
    
    Parameters
    ----------
    rho : numpy array
        matrix of the dimensions 2**N x 2**N(density matrix in our case)
    elem : int / iterable of ints
        number(s) of the electron(s) whose state(s) is to project the system 
        density matrix on

    Returns
    -------
    complex 2D array
        new density matrix of the dimensions 2**m x 2**m, m = N - dim(elem) 

    """

    dim=rho.shape[0]    #dimension of the matrix
    N = int(math.log2(rho.shape[0]))   
            #number of qubits encoded by density matrix
    
    if isinstance(elem, int):
        
        temp=[]
            #the idea behind bitwise shifts is that every matrix in tensor product
            #doubles the dimension of the resulting matrix, i. e. adds one binary 
            #digit to the number. It follows from the definition of Kronecker
            #product that the most significant bit defines the element of the
            #leftmost matrix in Kronecker product, and vice versa
        for i in range(0, dim):
            if (i >> (N - elem))%2==1:
                var=[]
                for j in range(0, dim):
                    if (j >> (N - elem))%2==1:
                        var.append(rho[i][j])
                temp.append(var)
            else:
                continue
    
        return  np.array(temp)
    
    elif isinstance(elem, (tuple,list,set)):
        temp = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            #consecutive application of the project_down operations 
        for el in elems_sorted:
            temp = project_down(temp, el)
        return temp
    
    else:
        print("Qubits that are traced out should be defined by a single int \
              number or a tuple of ints. Try again")


def partial_trace(rho, elem):
    """
    Finds partial trace with respect to the k^th qubit
    
    Parameters
    ----------
    rho : numpy array
        matrix of the dimensions 2**N x 2**N (density matrix in our case)
    elem : int / iterable of ints
        number(s) of the electron(s) whose state(s) is/are averaged out

    Returns
    -------
    complex 2D array
        new density matrix of the dimensions 2**m x 2**m, m = N - dim(elem) 

    """
    if isinstance(elem, int):
        return project_up(rho, elem) + project_down(rho, elem)
    
    elif isinstance(elem, (tuple, set, list)):
        temp = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            # consecutive application of the project_up operations 
        for el in elems_sorted:
            temp = partial_trace(temp, el)
        return temp
    else:
        print("Qubits that are traced out should be defined by a single int \
              number or a tuple of ints. Try again")

def matrix_sqrt(A):
    """
    Calculates a square root of the matrix

    Parameters
    ----------
    A : numpy array
        matrix to calculate the square root of

    Returns
    -------
    complex 2D array
        square root of the matrix

    """
    w,v=np.linalg.eig(A)
    
    return v @ np.diag(np.sqrt((1+0j)*w)) @ np.linalg.inv(v)
    












