import math
import numpy as np
from scipy.linalg import expm

########## Elementary Matrices ##########

X = np.array([[0,1],[1,0]],dtype=complex)
Y = np.array([[0,-1.j],[1.j,0]],dtype=complex)
Z = np.array([[1,0],[0,-1]],dtype=complex)
H = np.dot(math.sqrt(2), np.array([[1,1],[1,-1]],dtype=complex))
I = np.array([[1,0],[0,1]],dtype=complex)

def construct_elementary(numOfQubits, axis, qubit):
    """
    Given the number of qubits in the system, the axis of the elementary matrix,
    and the qubit it's acting on, this function results in the corresponding
    elementary matrix

    Parameters
    ----------
    numOfQubits : int
        Total number of qubits in the system
    axis : string
        Specify which pauli matrix the elementary matrix is based on
        options = ['x', 'y', 'z']
    qubit : int
        Specify which qubit the elementary matrix is about

    Returns
    -------
    An 2^n * 2^n elementary matrix, where n is the number of qubits

    """
    if axis == 'x':
        sigma = X
    elif axis == 'y':
        sigma = Y
    elif axis == 'z':
        sigma = Z
    
    if qubit != 1:
        output = I
        for i in range(qubit-1):
            output = np.kron(output, I)
        output = np.kron(output, sigma)
    else:
        output = sigma
    for i in range(qubit, numOfQubits):
        output = np.kron(output, I)
    return output

# test = construct_elementary(2, 'x', 1)
# print('elementary test: ', test)

########## Quantum Logic Gates from Elementary Matrices ##########

def construct_rotation(numOfQubits, axis, qubit, angle):
    """
    Given the number of qubits in the system, the axis of the elementary matrix,
    and the qubit it's acting on, and the rotation angle in radians, 
    this function results in the corresponding
    rotation matrix

    Parameters
    ----------
    numOfQubits : int
        Total number of qubits in the system
    axis : string
        Specify which pauli matrix the elementary matrix is based on
        options = ['x', 'y', 'z']
    qubit : int
        Specify which qubit the elementary matrix is about
    angle : float
        The rotation angle

    Returns
    -------
    An 2^n * 2^n matrix corresponding to a rotation
    
    TODO: question there's an overall phase introduced

    """
    if axis == 'x':
        sigma = X
    elif axis == 'y':
        sigma = Y
    elif axis == 'z':
        sigma = Z
    elementary = construct_elementary(numOfQubits, axis, qubit)
    return expm(-1.j * angle * elementary / 2)

# test = construct_rotation(1, 'x', 1, math.pi)
# print('RX constructed: ', test)
# print('RX:', expm(1.j * math.pi * X /2))

def construct_controlled(numOfQubits, control, target, operation):
    """
    Given the number of qubits in the system, the control bit,
    the target bit, and the specified operation, this function 
    results in a matrix of the corresponding controlled operation

    Parameters
    ----------
    numOfQubits : int
        Total number of qubits in the system
    control : int
        The number indcating the control qubit
    target : int
        The number indcating the target qubit
    operation: string
        Specify what operation is done on the target qubit
        options = ['X', 'Y', 'Z']

    Returns
    -------
    An 2^n * 2^n matrix corresponding to a CNOT gate

    """
    # matrix of the target operation
    if operation == 'X': 
        op = X
    if operation == 'Y': 
        op = Y
    elif operation == 'Z':
        op = Z

    # Useful control qubit matrices
    control1 = (I + Z)/2
    control2 = (I - Z)/2

    # the matrix is a sum of two cases
    # initialize both terms for the first qubit
    # When neither qubit is the first qubit
    if control != 1 and target != 1:
        first = I
        second = I
    # When the first qubit is the control qubit
    elif control == 1:
        first = control1
        second = control2
    # When the first qubit is the target qubit
    elif target == 1:
        first = I
        second = op
    # Loop through the rest of the 
    for i in range(numOfQubits-1):
        if i+2 == control:
            first = np.kron(first, control1)
            second = np.kron(second, control2)
        elif i+2 == target:
            first = np.kron(first, I)
            second = np.kron(second, op)
        else:
            first = np.kron(first, I)
            second = np.kron(second, I)
    output = first + second

    return output

# test = construct_controlled(2, 1, 2, 'X')
# print('CNOT constructed: ', test)
# test = construct_controlled(2, 1, 2, 'Z')
# print('CZ constructed: ', test)

SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=complex)
RSWAP = np.array([[1,0,0,0],[0,(1 + 1.j)/2,(1 - 1.j)/2,0],[0,(1 - 1.j)/2,(1 + 1.j)/2,0],[0,0,0,1]],dtype=complex)
CTRLX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
CTRLZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]],dtype=complex)


gates = {"X": X, "Y": Y, "Z": Z, "H": H, "SWAP": SWAP, 
"RSWAP": RSWAP, "CTRLX": CTRLX, "CTRLZ": CTRLZ}

########## Functions ##########    
def main():
    # open the quantum circuit file
    f = open('test.qcirc', 'r')
    line = f.readline()
    cnt = 1
    start = False
    while line:
        # print("Line {}: {}".format(cnt, line.strip()))
        lineL = line.split()

        if "Number of qubits" in line:
            n = int(lineL[-1])
            rho = np.zeros(2**n)
            rho[0] = 1
            # print(rho)
            start = True
        
        elif start: 
            gate = gates[lineL[0]]
            qubits = [int(i) for i in lineL[1:]]
            if len(qubits) == 1:
                if qubits[0] == 1: 
                    op = np.kron(gate, I)
                    rho = np.matmul(op,rho)
                    # print(rho)
                elif qubits[0] == 2:
                    op = np.kron(I, gate)
                    rho = np.matmul(op,rho)
                    # print(rho)
            elif len(qubits) == 2:
                rho = np.matmul(gate,rho)
                # print(rho)

        line = f.readline()
        cnt += 1

if __name__ == "__main__":
    main()