import os
import math
import numpy as np
from scipy.linalg import expm
import file_parsers

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
    if axis == 'X':
        sigma = X
    elif axis == 'Y':
        sigma = Y
    elif axis == 'Z':
        sigma = Z
    
    if qubit != 1:
        output = I
        for i in range(qubit-2):
            print(i)
            output = np.kron(output, I)
        output = np.kron(output, sigma)
    else:
        output = sigma
    for i in range(qubit, numOfQubits):
        output = np.kron(output, I)
    return output

test = construct_elementary(2, 'X', 2)
# test2 = construct_elementary(2, 'X', 2)
print('elementary test: ', test)
# print('elementary test2: ', test2)


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
        The rotation angle in radians

    Returns
    -------
    An 2^n * 2^n matrix corresponding to a rotation
    
    TODO: question there's an overall phase introduced

    """
    if axis == 'X':
        sigma = X
    elif axis == 'Y':
        sigma = Y
    elif axis == 'Z':
        sigma = Z
    elementary = construct_elementary(numOfQubits, axis, qubit)
    return expm(-1.j * angle * elementary / 2)

# test = construct_rotation(2, 'X', 1, math.pi)
# print('RX constructed: ', test)
# print('RZ:', expm(1.j * math.pi * Z /2))

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
    An 2^n * 2^n matrix corresponding to a controlled-R gate

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

# Function to generate all binary strings  
def generateAllBinaryStrings_helper(n, arr, i, out):  
    if i == n: 
        temp = []
        for i in range(0, n):  
            temp.append(arr[i])
        out.append(temp)
        return 
      
    # First assign "0" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 0
    generateAllBinaryStrings_helper(n, arr, i + 1, out)  
  
    # And then assign "1" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 1
    generateAllBinaryStrings_helper(n, arr, i + 1, out) 

def generateAllBinaryStrings(numOfQubits):
    """
    Returns a list of all possible states of a system with a specifc 
    number of qubits. Each state is represente as a list of length n
    """
    output = []
    arr = [None] * numOfQubits
    generateAllBinaryStrings_helper(numOfQubits, arr, 0, output)
    return output

def SWAP(numOfQubits, qubit1, qubit2):
    """
    Given the number of qubits in the system and the index of the
    two qubits being swapped, this function results in a matrix of 
    the corresponding SWAP operation

    Parameters
    ----------
    numOfQubits : int
        Total number of qubits in the system
    qubit1 : int
        the qubit being swapped with a smaller index
    qubit2 : int
        the qubit being swapped with a larger index

    Returns
    -------
    An 2^n * 2^n matrix corresponding to a SWAP gate
    """
    # initialize the matrix of the gate
    matrix = np.zeros((2**numOfQubits, 2**numOfQubits),dtype=complex)
    q1 = qubit1 - 1
    q2 = qubit2 - 1
    states = generateAllBinaryStrings(numOfQubits)
    for i in range(len(states)):
        s = states[i]
        if s[q1] == s[q2]:
            matrix[i][i] = 1
        else:
            # print(s)
            qubit1_value = s[q1]
            qubit2_value = s[q2]
            s[q1] = qubit2_value
            s[q2] = qubit1_value
            states = generateAllBinaryStrings(numOfQubits)
            swap_index = states.index(s)
            matrix[i][swap_index] = 1
    return matrix

def RSWAP(numOfQubits, qubit1, qubit2):
    """
    Given the number of qubits in the system and the index of the
    two qubits being swapped, this function results in a matrix of 
    the corresponding RSWAP operation

    Parameters
    ----------
    numOfQubits : int
        Total number of qubits in the system
    qubit1 : int
        the qubit being swapped with a smaller index
    qubit2 : int
        the qubit being swapped with a larger index

    Returns
    -------
    An 2^n * 2^n matrix corresponding to a RSWAP gate
    """
    # initialize the matrix of the gate
    matrix = np.zeros((2**numOfQubits, 2**numOfQubits),dtype=complex)
    q1 = qubit1 - 1
    q2 = qubit2 - 1
    states = generateAllBinaryStrings(numOfQubits)
    for i in range(len(states)):
        s = states[i]
        if s[q1] == s[q2]:
            matrix[i][i] = 1
        else:
            # print(s)
            qubit1_value = s[q1]
            qubit2_value = s[q2]
            s[q1] = qubit2_value
            s[q2] = qubit1_value
            states = generateAllBinaryStrings(numOfQubits)
            swap_index = states.index(s)
            matrix[i][swap_index] = (1-1.j)/2
            matrix[i][i] = (1+1.j)/2
    return matrix

########## Load Circuit ##########
def load_circuit(filename):
    # Load the qcirc file into a QuantumCircuit object
    qcirc = file_parsers.load_circuit(filename)
    qcirc.print_ideal_circuit()

    # number of qubits
    n = qcirc.n_qubits

    # circuit sequence
    circuit_seq = qcirc.circuit_sequence
    print(circuit_seq)

    # Initialize the system
    rho = np.zeros(2**n)
    rho[0] = 1
    print(rho)

    for op in circuit_seq:
        gate = op[1]
        if gate[0] == 'R' and gate[1] in ['X', 'Y', 'Z']:
            # print('gate[1]:', gate[1])
            gate_mat = construct_rotation(n, gate[1], op[2][0], int(gate[2:])*math.pi/180)
        elif gate[:4] == 'CTRL':
            gate_mat = construct_controlled(n, op[2][0], op[2][1], gate[4])
        elif gate == 'SWAP':
            gate_mat = SWAP(n, op[2][0], op[2][1])
        elif gate == 'RSWAP':
            gate_mat = RSWAP(n, op[2][0], op[2][1])
        # print(gate_mat)
        rho = np.matmul(gate_mat,rho)
        # rho = gate_mat @ rho @ gate_mat.conj().T

    print(rho)
    return rho

filename = 'test.qcirc'
load_circuit(filename)

# ########## Functions ##########    
# def main():
#     # open the quantum circuit file
#     f = open('test.qcirc', 'r')
#     line = f.readline()
#     cnt = 1
#     start = False
#     while line:
#         # print("Line {}: {}".format(cnt, line.strip()))
#         lineL = line.split()

#         if "Number of qubits" in line:
#             n = int(lineL[-1])
#             rho = np.zeros(2**n)
#             rho[0] = 1
#             # print(rho)
#             start = True
        
#         elif start: 
#             gate = gates[lineL[0]]
#             qubits = [int(i) for i in lineL[1:]]
#             if len(qubits) == 1:
#                 if qubits[0] == 1: 
#                     op = np.kron(gate, I)
#                     rho = np.matmul(op,rho)
#                     # print(rho)
#                 elif qubits[0] == 2:
#                     op = np.kron(I, gate)
#                     rho = np.matmul(op,rho)
#                     # print(rho)
#             elif len(qubits) == 2:
#                 rho = np.matmul(gate,rho)
#                 # print(rho)

#         line = f.readline()
#         cnt += 1

# if __name__ == "__main__":
#     main()