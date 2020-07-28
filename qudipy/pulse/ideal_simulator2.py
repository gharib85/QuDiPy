import os, sys
sys.path.append('../../')
import math
import numpy as np
import qudipy.qutils.matrices as mat
import qudipy.pulse.file_parsers as fp


########## Load Circuit ##########
def ideal_simulator(filename):
    # Load the qcirc file into a QuantumCircuit object
    qcirc = fp.load_circuit(filename)
    qcirc.print_ideal_circuit()

    # number of qubits
    n = qcirc.n_qubits

    # circuit sequence
    circuit_seq = qcirc.circuit_sequence
    print(circuit_seq)

    # Initialize density matrix of the system
    rho = np.zeros((2**n, 2**n), dtype = complex)
    rho[0][0] = 1
    print("The density matrix of the initial state is: ", rho)

    for op in circuit_seq:
        gate = op[1]
        if gate[:2] == 'RX':
            q = float(op[2][0])
            print("x correct: ", mat.x(n, 2))  
            print("x: ", mat.x(n, op[2][0]))        # TODO: wrong, this gives identity
            
            gate_mat = mat.rx(n, q, int(gate[2:]))
            
            print ("correct: ", mat.rx(2,2,180))
            print("calculated: ", gate_mat)
        elif gate[:2] == 'RY':
            gate_mat = mat.ry(n, op[2][0], int(gate[2:]))
        elif gate[:2] == 'RZ':
            gate_mat = mat.rz(n, op[2][0], int(gate[2:]))
        elif gate == 'CTRLX':
            gate_mat = mat.cnot(n,op[2][0], op[2][1])
        elif gate == 'CTRLZ':
            gate_mat = mat.cz(n,op[2][0], op[2][1])
        elif gate == 'SWAP':
            gate_mat = mat.swap(n, op[2][0], op[2][1])
        elif gate == 'RSWAP':
            gate_mat = mat.rswap(n, op[2][0], op[2][1])
        # print(gate_mat)
        rho = gate_mat @ rho @ gate_mat.conj().T

    print("The density matrix of the state after the circuit is: ", rho)
    return rho

def main():
    filename = 'test.qcirc'
    ideal_simulator(filename)

if __name__ == "__main__":
    main()