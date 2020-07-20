import math
import numpy as np


########## Quantum Logic Gates ##########
RX180 = np.array([[0,1],[1,0]],dtype=complex)
RY180 = np.array([[0,-1.j],[1.j,0]],dtype=complex)
RZ180 = np.array([[1,0],[0,-1]],dtype=complex)

H = np.dot(math.sqrt(2), np.array([[1,1],[1,-1]],dtype=complex))
I = np.array([[1,0],[0,1]],dtype=complex)

SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=complex)
RSWAP = np.array([[1,0,0,0],[0,(1 + 1.j)/2,(1 - 1.j)/2,0],[0,(1 - 1.j)/2,(1 + 1.j)/2,0],[0,0,0,1]],dtype=complex)
CTRLX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
CTRLZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]],dtype=complex)


gates = {"RX180": RX180, "RY180": RY180, "RZ180": RZ180, "H": H, "SWAP": SWAP, 
"RSWAP": RSWAP, "CTRLX": CTRLX, "CTRLZ": CTRLZ}

########## Functions ##########    
def main():
    # open the quantum circuit file
    f = open('test.qcirc', 'r')
    line = f.readline()
    cnt = 1
    start = False
    while line:
        print("Line {}: {}".format(cnt, line.strip()))
        lineL = line.split()

        if "Number of qubits" in line:
            n = int(lineL[-1])
            rho = np.zeros(2**n)
            rho[0] = 1
            print(rho)
            start = True
        
        elif start: 
            gate = gates[lineL[0]]
            qubits = [int(i) for i in lineL[1:]]
            if len(qubits) == 1:
                if qubits[0] == 1: 
                    op = np.kron(gate, I)
                    rho = np.matmul(op,rho)
                    print(rho)
                elif qubits[0] == 2:
                    op = np.kron(I, gate)
                    rho = np.matmul(op,rho)
                    print(rho)
            elif len(qubits) == 2:
                rho = np.matmul(gate,rho)
                print(rho)

        line = f.readline()
        cnt += 1

if __name__ == "__main__":
    main()