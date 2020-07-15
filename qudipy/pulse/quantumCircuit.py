"""
Class for a quantum circuit
"""

class QuantumCircuit:
    
    def __init__(self, circuit_name, n_qubits, pulse_dict):
        
        # Name of circuit file
        self.name = circuit_name
        # Number of qubits in the circuit
        self.n_qubits = n_qubits
        # Loaded gates 
        self.gates = pulse_dict
        
        # 
        self.circuit_sequence = []
        
        # Index to track which gate in sequence we are on
        self.curr_gate_idx = 0
        
    def add_gate(self, gate_name, ideal_gate, used_qubits):
        '''
        This function adds a gate into the quantum circuit sequence.

        Parameters
        ----------
        gate_name : string
            Name of gate being added to sequence.
        used_qubits : int list
            Indices of qubits acted on by the gate.

        Returns
        -------
        None.

        '''
        
        # Check if a single qubit index was loaded or if it was a list.
        # If not a list, make it one.
        if isinstance(used_qubits, list) == False:
            used_qubits = [used_qubits]
        
        # Make sure used_qubits contains only ints
        used_qubits = [int(qubit_idx) for qubit_idx in used_qubits]
            
        # Check that the gate we are adding does not have an invalid qubit
        # index (i.e. outside of the allowable values)
        if (any(qubit_idx > self.n_qubits for qubit_idx in used_qubits) or
            any(qubit_idx < 1 for qubit_idx in used_qubits)):
            raise ValueError("Problem loading circuit file: " +
                            f"{self.name}.\n" +
                            f"Gate {gate_name} could not be loaded as the " +
                            f"affected qubit indices {used_qubits}\n are " +
                            " greater than the number of qubits in the circuit " +
                            f"({self.n_qubits}) or is <= 0.\n")
        
        # Add the gate to the circuit sequence
        self.circuit_sequence.append([gate_name, ideal_gate, used_qubits])
        
    def get_next_gate(self):
        '''
        Get the next gate in the quantum circuit sequence. If no more exist, 
        return None.

        Returns
        -------
        next_gate : [string, int list]
            Returns a list containing the next gate name and the affected 
            qubits by the gate.

        '''
        
        try:
            next_gate = self.circuit_sequence[self.curr_gate_idx]
            self.curr_gate_idx += 1
        except:
            next_gate = None
        
        return next_gate