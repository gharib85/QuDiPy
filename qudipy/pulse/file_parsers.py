"""
Functions for handling control pulse files .ctrlp
"""

import pandas as pd
import numpy as np
import os
from qudipy.pulse import ControlPulse, QuantumCircuit

def _load_one_pulse(f_name):
    '''
    This function takes a single .ctrlp file and extracts the pulse information.

    Parameters
    ----------
    f_name : string
        Full file path to the control pulse file to load.

    Returns
    -------
    ctrl_pulse : ControlPulse object
        Control pulse object containing all information loaded from the file.

    '''
        
    # Check file extension
    if f_name[-6::] != ".ctrlp":
        raise ValueError("Unrecognized control pulse file type..." +
                         " Expected .ctrlp file.")
    
    # Open file and extract pulse name
    f = open(f_name, "r")
    pulse_name = os.path.basename(f.name).split('.')[0]
    
    # Count the number of lines which don't the control pulse vs time information
    line_cnt = 0
    # Initialize ideal gate to idenity
    ideal_gate = "I"
    for x in f:
        # Keep track of how many lines to skip for reading in the pulse table
        line_cnt += 1
        # Parse line by line (don't care about ordering except that Control 
        # pulses is the last line)
        if "Control pulses:" in x:
            break
        elif "type:" in x:
            # Remove any leading/ending white space and convert to lowercase
            pulse_type = x.split(":")[1].strip().lower()
        elif "length" in x:
            pulse_length = int(x.split(":")[1].strip().split(" ")[0])
        elif "Ideal" in x:
            ideal_gate = x.split(":")[1].strip().upper()
            
    # Initialize the control pulse object
    ctrl_pulse = ControlPulse(pulse_name, pulse_type, pulse_length, ideal_gate)
    
    # Read the rest of the pulse file to get the pulse information
    df = pd.read_csv(f_name, skiprows = line_cnt)
    
    # Loop over each column in the pulse table and add control variables to 
    # control pulse object
    ctrl_var_names = df.columns
    for ctrl_var in ctrl_var_names:
        ctrl_pulse.add_control_variable(ctrl_var, df[ctrl_var])
    
    return ctrl_pulse

def load_pulses(f_names):
    '''
    This function takes multiple .ctrlp files as inputs and constructs a
    dictionary containing every control pulse's information

    Parameters
    ----------
    f_names : list of strings
        Full file path to the control pulse file to load.

    Returns
    -------
    pulse_dict : dictionary of ControlPulse objects
        Dictionary containing many control pulse objects
        
    '''
    
    pulse_dict = {}
    
    # Check if only a single file was passed.
    # If it was and was not contained in a list already, then wrap it in a list.
    if isinstance(f_names, list) == False:
        f_names = [f_names]
    
    # Loop through each file, load the pulse then add to the pulse dictionary.
    for f in f_names:
        curr_pulse = _load_one_pulse(f)
        
        pulse_dict[curr_pulse.name] = curr_pulse
        
    return pulse_dict
      
def load_circuit(f_name, gate_dict):
    '''
    This function takes in a single quantum circuit .qcirc file as input and 
    constructs a quantum circuit object

    Parameters
    ----------
    f_name : string
        Full file path to the circuit file.

    Returns
    -------
    q_circ : QuantumCircuit object
        Class containing information loaded from the .qcirc file required for
        simulation.

    '''
    
    # Check file extension
    if f_name[-6::] != ".qcirc":
        raise ValueError("Unrecognized quantum circuit file type..." +
                         " Expected .qcirc file.")
    
    # Open the file and extract the circuit name
    f = open(f_name, "r")
    circuit_name = os.path.basename(f.name).split('.')[0]

    for x in f:
        # Parse line by line
        if "Number of qubits:" in x:
            n_qubits = int(x.split(":")[1])
            
            # Initialize the quantum circuit object
            q_circ = QuantumCircuit(circuit_name, n_qubits, gate_dict)       
        else:
            # Parse the line information
            temp = x.strip().split(" ")
            
            gate_name = temp[0]
            
            # Check that the gate name is one that was actually loaded
            if gate_name not in q_circ.gates.keys():
                raise ValueError("Problem loading circuit file: " +
                                 f"{circuit_name}.\n" +
                                 f"Gate {gate_name} could not be loaded as " +
                                 "the corresponding pulse was not loaded nor " + 
                                 " is the gate name a ideal gate keyword.\n" +
                                 "Check .qcirc file for typos or load the " +
                                 "corresponding pulse file.")
            
            # Get the corresponding ideal gate
            ideal_gate = q_circ.gates[gate_name].ideal_gate
            # Check that it's a valid ideal gate keyword
            if not _check_ideal_gate(ideal_gate):
                raise ValueError("Problem loading circuit file: " +
                                 f"{circuit_name}.\n" +
                                 f"Gate {gate_name} was not loaded as the " +
                                 "ideal gate keyword was not recognized.")
            
            gate_acting_qubits = temp[1:]
            
            gate_acting_qubits = [int(qubit_idx) for qubit_idx in gate_acting_qubits]
            q_circ.add_gate(gate_name, ideal_gate, gate_acting_qubits)

    return q_circ            
    
def _check_ideal_gate(gate_name, qubit_idx=[]):
    '''
    This function checks if the supplied gate_name is a valid ideal gate
    keyword used to simulate an ideal quantum circuit. 
    Current supported keywords are
    I, RX###, RY###, RZ###, H, CTRLX, CTRLY, CTRLZ, SWAP, RSWAP
    where ### in the R gates indicates the gate's rotation angle in degrees.

    Parameters
    ----------
    gate_name : string
        Gate name to be tested.
    qubit_idx : list of ints 
        Indices of qubits used by gate

    Returns
    -------
    boolean

    '''
    
    # Quick check by looking at gate name length
    if len(gate_name) not in [1,4,5]:
        return False
    
    # Check I gate:
    if gate_name == "I":
        return True
    
    # Check H gate
    if gate_name == "H":
        return True
    
    # Check for a R gate first
    if gate_name[0] == "R" and gate_name[1] in ["X","Y","Z"]:
        # Now check that the next three characters are ints
        for idx in range(2,5):
            try:
                int(gate_name[idx])
            except:
                return False
        return True
    
    # Check CTRL gates
    if gate_name[:4] == "CTRL" and gate_name[4] in ["Z","X","Y"]:
        # Now check qubit number, must be an even number of used qubits
        if len(qubit_idx) == 0 or np.mod(len(qubit_idx),2) == 0:
            return True
    
    # Check SWAP and RSWAP
    if gate_name == "SWAP" or gate_name == "RSWAP":
        # Now check qubit number, must be an even number of used qubits
        if len(qubit_idx) == 0 or np.mod(len(qubit_idx),2) == 0:
            return True
    
    # Otherwise
    return False
    
if __name__ == "__main__":
    path_to_pulses = "/Users/simba/Documents/GitHub/Silicon-Modelling/tutorials/Tutorial data/Control pulses/"
    file_pulse = ["ROTX_1_2.ctrlp", "CTRLZ_3_4.ctrlp", "H_1_2_4.ctrlp",
                  "ROTX_3.ctrlp", "CTRLX_2_3.ctrlp"]
    pulse_files = []
    for p in file_pulse:
        pulse_files.append(path_to_pulses + p)
    
    file_circuit = '/Users/simba/Documents/GitHub/Silicon-Modelling/tutorials/Tutorial data/Quantum circuits/test_circuit.qcirc'
    
    pulse_dict = load_pulses(pulse_files)
    
    circuit = load_circuit(file_circuit, pulse_dict)
    
    test = "CTRLZ"
    print(_check_ideal_gate(test,[1,2,4]))
    
    # TODO: Add a dummy ascii printout for the quantum circuit.
    
    
    

