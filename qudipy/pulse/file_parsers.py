"""
Functions for handling control pulse files .ctrlp
"""

import pandas as pd
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
            
    # Initialize the control pulse object
    ctrl_pulse = ControlPulse(pulse_name, pulse_type, pulse_length)
    
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
            
            gate_acting_qubits = temp[1:]
            
            gate_acting_qubits = [int(qubit_idx) for qubit_idx in gate_acting_qubits]
            q_circ.add_gate(gate_name, gate_acting_qubits)

    return q_circ            
    
    
if __name__ == "__main__":
    file_pulse = '/Users/simba/Documents/GitHub/Silicon-Modelling/tutorials/Tutorial data/Control pulses/ROTX_1_2.ctrlp'
    file_circuit = '/Users/simba/Documents/GitHub/Silicon-Modelling/tutorials/Tutorial data/Quantum circuits/test_circuit.qcirc'
    
    pulse_obj = load_pulses(file_pulse)
    
    circuit = load_circuit(file_circuit, pulse_obj)
    
    # TODO: check that loaded gate name in add_gate is actually avaiable in
    # in the pulse dictionary.
    
    
    
    
    

