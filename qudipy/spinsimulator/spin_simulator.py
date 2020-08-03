# -*- coding: utf-8 -*-
"""

- Script for calculating spin evolution

@author: bkhromet
"""

import numpy as np

from math import pi, log2, exp, sin, cos

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))


import qudipy.qutils.matrices as matr
import qudipy.qutils.math as mth

import qudipy.circuit as circ
from qudipy.utils.constants import Constants 

#defining system constants
cst = Constants()       #TODO add input("Enter the system name")

#helper functions

def p(B_0, T):
    """
    
    Parameters
    ----------
    B_0 : Zeeman field 
    T : temperature

    Returns
    -------
    float
        population of the spin-up state

    """
    
    return 1/(exp(2*cst.muB*B_0/(cst.kB*T))+1)
    

    

def J_sigma_product(N, k1, k2):
    """
    Matrix coupled to the exchange parameter J_{k_1, k_2}. Takes into account
    that there is no exchange with itself
    
    Parameters
    ----------
    N : number of electrons in the system
    k1, k2 : positions of the sigma matrices(have values from 1 to N)

    Returns
    -------
    numpy array
        i/4\cst.hbar \vec{sigma_k1} \cdot \vec{sigma_k2}

    """    
    if k1==k2:
        return np.zeros((2**N, 2**N))
    else:
        return 1/(4*cst.hbar)*matr.sigma_product(N, k1, k2)
    
    
def x_sum(N):
    """
    Sum highligted with green in equation 2.1 of the "Simulator plan"
    
    Parameters
    ----------
    N : number of electrons in the system

    Returns
    -------
    numpy array
        Sum of X_k-matrices for all k\in[1,N] weighted by cst.muB/cst.hbar

    """
    return cst.muB/cst.hbar*sum( matr.x(N,k) for k in range(1, N+1) )


def y_sum(N):
    """
    Sum highligted with cyan in equation 2.1 of the "Simulator plan"
    
    Parameters
    ----------
    N : number of electrons in the system

    Returns
    -------
    numpy array
        Sum of Y_k-matrices for all k\in[1,N] weighted by cst.muB/cst.hbar

    """
    return cst.muB/cst.hbar*sum( matr.y(N,k) for k in range(1, N+1) )


def z_sum_omega(N, B_0, f_rf):
    """
    Sum highligted with brown in equation 2.1 of the "Simulator plan"
    
    Parameters
    ----------
    N : number of electrons in the system

    Returns
    -------
    numpy array
        Sum of Z_k-matrices for all k\in[1,N] weighted by i(omega-omega_tilde)

    """
    return (1*(cst.muB*B_0/cst.hbar-pi*f_rf)
            *sum( matr.z(N,k) for k in range(1, N+1)))


def z_sum_p(N, B_0, T, T_1):
    """
    Sum highligted with red in equation 2.1 of the "Simulator plan"
    
    Parameters
    ----------
    N : int
        number of electrons in the system
    
    B_0 : float
        Zeeman field
    
    T : float
            temperature
    
    T1 : float
            spin relaxation time
    
    Returns
    -------
    numpy array
        DESCRIPTION.

    """
    return (2*p(B_0,T)-1)/T_1*sum( matr.z(N,k) for k in range(1, N+1) )
    

#list of dictionaries of constant matrices


def const_dict(N_0, T, B_0, f_rf, T_1):
    """
    List of dictionaries of all constant matrices
    
    Parameters
    ----------
    N_0 : int
        The maximal number of quantum dots in the system that host electrons
    ... (see previous functions)

    Returns
    -------
    List of dictionaries. Entries of each dictionary:
        - "Xs" - list of X_k
        - "Ys" - list of Y_k
        - "Zs" - list of Z_k
        - "sigma_pluses" - array of matr.sigma_plus_k_l
        - "sigma_minuses" - list of matr.sigma_minus_k
        - "J_sigma_products" - (symmetric) matrix of (i/4\cst.hbar \vec{sigma_k1} \cdot \vec{sigma_k2} )
        - "x_sum" (multiplied by cst.muB/cst.hbar)
        - "y_sum" (multiplied by cst.muB/cst.hbar)
        - "z_sum_omega"
        - "Z_sum_p"
            
    """
    temp = []
    for N in range(1,N_0+1):
        
        Xs=[matr.x(N,k) for k in range(1,N+1)]
        Ys=[matr.y(N,k) for k in range(1,N+1)]
        Zs=[matr.z(N,k) for k in range(1,N+1)]
        
        sigma_pluses=[matr.sigma_plus(N,k) for k in range(1,N+1)]
        sigma_minuses=[matr.sigma_minus(N,k) for k in range(1,N+1)]
        
        J_sigma_products=[[J_sigma_product(N,k1,k2) for k2   in range(1,N+1)] for k1 in range(1,N+1)] 
                       
        temp.append({"Xs":Xs, "Ys":Ys, "Zs":Zs,  "sigma_pluses":sigma_pluses, "sigma_minuses":sigma_minuses, "J_sigma_products":J_sigma_products, "x_sum":x_sum(N), "y_sum":y_sum(N), "z_sum_omega":z_sum_omega(N, B_0, f_rf), "z_sum_p":z_sum_p(N,B_0,T,T_1)})
    
    return temp


#helper functions



class SpinSys:
        
    def __init__(self, rho, N_0=None, 
                 sys_param_dict=None):
        """
            
        The class describes the spin system. Attributes comprise the system
        density matrix and constant system parameters. Methods construct Hamiltonian
        and time-evolve the system with pulses
        
        ----------
        rho : numpy 2D array
            Density matrix. Initialized with the value before pulse
            
        N_0 : int
            Maximal number of electrons in the system (could be bigger
            than the dimensions of rho ) 
            
        sys_param_dict : dictionary
            System parameter dictionary that might contain:
        T_1: float
            spin relaxation time [s]
        T_2: float
            spin dephasing time [s]
        B_0 : float
            Zeeman field [T]
        T : float
            Temperature [K]. 
        f_rf: float
            RF field frequency [Hz]              
       
            
        Returns
        -------
        None.

        """
        
        self.rho = rho
        if N_0 == None:
            self.N_0 = int(log2(rho.shape[0]))
        else:
            self.N_0 = N_0
        
        if sys_param_dict is not None:
            self.T_1 = sys_param_dict.get("T_1",1)      
                #1 s is default value (chosen randomly)
            self.T_2 = sys_param_dict.get("T_2",1E-3)   
                #1 ms is default value (chosen randomly)
            self.B_0 = sys_param_dict.get("B_0", 0) 
                # no Zeeman field by default
            self.T = sys_param_dict.get("T", 4)
                #Default value is liquid He temperature
            self.f_rf = sys_param_dict.get("f_rf", 0)
            
        else:
            self.T_1 = 1
            self.T_2 = 1E-3        
            self.B_0 = 0
            self.T = 4
            self.f_rf = 0
            self.cdict = const_dict(self.N_0, 4, 0, 0, 1)
       
        self.cdict = const_dict(self.N_0, self.T,
                                self.B_0, self.f_rf,
                                self.T_1)
        

        
        #instance attributes
        
    def hamiltonian(self, const_dict_N, pulse_params=None):
        """
        Builds the system hamiltonian at a particular point of time

        Parameters
        ----------
        const_dict_N: dict
            dictionary of constant matrices 2**N x 2**N
        pulse_params : dictionary
             dictionary of values of delta_g[i], 
        J[i], B_x, B_y at a particular point of time


        Returns
        -------
        Numpy array that represents Hamiltonian

        """
        if pulse_params is None:
            return 0
        else:
            
            N = int(log2(const_dict_N["x_sum"].shape[0])) 
            
            # building a list of delta_g values           
            delta_g=[0]*N
            
            delta_g_dict = {int(k[8:]):v for (k,v) in pulse_params.items() 
                     if k[:8]=="delta_g_"}      
                        #TODO could be necessary to update 8 to a diff. number
            for i in range(1, N+1):
                delta_g[i-1]=delta_g_dict.get(i, 0)
            
            # building a list of J values
            if N==1:
                J=0
            else:
                J=[0]*(N-1)
                
                J_dict=  {int(k[2:]):v for (k,v) in pulse_params.items() 
                         if k[:2]=="J_"}
                for i in range(1, N):
                    J[i-1]=J_dict.get(i, 0)
            
            #Incorporating values of Bx, By
                    
            B_x = pulse_params.get("B_x", 0)
            B_y = pulse_params.get("B_y", 0)
            phi = pulse_params.get("phi", 0)
            
            ham = const_dict_N["z_sum_omega"].copy()
                #copy is in order not to modify the const_dict_N entries
            
            if delta_g != [0]*N:
                for k in range(N):
                    ham += cst.muB/(2*cst.hbar)*self.B_0*delta_g[k] * const_dict_N["Zs"][k]
                
            if J != [0]*(N-1) and J != 0:
                for k in range(N):
                    ham += J[k]*const_dict_N["J_sigma_products"][k][k+1]
            
            if B_x != 0:
                ham += const_dict_N["x_sum"]* B_x * cos(phi)
            
            if B_y != 0:
                ham += const_dict_N["y_sum"]* B_y * sin(phi)
                
        return ham
    

    def lindbladian(self, rho_mod, const_dict_N, pulse_params=None):
        """
        Creates right-hand side of the Lindblad equation

        Parameters
        ----------
        rho_mod: numpy array
            intermediate density matrices the Runge-Kutta method is used for
        const_dict_N: dict
            dictionary of constant matrices 2**N x 2**N

        Returns
        -------
        Numpy array of the right-hand side of the equation

        """
        N = int(log2(const_dict_N["x_sum"].shape[0])) 
            #automatically calculating the size
        ham = np.zeros((2**N, 2**N))
        if pulse_params is not None:
            ham = ham + self.hamiltonian(const_dict_N, pulse_params)
        
        lin = ( 1j *( rho_mod.dot(ham) - ham.dot(rho_mod) ) + const_dict_N["z_sum_p"] - (2/self.T_1 + 1/(2*self.T_2)) * N * rho_mod )
        
        for k in range(N):
            lin += (  p(self.B_0, self.T) / self.T_1 * np.dot(const_dict_N["sigma_pluses"][k] , rho_mod.dot(const_dict_N["sigma_minuses"][k])) +  (1 - p(self.B_0, self.T)) / self.T_1 * np.dot(const_dict_N["sigma_minuses"][k], rho_mod.dot(const_dict_N["sigma_pluses"][k])))
 
        return lin
        
    
    def evolve(self, pulses, 
               rhos_expected=None, is_fidelity=False, is_purity=False,
               track_qubits=None, are_Bloch_vectors=False,
                track_points_per_pulse=100
               ):
        """
        Function that performs spin system evolution under the external pulse,
        i.e. updates the system density matrix

        Parameters
        ----------
        pulses : dictionary of pulse details (which is also dictionary)
            at different points of time
        
        optional arguments:
        
        rhos_expected: numpy array/ iterable of numpy arrays 
            The theoretical (expected) density matrix after the system
            evolution under a pulse 
            
        is_fidelity : bool
            Indicates whether to calculate and return the 
            fidelity of the density matrix after the pulse with rho_expected
        is_purity : bool
            Indicates whether to calculate and return the purity
            of the density matrix after the pulse
        track_qubits : int / iterable of ints
            Positions(s) of qubit(s) to track during the pulses 
            (i. e. evaluate density submatrices for them)
        are_Bloch_vectors: bool
            Indicates whether to calculate Bloch vectors for the tracked 
            qubits    
        track_points_per_pulse: int
            Number of submatrices to save per pulse 
            
        Returns
        -------
        dictionary containing values of fidelity and purity, if they are 

        """
        ret_dict = {"pulse": tuple(pulses.keys())}
        #dictionary to be returned; copied in order to retain the
        if rhos_expected is not None:
            rhodict = rhos_expected.copy()
        else:
            rhodict = None

        if is_fidelity:
            ret_dict["fidelity"]=[]
        
        if is_purity:
            ret_dict["purity_simulated"] = []
            if rhos_expected is not None:
                ret_dict["purity_expected"] = []
        

        if track_qubits is not None:
            
            track_keys = list(self.track_subsystem(
                track_qubits, are_Bloch_vectors).keys())
            
            ret_dict.update({k:[] for k in track_keys})
            ret_dict["track_time"] = []   
      
        time_counter=0
        
        for pulsename in pulses:
            pulse = pulses[pulsename].copy()
            
            N = int(log2(self.rho.shape[0]))
            
            cdict_N = self.cdict[N-1]
            pulse_time = pulse.pop("pulse_time", 0)
            
            #number of values of delta_g, B_x, etc contained in the pulse.
            num_values = np.shape(pulse[list(pulse.keys())[0]])[0]
            
            if pulse_time==0:       
                delta_t=self.T_2/100   #TODO change to an appropriate number
            else:
                delta_t = pulse_time /  num_values
                
            #checking if dimensions of arrays in pulse are consistent
            dim_correct=True
            for var in pulse:
                dim_correct &= ( 
                    pulse_time / np.shape(pulse[var])[0] == delta_t)
            if not dim_correct:
                print("Inconsistent dimensions of the pulse entry. Please try again")
                break
            else:
                ret_dict["track_time"].append(time_counter)
                for key in track_keys:
                    ret_dict[key].append(self.track_subsystem( 
                        track_qubits, are_Bloch_vectors)[key])
                        
                for n in range(1, num_values):
                    #Runge-Kutta method
                    cur_pulse = {k:v[n-1] for (k,v) in pulse.items()}
                    next_pulse={k:v[n] for (k,v) in pulse.items()}
                    avg_pulse={k:(0.5*(v[n-1]+v[n])) for (k,v) in pulse.items()}
                    
                    K1 = self.lindbladian(self.rho, cdict_N, cur_pulse)
                    K2 = self.lindbladian(self.rho + 0.5*delta_t*K1, cdict_N, avg_pulse )
                    K3 = self.lindbladian(self.rho + 0.5*delta_t*K2, cdict_N, avg_pulse )
                    K4 = self.lindbladian(self.rho + delta_t*K3, cdict_N, next_pulse )
                    #updating density matrix
                    
                    self.rho = self.rho + delta_t/6*(K1+2*K2+2*K3+K4)
                    
                    #implementing tracking:
                    if track_qubits is not None:
                        track_step = max(int(num_values /  track_points_per_pulse), 1)
                        
                        if n % track_step == 0:
                            ret_dict["track_time"].append(time_counter + n*delta_t)
                            for key in track_keys:
                                ret_dict[key].append(self.track_subsystem( 
                                    track_qubits, are_Bloch_vectors)[key])
                            
                time_counter += pulse_time
                
                                      
        #implementing additional functions
                    #fidelity
                if is_fidelity and rhodict is not None:
                    if isinstance(rhodict, dict):
                        ret_dict["fidelity"].append(
                            self.fidelity(rhodict[pulsename]))
                    elif isinstance(rhodict, (tuple, list)):
                        ret_dict["fidelity"].append( 
                                    self.fidelity(rhodict.pop(0)))
                        
                    # purity
                if is_purity:
                        purdic = {}
                        if isinstance(rhodict, dict):
                            purdic = self.purity(
                                        rho_expected=rhodict[pulsename])
                        elif isinstance(rhodict, (tuple, list)):
                            purdic = self.purity(rho_expected=rhodict.pop(0))
                        for key in purdic:
                                ret_dict[key].append(purdic[key])
                
        # tracked single-qubit subsystems
        if track_qubits is not None:
                    
            #case of only one expected matrix
            if is_fidelity and isinstance(rhodict, np.ndarray):
                ret_dict["fidelity"] = self.fidelity(rhodict[pulsename])
                
            if is_purity:
                if isinstance(rhodict, np.ndarray) or rhodict == None:
                    purdic = self.purity(rho_expected=rhodict)
                    for key in purdic:
                        ret_dict[key].append(purdic[key])
                        
                    
        
        return ret_dict


# optionally called functions

      
    
    def fidelity(self, rho_expected):
        """
        Calculates fidelity of the system density matrix with respect to 
        the expected one.
    
        Parameters
        ----------
        rho_expected : numpy array
            The theoretical (expected) density matrix after the system
            evolution under a pulse
    
        Returns
        -------
        Float value of fidelity defined in simulator_plan    TODO: add formula
    
        """
        
        return np.trace(mth.matrix_sqrt(
                    mth.matrix_sqrt(rho_expected) 
                             @ self.rho @ mth.matrix_sqrt(rho_expected)))**2
        
    
    def purity(self, rho_expected=None):
        """
        Calculates purity tr(rho^2)
    
        Returns
        -------
        Float value of purity
    
        """
        if rho_expected is None:
            return {"purity_simulated" : np.trace( self.rho @  self.rho)}
        else:
            return {"purity_simulated" : np.trace( self.rho @  self.rho),
                    "purity_expected" : np.trace(rho_expected @ rho_expected) }

                
    
    def track_subsystem(self, track_qubits, 
                          are_Bloch_vectors):
        """
        Gives the system submatrices and Bloch vectors if tracked

        Parameters
        ----------
        pulse : dict
            Parameters of a (single) pulse
        
        track_qubits : int/iterable
            Defines the qubits whose density submatrices are tracked
        
        are_Bloch_vectors: bool 
            Indicates whether to calculate Bloch vector component(s)
        Returns
        -------
        dictionary containing subsystem density matrices and, possibly, their 
        Bloch vectors

        """
        ret_dict = {}
        
        N = int(log2(self.rho.shape[0]))  
        Nset = set(range(1,N+1))
        trqub = {}
        
        if isinstance(track_qubits, int):
            trqub = {track_qubits}
        elif isinstance(track_qubits, tuple):
            trqub = track_qubits
        elif isinstance(track_qubits, (list, set)):
            trqub = track_qubits.copy()
        else:
            print("error")
            
        for qub in trqub:
            submatrix = mth.partial_trace(self.rho, (Nset-set(trqub)))
            subm = "submatrix_{}".format(qub)
            ret_dict[subm] = submatrix
            if are_Bloch_vectors:
                ret_dict["sigma_x_{}".format(qub)] = np.trace(submatrix @ matr.PAULI_X)
                ret_dict["sigma_y_{}".format(qub)] = np.trace(submatrix @ matr.PAULI_Y)
                ret_dict["sigma_z_{}".format(qub)] = np.trace(submatrix @ matr.PAULI_Z)
            
        return ret_dict
            
            
        