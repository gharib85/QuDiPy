# -*- coding: utf-8 -*-
"""

- Script for calculating spin evolution

@author: bkhromet
"""

import numpy as np

from math import pi, log2, exp, sin, cos, prod

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import warnings

import qudipy.qutils.matrices as matr
import qudipy.qutils.math as mth

from qudipy.circuit import ControlPulse
from qudipy.utils.constants import Constants 


#material system is chosen to be GaAs by default because such parameters as 
#effective mass or dielectric constant do not matter for spin simulations;
#this could be changed later if needed

cst = Constants("GaAs")       

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
    The following numpy 2D array:
        1/4\cst.hbar \vec{sigma_k1} \cdot \vec{sigma_k2}

    """    
    if k1==k2:
        return np.zeros((2**N, 2**N))
    else:
        return 1 / (4*cst.hbar)*matr.sigma_product(N, k1, k2)
    
    
def x_sum(N):
    """
    Sum highligted with green in equation 2.1 of the "Simulator plan". Used to 
    reduce the total number of multiplications during the time evolution 
    
    Parameters
    ----------
    N : number of electrons in the system

    Returns
    -------
    numpy 2D array
        Sum of X_k-matrices for all k in [1,N] weighted by cst.muB/cst.hbar

    """
    return cst.muB/cst.hbar*sum( matr.x(N,k) for k in range(1, N+1) )


def y_sum(N):
    """
    Sum highligted with cyan in equation 2.1 of the "Simulator plan". Used to 
    reduce the total number of multiplications during the time evolution 
    
    Parameters
    ----------
    N : number of electrons in the system

    Returns
    -------
    numpy 2D array
        Sum of Y_k-matrices for all k in[1,N] weighted by cst.muB/cst.hbar

    """
    return cst.muB/cst.hbar*sum( matr.y(N,k) for k in range(1, N+1) )


def z_sum_omega(N, B_0, f_rf):
    """
    Sum highligted with brown in equation 2.1 of the "Simulator plan". Used to 
    reduce the total number of multiplications during the time evolution 
    simulation   
    
    Parameters
    ----------
    N : number of electrons in the system

    Returns
    -------
    numpy 2D array
        Sum of Z_k-matrices for all k in [1,N] weighted by i(omega-omega_rf)/2

    """
    return ((cst.muB*B_0/cst.hbar-pi*f_rf)
            *sum( matr.z(N,k) for k in range(1, N+1)))


def z_sum_p(N, B_0, T, T_1):
    """
    Sum highligted with red in equation 2.1 of the "Simulator plan". Used to 
    reduce the total number of multiplications during the time evolution 
    simulation
    
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
    numpy 2D array
        Sum of Z_k-matrices for all k in [1,N] weighted by (2*p(B_0,T)-1)/T_1


    """
    return (2*p(B_0,T)-1)/T_1*sum(matr.z(N,k) for k in range(1, N+1))
    

#list of dictionaries of constant matrices


def const_dict(N_0, T, B_0, f_rf, T_1):
    """
    List of dictionaries of all constant matrices
    
    Parameters
    ----------
    N_0 : int
        The maximal number of quantum dots in the system that host electrons
    ... (see previously defined functions)

    Returns
    -------
    List of dictionaries. Entries of each dictionary:
        - "Xs" - list of X_k
        - "Ys" - list of Y_k
        - "Zs" - list of Z_k
        - "sigma_pluses" - array of matr.sigma_plus_k_l
        - "sigma_minuses" - list of matr.sigma_minus_k
        - "J_sigma_products" - (symmetric) matrix of 
            (i/4\cst.hbar \vec{sigma_k1} \cdot \vec{sigma_k2} )
        - "x_sum" (multiplied by cst.muB/cst.hbar)
        - "y_sum" (multiplied by cst.muB/cst.hbar)
        - "z_sum_omega"
        - "Z_sum_p"
    Each dictionary corresponds to a different dimensionality of the system 
    (from 1 to N0)
    """
    temp = []
    for N in range(1,N_0+1):
        
        Xs=[matr.x(N,k) for k in range(1,N+1)]
        Ys=[matr.y(N,k) for k in range(1,N+1)]
        Zs=[matr.z(N,k) for k in range(1,N+1)]
        
        sigma_pluses=[matr.sigma_plus(N,k) for k in range(1,N+1)]
        sigma_minuses=[matr.sigma_minus(N,k) for k in range(1,N+1)]
        
        J_sigma_products=[[J_sigma_product(N,k1,k2) for k2 in range(1,N+1)] 
                          for k1 in range(1,N+1)] 
                       
        temp.append({"Xs":Xs, "Ys":Ys, "Zs":Zs,  "sigma_pluses":sigma_pluses, 
                     "sigma_minuses":sigma_minuses, 
                     "J_sigma_products":J_sigma_products,
                     "x_sum":x_sum(N), "y_sum":y_sum(N), 
                     "z_sum_omega":z_sum_omega(N, B_0, f_rf), 
                     "z_sum_p":z_sum_p(N,B_0,T,T_1)})
    
    return temp



class SpinSys:
    """
    The class describes the spin system. Attributes comprise the system
    density matrix and constant system parameters. Methods construct 
    Hamiltonian  and time-evolve the system with pulses
    
    Attributes
    ----------
    rho : 2D array
        Density matrix. Initialized with the value before pulse
            
    N_0 : int
        Maximal possible number of electrons in the system (could be bigger
            than the dimensions of rho; useful in case we load new electrons) 
    time: float
        Point in time at which the system is defined
    T_1: float
        Spin relaxation time [s]
    T_2: float
        Spin dephasing time [s]
    B_0 : float
        Zeeman field [T]
    T : float
        Temperature [K]. 
    f_rf: float
        RF field frequency [Hz]  
    cdict: dict of 2D arrays
        Collection of costant matrices used to speed up the simulation
    Methods
    ----------
    hamiltonian(cdict_N, pulse_params=None):
        Creates system Hamiltonian at a particular point of time
    lindbladian(rho_mod, const_dict_N, pulse_params=None):
        Calculates the right-hand side of the Lindblad equation
        at a particular point of time
    evolve(pulse, 
               rho_reference=None, is_fidelity=False, is_purity=False,
               track_qubits=None, are_Bloch_vectors=False,
                track_points_per_pulse=100):
        Performs system evolution under a given pulse. Outputs a dictionary of
        the specified tracking parameters
    fidelity(rho_reference):
        Calculates fidelity of a 2D matrix with respect to the reference one
    purity():
        Calculates purity (tr(rho^2)) of a 2D matrix rho
    track_subsystem(self, track_qubits, are_Bloch_vectors=False):
        Gives the specified system submatrices and Bloch vectors (if tracked)
  
    """
        
    def __init__(self, rho, N_0=None, sys_param_dict=None, time=0):
        """
        Initializes the SpinSys class object
        
        Parameters
        ----------
        rho : numpy 2D array
            Density matrix. Initialized with the value before pulse
            
        N_0 : int
            Maximal number of electrons in the system (could be bigger
            than the dimensions of rho) 
            
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
       
        time: float
            Point in time at which rho is defined
        
        Returns
        -------
        None.

        """
        
        self.rho = rho
        self.time = time
        
        if N_0 == None: 
            #assuming the current dimensionality of rho is maximum possible
            #throughout the system evolution
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
                #the step is needed to carefully deal with None variables
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
        Builds the system Hamiltonian at a particular point of time

        Parameters
        ----------
        const_dict_N: dict
            dictionary of constant matrices 2**N x 2**N
        pulse_params : dictionary
             dictionary of values of delta_g[i], 
             J[i], B_rf taken at a particular point of time


        Returns
        -------
        Numpy 2D array that represents Hamiltonian

        """
        if pulse_params is None:
            return 0
        else:

            N = int(log2(const_dict_N["x_sum"].shape[0])) 
            
            # building a list of delta_g_i values, i=1...N           
            delta_g=[0]*N
            
            delta_g_dict = {int(k[8:]):v for (k,v) in pulse_params.items() 
                     if k[:8]=="delta_g_"}      
                        #TODO could be necessary to update 8 to a diff. number
            for i in range(1, N+1):
                delta_g[i-1] = delta_g_dict.get(i, 0)
            
            # building a list of J_{i,i+1} values, i=1...N-1. For now, we  
            # assume that the qubits are numbered sequentially so that all 
            # nearest-neighbor exchange parameters could be indexed with
            # a single index i
                
                
            if N==1:
                J=[]
            else:
                J=[0]*(N-1)
                
                J_dict=  {int(k[2:]):v for (k,v) in pulse_params.items() 
                         if k[:2]=="J_"}
                for i in range(1, N):
                    J[i-1] = J_dict.get(i, 0)
            
            #Incorporating values of B_rf
                    
            B_rf = pulse_params.get("B_rf", 0)
            phi = pulse_params.get("phi", 0)
            
            ham = const_dict_N["z_sum_omega"].copy()
                #copy is in order not to modify the const_dict_N entries; 
                #the term z_sum_omega is always present in the Hamiltonian
            
            if delta_g != [0]*N:
                for k in range(N):
                    ham = ham + (cst.muB/(2*cst.hbar)*self.B_0*delta_g[k] *
                            const_dict_N["Zs"][k])
                
            if J != [0]*(N-1):
                for k in range(N-1):
                    ham = ham + J[k] * const_dict_N["J_sigma_products"][k][k+1]
            
            if B_rf != 0:
                ham = ham + const_dict_N["x_sum"]* B_rf * cos(phi)
                ham = ham + const_dict_N["y_sum"]* B_rf * sin(phi)
              
        return ham
    

    def lindbladian(self, rho_mod, const_dict_N, pulse_params=None):
        """
        Creates right-hand side of the Lindblad equation at a particular
        point of time

        Parameters
        ----------
        rho_mod: 1D array 
            intermediate density matrices the Runge-Kutta method is used for
        const_dict_N: dict
            dictionary of constant matrices 2**N x 2**N
        pulse_params: 1D array
            list of pulse parameters

        Returns
        -------
        2D array: the right-hand side of the Lindblad equation

        """
        N = int(log2(const_dict_N["x_sum"].shape[0])) 
            #automatically calculating the size
        
        ham = np.zeros((2**N, 2**N), dtype=complex)
        if pulse_params is not None:
            ham  = ham + self.hamiltonian(const_dict_N, pulse_params)
        
        lin = ( 1j *( rho_mod @ ham - ham @ rho_mod) + const_dict_N["z_sum_p"] 
               - (2/self.T_1 + 1/(2*self.T_2)) * N * rho_mod )
        
        for k in range(N):
            lin  = lin + (  p(self.B_0, self.T) / self.T_1 * 
                    const_dict_N["sigma_pluses"][k] @ 
                    rho_mod.dot(const_dict_N["sigma_minuses"][k]) +  
                    (1 - p(self.B_0, self.T)) / self.T_1 * 
                    const_dict_N["sigma_minuses"][k] @
                    rho_mod.dot(const_dict_N["sigma_pluses"][k]))
 
        return lin
        
    
    def evolve(self, pulses, 
               rho_reference=None, is_fidelity=False, is_purity=False,
               track_qubits=None, are_Bloch_vectors=False,
                track_points_per_pulse=100
               ):
        """
        Function that performs spin system evolution under the external pulse,
        i.e. updates the system density matrix

        Parameters
        ----------
        pulses : a ControlPulse object, or tuple/list of such objects
            contains pulse parameters at different points of time 
                 
        rho_reference: 2D array
            The density matrix (initial, anticipated final, etc.) to compare 
            with the density matrix during the simulation
            
        is_fidelity : bool
            Indicates whether to track the fidelity of the density matrix with 
            respect to the specified rho_reference 
            
        is_purity : bool
            Indicates whether to track the purity
            of the density matrix 
            
        track_qubits : int / iterable of ints
            Number(s) of qubit(s) to track during the pulses 
            (i. e. evaluate density submatrices for them)
            
        are_Bloch_vectors: bool
            Indicates whether to calculate Bloch vectors for the tracked 
            qubits along with their density submatrices
            
        track_points_per_pulse: int
            Number of points of time during the system evolution to save
            parameters indicated above 
            
        Returns
        -------
        dictionary containing values of fidelity, purity and one-qubit 
        submatrices (if they are tracked) as 1D arrays

        """
        ret_dict = {}    #dictionary to be returned 
              
        #calculating the values at the beginning of the evolution
        if is_fidelity:
            if rho_reference is not None:
                rho_ref = rho_reference.copy()
                    #copied just in case to keep the 
                    #passed reference matrix unchanged
                ret_dict["fidelity"]=[self.fidelity(rho_ref)]
            else:
                print("The reference matrix is not specified. "
                      "Fidelity has not been calculated")
                    
        if is_purity:
            ret_dict["purity"] = [self.purity()]
            
        if track_qubits is not None:
            track_dict = self.track_subsystem(
                track_qubits, are_Bloch_vectors)
            track_keys = track_dict.keys()
            ret_dict.update({k:[track_dict[k]] for k in track_keys})
        elif are_Bloch_vectors:
            warnings.warn("Qubits are not specified; Bloch vectors are not tracked")
            
        ret_dict["time"] = [self.time]
                     
        N = int(log2(self.rho.shape[0]))
        cdict_N = self.cdict[N-1]
      
        # nested function that deals with the evolution of a single pulse,
        # or recursively calls each single pulse one after another
        def nested_evolve(pulse, retdict):
            #evolution when a single pulse is specified
            
            if isinstance(pulse, ControlPulse):
                                
                si_pulse_length = pulse.length*1.0e-12  
                    # to comply with Brandon's code which uses picoseconds
                
                #number of values of delta_g, B_rf, etc contained in the pulse.
                #To calculate it, a control parameter (B_rf, etc)
                #is chosen randomly 
                
                num_values = np.shape(pulse.ctrl_pulses
                                      [list(pulse.ctrl_names)[0]])[0]
                
                if si_pulse_length==0:       
                    delta_t=self.T_2/100   #TODO change to an appropriate 
                                            #number; 100 is chosen randomly
                else:
                    delta_t = 1.0* si_pulse_length /  (num_values-1)
                    
                #checking if dimensions of arrays in pulse are consistent
                dim_correct=True
                for name in pulse.ctrl_names:
                    dim_correct &= ( si_pulse_length / 
                                    (np.shape(pulse.ctrl_pulses[name])[0]-1) == 
                                    delta_t)
                
                if not dim_correct:
                    raise ValueError("Inconsistent dimensions of the pulse "
                                     "entry. Please try again")
        
                else:
                                       
                    for n in range(1, num_values):
                        #Runge-Kutta method
                        prev_pulse = {k:v[n-1] for (k,v) 
                                       in pulse.ctrl_pulses.items()}
                        cur_pulse={k:v[n] for (k,v) 
                                       in pulse.ctrl_pulses.items()}
                        avg_pulse={k:(0.5*(v[n-1]+v[n])) for (k,v) 
                                       in pulse.ctrl_pulses.items()}
                        
                        K1 = self.lindbladian(self.rho, cdict_N, prev_pulse)
                        K2 = self.lindbladian(self.rho + 0.5*delta_t*K1, 
                                                  cdict_N, avg_pulse )
                        K3 = self.lindbladian(self.rho + 0.5*delta_t*K2, 
                                                  cdict_N, avg_pulse )
                        K4 = self.lindbladian(self.rho + delta_t*K3, 
                                                  cdict_N, cur_pulse)
                        #updating density matrix
                        
                        self.rho = self.rho + delta_t/6*(K1+2*K2+2*K3+K4)
            #______________________________________________________________
                        # defining when to write values of fidelity, etc.
                        
                        # the following formulas are correct for the case
                        # track_points_per_pulse > 3
                        if (num_values-1) % (track_points_per_pulse-1) == 0:
                            track_step = max(int((num_values-1) /  
                                                (track_points_per_pulse-1)), 1)
                        else:
                            track_step = max(int((num_values-1) /  
                                                (track_points_per_pulse-2)), 1)
                        
                        #writing values in the output dictionary
                        if n % track_step == 0 or n == num_values-1:
                            #update time point
                            retdict["time"].append(self.time + n*delta_t)
          
                            #implementing tracking:
                            if track_qubits is not None:
                                for key in track_keys:
                                    retdict[key].append(self.track_subsystem( 
                                        track_qubits, are_Bloch_vectors)[key])  
                            if is_fidelity and rho_reference is not None:    
                                retdict["fidelity"].append(
                                    self.fidelity(rho_ref))
                            if is_purity:
                                retdict["purity"].append(self.purity())
        
                #updating the time attribute after the pulse
                self.time += si_pulse_length
            #recursive call of pulses embedded in the "pulse" iterable
            elif (isinstance(pulse, (list,tuple))):
                for p in pulse:
                    nested_evolve(p, retdict)
            
            else:
                raise ValueError("The pulse input format is incorrect."
                                 " Please try again")
        
        nested_evolve(pulses, ret_dict)
 
        #making all lists 1D arrays
        return {key:np.array(val) for key, val in ret_dict.items()}


# functions called when optional parameters are specified as True

    def fidelity(self, rho_reference):
        """
        Calculates fidelity of the system density matrix with respect to 
        the reference one
    
        Parameters
        ----------
        rho_expected : 2D array
            The density matrix (initial, anticipated final, etc.) to compare 
            with the density matrix during the simulation
    
        Returns
        -------
        Float value of fidelity defined in simulator_plan document
    
        """
        
        return np.real(np.trace(mth.matrix_sqrt(
                    mth.matrix_sqrt(rho_reference) 
                             @ self.rho @ mth.matrix_sqrt(rho_reference)))**2)
        
    
    def purity(self):
        """
        Calculates purity tr(rho^2)
    
        Returns
        -------
        Float value of density matrix purity
    
        """
        return np.real(np.trace( self.rho @  self.rho))
        

                
    
    def track_subsystem(self, track_qubits=None, are_Bloch_vectors=False):
        """
        Gives the specified system submatrices and Bloch vectors (if tracked)

        Parameters
        ----------
        pulse : dict
            Parameters of a (single) pulse
        track_qubits : int/iterable of ints
            Defines the qubits whose density submatrices are tracked
        are_Bloch_vectors: bool 
            Indicates whether to calculate Bloch vector component(s)
        Returns
        -------
        dictionary containing subsystem density matrices, and their 
        Bloch vectors if specified

        """
        ret_dict = {}
        
        N = int(log2(self.rho.shape[0]))  
            #automatically calculating the size
        
        Nset = set(range(1,N+1))
        trqub = set()   # a set of tracked qubits; using set automatically 
                        # discard possible repetitions
        
        ifint = isinstance(track_qubits, int)    #track a single qubit
        ifiterable = (isinstance(track_qubits, (tuple,list, set))
              and prod(isinstance(val, int) for val in track_qubits))
                #track multiple qubits
        
        if not (ifint or ifiterable):
            raise ValueError("The tracked qubits should be properly specified"  
                             "by an int or an iterable of ints. None of the"  
                                 "qubits has been tracked")
        else:
            if ifint:
                trqub = {track_qubits}
            if ifiterable :
                trqub = set(track_qubits) 

            for qub in trqub:
                submatrix = mth.partial_trace(self.rho, (Nset-{qub}))
                subm = "submatrix_{}".format(qub)
                ret_dict[subm] = submatrix
                if are_Bloch_vectors:
                    ret_dict["sigma_x_{}".format(qub)] = np.trace(submatrix 
                                                                @ matr.PAULI_X)
                    ret_dict["sigma_y_{}".format(qub)] = np.trace(submatrix 
                                                                @ matr.PAULI_Y)
                    ret_dict["sigma_z_{}".format(qub)] = np.trace(submatrix 
                                                                @ matr.PAULI_Z)
                
        return ret_dict
            
            
        