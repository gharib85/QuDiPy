'''
File used to generate and plot charge stability diagrams from provided potentials using the Hubbrad model
For more information about the method, see the references https://doi.org/10.1103/PhysRevB.83.235314 and https://doi.org/10.1103/PhysRevB.83.161301
'''

import copy
import numpy as np
import pandas as pd
import itertools
from scipy import linalg as la
from ..utils.constants import Constants

class HubbardCSD:
    '''
    Initialize the charge stability diagram class which generates charge stability diagrams based on given capacitance parameters.
    Based on section III in https://doi.org/10.1103/PhysRevB.83.235314.
    This class is intended for use with NextNano potentials to simulate charge stability diagrams of various designs, but can also be used with analytic potentials.
    '''
    def __init__(self, n_sites, n_e, h_mu=False, h_t=False, h_u=False, **kwargs):
        '''

        Parameters
        ----------
        n_sites: Number of sites that a hamilt
        n_e: Maximum number of electrons in the system. Must be less than or equal to 2*n_sites

        Keyword Arguments
        -----------------
        h_mu: Whether to include chemical potential term in Hamiltonian when Hamiltonian is created (default False)
        h_t: Whether to include tunnel coupling term in Hamiltonian when Hamiltonian is created (default False)
        h_u: Whether to include coulomb repulsion repulsion in Hamiltoninan when hamiltonian is created (default False)

        Returns
        -------
        None
        '''


        const = Constants()
        self.e = const.e
        # Save which parts of Hamiltonian to be used for later
        self.h_mu = h_mu
        self.h_t = h_t
        self.h_u = h_u

        for key, item in kwargs.items():
            self.__setattr__(key, item)

        if self.h_mu is False:
            raise Exception("Hamiltonian will be independent of gate volatges so no charge stability diagram can be generated")

        # Check that number of electrons doesn't exceed twice the amount of sites
        if n_e > 2 * n_sites:
            raise Exception(f"Number of electrons ({n_e}) exceeds twice the amount of sites ({n_sites}) allowed")
        else:
            self.n_e = n_e
            self.n_sites = n_sites

        # Generates the basis to be used
        self._generate_basis()

        # These next steps generate the fixed portion of the Hamiltonian, which is created on initialization since it is independent of voltage

        # First, generate the matrix of the correct size
        self.fixed_hamiltonian = np.zeros((len(self.basis),len(self.basis)))

        # Then add the component to the fixed portion of the Hamiltonian that you want to consider
        if h_t is True:
            self.fixed_hamiltonian += self._generate_h_t()

        if h_u is True:
            self.fixed_hamiltonian += self._generate_h_u()

    def generate_csd(self, v_g1_max, v_g2_max, v_g1_min=0, v_g2_min=0, num=100):
        '''Generates the charge stability diagram between v_g1(2)_min and v_g1(2)_max with num by num data points in 2D

        Parameters
        ----------
        v_g1_max: maximum voltage on plunger gate 1
        v_g2_max: maximum voltage on plunger gate 2

        Keyword Arguments
        -----------------
        v_g1_max: minimum voltage on plunger gate 1 (default 0)
        v_g2_max: minimum voltage on plunger gate 2 (default 0)
        c_cs_1: coupling between charge sensor and dot 1 (default to None)
        c_cs_2: coupling between charge sensor and dot 2 (default to None)
        num: number of voltage point in 1d, which leads to a num^2 charge stability diagram (default 100)
        plotting: flag indicating whether charge stability diagram should be plotted after completion (default False)

        Returns
        -------
        None
        '''

        # Stores parameters for late
        self.num = num
        self.v_g1_min = v_g1_min
        self.v_g1_max = v_g1_max
        self.v_g2_min = v_g2_min
        self.v_g2_max = v_g2_max

        # Generate voltage points to sweep over
        self.v_1_values = [round(self.v_g1_min + i/num * (v_g1_max - self.v_g1_min), 6) for i in range(num)]
        self.v_2_values = [round(self.v_g2_min + j/num * (v_g2_max - self.v_g2_min), 6) for j in range(num)]

        # Loop over all voltage point combinations in list comprehension
        occupation = [[[self._lowest_energy(v_1, v_2)] for v_1 in self.v_1_values] for v_2 in self.v_2_values]

        # Create a num by num DataFrame from occupation data information as entries
        self.occupation = pd.DataFrame(occupation, index=self.v_1_values, columns=self.v_2_values)

    def _lowest_energy(self, v_1, v_2):
        mu_1, mu_2 = self._volt_to_chem_pot(v_1, v_2)
        h_mu = np.zeros(self.fixed_hamiltonian.shape)

        for i in range(self.fixed_hamiltonian.shape[0]):
            state_1 = self.basis[i]

            result = 0
            for k in range(len(self.basis_labels)):
                if k == 0 or k == 1:
                    result += - mu_1 * self._inner_product(state_1, self._number(state_1, k))
                if k == 2 or k == 3:
                    result += - mu_2 * self._inner_product(state_1, self._number(state_1, k))

            h_mu[i][i] = result

        current_hamiltonian = self.fixed_hamiltonian + h_mu
        eigenvals, eigenvects = la.eig(current_hamiltonian)
        eigenvects = np.transpose(eigenvects) # To get column eigenvectors not column entries
        eigenvals = np.real(eigenvals) # Needs to be cast to real (even though Hamiltonian is Hermitian so eigenvalues are real)
        lowest_eigenvect = np.squeeze(eigenvects[np.argmin(eigenvals)])
        lowest_eigenvect = lowest_eigenvect/la.norm(lowest_eigenvect)
        lowest_eigenvect_prob = np.real(lowest_eigenvect * np.conj(lowest_eigenvect))
        occupation_list = []
        for i in range(self.n_sites):
            occupation_list.append((lowest_eigenvect_prob * getattr(self, 'basis_occupation_' + str(i))).sum())
        return tuple(occupation_list)

    def _generate_basis(self):
        '''Creates the basis of all possible states given the constraints on the number of sites and number of electrons

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Compute all possible occupations with the cartesian product, and then
        # remove all states that exceed the number of electron specified
        # TODO make this more efficient so we only generate the states we want
        all_combos = list(itertools.product(*[[0,1] for i in range(self.n_sites * 2)])) # * 2 is for spin degeneracy (could add another *2 for valleys)

        basis = []
        for combo in all_combos:
            if sum(combo) <= self.n_e:
                basis.append(list(combo))

        # Labels each index in the basis state with site number and spin direction (could add valley states)
        self.sites = [f'site_{n+1}' for n in range(self.n_sites)]
        self.spins = ['spin_up', 'spin_down']
        for i in range(self.n_sites):
            j = 2*i
            setattr(self, 'basis_occupation_' + str(i), np.array([sum(x[j:j+2]) for x in basis]))
        basis_labels = list(itertools.product(self.sites, self.spins))

        self.basis = basis
        self.basis_labels = basis_labels

        return

    def _generate_h_t(self):
        h_t = np.zeros(self.fixed_hamiltonian.shape)
        for i in range(self.fixed_hamiltonian.shape[0]):
            for j in range(i):
                state_1 = self.basis[i]
                state_2 = self.basis[j]

                if sum(state_1) != sum(state_2):
                    continue # No tunnel coupling between states with different charge occupation
                
                if sum(state_1[::2]) != sum(state_2[::2]):
                    continue # No tunnel coupling between states with different number of spins in each orientation

                result = 0
                for k in range(len(self.basis_labels)):
                    for l in range(k):
                        result += self.t_12 * self._inner_product(state_1, self._create(self._annihilate(state_2, k), l))
                        result += self.t_12 * self._inner_product(state_1, self._create(self._annihilate(state_2, l), k))

                h_t[i][j] = -result
                h_t[j][i] = -result #Since matrix is symmetric
        return h_t

    def _generate_h_u(self):
        h_u = np.zeros(self.fixed_hamiltonian.shape)
        for i in range(self.fixed_hamiltonian.shape[0]):
                state_1 = self.basis[i]

                result = 0
                for k in range(len(self.basis_labels)):
                    for l in range(k):
                        for j in range(self.n_sites):
                            if self.basis_labels[k][0] == 'site_' + str(j+1) and self.basis_labels[l][0] == 'site_' + str(j+1): # check if electrons are on same site
                                result += getattr(self, 'U_' + str(j+1) + str(j+1)) * self._inner_product(state_1, self._number(self._number(state_1, k), l))
                                break
                        else:
                            for j in range(self.n_sites):
                                for m in range(j):
                                    result += getattr(self, 'U_' + str(m+1) + str(j+1)) * self._inner_product(state_1, self._number(self._number(state_1, k), l))
                                    break
                                else:
                                    continue
                                break


                h_u[i][i] = result
        return h_u

    def _volt_to_chem_pot(self, v_1, v_2):
        '''
        Converts from supplied voltages to chemical potential for 2 coupled dots, assuming the capacitance model is valid.
        Requires self charging and coulomb repulsion energy terms to be loaded already

        Parameters
        ----------
        v_1: voltage on plunger gate 1
        v_2: voltage on plunger gate 2

        Keyword Arguments
        -----------------
        None

        Returns
        -------
        Chemical potentials mu_1 and m_2 on site 1 and site 2 respectively
        '''
        alpha_1 = ((self.U_22 - self.U_12) * self.U_11) / (self.U_11 * self.U_22 - self.U_12**2)
        alpha_2 = ((self.U_11 - self.U_12) * self.U_22) / (self.U_11 * self.U_22 - self.U_12**2)
        mu_1 = (alpha_1 * v_1 + (1 - alpha_1) * v_2)
        mu_2 = ((1 - alpha_2) * v_1 + alpha_2 * v_2)
        return mu_1, mu_2

    def _inner_product(self, state_1, state_2):
        '''
        Computes the inner product
        '''
        if state_1==None or state_2==None: # Deals with cases where the coefficient of state is 0, so the inner product is multiplied by 0
            return 0
        elif state_1 == state_2:
            return 1
        else:
            return 0

    def _create(self, state, position):
        '''
        docstring
        '''
        state = copy.copy(state)
        if state == None:
            pass # keep state as None
        elif state[position] == 0:
            state[position] = 1
        else:
            state = None
        return state

    def _annihilate(self, state, position):
        '''
        docstring
        '''
        state = copy.copy(state)
        if state == None:
            pass
        elif state[position] == 1:
            state[position] = 0
        else:
            state = None
        return state

    def _number(self, state, position):
        '''
        docstring
        '''
        return self._create(self._annihilate(state, position), position)

    def load_potential(self, file_path):
        # TODO
        pass
