'''
File used to generate and plot charge stability diagrams from provided potentials using the Hubbrad model
For more information about the method, see the references https://doi.org/10.1103/PhysRevB.83.235314 and https://doi.org/10.1103/PhysRevB.83.161301
'''

import math
import sys
import copy
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import itertools
from scipy import linalg as la
from scipy.ndimage import gaussian_filter
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
        self.basis, self.basis_labels = self._generate_basis()

        # These next steps generate the fixed portion of the Hamiltonian, which is created on initialization

        # First, generate the matrix of the correct size
        self.fixed_hamiltonian = np.zeros((len(self.basis),len(self.basis)))

        # Then add the component to the fixed portion of the Hamiltonian that you want to consider
        if h_t is True:
            self.fixed_hamiltonian += self._generate_h_t()

        if h_u is True:
            self.fixed_hamiltonian += self._generate_h_u()

    def generate_csd(self, v_g1_max, v_g2_max, v_g1_min=0, v_g2_min=0, c_cs_1=None, c_cs_2=None, num=100, plotting=False, blur=False, blur_sigma=0):

        self.num = num
        self.v_g1_min = v_g1_min
        self.v_g1_max = v_g1_max
        self.v_g2_min = v_g2_min
        self.v_g2_max = v_g2_max

        self.v_1_values = [round(self.v_g1_min + i/num * (v_g1_max - self.v_g1_min), 4) for i in range(num)]
        self.v_2_values = [round(self.v_g2_min + j/num * (v_g2_max - self.v_g2_min), 4) for j in range(num)]

        data = []
        for v_1 in self.v_1_values:
            for v_2 in self.v_2_values:
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
                eigenvals = np.real(eigenvals) # Needs to be cast to real (even though Hamiltonian is Hermitian so eigenvalues are real)
                lowest_eigenvect = np.squeeze(eigenvects[np.argmin(eigenvals)])
                lowest_eigenvect = lowest_eigenvect/la.norm(lowest_eigenvect)
                lowest_eigenvect_prob = lowest_eigenvect * np.conj(lowest_eigenvect)
                # e_vals = []
                # for i in range
                occupation_1 = (lowest_eigenvect_prob * self.basis_occupation_1).sum()
                occupation_2 = (lowest_eigenvect_prob * self.basis_occupation_2).sum()
                current = occupation_1 * c_cs_1 + occupation_2 * c_cs_2
                data.append([v_1, v_2, current])

        data = np.real(data) # Needs to be cast to real to avoid problems plotting
        df = pd.DataFrame(data, columns=['V_g1', 'V_g2', 'Current'])
        self.csd = df.pivot_table(index='V_g1', columns='V_g2', values='Current')

        # Show plots if flag is set to True
        if plotting is True:

            # Toggles colorbar if charge sensor information is given
            cbar_flag = False
            if (c_cs_1 is not None) and (c_cs_2 is not None):
                cbar_flag = True

            # Plot the chagre stability diagram
            p1 = sb.heatmap(self.csd, cbar=cbar_flag, xticklabels=int(
                num/5), yticklabels=int(num/5), cbar_kws={'label': 'Current (arb.)'})
            p1.axes.invert_yaxis()
            p1.set(xlabel=r'V$_2$', ylabel=r'V$_1$')
            plt.show()

            # # Plot the "derivative" of the charge stability diagram
            # p2 = sb.heatmap(df_der, cbar=cbar_flag, xticklabels=int(
            # num/5), yticklabels=int(num/5))
            # p2.axes.invert_yaxis()
            # p2.set(xlabel=r'V$_1$', ylabel=r'V$_2$')
            # plt.show()

    def _generate_basis(self):

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
        self.basis_occupation_1 = np.array([sum(x[:2]) for x in basis])
        self.basis_occupation_2 = np.array([sum(x[2:4]) for x in basis])
        basis_labels = list(itertools.product(self.sites, self.spins))

        return basis, basis_labels

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
                term_1 = 0
                term_2 = 0
                for k in range(len(self.basis_labels)):
                    for l in range(k):
                        term_1 += self.t * self._inner_product(state_1, self._create(self._annihilate(state_2, k), l))
                        term_2 += self.t * self._inner_product(state_1, self._create(self._annihilate(state_2, l), k))

                result += term_1 + term_2
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
                        if self.basis_labels[k][0] == 'site_1' and self.basis_labels[l][0] == 'site_1': # check if electrons are on same site
                            result += self.U_1 * self._inner_product(state_1, self._number(self._number(state_1, k), l))
                        elif self.basis_labels[k][0] == 'site_2' and self.basis_labels[l][0] == 'site_2': # check if electrons are on same site
                            result += self.U_2 * self._inner_product(state_1, self._number(self._number(state_1, k), l))
                        else:
                            result += self.U_12 * self._inner_product(state_1, self._number(self._number(state_1, k), l))

                h_u[i][i] = result
        return h_u

    def _volt_to_chem_pot(self, v_1, v_2):
        alpha_1 = ((self.U_2 - self.U_12) * self.U_1) / (self.U_1 * self.U_2 - self.U_12**2)
        alpha_2 = ((self.U_1 - self.U_12) * self.U_2) / (self.U_1 * self.U_2 - self.U_12**2)
        mu_1 = (alpha_1 * v_1 + (1 - alpha_1) * v_2)
        mu_2 = ((1 - alpha_2) * v_1 + alpha_2 * v_2)
        return mu_1, mu_2

    def _inner_product(self, state_1, state_2):
        '''
        docstring
        '''
        if state_1==None or state_2==None: # Deals with cases where eigenvalue of number is 0, so the inner product is multiplied by 0
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
