'''
File used to generate and plot charge stability diagrams from provided potentials using the Hubbrad model
For more information about the method, see the references https://doi.org/10.1103/PhysRevB.83.235314 and https://doi.org/10.1103/PhysRevB.83.161301
'''

import math
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
from scipy.ndimage import gaussian_filter

class HubbardCSD:
    '''
    Initialize the charge stability diagram class which generates charge stability diagrams based on given capacitance parameters.
    Based on section III in https://doi.org/10.1103/PhysRevB.83.235314.
    This class is intended for use with NextNano potentials to simulate charge stability diagrams of various designs, but can also be used with analytic potentials.
    '''
    def __init__(self, n_sites, n_e, h_mu=False, h_t=False, h_u=False):
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

        # Save which parts of Hamiltonian to be used for later
        self.h_mu = h_mu
        self.h_t = h_t
        self.h_u = h_u

        if self.h_mu is False:
            raise Exception("Hamiltonian will be independent of gate volatges so no charge stability diagram can be generated")

        # Check that number of electrons doesn't exceed twice the amount of sites
        if n_e >= 2 * n_sites:
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

    def _generate_basis(self):

        # Compute all possible occupations with the cartesian product, and then 
        # remove all states that exceed the number of 
        all_combos = list(itertools.product(*[[0,1] for i in range(self.n_sites * 2)])) # * 2 is for spin degeneracy (could add anouth *2 for valleys)

        basis = []
        for combo in all_combos:
            if sum(combo) <= self.n_e:
                basis.append(combo)

        # Labels each index in the basis state with site number and spin direction (could add valley states)
        site = [f'site_{n+1}' for n in range(self.n_sites)]
        spin = ['spin_up', 'spin_down']
        basis_labels = list(itertools.product(site, spin))

        return basis, basis_labels


    def _generate_h_t(self):
        pass

    def _generate_h_u(self):
        pass


    def _lowest_energy(self, v_g1, v_g2):
        pass

    def load_potential(self, file_path):
        # TODO
            pass
