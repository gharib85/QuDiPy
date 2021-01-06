"""
Hamiltonian generalization class

author: aaranyaalex
"""
import numpy as np
import qudipy.potential as pot
from qudipy.qutils.solvers import build_1DSE_hamiltonian, build_2DSE_hamiltonian
from scipy.sparse.linalg import eigs
from types import SimpleNamespace

class HamGen:

    def ham_interp(self, i_params):
        params = self.extract_dict(i_params)
        compat_types = ['RealSpace', 'effOrbital', 'effSpin']
        compat_params = [['PotInterp', 'V'], ['TC', 'Eps', 'OSplit'], ['TC', 'Eps', 'VPhase', 'VSplit', 'Ez']]

        if not hasattr(params, 'HamType'):
            # Assign a HamType based on the available parameter attributes
            params.HamType = [compat_types[compat_params.index(row)] for row in compat_params
                              if(set(row).issubset(set(params.__dict__)))]
            # Raise error if no HamType can be formed
            if not params.HamType:
                raise ValueError('Supplied parameters are insufficient to form supported Hamiltonians')
        else:
            if 'RealSpace' in params.HamType:
                potential = params.PotInterp(params.V)
                gparams = pot.GridParameters(params.PotInterp.xcoords, potential=potential)
                if gparams.grid_type == '1D':
                    H_RS = build_1DSE_hamiltonian(params.PotInterp.constants, gparams)
                elif gparams.grid_type == '2D':
                    H_RS = build_2DSE_hamiltonian(params.PotInterp.constants, gparams)

            if 'effOrbital' in params.HamType:
                n_dots = min(len(params.TC)+1, len(params.Eps), len(params.OSplit))

                if n_dots < max(len(params.TC)+1, len(params.Eps), len(params.OSplit)):
                    params.TC = params.TC[:n_dots-1]
                    params.Eps = params.Eps[:n_dots]
                    params.OSplit = params.OSplit[:n_dots]
                    print(f'Supplied data inconsistent, calculating H for a {n_dots}QD-system')

                # form diagonals from parameters
                d_0 = np.insert(np.array(params.OSplit)+np.array(params.Eps), slice(None, None, 1), params.Eps)
                d_1 = np.insert(np.array(params.TC), slice(None, None, 1), np.zeros(n_dots+1))
                d_2 = np.repeat(np.array(params.TC), 2)
                d_3 = np.insert(np.zeros(n_dots-1), slice(None, None, 1), np.array(params.TC))

                effH_O = np.diag(d_0) + np.diag(d_1, 1) + np.diag(d_1, -1) + np.diag(d_2, 2) + np.diag(d_2, -2) + \
                         np.diag(d_3, 3) + np.diag(d_3, -3)

            if 'effSpin' in params.HamType:

            else: # throw error, inputted type not compatible

        return H_RS, effH_O, effH_S

    def extract_dict(self, params):
        # compatible parameters to create hamiltonians
        compat = ['V', 'PotInterp', 'HamType', 'TC', 'Eps', 'OSplit', 'VPhase', 'VSplit', 'Ez']
        # check if parameters in dictionary are formatted properly
        if all(item in list(params.keys()) for item in compat):
            # send dictionary elements to variable
            extracted_data = SimpleNamespace(**params)
        else:
            # show user the improperly formatted keys
            invalid_keys = np.setdiff1d(list(params.keys()), compat)
            raise ValueError(f'Supplied dictionary keys {invalid_keys} are ' +
                             'invalid.\nCompatible keys are:\n' +
                             f'{compat}')
        return extracted_data

    def eigs(self, nsols=1):
        e_ens, e_vecs = eigs(self.ham, k=nsols, sigma=self.gparams.potential.min())
        idx = e_ens.argsort()
        e_ens = e_ens[idx]
        e_vecs = e_vecs[:, idx]

        return e_ens, e_vecs

    def expectation_vals(self, wf1, wf2):
        wf1 = wf1.reshape(1, -1).conj()
        wf2 = wf2.reshape(-1, 1)

        E = np.dot(wf1, np.dot(self.ham, wf2))

        return E