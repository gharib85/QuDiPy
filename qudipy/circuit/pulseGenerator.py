"""
Constant-adiabaticity pulse generator

Reference: B. Buonacorsi, B. Shaw, J. Baugh. (2020) doi.org/10.1103/PhysRevB.102.125406

@author: aaranyaalex
"""

import numpy as np
import matplotlib.pyplot as plt
from qudipy.qutils.solvers import solve_schrodinger_eq
from scipy.optimize import fminbound
from qudipy.qutils.math import inner_prod
import qudipy.potential as pot
import qudipy as qd
from tqdm import tqdm
from bisect import insort


class PulseGen:

    def __init__(self, ctrl_pulse, pot_interp):
        self.init_ctrl = ctrl_pulse.copy()
        self.init_ctrl.ctrl_time = None
        self.init_ctrl.set_pulse_length(1.0)
        self.init_ctrl._generate_ctrl_interpolators()
        self.pot_interp = pot_interp

    def optimize_ap(self, target_ap, n=0, n_pts=500, di=1E-6, m=list(np.arange(5))):
        """
        Method for generating a control pulse sequence with constant adiabaticity

        Parameters
        ----------
        target_ap : float
            Desired value of approximate adiabatic parameter to be satisfied by the control pulse.

        Keyword Arguments
        -----------------
        n : int, optional
            Target state for the evolution, optimized for constant adiabaticity.
            Default is the ground state, n=0.

        n_pts : int, optional
            Amount of interpolated control pulse points to be optimized to fit the adiabatic parameter. The
            ControlPulse object "self.init_ctrl" is interpolated to have n points.
            Default is 500.

        di : float, optional
            Small shift of interpolated pulse index. Used to calculated shifted pulses and approximate ground-state
            derivative with a 5 point stencil.
            Default is 1E-6.

        m : list, optional
            Sorted list of integers representing the accessible states wished to be used to optimize the adiabatic
            parameter.
            Default is [0, 1, 2, 3, 4]

        Returns
        -------
            None. 2D array of time points (in seconds) and control pulse values stored in self.adia_pulse.

        """

        consts = self.pot_interp.constants
        X = self.pot_interp.x_coords
        # indices for interpolated control pulses
        indices = np.linspace(0, 1, num=n_pts)
        # list of accessible states should contain target state
        if n not in m:
            insort(m, n)

        volt_vec = np.zeros((n_pts, self.init_ctrl.n_ctrls), dtype=complex)
        state = np.zeros((len(X), 5), dtype=complex)
        energies =np.zeros((len(m), 5), dtype=complex)
        wfns = np.zeros((len(X), len(m), 5), dtype=complex)
        di_dt = np.zeros((n_pts - 1), dtype=float)
        times = np.zeros(n_pts)

        def adiabatic(log_di_dt, psi, psi_target, e_ens, ap, n, m, gparams):
            """
            Helper function to calculate approximate adiabatic parameter. Is optimized to satisfy the target value.

            Parameters
            ----------
            log_di_dt : float
                Derivative of control pulse index with respect to control pulse time, value to be optimized.
                Logarithmic transformation is used for ease of convergence.

            psi : 2D complex array
                Array of wavefunctions (eigenvectors) for all accessible states at a given control pulse index.
                Size: x by length(m)
                    x --> x-coordinates of wavefunction
                    m --> list of specified, accessible excited states

            psi_target : 2D complex array
                Derivative of target state wavefunction with respect to a control pulse index. Approximated with
                five point stencil.
                Size: x by 1

            e_ens: 1D vector
                Eigenenergies of corresponding accessible states. e_ens[m] corresponds to psi[:, m]
                Size: length(m)

            ap: float
                Desired adiabatic parameter.

            n: integer
                Target state for evolution of optimization.

            m: integer list
                List corresponding to solved accessible states, includes target state (n)

            gparams: GridParameters Class object
                Contains grid and potential information of wavefunctions.

            Returns
            -------
            xi - ap : float
                Difference between actual adiabatic parameter for parameter set, and desired value.
            """

            xi = 0
            # log transformation
            didt = np.array([10], dtype=float) ** log_di_dt

            # iterate over all excited states, except target state, n
            n_idx = m.index(n)
            for num_state in range(len(m)):
                if num_state != n_idx:
                    ip = inner_prod(gparams, psi[:, num_state], didt * psi_target)
                    xi += consts.hbar * np.abs(ip / (e_ens[n_idx] - e_ens[num_state]))

            return np.abs(xi - ap)

        # for all interpolated voltage configurations
        for i in tqdm(range(n_pts-1)):
            curr_time_idx = (i+1)/n_pts

            # 5 point stencil for dpsi0/di
            for pt, shift in enumerate([-2*di, -di, 0, di, 2*di]):
                idx_point = curr_time_idx + shift
                # calculate eigenstates for point in stencil
                int_pot = self.pot_interp(self.init_ctrl([idx_point]))
                gparams = pot.GridParameters(X, potential=int_pot)
                e_ens, e_vecs = solve_schrodinger_eq(consts, gparams, n_sols=max(m)+1)
                # store wf and energies only for specified accessible states
                state[:, pt] = e_vecs[:, n]
                energies[:, pt] = np.real(e_ens[m])
                wfns[:, :, pt] = e_vecs[:, m]

            # Fix phases of target states for the stencil to all have the same global phase
            for j in [0, 1, 3, 4]:
                if inner_prod(gparams, state[:, j], state[:, 2]) < 0:
                    state[:, j] = -state[:, j]

            # compute derivative
            dpsi0_di = (state[:, 0] - 8 * state[:, 1] + 8 * state[:, 3] - state[:, 4]) / (12 * di)
            # optimize di/dt to achieve adiabatic parameter
            di_dt[i] = fminbound(adiabatic, -20, 20, args=(wfns[:, :, 2], dpsi0_di, energies[:, 2], target_ap, n, m,
                                                           gparams), xtol=1E-6)

        # calculate new control pulse time indices
        times[1:] = np.cumsum(di/(10**di_dt))  # in seconds
