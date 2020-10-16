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
from tqdm import tqdm


class PulseGen:

    def __init__(self, ctrl_pulse, pot_interp):
        self.init_ctrl = ctrl_pulse.copy()
        # convert pulse times to indices
        self.init_ctrl.ctrl_time = None
        self.init_ctrl.set_pulse_length(1.0)
        self.init_ctrl._generate_ctrl_interpolators()

        self.pot_interp = pot_interp
        self.adia_pulse = []
        self.m = 1

    def optimize_ap(self, desired_ap, n=500, di=1E-6, m=5, plot_ap=False):
        """
        Method for generating a control pulse sequence with constant adiabaticity

        Parameters
        ----------
        desired_ap : float
            Desired value of approximate adiabatic parameter to be satisfied by the control pulse.

        Keyword Arguments
        -----------------
        n : int, optional
            Amount of interpolated control pulse points to be optimized to fit the adiabatic parameter. The
            ControlPulse object "self.init_ctrl" is interpolated to have n points.
            Default is 500.

        di : float, optional
            Small shift of interpolated pulse index. Used to calculated shifted pulses and approximate ground-state
            derivative with a 5 point stencil.
            Default is 1E-6.

        m : int, optional
            Number of accessible excited states of the system.
            Default is 5.

        plot_ap : bool, optional
            If True, will generate a plot of the full constant-adiabaticity control sequence

        Returns
        -------
            None. 2D array of time points (in seconds) and control pulse values stored in self.adia_pulse.

        """

        consts = self.pot_interp.constants
        X = self.pot_interp.x_coords
        # indices for interpolated control pulses
        indices = np.linspace(0, 1, num=n)

        volt_vec = np.zeros((n, self.init_ctrl.n_ctrls), dtype=complex)
        grounds = np.zeros((len(X), 5), dtype=complex)
        energies =np.zeros((m, 5), dtype=complex)
        wfns = np.zeros((len(X), m, 5), dtype=complex)
        di_dt = np.zeros((n - 1), dtype=float)
        times = np.zeros(n)

        def adiabatic(log_di_dt, psi, psi_ground, e_ens, desired_ap, gparams):
            """
            Helper function to calculate approximate adiabatic parameter. Is optimized to satisfy the desired value.

            Parameters
            ----------
            log_di_dt : float
                Derivative of control pulse index with respect to control pulse time, value to be optimized.
                Logarithmic transformation is used for ease of convergence.
            psi : 2D complex array
                Array of wavefunctions (eigenvectors) for all accessible states at a given control pulse index.
                Size: x by m
                    x --> x-coordinates of wavefunction
                    m --> number of accessible excited states
            psi_ground : 2D complex array
                Derivative of ground state wavefunction with respect to a control pulse index. Approximated with
                five point stencil.
                Size: x by 1
            e_ens: 1D vector
                Eigemenergies of corresponding accessible states. e_ens[m] corresponds to psi[:, m]
                Size: m
            desired_ap: float
                Desired adiabatic parameter.
            gparams: GridParameters Class object
                Contains grid and potential information of wavefunctions.

            Returns
            -------
            xi - desired_ap : float
                Difference between actual adiabatic parameter for parameter set, and desired value.
            """

            xi = 0
            # log transformation
            didt = np.array([10], dtype=float) ** log_di_dt

            # iterate over all excited states, except ground
            for m in range(len(e_ens) - 2):
                ip = inner_prod(gparams, psi[:, m + 1], didt * psi_ground)
                xi += consts.hbar * np.abs(ip / (e_ens[0] - e_ens[m + 1]))

            return np.abs(xi - desired_ap)

        for idx, ctrl in enumerate(self.init_ctrl.ctrl_names):
            # generate array of interpolated control pulses
            volt_vec[:, idx] = self.init_ctrl.ctrl_interps[ctrl](indices)

        # for all interpolated voltage configurations
        for i in tqdm(range(n-1)):
            curr_time_idx = (i+1)/n

            # 5 point stencil for dpsi0/di
            for pt, shift in enumerate([-2*di, -di, 0, di, 2*di]):
                idx_point = curr_time_idx + shift
                # calculate eigenstates for point in stencil
                int_pot = self.pot_interp(self.init_ctrl([idx_point]))
                gparams = pot.GridParameters(X, potential=int_pot)
                e_ens, e_vecs = solve_schrodinger_eq(consts, gparams, n_sols=m)
                # store wf and energies for all accessible states
                grounds[:, pt] = e_vecs[:, 0]
                energies[:, pt] = np.real(e_ens)
                wfns[:, :, pt] = e_vecs

            # Fix phase of ground states shifted in stencil
            for j in [0, 1, 3, 4]:
                if inner_prod(gparams, grounds[:, j], grounds[:, 2]) < 0:
                    grounds[:, j] = -grounds[:, j]

            # compute derivative
            dpsi0_di = (grounds[:, 0] - 8 * grounds[:, 1] + 8 * grounds[:, 3] - grounds[:, 4]) / (12 * di)
            # optimize di/dt to achieve adiabatic parameter
            di_dt[i] = fminbound(adiabatic, -20, 20, args=(wfns[:, :, 2], dpsi0_di, energies[:, 2], desired_ap, gparams),
                                 xtol=1E-6)

        # calculate new control pulse time indices
        times[1:] = np.cumsum(di/(10**di_dt))  # in seconds
        self.adia_pulse = np.column_stack((times, volt_vec))

        if plot_ap:
            plt.plot(times/1e-12, volt_vec[:, 0], '-ro', markersize=2)
            plt.plot(times/1e-12, volt_vec[:, 1], '-bo', markersize=2)
            plt.plot(times/1e-12, volt_vec[:, 2], '-go', markersize=2)
            plt.xlabel('Time [ps]')
            plt.show()
