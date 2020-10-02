"""
Constant adiabaticity pulse generator

@author: aaranyaalex
"""

import numpy as np
import matplotlib.pyplot as plt
from qutils.solvers import solve_schrodinger_eq
import potential as pot
from scipy.interpolate import interp1d, fmin
from scipy.constants import hbar


class PulseGen:

    def __init__(self, ctrl_pulse, pot_interp, adia_param):
        self.init_ctrl = ctrl_pulse
        self.pot_interp = pot_interp
        self.desired_ap = adia_param

    def optimizeAP(self, pot_interp):
        # Get the x-coordinates
        X = pot_interp.x_coords
        consts = pot_interp.constants
        n_ctrls = self.init_ctrl.n_ctrls
        # interpolation of ctrl pulses to match length of pot_interp
        times = np.linspace(0, len(X)-1, num=len(X))  # arbitrary
        interp_ctrl = [interp1d(times, self.init_ctrl[:, i]) for i in range(n_ctrls)]

        # helper function for adiabaticity calculation
        def adiabatic(dt, psi, psi_ground, e_ens, desired_ap):
            xi = np.zeros(len(psi))
            for m in range(2*n_ctrls-1):
                xi += hbar * np.abs(np.divide(np.vdot(psi[m+1, :], dt * psi_ground), (e_ens[0, :] - e_ens[m+1, :])))
            return xi - desired_ap

        # loop through all interpolated voltage configs
        h = 1e-6  # in V
        coeff = [1, -8, +8, -1]  # for 5 pt stencil
        dpsi_dv = 0
        for i in range(n_ctrls):
            potential = [interp_ctrl[i](times) + (j * h) for j in [-2, -1, 0, 1, 2]]
            gparams = [pot.GridParameters(X, potential= potential[k]) for k in range(5)]
            # evecs same length has potinterp, m is the number of accessible states
            for s in range(5):
                if s == 2:
                    # center of 5 point stencil, store wf and energies for all accessible states
                    ce_ens, ce_vecs = solve_schrodinger_eq(consts, gparams[s], n_sols=2*n_ctrls-1)
                else:
                    # edges of stencil, only requires ground state
                    _, e_vecs = solve_schrodinger_eq(consts, gparams[s], n_sols=1)
                    dpsi_dv += e_vecs[:, 0] * coeff[s]

            # approximate the derivative of ground state
            dpsi_dv = dpsi_dv/(12*h)
            dvi_dt = np.gradient(potential[2], edge_order=2)
            # find optimal dv/dt to achieve adiabatic parameter, set current dv/dt as initial guess
            grad = fmin(adiabatic, dvi_dt, args=(ce_vecs, dpsi_dv, ce_ens, self.desired_ap))
            # change the time indices
            opt_times = times
            opt_times[1:] = times[:-1] + np.divide(np.diff(potential[2]), grad[1:])
