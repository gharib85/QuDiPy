## intiilize modules
import os, sys
## Add folder to path
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append("C:\\Users\\steph\\Documents\\GitHub\\QuDiPy")

import qudipy as qd
import qudipy.potential as pot
import qudipy.qutils as qt

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz

consts = qd.Constants("Si/SiO2")

# First define the x-coordinates
x = np.linspace(-70,70,301)*1E-9
# Define harmonic oscillator frequency
omega = 5E12
# Now construct the harmonic potential
harm_pot = 1/2*consts.me*omega**2*np.square(x-10e-9)
    
# Create a GridParameters object
gparams = pot.GridParameters(x, potential=harm_pot)

# Pass sparams, gparams to the solve_schrodinger_eq qutils method to obtain the eigenvalues and eigenvectors
e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=6)

# Get ground state wavefunction and cast to real (should already be real)
ground_state = np.real(e_vecs[:,0])

# Plot the wavefunction
plt.plot(x, ground_state)
plt.show()

# Plot the cumulative integral of the magnitude squared of the wavefunction
integral = cumtrapz(ground_state**2, x, initial=0)
plt.plot(x, integral)
plt.show()

# Define electric field
e_field = np.linspace(0,1,301)

# Evaluate integral over value
integral = cumtrapz(ground_state.conj() * e_field * ground_state, x, initial=0)
plt.plot(x, integral)
plt.show()




