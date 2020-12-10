## intiilize modules
import os, sys
## Add folder to path
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append("C:\\Users\\Azharuddin mohammed\\Documents\\GitHub\\QuDiPy")

import qudipy as qd
import qudipy.potential as pot
import qudipy.qutils as qt

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz
consts = qd.Constants("Si/SiO2")

arr = np.array([[1,2],[3,4]])
arr[:,1] =  arr[:,1] + 10
print(arr)


# First define the x(y)-coordinates
x = np.linspace(-60,60,201)*1E-9
y = np.linspace(-60,60,201)*1E-9
# Define harmonic oscillator frequency
omega = 5E12

# Now initialize the GridParameters class
gparams = pot.GridParameters(x, y)

# Now construct the harmonic potential
# Change the origin of the potential by modifying it in the harm_pot function 
harm_pot = 1/2*consts.me*omega**2*(np.square(gparams.x_mesh-10E-9) + np.square(gparams.y_mesh))

# Update the potential for gparams
gparams.update_potential(harm_pot)

# Pass sparams, gparams to the solve_schrodinger_eq qutils method to obtain the eigenvalues and eigenvectors
# Get ground state wavefunction and cast to real (should already be real) 
e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)
ground_state = np.real(e_vecs)

# Get the groundstate wavefunction probabilty density 
# Get rid of extra set empty arrays as well compute probabiltiy outside of the the graphing function 
probability = (np.squeeze(np.real(np.multiply(ground_state,ground_state.conj()))))

# Plot the ground state wavefunction 2D
fig, ax = plt.subplots()
ax.imshow(probability,cmap='viridis',extent=[gparams.x.min()/1E-9, gparams.x.max()/1E-9,gparams.y.min()/1E-9, gparams.y.max()/1E-9])
ax.set_title("State: 0 ")
ax.set(xlabel='x-coords [nm]',ylabel='y-coords [nm]')
plt.show()

# Trying to integrate the groundstate first the x direction then the y direction 
#integral = cumtrapz([cumtrapz(probability_x, x,initial=0) for probability_x in probability],y,initial=0) # this method is wrong !fix!
integral = np.array([cumtrapz(probability_x, x,initial=0) for probability_x in probability])
n = cumtrapz(integral[:,200],y,initial=0) 
plt.plot(x,n)
plt.show()

""""
# Plot the integral of the groundstate wf in 2D
fig, ax = plt.subplots()
ax.imshow(integral,cmap='viridis',extent=[gparams.x.min()/1E-9, gparams.x.max()/1E-9,gparams.y.min()/1E-9, gparams.y.max()/1E-9])
ax.set_title("State: integral of ground state ")
ax.set(xlabel='x-coords [nm]',ylabel='y-coords [nm]')
plt.show()
"""

# Define electric field 
electric_x = np.linspace(0,1,201)
electric_y = np.linspace(0,1,201)
X,Y = np.meshgrid(x,y)
electric_field = X + Y


# Define expectation value
# The wavefunction has an empty array and I used the dot and transpose to "properly" multiply the arrays 
expectation_intergrand = np.dot(np.dot(np.transpose(np.squeeze(np.real(ground_state.conj()))), electric_field), np.squeeze(np.real(ground_state)))
#expectation_integral = cumtrapz([cumtrapz(expectation_y, y,initial=0) for expectation_y in expectation_intergrand ],x,initial=0) # this method is wrong !fix!
expectation_integral = np.array([cumtrapz(expectation_x, x,initial=0) for expectation_x in expectation_intergrand])
print(expectation_integral.shape)
n = cumtrapz(expectation_integral[:,200],y,initial=0) 
plt.plot(x,n)
plt.show()

"""
# Plot expectation value in 2D
fig, ax = plt.subplots()
ax.imshow(expectation_integral,cmap='viridis',extent=[gparams.x.min()/1E-9, gparams.x.max()/1E-9,gparams.y.min()/1E-9, gparams.y.max()/1E-9])
ax.set_title("State: integral of ground state ")
ax.set(xlabel='x-coords [nm]',ylabel='y-coords [nm]')
plt.show()
"""

