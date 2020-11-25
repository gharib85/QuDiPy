## intiilize modules
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append("C:\\Users\\Azharuddin mohammed\\Documents\\GitHub\\QuDiPy")

import qudipy as qd
import qudipy.potential as pot
import qudipy.qutils as qt

import numpy as np
import matplotlib.pyplot as plt

## Set up varibles 
x = np.arange(-3e-9, 3e-9, 0.1e-9)
y = np.arange(-3e-9, 3e-9, 0.1e-9)


## get electric field 
def create_electric_field(x,y):
    # function at fixed z axis ex. z = 2
    xx,yy = np.meshgrid(x,y)
    z = 0
    q = 1.6e-19
    E1 = (4*np.pi*q)*(((xx-2e-9)**2+yy**2+z**2)**-1.5)
    E2 = (4*np.pi*q)*(((xx+2e-9)**2+yy**2+z**2)**-1.5)
    return E1+E2

f = create_electric_field(x,y) 
plt.contour(x,y,f) 
plt.show()
print(f.shape)

#  initalize potential
V1 = [0.2, 0.22, 0.24, 0.26]
V2 = [0.2, 0.22, 0.24, 0.26, 0.28]

ctrl_vals = [V1, V2]
ctrl_names = ['V1','V2']

def create_potentials(cvals, gparams):
    x = gparams.x_mesh
    y = gparams.y_mesh
    v_1 = cvals[0]
    v_2 = cvals[1]
    # Potential here is a quartic confinement in x (for 2 dots to form) and a parabolic confinement in y
    # There is also a linear term in x proportional to the difference in voltage  
    potential = 1e17 * (x + 2e-9)**2 * (x - 2e-9)**2 + y**2 + 1e-9 * (v_2 - v_1) * x
    return potential

analytic_potential = pot.load_data.analytical_potential(ctrl_vals, ctrl_names, create_potentials, x, y)

# Now specify the potentials to interpolate around
v_vec = [0.23, 0.23]

# Build the interpolator the same way as usual
# This behaves exactly like any other potential interpolator
pot_interp = pot.build_interpolator(analytic_potential, constants=qd.Constants("Si/SiO2"))


# Plot the 2D potential it shows the  WAVE FUNCTION
pot_interp.plot(v_vec, plot_type='1D', y_slice=0e-9, show_wf=True, wf_n=0)
consts = qd.Constants("Si/SiO2")

## get ground state wavefunction
## getting wave function

# Now initialize the GridParameters class
gparams = pot.GridParameters(x, y)

# Now construct the non interpolated analytic potential
x = gparams.x_mesh
y = gparams.y_mesh
v_1 = 0.2
v_2 = 0.2
non_interpolated_analytic_potential = 1e17 * (x + 2e-9)**2 * (x - 2e-9)**2 + y**2 + 1e-9 * (v_2 - v_1) * x

# Update the potential for gparams
gparams.update_potential(non_interpolated_analytic_potential)
consts = qd.Constants("Si/SiO2")

# Eigenenergy and wavefunction
e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=1)

# Plot Wave function
plt.imshow(np.real(np.multiply(e_vecs[:,:,0],e_vecs[:,:,0].conj())), cmap='viridis', extent=[gparams.x.min()/1E-9, gparams.x.max()/1E-9, gparams.y.min()/1E-9, gparams.y.max()/1E-9])
plt.title("State: Ground state ")
plt.xlabel('x-coords [nm]')
plt.ylabel('y-coords [nm]')
plt.show()

wavefunction = e_vecs 
conjugate_wavefunction = e_vecs.conj()

def expectation_value(e_vecs,electric_field):
    wavefunction = e_vecs 
    conjugate_wavefunction = e_vecs.conj()
    m = wavefunction*electric_field*conjugate_wavefunction
    return expectation_value

print(wavefunction*f*conjugate_wavefunction)
print(wavefunction*conjugate_wavefunction)