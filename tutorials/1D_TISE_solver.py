# REQUIRES PYTHON 3.6 OR GREATER

'''
Solve the time-independent Schrodinger-Equation H|Y> = E|Y> where H is
the single-electron 1 dimensional Hamiltonian.

'''

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

# Specify some constants
hbar = 6.626E-34/(2*np.pi) # Reduced planck's constant
me = 9.11E-31 # Free electron mass
e = 1.602E-19 # Electron charge
    
def build_1DSE_hamiltonian(x_coords, potential):
    ''' 
    Build a single electron Hamilonian for the 1-dimensional potential 
    specified in the gparams class. The laplacian operator is approximated by
    using a 1D 3-point stencil. The Hamilonian assumes a natural ordering 
    format along the main diagonal.
    
    Parameters
    ----------
    x_coords : 1D array
        Coordinates along x axis. Code assumes a uniform grid spacing.
    potential : 1D array
        Potential along x axis. 
        
    Returns
    -------
    ham_1D : sparse 2D array
        1-dimensional Hamtilonian. The diagonal elements are in natural
        ordering format
    '''
    
    nx = len(x_coords)
    # Assume a uniform grid
    dx = x_coords[1] - x_coords[0]
    
    # Build potential energy hamiltonian term
    PE_1D = sparse.diags(potential)
    
    # Build the kinetic energy hamiltonian term
    
    # Construct dummy block matrix B
    KE_1D = sparse.eye(nx)*(-2/(dx**2))
    # Add the +/-1 off diagonal entries for the 1/dx^2 elements
    KE_1D = KE_1D + sparse.diags(np.ones(nx-1)/(dx**2),-1)
    KE_1D = KE_1D + sparse.diags(np.ones(nx-1)/(dx**2),1)
    
    # Scale by appropriate units
    KE_1D = -hbar**2/(2*me)*KE_1D    
        
    # Assemble the full Hamiltonian with potential and kinetic terms
    ham_1D = PE_1D + KE_1D
    
    return ham_1D

#******************************#
#*********** INPUTS ***********#
#******************************#

# Number of grid points. The higher the number, the slow the computation time,
# but the results will be more accurate. Since this is 1D, computation time is
# neglible.
ngrid = 501

# Specify the x-coordinates
min_x = -90E-9 # -70 nm
max_x = 90E-9 # 70 nm
x = np.linspace(min_x, max_x, ngrid)

# Define harmonic oscillator frequency
# Number of eigenstates you wish to find
n_sols = 4

# Construct the potential
# Comment out the potentials you don't want to use.

# OPTION 1: Harmonic potential
omega = 2.5E11
potential = 1/2*me*omega**2*np.square(x)

# OPTION 2: Infinite square potential
L = 120E-9
potential = np.zeros(ngrid)
potential[np.logical_or(x<(-L/2),x>(L/2))] = 1E10

# OPTION 3: Semi-finite square potential
#L = 120E-9
potential = np.zeros(ngrid)
potential[x<(-L/2)] = 1E10
potential[x>(L/2)] = 6E-23

potential[x<(0)] = 1E10
potential[x>(0)] = np.abs(x[x>0])*8E-23/90E-9

#******************************#
#********** MAIN CODE *********#
#******************************#

# Build the respective Hamiltonian
hamiltonian = build_1DSE_hamiltonian(x, potential)
    
# Solve the time independent schrodinger equation
# Eigenvalue problem H|Psi> = E|Psi>
eig_ens, eig_vecs = eigs(hamiltonian.tocsc(), k=n_sols, M=None,
                                       sigma=potential.min())

# Sort the eigenvalues in ascending order (if not already)
idx = eig_ens.argsort()   
eig_ens = eig_ens[idx]
eig_vecs = eig_vecs[:,idx]

# Normalize the wavefunctions
for idx in range(n_sols):
    curr_wf = eig_vecs[:,idx]
    
    norm_val = np.trapz(np.multiply(curr_wf.conj(),curr_wf), x=x)

    eig_vecs[:,idx] = curr_wf/np.sqrt(norm_val)
    
# Print eigenenergies
print(f"Energies = {np.real(eig_ens)/e} eV")
# Print difference between eigenenergies
print(f"Calculated energy diffs = {np.diff(np.real(eig_ens))/e} eV")
    
#******************************#
#************ PLOT ************#
#******************************#
# Will plot the potential, eigenvalues and eigenfunctions
fig, ax = plt.subplots()
plt.plot(x/1E-9,potential/e,color='k')
# Calculate max amplitude of eigenfunctions for pretty plotting
max_amp = np.real(np.diff(eig_ens).min())/2/e
for idx in range(n_sols):
    
    # Plot ith eigenenergy
    plt.plot(np.array([min(x),max(x)])/1E-9,
             np.real([eig_ens[idx],eig_ens[idx]])/e,
             color='k',linestyle='--')
    
    # Plot ith eigenwavefunction
    curr_wf = np.real(eig_vecs[:,idx])
    
    if curr_wf[int(ngrid/2)] < 0:
        curr_wf = -curr_wf
        
    curr_wf = curr_wf/(max(curr_wf) - min(curr_wf))
    plt.plot(x/1E-9, curr_wf*max_amp + np.real(eig_ens[idx])/e)

ax.set(xlabel='x-coords [nm]',ylabel='Energy [eV]')
plt.ylim(-max_amp, np.real(max(eig_ens))/e+4*max_amp)

plt.show()









