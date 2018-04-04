#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion equation in 1D - Crank-Nicholson method implemented
CFD - Pontificia Universidad Javeriana
March 2018
@author: Antonio Preziosi-Ribero
"""

import numpy as np
import scipy.sparse as SP
from scipy.sparse.linalg import spsolve
import Analyt as AN
import matplotlib.pyplot as plt
from matplotlib import style

# ==============================================================================
# Variable declaration
# ==============================================================================

X0 = 0.                                     # Initial x coordinate
XL = 5.                                     # Final x coordinate 
M = 10                                      # Mass injected
xo = 2                                      # Injection coordinate
Dx = 0.3                                    # Diffusion coefficient
T0 = 1.                                     # Initial time
Tf = 5.                                     # Final time

# ==============================================================================
# Numerical parameters
# theta = 0 --> Fully explicit
# theta = 1 --> Fully implicit
# ==============================================================================

Sx = 0.3
theta = 0.50                                # C-N ponderation factor
N = 81                                      # Nodes in the domain
L = XL - X0                                 # Domain length

# Generating vector with discretization points
x = np.linspace(X0, XL, N) 

dx = x[1] - x[0]                            # Calculating spacing           
dT = Sx * ( dx ** 2) / (Dx)                 # Calculating timestep size
npt = int(np.ceil((Tf - T0) / (dT)))        # Number of timesteps

# Generating vector that stores error in time
ert = np.zeros(int(npt))

# ==============================================================================
# Generate matrix for implicit version (Dirichlet boundary conditions)
# ==============================================================================

K = SP.lil_matrix((N, N))

K[0, 0] = 1
K[N - 1, N - 1]  = 1

for i in range(1, N - 1):
    
    K[i, i] = 1 + 2 * Sx
    K[i, i + 1] = -Sx
    K[i, i - 1] = -Sx

# ==============================================================================
# Generating initial condition
# ==============================================================================

C = AN.difuana(M, L, Dx, x, xo, T0)
C1 = np.zeros(N)
Cmax = np.max(C)

# Plotting initial condition
plt.ion()
plt.figure(1, figsize=(11, 8.5))
style.use('ggplot')

plt.subplot(1, 1, 1)
plt.plot(x, C)
plt.title('Initial condition')
plt.xlabel(r'Distance $(m)$')
plt.ylabel(r'Concentration $ \frac{kg}{m} $')
plt.pause(3)
plt.close(1)

# ==============================================================================
# Entering the time loop
# ==============================================================================

plt.ion()
plt.figure(1, figsize=(11, 8.5))
style.use('ggplot')

for t in range(1, npt + 1):
    
    # Generating analytical solution
    Ca = AN.difuana(M, L, Dx, x, xo, T0 + t * dT)
    
    # Explicit internal part
    for i in range(1, N - 1):
        
        C1[i] = Sx * C[i - 1] + (1 - 2 * Sx) * C[i] + Sx * C[i + 1]
        
    # Imposing boundary conditions
    C1[0] = Ca[0]
    C1[N - 1] = Ca[N - 1]
    
    # Implicit part of the solver
    
    #Imposing boundary conditions
    C[0] = Ca[0]
    C[N - 1] = Ca[N - 1]
    
    # Solving system of linear equations
    C1i = spsolve(K, C)
    
    C1t = (1 - theta) * C1 + theta * C1i
    
    # Estimating error
    err = np.abs(C1t - Ca)
    ert[t] = np.linalg.norm(err)
    
    # Plotting numerical solution and comparison with analytical
    plt.clf()
    
    plt.subplot(2, 2, 1)
    plt.plot(x, C1t, 'b')
    plt.xlim([X0, XL])
    plt.ylim([0, Cmax])
    plt.ylabel(r'Concentration $ \frac{kg}{m} $')
    plt.title('Numerical solution')
    
    plt.subplot(2, 2, 2)
    plt.plot(x, Ca)
    plt.xlim([X0, XL])
    plt.ylim([0, Cmax])
    plt.title('Analytical solution')
    
    plt.subplot(2, 2, 3)
    plt.semilogy(x, err)
    plt.xlim([X0, XL])
    plt.ylim([1e-8, 1e2])
    plt.ylabel('Absolute error')
    plt.title('Error')
    
    plt.subplot(2, 2, 4)
    plt.semilogy(np.linspace(T0, Tf, npt), ert)
    plt.xlim([T0 - 0.2, Tf + 0.2])
    plt.ylim([1e-8, 1e2])
    plt.title('Error evolution')
    
    plt.draw()
    titulo = str(theta)
    titulo = 'Finite differences CN factor = ' + titulo
    plt.suptitle(titulo)
    plt.pause(0.2)
    
    # Preparing for next timestep   
    C = C1t
    
    