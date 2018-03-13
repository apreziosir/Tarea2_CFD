#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion equation in 1D
CFD - Pontificia Universidad Javeriana
March 2018
@author: Antonio Preziosi-Ribero
"""

import numpy as np
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
theta = 0.                                  # C-N ponderation factor
N = 11                                      # Nodes in the domain
L = XL - X0                                 # Domain length

# Generating vector with discretization points
x = np.linspace(X0, XL, N) 

dx = x[1] - x[0]                            # Calculating spacing           
dT = Sx * ( dx ** 2) / (Dx)                 # Calculating timestep size
npt = np.ceil((Tf - T0) / (dT))             # Number of timesteps

# Generating vector that stores error in time
ert = np.zeros(int(npt))

# ==============================================================================
# Generating initial condition
# ==============================================================================

C = AN.difuana(M, L, Dx, x, xo, T0)
C1 = np.zeros(N)

# Plotting initial condition
plt.ion()
plt.figure(1)
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
plt.figure(1)
style.use('ggplot')

for t in range(1, npt + 1):
    
    # Generating analytical solution
    Ca = AN.difuana(M, L, Dx, x, xo, T0 + t * dT)
    
    # Explicit internal part
    for i in range(1, N):
        
        C1[i] = Sx * C[i - 1] + (1 - 2 * Sx) * C[i] + Sx * C[i + 1]
        
    # Imposing boundary conditions
    C1[0] = Ca[0]
    C1[N - 1] = Ca[N - 1]
    
    # Plotting numerical solution and comparison with analytical
    plt.clf()
    
    plt.subplot(2, 2, 1)
    plt.plot(x, C1)
    plt.title('Numerical solution')
    
    
    # Calculating error
    
    C = C1
    
    