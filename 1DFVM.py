#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D unsteady diffusion problem solved with Finite Volume Method
@author: Antonio Preziosi-Ribero
CFD Pontificia Universidad Javeriana - March 2018
"""

import numpy as np
import Analyt as AN
import Auxiliary as AUX
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

Sx = 0.2
theta = 0.                                  # C-N ponderation factor
N = 10                                      # Volumes in the domain
L = XL - X0                                 # Domain length
dx = L / N                                  # Calculating spacing           
xn = np.zeros(N)                            # Node coordinates vector                            

# Filling vector of node coordinates
for ic in range(0, N):
    
    xn[ic] = ic * dx + dx / 2

# Calculating dT and number of timesteps
dT = Sx * ( dx ** 2) / (Dx)                 # Calculating timestep size
npt = int(np.ceil((Tf - T0) / (dT)))        # Number of timesteps

# Generating vector that stores error in time
ert = np.zeros(int(npt))

# ==============================================================================
# Generating initial condition
# ==============================================================================

C = AN.difuana(M, L, Dx, xn, xo, T0)
C1 = np.zeros(N)
Cmax = np.max(C)

# Plotting initial condition
plt.ion()
plt.figure(1, figsize=(11, 8.5))
style.use('ggplot')

plt.subplot(1, 1, 1)
plt.plot(xn, C)
plt.title('Initial condition')
plt.xlabel(r'Distance $(m)$')
plt.ylabel(r'Concentration $ \frac{kg}{m} $')
#plt.pause(3)
#plt.close(1)