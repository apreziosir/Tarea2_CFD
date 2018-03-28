#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D unsteady diffusion solved by Finite Element Method (FEM), with 3 nodes 
elements and forward Euler in time.
@author: Antonio Preziosi-Ribero
CFD- Pontificia Universidad Javeriana
Marzo de 2018
"""

import numpy as np
import scipy.sparse as SP
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
# ==============================================================================

nn = 11                                 # Number of nodes (must be odd and >= 3)
ne = int((nn - 1) / 2)                  # Number of elements calculated
h = (XL - X0) / ne                      # cell size

# Set up mesh

# Nodes coordinates
xn = np.linspace(X0, XL, nn)

# Nodes connectivity matrix
n0 = np.arange(0, nn - 2, 2)
n1 = n0 + 1
n2 = n0 + 2
nconn = np.zeros((ne, 3))
nconn[:, 0] = n0
nconn[:, 1] = n1
nconn[:, 2] = n2

# ==============================================================================
# Building matrix and RHS for solving
# ==============================================================================

# Initializing global matrix and global RHS
dLHS = SP.lil_matrix((nn, nn))
RHS = np.zeros(nn)

# Loop over elements to assemble diffusion matrix

plt.ion()
plt.figure(1, figsize=(11, 8.5))
style.use('ggplot')

for ielt in range(0, ne):
    
    # Creating element diffusion matrix, assuming element size h    
    eLHS = ((Dx * h) / 3) * np.array([[7/4, -2, 1/4],[-2, 4, -2], [1/4, \
           -2, 7/4]])
    
    # Assemble to global LHS matrix
    # Loop over values in element matrix (i,j) and assemble to global
    # locations (I,J)
    for i in range(0, 3):
        I = nconn[ielt, i]
        
        for j in range(0, 3):
            
            J = nconn[ielt, j]
            
            dLHS[I, J] = dLHS[I, J] + eLHS[i, j]
            
            plt.clf()
            plt.spy(dLHS)
            plt.draw()
            plt.pause(0.2)

# Loop over elements to assemble mass matrix
            
mLHS = SP.lil_matrix((nn, nn))

plt.ion()
plt.figure(2, figsize=(11, 8.5))
style.use('ggplot')

for ielt in range(0, ne):
    
    # Creating element diffusion matrix, assuming element size h    
    eLHS = (h / 15) * np.array([[2, 1, -1/2],[1, 8, 1], [-1/2, 1, 2]])
    
    # Assemble to global mLHS matrix
    # Loop over values in element matrix (i,j) and assemble to global
    # locations (I,J)
    for i in range(0, 3):
        I = nconn[ielt, i]
        
        for j in range(0, 3):
            
            J = nconn[ielt, j]
            
            mLHS[I, J] = mLHS[I, J] + eLHS[i, j]
            
            plt.clf()
            plt.spy(mLHS)
            plt.draw()
            plt.pause(0.2)

