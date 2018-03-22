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

