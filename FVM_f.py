#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D unsteady diffusion problem solved with Finite Volume Method
@author: Antonio Preziosi-Ribero
CFD Pontificia Universidad Javeriana - March 2018
"""

import numpy as np
import scipy.sparse as SP
from scipy.sparse.linalg import spsolve
import Analyt as AN
import Auxiliary as AUX
import matplotlib.pyplot as plt
from matplotlib import style

def FVM1D(Sx, N):

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
    
#    Sx = 0.3
    theta = 0.5                                # C-N ponderation factor
#     N = 41                                      # Volumes in the domain
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
    ert = np.zeros(int(npt) + 1)
    
    # ==============================================================================
    # Generating matrix for implicit solver
    # ==============================================================================
    
    K = SP.lil_matrix((N, N))
    
    K[0, 0] = 1 + 3 * Sx
    K[0, 1] = -Sx
    K[N - 1, N - 1]  = 1 + 3 * Sx
    K[N - 1, N - 2] = -Sx
    
    for i in range(1, N - 1):
        
        K[i, i] = 1 + 2 * Sx
        K[i, i + 1] = -Sx
        K[i, i - 1] = -Sx
    
    # ==============================================================================
    # Generating initial condition
    # ==============================================================================
    
    C = AN.difuana(M, L, Dx, xn, xo, T0)
    C1 = np.zeros(N)
    Cmax = np.max(C)
    
#    # Plotting initial condition
#    plt.ion()
#    plt.figure(1, figsize=(11, 8.5))
#    style.use('ggplot')
#    
#    plt.subplot(1, 1, 1)
#    plt.plot(xn, C)
#    plt.title('Initial condition')
#    plt.xlabel(r'Distance $(m)$')
#    plt.ylabel(r'Concentration $ \frac{kg}{m} $')
    
    # =============================================================================
    # Starting time loop 
    # =============================================================================
    
#    plt.ion()
#    plt.figure(1, figsize=(11, 8.5))
#    style.use('ggplot')
    
    for t in range(1, npt + 1):
        
        # Generating analytical solution
        Ca = AN.difuana(M, L, Dx, xn, xo, T0 + t * dT)
        
        # Estimating solution C1 in t
        CL = AN.difuana(M, L, Dx, X0, xo, T0 + (t - 1) * dT)
        CR = AN.difuana(M, L, Dx, XL, xo, T0 + (t - 1) * dT)
        spa = AUX.FVev_sp(C, Dx, dx, CL, CR)
        
        C1 = C + dT * spa
        
        # Implicit part of the solution
        # Imposing boundary conditions
        C[0] = C[0] + AN.difuana(M, L, Dx, X0, xo, T0 + t * dT) * Sx * 2
        C[len(xn) - 1] = C[len(xn) - 1] + AN.difuana(M, L, Dx, XL, xo, T0 + t * dT)\
        * Sx * 2
        
        # Solving linear system
        C1i = spsolve(K, C)
        
        # Crank-Nicholson ponderation
        C1t = (1 - theta) * C1 + theta * C1i
        
        # Estimating error
        err = np.abs(C1t - Ca)
        ert[t] = np.linalg.norm(err)
        
#        # Plotting numerical solution and comparison with analytical
#        plt.clf()
#        
#        plt.subplot(2, 2, 1)
#        plt.plot(xn, C1t, 'b')
#        plt.xlim([X0, XL])
#        plt.ylim([0, Cmax])
#        plt.ylabel(r'Concentration $ \frac{kg}{m} $')
#        plt.title('Numerical solution')
#        
#        plt.subplot(2, 2, 2)
#        plt.plot(xn, Ca)
#        plt.xlim([X0, XL])
#        plt.ylim([0, Cmax])
#        plt.title('Analytical solution')
#        
#        plt.subplot(2, 2, 3)
#        plt.semilogy(xn, err)
#        plt.xlim([X0, XL])
#        plt.ylim([1e-8, 1e2])
#        plt.ylabel('Absolute error')
#        plt.title('Error')
#        
#        plt.subplot(2, 2, 4)
#        plt.semilogy(np.linspace(T0, Tf, npt), ert)
#        plt.xlim([T0 - 0.2, Tf + 0.2])
#        plt.ylim([1e-8, 1e2])
#        plt.title('Error evolution')
#        
#        plt.draw()
#        plt.pause(0.2)
        
        # Preparing for next timestep   
        C = C1t
        
    return ert