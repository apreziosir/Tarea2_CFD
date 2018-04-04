#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D unsteady diffusion solved by Finite Element Method (FEM), with 3 nodes 
elements and backward Euler in time.
@author: Antonio Preziosi-Ribero
CFD- Pontificia Universidad Javeriana
Marzo de 2018
"""

import numpy as np
import scipy.sparse as SP
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import Analyt as AN
import matplotlib.pyplot as plt
from matplotlib import style

def FEM1D(Sx, nn):
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
    
    L = XL - X0
#    nn = 21                        # Number of nodes (must be odd and >= 3)
    ne = int((nn - 1) / 2)          # Number of elements calculated
    h = (XL - X0) / ne              # cell size
#    Sx = 0.01
    
    # Nodes coordinates
    xn = np.linspace(X0, XL, nn)
    
    # Calculating timestep size for the Sx parameter. 
    dT = Sx * (xn[2] - xn[0]) / Dx
    nT = int((Tf - T0) / dT)
    
    ert = np.zeros(int(nT))

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
    del(n0, n1, n2)
    
# ==============================================================================
# Building matrix and RHS for solving
# ==============================================================================
    
    # Initializing global matrix and global RHS
    dLHS = SP.lil_matrix((nn, nn))
    RHS = np.zeros(nn)
    
    # Loop over elements to assemble diffusion matrix
    
#    plt.ion()
#    plt.figure(1, figsize=(11, 8.5))
#    style.use('ggplot')
    
    for ielt in range(0, ne):
        
        # Creating element diffusion matrix, assuming element size h 
        # (from Gresho)   
        eLHS = (1 / (3 * h)) * np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]])
        
        # Assemble to global LHS matrix
        # Loop over values in element matrix (i,j) and assemble to global
        # locations (I,J)
        for i in range(0, 3):
            I = nconn[ielt, i]
            
            for j in range(0, 3):
                
                J = nconn[ielt, j]
                
                dLHS[I, J] = dLHS[I, J] + eLHS[i, j]
                
# ==============================================================================
# Mass matrix proposition
# ==============================================================================
    
    # Loop over elements to assemble mass matrix
                
    mLHS = SP.lil_matrix((nn, nn))
    
    for ielt in range(0, ne):
        
        # Creating element mass matrix, assuming element size h    
        eLHS = (h / 30) * np.array([[4, 2, -1],[2, 16, 2], [-1, 2, 4]])
        
        # Assemble to global mLHS matrix
        # Loop over values in element matrix (i,j) and assemble to global
        # locations (I,J)
        for i in range(0, 3):
            I = nconn[ielt, i]
            
            for j in range(0, 3):
                
                J = nconn[ielt, j]
                
                mLHS[I, J] = mLHS[I, J] + eLHS[i, j]
    
    # Left hand side of the system (this does not vary in time)            
    LHSd = dT * Dx * dLHS + mLHS 
    
    # Imposing boundary conditions in the matrix 
    LHSd[0, :] = 0
    LHSd[0, 0] = 1
    LHSd[nn - 1, :] = 0
    LHSd[nn - 1, nn - 1] = 1
    
    LHSd = SP.csr_matrix.tocsr(LHSd)
    
    # Generating initial condition
    C = AN.difuana(M, L, Dx, xn, xo, T0)
    
    C1 = np.zeros(nn)
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
#    plt.draw()
#    plt.pause(1.5)
       
    # Entering time loop 
    
    for t in range(1, nT):
        
        # Generating analytical solution
        Ca = AN.difuana(M, L, Dx, xn, xo, T0 + t * dT)
        
        # Setting up right hand side
        RHS = mLHS * C
        RHS[0] = AN.difuana(M, L, Dx, X0, xo, T0 + t * dT)
        RHS[nn - 1] = AN.difuana(M, L, Dx, XL, xo, T0 + t * dT)
        
        # Solving system (matrix vector multiplication)
        C1 = spsolve(LHSd, RHS)
        
        # Estimating error
        err = np.abs(C1 - Ca)
        ert[t] = np.linalg.norm(err)
        
#        # Plotting numerical solution and comparison with analytical
#        plt.clf()
#        
#        plt.subplot(2, 2, 1)
#        plt.plot(xn, C1, 'b')
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
#        plt.semilogy(np.linspace(T0, Tf, nT), ert)
#        plt.xlim([T0 - 0.2, Tf + 0.2])
#        plt.ylim([1e-8, 1e2])
#        plt.title('Error evolution')
#        
#        plt.draw()
#        titulo = 'Finite element solution implicit'
#        plt.suptitle(titulo)
#        plt.pause(0.2)
        
        # Preparing for next timestep   
        C = C1
        
    return ert