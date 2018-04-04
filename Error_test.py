#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that compares the results from the different discretization schemes used
in the work done
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""


# Importing the functions that calculate with different discretizations
from FDM_f import FDM1D
from FVM_f import FVM1D
from FEM_f import FEM1D

# Importing libraries to plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


X0 = 0.
XL = 5.

# Declaring variables to be run and then plotted
Sx = np.array([0.4, 0.3, 0.1, 0.01])
Nx = np.array([11, 21, 51, 101, 201, 501])

# Matrices to store maximum errors of each run
MD = np.zeros((np.size(Sx), np.size(Nx)))
MV = np.zeros((np.size(Sx), np.size(Nx)))
ME = np.zeros((np.size(Sx), np.size(Nx)))

# Looping to find maximum error of each run
for i in range(0, np.size(Sx)):
    
    for j in range(0, np.size(Nx)):
        
        MD[i, j] = np.max(FDM1D(Sx[i], Nx[j]))
        MV[i, j] = np.max(FVM1D(Sx[i], Nx[j]))
        ME[i, j] = np.max(FEM1D(Sx[i], Nx[j]))

# Plotting the error curves for each Sx
plt.figure(1, figsize=(11, 8.5))
style.use('ggplot')        

plt.subplot(2, 2, 1)
line1 = plt.semilogy(Nx, MD[0,:], 'b', label='FDM')
line2 = plt.semilogy(Nx, MV[0,:], 'r', label='FVM')
line3 = plt.semilogy(Nx, ME[0,:], 'g', label='FEM')
plt.xlim([np.min(Nx), np.max(Nx)])
plt.ylim([1e-5, 1e0])
plt.ylabel(r'Infinity error norm')
plt.legend(loc=3)
plt.title('Sx = ' + str(Sx[0])) 
#
plt.subplot(2, 2, 2)
line1 = plt.semilogy(Nx, MD[1,:], 'b', label='FDM')
line2 = plt.semilogy(Nx, MV[1,:], 'r', label='FVM')
line3 = plt.semilogy(Nx, ME[1,:], 'g', label='FEM')
plt.xlim([np.min(Nx), np.max(Nx)])
plt.ylim([1e-5, 1e0])
plt.legend(loc=3)
plt.title('Sx = ' + str(Sx[1]))
#
plt.subplot(2, 2, 3)
line1 = plt.semilogy(Nx, MD[2,:], 'b', label='FDM')
line2 = plt.semilogy(Nx, MV[2,:], 'r', label='FVM')
line3 = plt.semilogy(Nx, ME[2,:], 'g', label='FEM')
plt.xlim([np.min(Nx), np.max(Nx)])
plt.ylim([1e-5, 1e0])
plt.ylabel(r'Infinity error norm')
plt.xlabel('Number of nodes in discretization')
plt.legend(loc=3)
plt.title('Sx = ' + str(Sx[2]))
#
plt.subplot(2, 2, 4)
line1 = plt.semilogy(Nx, MD[3,:], 'b', label='FDM')
line2 = plt.semilogy(Nx, MV[3,:], 'r', label='FVM')
line3 = plt.semilogy(Nx, ME[3,:], 'g', label='FEM')
plt.xlim([np.min(Nx), np.max(Nx)])
plt.ylim([1e-5, 1e0])
plt.xlabel('Number of nodes in discretization')
plt.legend(loc=3)
plt.title('Sx = ' + str(Sx[3]))
plt.draw()

plt.suptitle('Discretization comparison')