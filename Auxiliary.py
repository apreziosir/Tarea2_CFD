#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions associated with Finite Differences Method for solving a 1D diffusion 
problem
@author: Antonio-Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""

# =============================================================================
# Space evaluation function for Finite Difference Method - explicit
# =============================================================================

def FDev_sp(C0, Dx, dx):
    
    C = 0 * C0
    
    r = len(C0)
    
    for i in range(1, r - 1):
        
        C[i] = (Dx / dx ** 2) * (C0[i + 1] - 2 * C0[i] + C0[i - 1])
    
    return C

# =============================================================================
# Space evaluation for Finite Volume method - explicit
# =============================================================================
    
def FVev_sp(C0, Dx, dx, CL, CR):
    
    C = 0 * C0
    
    r = len(C0)
    
    C[0] = C0[1] * (Dx / dx ** 2) - (3 * Dx / dx ** 2) * C0[0] + (2 * Dx / \
    dx ** 2) * CL
    
    C[r - 1] = C0[r - 2] * (Dx / dx ** 2) - (3 * Dx / dx ** 2) * C0[r - 1] + \
    (2 * Dx / dx ** 2) * CR
    
    for i in range(1, r - 1):
    
        C[i] = (Dx / dx ** 2) * (C0[i + 1] - 2 * C0[i] + C0[i - 1])   
    
    return C