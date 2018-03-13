#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytical solution to 2D transport equation
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana 
"""

import numpy as np 

# ==============================================================================
# Analytical solution to de one dimensional transport equation in a rectangular
# domain with constant velocity fields
# ==============================================================================

def difuana(M, A, Dx, x, x0, t):
    
    temp1 = A * np.sqrt(4 * np.pi * Dx * t);
    
    C = (M / temp1) * np.exp(-(x - x0) ** 2 / (4 * Dx * t))
    
    return C 

