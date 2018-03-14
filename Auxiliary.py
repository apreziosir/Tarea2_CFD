#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions associated with Finite Differences Method for solving a 1D diffusion 
problem
@author: Antonio-Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""

import numpy as np


def FDev_sp(C0, Dx, dx):
    
    C = 0 * C0
    
    r = len(C0)
    
    for i in range(1, r - 1):
        
        C[i] = (Dx / dx ** 2) * (C0[i + 1] - 2 * C0[i] + C0[i - 1])
    
    return C