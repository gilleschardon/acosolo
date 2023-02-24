#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:36:14 2023

@author: gilleschardon
"""

import numpy as np

def freefield(PX, PS, k, normalized=False):
      
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[: ,1] - PS[1]
        dz = PX[:, 2] - PS[2]
        
    else:    
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T
        dz = PX[:, 2:3] - PS[:, 2:3].T

    d = np.sqrt(dx*dx + dy*dy + dz*dz);
        
    D = np.exp( -1j * k * d) / d
    
    if normalized:
        D = D / np.sqrt(np.real(np.sum(D * np.conj(D), axis=0)))
        
    return D

def freefield2D(PX, PS, Z, k, normalized=False):
      
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[: ,1] - PS[1]
        
    else:    
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T

    d = np.sqrt(dx*dx + dy*dy + Z*Z);
        
    D = np.exp( -1j * k * d) / d
    
    if normalized:
        D = D / np.sqrt(np.real(np.sum(D * np.conj(D), axis=0)))
        
    return D