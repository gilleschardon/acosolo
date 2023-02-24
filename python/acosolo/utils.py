#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:46:27 2023

@author: gilleschardon
"""

import numpy as np
from numpy.random import normal
from scipy.linalg import sqrtm


def square_array(aperture, N, center = [0,0,0], axis='z'):
    
    c = np.linspace(-aperture/2, aperture/2, N)
    
    if axis == 'x':
        xg, yg, zg = np.meshgrid(c + center[0], c + center[1], center[2])
    elif axis == 'y':
        xg, yg, zg = np.meshgrid(c + center[0], center[1], c + center[2])
    elif axis == 'z':
        xg, yg, zg = np.meshgrid(c + center[0], c + center[1], center[2])     
        
    return np.vstack([xg.ravel(), yg.ravel(), zg.ravel()]).T

def grid3D(lb, ub, step):
    xx = np.linspace(lb[0], ub[0], int((ub[0]-lb[0]) // step + 1))
    yy = np.linspace(lb[1], ub[1], int((ub[1]-lb[1]) // step + 1))
    zz = np.linspace(lb[2], ub[2], int((ub[2]-lb[2]) // step + 1))

    dims = [xx.size, yy.size, zz.size]

    Xg, Yg, Zg = np.meshgrid(xx, yy, zz)
    
    return np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T, dims

def generate_noise(Nm, Ns, sigma2):
    noise = normal(scale = np.sqrt(sigma2/2), size=(Nm, Ns)) + 1j * normal(scale = np.sqrt(sigma2/2), size=(Nm, Ns))    
    return noise

def generate_source(g, Ns, p):
    sig = g[:, np.newaxis] @ (normal(scale = np.sqrt(p/2), size=(1, Ns)) + 1j * normal(scale = np.sqrt(p/2), size=(1, Ns)))    
    return sig

def generate_correlated_sources(G, Ns, Sigma_source):
    Ssqrt = sqrtm(Sigma_source)
    sig = G @ Ssqrt @ ((normal(scale = np.sqrt(1/2), size=(G.shape[1], Ns)) + 1j * normal(scale = np.sqrt(1/2), size=(G.shape[1], Ns)))) 
    return sig

def scm(sig):
    return sig @ sig.conj().T / sig.shape[1]

def normalize(A):
    return A / np.sqrt(np.real(np.sum(A * np.conj(A), axis=0)))