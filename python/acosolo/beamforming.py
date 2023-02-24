#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:39:19 2023

@author: gilleschardon
"""

import numpy as np
import scipy.optimize
from scipy.linalg import norm
from .utils import normalize

def bmap_unc(Sigma, A):
    '''Beamforming map, with matrix of source vectors A'''
    A_norm = normalize(A)
    bf_map = np.real(  np.sum((A_norm.conj().T @ Sigma) * A_norm.T, 1))
    return bf_map

def bf_unc_crit(Sigma, g, x):
    '''Beamfoming criterion (MLE)'''
    gx = normalize(g(x))        
    return np.real(gx.conj().T @ Sigma @ gx)
    
def bf_unc_crit_async(Sigmas, gs, Nsnaps, x):
    '''Beamfoming criterion (MLE)'''
    
    bf = 0    
    for n in range(len(Sigmas)):
        gx = normalize(gs[n](x))
        bf = bf + np.real(gx.conj().T @ Sigmas[n] @ gx) * Nsnaps[n]
        
    return bf
    
def mle_unc(Sigma, g, X_init, sigma2 = None, A_init=None):
    '''MLE unconditional'''
    if not A_init:
        A_init = g(X_init)
        
    map_init = bmap_unc(Sigma, A_init)
    
    idx = np.argmax(map_init)
    Xinitbf = X_init[idx, :]
    
    objfun = lambda x : - bf_unc_crit(Sigma, g, x)
    res = scipy.optimize.minimize(objfun, Xinitbf)

    X = res.x
    gX = g(X)
    ngX2 = (np.real(np.sum(gX * gX.conj())))
    
    B = np.real(gX.conj().T @ Sigma @ gX)
 
    if sigma2:   
        p = (B / ngX2 - sigma2) / ngX2    
        return X, p
    
    else:
        N = Sigma.shape[0]
        sigma2 = (np.trace(Sigma) - B/ngX2) / (N - 1)
        p = (B/ngX2 - sigma2) / ngX2       
        return X, p, sigma2
    
    
def mle_unc_async_relax(Sigmas, gs, X_init, sigma2, Nsnaps, A_init=None):
    '''MLE unconditional, relax async'''
    if not A_init:
        A_init = []
        for n in range(len(Sigmas)):
            A_init.append(gs[n](X_init))
        
    map_init = bmap_unc(Sigmas[0], A_init[0]) * Nsnaps[0]
    
    for n in range(1, len(Sigmas)):
        map_init = map_init + bmap_unc(Sigmas[n], A_init[n]) * Nsnaps[n]
    
    idx = np.argmax(map_init)
    Xinitbf = X_init[idx, :]
    
    objfun = lambda x : - bf_unc_crit_async(Sigmas, gs, Nsnaps, x)
    res = scipy.optimize.minimize(objfun, Xinitbf)
    X = res.x
    
    p = 0   
    for n in range(len(Sigmas)):       
        gX = gs[n](X)
        ngX2 = (np.real(np.sum(gX * gX.conj())))   
        B = np.real(gX.conj().T @ Sigmas[n] @ gX)
        p = p + (B / ngX2 - sigma2) / ngX2 * Nsnaps[n]        
    p = p / np.sum(Nsnaps)
    
    return X, p

def mle_unc_async_strict(Sigmas, gs, X_init, sigma2, Nsnaps, A_init=None, output_relaxed=False):
    '''MLE unconditional, strict async'''

    # initialization using the relaxed model
    Xr, pr = mle_unc_async_relax(Sigmas, gs, X_init, sigma2, Nsnaps, A_init=None)
    
    # objective function
    def obj(Xp):
        X = Xp[:-1]
        p = Xp[-1]       
        obj = 0   
        for n in range(len(Sigmas)):    
            gx = gs[n](X)
            objl =(- p * (np.real(gx.T.conj() @ Sigmas[n] @ gx)) / (sigma2 * (sigma2 + p * norm(gx)**2)) + np.log(sigma2 + p * norm(gx)**2)) * Nsnaps[n]
            obj = obj + objl           
        return obj
    
    XPinit = np.hstack([Xr, pr])    
    bounds = scipy.optimize.Bounds([-np.inf, -np.inf, -np.inf, 0], [+np.inf, +np.inf, +np.inf, +np.inf])
    res = scipy.optimize.minimize(obj, XPinit, bounds=bounds)     
    X = res.x[:-1]
    p = res.x[-1]
    
    if output_relaxed:
        return X, p, Xr, pr
    else:
        return X, p