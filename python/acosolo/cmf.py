#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:36:35 2020

@author: gilleschardon
"""

import numpy as np
import scipy.linalg as la
import time
from .utils import normalize

# some fast matrix products

# beamforming
def proddamastranspose(D, Data):    
    x = np.real( np.sum( (D.conj().T @ Data) * D.T, 1))    
    return x

# product by the DAMAS matrix
def proddamas(D, x, support=None):    
    if support is None:
        z = D @ (x * D.conj()).T
    else:
        z = D[:, support] @ (x[support] * D[:, support].conj()).T    
    return proddamastranspose(D, z)

def proddamasdr(D, x, support=None):    
    if support is None:
        z = D @ (x * D.conj()).T
    else:
        z = D[:, support] @ (x[support] * D[:, support].conj()).T
    z = z - np.diag(np.diag(z))    
    return proddamastranspose(D, z)
    
# local unconstrained least-squares problem
def solve_ls(D, bf, support):
    Gram = np.abs(D[:, support].conj().T @ D[:, support]) ** 2
    return la.solve(Gram, bf[support], assume_a='pos'), Gram

# local unconstrained least-squares problem for diagonal removal
def solve_ls_dr(D, bf, support):
    aD2 = np.abs(D[:, support] ** 2)
    Gram = np.abs(D[:, support].conj().T @ D[:, support]) ** 2 - aD2.T @ aD2
    return la.solve(Gram, bf[support], assume_a='pos'), Gram

## CMF-NNLS
def cmf_nnls_lh(D, Data, lreg = 0, dr = True, norm = True):
       
    norms = np.ones(D.shape[1])
    if dr:        
        if norm:
            norms = (np.sum(abs(D)**2, 0)**2 - np.sum(np.abs(D)**4, 0))
        Dnorm = D / np.tile(norms, (D.shape[0], 1))

        Dataloc = Data - np.diag(np.diag(Data))
        prodA = lambda D, x, support : proddamasdr(Dnorm, x, support)
        solve = lambda D, b, support : solve_ls_dr(Dnorm, b, support)

    else:
        Dataloc = Data
        if norm:
            norms =  np.sqrt(np.sum(np.abs(D)**2, axis=0))
            
        Dnorm = D / np.tile(norms, (D.shape[0], 1))
            
        prodA = lambda D, x, support : proddamas(Dnorm, x, support)
        solve = lambda D, b, support : solve_ls(Dnorm, b, support)
    
    bf = proddamastranspose(Dnorm, Dataloc) - lreg
    
    x, unique = lawson_hanson(prodA, solve, Dnorm, bf)

    return x / (norms**2)
#        


## Lawson-Hanson solver, custom implementation

# ||Mx - y||_2^2, with M described by D
# prodA(D, x, support) = M*x
# solve(D, b, support) solve the local LS problem
# b = M'*y
def lawson_hanson(prodA, solve, D, b, verbose = True):
    
    T0 = time.perf_counter()
    
    n = D.shape[1]
    R = np.ones([n], dtype=bool) # active set
    N = np.arange(n);
    x = np.zeros([n])
    
    # residual
    
    w = b
    
    it = 0
    
    while np.any(R) and (np.max(w[R]) > 0):
        if verbose:
            print(f"iter {it} tol {np.max(w[R]):.2}")
        it = it + 1

        # update of the active set        
        idx = np.argmax(w[R])
        Ridx = N[R]
        idx = Ridx[idx]        
        R[idx] = 0
        
        # least-square in the passive set        
        s = np.zeros(x.shape)      
        s[~R], Gram = solve(D, b, ~R)
        
        # removal of negative coefficients
        while np.min(s[~R]) <= 0:
            
            Q = (s <= 0) & (~R)
            alpha = np.min(x[Q] / (x[Q] - s[Q]))
            x = x + alpha * (s - x)
            R = (( x <= np.finfo(float).eps) & ~R) | R
            
            s = np.zeros(x.shape)
            s[~R], Gram = solve(D, b,  ~R)

            
        # update of the solution
        x = s
        # update of the residual
        w = b - prodA(D, x, ~R)
        
    try:
        la.cholesky(Gram)
        if verbose:
            unique = True
            print(f'Solution is unique, T = {time.perf_counter() - T0:.2}')
    except e:
        if verbose:
            unique = False
            print(f'Solution is not unique, T = {time.perf_counter() - T0:.2}')
                   
    return x, unique



def CMF_OLS_correl(Sigma, A, Niter):
    
    
    
    N = Sigma.shape[0]
    Aorth = np.zeros([N, 0])
    sel = []
    
    for n in range(Niter):
        AA = Aorth@Aorth.T.conj()
        Sigmaproj = AA @ Sigma @ AA
        res = Sigma - Sigmaproj
        
        projA = A - AA @ A
        projA = normalize(projA)# / np.sqrt(np.sum(projA*projA.conj(), axis=0))
        
        SAorth = Sigma @ Aorth
        nproj1 = np.abs(np.sum((projA.T.conj() @ Sigma) * (projA.T), axis=1))**2
        nproj2 = 2 * np.sum(np.abs(projA.T.conj() @ SAorth)**2, axis=1)
        
        nproj = nproj1 + nproj2
        
        nproj[sel] = -np.inf
                
        idx = np.argmax(nproj)
        
        sel.append(idx)
        
        Aorth = np.hstack([Aorth, projA[:, idx:idx+1]])
        
    Asources = A[:, sel]
    pAs = la.pinv(Asources)
    Sigma_sources = pAs @ Sigma @ pAs.T.conj()
    
    return Sigma_sources, sel