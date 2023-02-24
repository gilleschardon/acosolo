#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:43:59 2023

@author: gilleschardon
"""

import numpy as np
from scipy.linalg import inv
import scipy.optimize

def gridless_cmf(Sigma, g, X_init, Niter, box, A_init=None, mode="comet2"):

    # generates the dictionary on the grid, if not given as an argument
    if not A_init:
        A_init = g(X_init)
        
    A_init_norm = A_init / np.sqrt(np.real(np.sum(A_init * np.conj(A_init), axis=0)))
        
    # spatial dimension
    dim = X_init.shape[1]

    # variables
    Xs = np.zeros([0, dim])
    Ps = np.zeros([0])
    sigma2 = 1
    lsigma2 = 0.001 # lower bound on sigma2, to avoid singular covariance matrices

    for iter in range(Niter):
        
        print(f"Iter {iter+1}/{Niter}")
        
        # select a new source
        Xnew, nu = maximize_nu(Sigma, g, Xs, Ps, sigma2, lsigma2, X_init, A_init_norm, box, mode)
        Xs = np.vstack([Xs, Xnew])
                
        # optimize the amplitudes
        Ps, sigma2 = optimize_amplitudes(Sigma, g, Xs, Ps, sigma2, lsigma2, mode)
        # optimize the amplitudes and positions
        Xs, Ps, sigma2 = optimize_amplitudes_positions(Sigma, g, Xs, Ps, sigma2, lsigma2, box, dim, mode)
        
    return Xs, Ps, sigma2

def maximize_nu(Sigma, g, Xs, Ps, sigma2, lsigma2, X_init, A_init, box, mode):
    # maximizes the nu criterion for a new source
    
    Gloc = g(Xs)
    sigma2p = lsigma2+sigma2
    
    if mode == 'cmf':
        Sigma_est = Gloc @ (Gloc * Ps).T.conj() + sigma2p * np.eye(Sigma.shape[0])
        RR = Sigma - Sigma_est
    else:
        if Gloc.shape[1] > 0:      
            Sigma_est_inv = np.eye(Sigma.shape[0]) / sigma2p - Gloc @ inv(np.diag(1/Ps) + Gloc.T.conj() @ Gloc / (sigma2p)) @ Gloc.T.conj() / sigma2p**2
        else:
            Sigma_est_inv = np.eye(Sigma.shape[0]) / sigma2p # no source

        RR = Sigma_est_inv @ Sigma @ Sigma @ Sigma_est_inv
    
    # global optimization on a grid
    nugrid = np.real(np.sum( A_init.conj() * (RR @ A_init), axis=0) - np.sum(np.abs(A_init**2), axis=0))
    idx = np.argmax(nugrid)
    Xgrid = X_init[idx, :]
    
    # local optimization
    
    # objective function
    def nuobj(X):  
        gx = g(X)    
        gx = gx / np.sqrt(np.sum(gx*gx.conj()))
        return - np.real(gx.T.conj() @ RR @ gx)
   
    boxbounds = scipy.optimize.Bounds(box[0, :]+0.01, box[1, :]-0.01)    
    res = scipy.optimize.minimize(nuobj, Xgrid, bounds=boxbounds)

    return res.x, - res.fun

def optimize_amplitudes(Sigma, g, Xs, Ps, sigma2, sigma2l, mode):    
    Gloc = g(Xs)
    objfunamp = lambda x : amplitudesobj(Sigma, Gloc, x, sigma2l, mode)
    init = np.hstack([Ps, 0.001, sigma2])    
    res = scipy.optimize.minimize(objfunamp, init, bounds=scipy.optimize.Bounds(np.zeros([Ps.shape[0]+2])))
    return res.x[:-1], res.x[-1]
    
def amplitudesobj(Sigma, Gloc, x, sigma2l, mode):
    # unpacking    
    Ps = x[:-1]
    sigma2p = x[-1]
    
    I = np.eye(Sigma.shape[0])    
    Sigma_est = Gloc @ (Gloc * Ps).T.conj() + (sigma2l+sigma2p) * I
    
    if mode=='cmf':
        obj = np.sum(np.abs(Sigma-Sigma_est)**2)
        return obj
    
    # avoids instability with almost zero powers
    sup = Ps > 0.0001
    
    if np.max(Ps) > 0.0001:
        Sigma_est_inv = I /  (sigma2l+sigma2p) - Gloc[:, sup] @ inv(np.diag(1/Ps[sup]) + Gloc[:, sup].T.conj() @ Gloc[:, sup] / ( (sigma2l+sigma2p))) @ Gloc[:, sup].T.conj() /  (sigma2l+sigma2p)**2
    else:
        Sigma_est_inv = I /  (sigma2l+sigma2p)
               
    obj = np.real(np.trace(Sigma @ Sigma_est_inv @ Sigma) + np.trace(Sigma_est))

    return obj

def optimize_amplitudes_positions(Sigma, g, Xs, Ps, sigma2, sigma2l, box, dim, mode):
    
    xinit = np.hstack([Xs.ravel(), Ps, sigma2])
    Ns = Ps.shape[0]
    
    lb = np.hstack([ np.kron(np.ones([Ns]), box[0, :]), np.zeros([Ns+1])])
    ub = np.hstack([ np.kron(np.ones([Ns]), box[1, :]), np.inf * np.ones([Ns+1])])
    boxbounds = scipy.optimize.Bounds(lb, ub)

    objfun = lambda x : ampposobj(Sigma, g, x, sigma2l, dim, mode)
    options = {"disp": True, "iprint" : 1}
    res = scipy.optimize.minimize(objfun, xinit, bounds=boxbounds, options=options)
   
    # unpacking
    Xs = np.reshape(res.x[:dim*Ns], [Ns, dim])
    Ps = res.x[dim*Ns : -1]
    sigma2 = res.x[-1]

    return Xs, Ps, sigma2

def ampposobj(Sigma, g, x, sigma2l, dim, mode):
    
    Ns = (x.shape[0] - 1)// (dim + 1)

    # unpacking
    X = np.reshape(x[:dim*Ns], [Ns, dim])
    Psigma2 = x[dim*Ns :]
  
    Gloc = g(X)

    obj = amplitudesobj(Sigma, Gloc, Psigma2, sigma2l, mode)
    
    return obj