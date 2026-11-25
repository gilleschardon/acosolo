#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Covariance Matrix Fitting (CMF) for sparse source localization.

Implements the NNLS (non-negative least squares) formulation of CMF
using a custom Lawson-Hanson solver, diagonal-removed variants, and
orthogonal least squares (OLS) for localizing coherent/correlated sources.

Also includes the DAMAS-MI iterative greedy solver and helper functions
for computing gradient components.

Key functions
-------------
cmf_nnls_lh : NNLS-CMF via Lawson-Hanson active-set solver (returns x)
CMF_OLS_correl : OLS selection for correlated sources
proddamas / proddamasdr : Fast DAMAS matrix-vector products
solve_ls / solve_ls_dr : Local least-squares (with/without diagonal removal)
lawson_hanson : Core NNLS solver (returns x, obj, unique)
DAMAS_MI : Greedy MI solver with relative convergence tolerance
compute_a / compute_a_dr : Gradient components for DAMAS-MI
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import time
from .utils import normalize

# some fast matrix products

def proddamastranspose(G: np.ndarray, Data: np.ndarray) -> np.ndarray:
    """Beamforming: np.real(sum(conj(G)^T @ Data * G^T, 1))."""
    return np.real(np.sum((G.conj().T @ Data) * G.T, 1))


def proddamas(G: np.ndarray, x: np.ndarray,
              support: np.ndarray | None = None) -> np.ndarray:
    """Product by the DAMAS matrix."""
    if support is None:
        z = G @ (x * G.conj()).T
    else:
        z = G[:, support] @ (x[support] * G[:, support].conj()).T
    return proddamastranspose(G, z)


def proddamasdr(G: np.ndarray, x: np.ndarray,
                support: np.ndarray | None = None) -> np.ndarray:
    """DAMAS product with diagonal removal."""
    if support is None:
        z = G @ (x * G.conj()).T
    else:
        z = G[:, support] @ (x[support] * G[:, support].conj()).T
    z = z - np.diag(np.diag(z))
    return proddamastranspose(G, z)


def solve_ls(G: np.ndarray, bf: np.ndarray, support: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Local unconstrained least-squares problem."""
    Gram = np.abs(G[:, support].conj().T @ G[:, support]) ** 2
    return la.solve(Gram, bf[support], assume_a='pos'), Gram


def solve_ls_dr(G: np.ndarray, bf: np.ndarray, support: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Local unconstrained least-squares with diagonal removal."""
    aG2 = np.abs(G[:, support] ** 2)
    Gram = np.abs(G[:, support].conj().T @ G[:, support]) ** 2 - aG2.T @ aG2
    return la.solve(Gram, bf[support], assume_a='pos'), Gram


def cmf_nnls_lh(Sigma: np.ndarray, G: np.ndarray, lreg: float = 0, dr: bool = True,
                norm: bool = True, verbose: bool = False) -> np.ndarray:
    """CMF-NNLS via Lawson-Hanson active-set solver.
    
    Parameters
    ----------
    Sigma : ndarray (Nm, Nm)
        Covariance matrix.
    G : ndarray (Nm, Ng)
        Steering vector matrix (dictionary).
    lreg : float
        L2 regularization term.
    dr : bool
        Diagonal removal (default True).
    norm : bool
        Column normalization (default True).
    verbose : bool
        Print convergence info (default False).
        
    Returns
    -------
    x : ndarray (Ng,)
        Source power estimate.
    """
    norms = np.ones(G.shape[1])
    if dr:
        if norm:
            # Fourth root, so that the diagonal-removed Gram matrix
            # (||g||^4 - sum|g|^4) / norms^4 has a unit diagonal.
            norms = (np.sum(np.abs(G)**2, 0)**2
                     - np.sum(np.abs(G)**4, 0)) ** 0.25
        Gnorm = G / np.tile(norms, (G.shape[0], 1))
        Dataloc = Sigma - np.diag(np.diag(Sigma))
        prodA = lambda x, support: proddamasdr(Gnorm, x, support)
        solve = lambda b, support: solve_ls_dr(Gnorm, b, support)
    else:
        Dataloc = Sigma
        if norm:
            norms = np.sqrt(np.sum(np.abs(G)**2, axis=0))
        Gnorm = G / np.tile(norms, (G.shape[0], 1))
        prodA = lambda x, support: proddamas(Gnorm, x, support)
        solve = lambda b, support: solve_ls(Gnorm, b, support)
    
    bf = proddamastranspose(Gnorm, Dataloc) - lreg
    x, _, _ = lawson_hanson(prodA, solve, bf, G.shape[1], verbose=verbose)
    return x / (norms**2)


def lawson_hanson(prodA, solve, b: np.ndarray, dim: int,
                  verbose: bool = True) -> tuple[np.ndarray, list[float], bool]:
    """Lawson-Hanson NNLS solver (custom implementation).
    
    Solves min ||Mx - y||_2^2 with x >= 0, where M is defined implicitly
    by the prodA and solve functions.
    
    Parameters
    ----------
    prodA : callable(x, support) -> z
        Matrix-vector product M*x, using only the atoms in ``support``
        (boolean mask).
    solve : callable(b, support) -> (s, Gram)
        Local least-squares solver restricted to ``support``; returns the
        solution on the support and the local Gram matrix.
    b : ndarray
        Gradient vector (M'*y).
    dim : int
        Number of dictionary atoms.
    verbose : bool
        Print convergence info (default True).
        
    Returns
    -------
    x : ndarray
        Non-negative solution.
    obj : list of float
        Objective history at each iteration, up to the constant
        ||y||^2 / 2 (empty if x = 0 is returned immediately).
    unique : bool
        Whether the solution is unique (Cholesky of the final Gram matrix
        succeeded). True when no gradient entry is positive and x = 0 is
        returned with an empty support.
    """
    T0 = time.perf_counter()
    
    n = dim
    R = np.ones([n], dtype=bool)  # active set
    N = np.arange(n)
    x = np.zeros([n])
    
    w = b
    it = 0
    obj = []
    Gram = None

    while np.any(R) and (np.max(w[R]) > 0):
        if verbose:
            print(f"iter {it} tol {np.max(w[R]):.2}")
        it += 1

        idx = np.argmax(w[R])
        Ridx = N[R]
        idx = Ridx[idx]
        R[idx] = 0
        
        s = np.zeros(x.shape)
        s[~R], Gram = solve(b, ~R)
        
        while np.min(s[~R]) <= 0:
            Q = (s <= 0) & (~R)
            alpha = np.min(x[Q] / (x[Q] - s[Q]))
            x = x + alpha * (s - x)
            R = ((x <= np.finfo(float).eps) & ~R) | R
            s = np.zeros(x.shape)
            s[~R], Gram = solve(b, ~R)

        x = s
        pA = prodA(x, ~R)
        w = b - pA
        obj.append(np.dot(x, pA / 2 - b))
        
    if Gram is None:
        # No positive gradient entry: x = 0 is optimal and the support is
        # empty, so there is no Gram matrix to check.
        unique = True
        if verbose:
            print(f'Solution is x = 0, T = {time.perf_counter() - T0:.2}')
        return x, obj, unique

    try:
        la.cholesky(Gram)
        unique = True
        if verbose:
            print(f'Solution is unique, T = {time.perf_counter() - T0:.2}')
    except np.linalg.LinAlgError:
        unique = False
        if verbose:
            print(f'Solution is not unique, T = {time.perf_counter() - T0:.2}')
                   
    return x, obj, unique


def CMF_OLS_correl(Sigma: np.ndarray, G: np.ndarray, Niter: int
                   ) -> tuple[np.ndarray, list[int]]:
    """OLS selection for correlated sources.
    
    Iteratively selects the most correlated steering vector, orthogonalizes
    against previously selected ones, and builds a signal subspace.
    
    Parameters
    ----------
    Sigma : ndarray (Nm, Nm)
        Covariance matrix.
    G : ndarray (Nm, Ng)
        Steering vector matrix.
    Niter : int
        Number of sources to select.
        
    Returns
    -------
    Sigma_sources : ndarray
        Estimated source covariance matrix in the source subspace.
    sel : list of int
        Indices of selected dictionary atoms.
    """
    N = Sigma.shape[0]
    Gorth = np.zeros([N, 0])
    sel = []
    
    for n in range(Niter):
        GG = Gorth @ Gorth.T.conj()
        
        projG = G - GG @ G
        projG = normalize(projG)
        
        SGorth = Sigma @ Gorth
        nproj1 = np.abs(np.sum((projG.T.conj() @ Sigma) * (projG.T), axis=1))**2
        nproj2 = 2 * np.sum(np.abs(projG.T.conj() @ SGorth)**2, axis=1)
        
        nproj = nproj1 + nproj2
        nproj[sel] = -np.inf
                
        idx = np.argmax(nproj)
        sel.append(idx)
        Gorth = np.hstack([Gorth, projG[:, idx:idx+1]])
        
    Gsources = G[:, sel]
    pGs = la.pinv(Gsources)
    Sigma_sources = pGs @ Sigma @ pGs.T.conj()
    
    return Sigma_sources, sel


def compute_a(Gconj: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Gradient component for DAMAS-MI (no diagonal removal)."""
    aa = np.abs(Gconj.T @ g)
    return np.square(aa)


def compute_a_dr(Gconj: np.ndarray, G2: np.ndarray, g: np.ndarray,
                 g2: np.ndarray) -> np.ndarray:
    """Gradient component for DAMAS-MI with diagonal removal."""
    aa = np.abs(Gconj.T @ g)
    return np.square(aa) - G2.T @ g2


def DAMAS_MI(Sigma: np.ndarray, G: np.ndarray, niter: int, dr: bool = False,
             rel_eps: float = 0) -> np.ndarray:
    """Greedy MI solver with relative convergence tolerance.
    
    Parameters
    ----------
    Sigma : ndarray (Nm, Nm)
        Covariance matrix. Its diagonal is removed internally if dr=True.
    G : ndarray (Nm, Ng)
        Steering vector matrix (dictionary), not necessarily normalized;
        columns are normalized internally.
    niter : int
        Maximum number of iterations.
    dr : bool
        Diagonal removal mode.
    rel_eps : float
        Relative improvement threshold for early stopping.
        
    Returns
    -------
    x : ndarray (Ng,)
        Source power estimate.
    """
    norms2 = np.sum(np.square(np.abs(G)), axis=0)
    
    if dr:
        Sigmaloc = Sigma - np.diag(np.diag(Sigma))
        norms4 = np.square(norms2) - np.sum(np.square(np.square(np.abs(G))), axis=0)
        G = G / np.sqrt(np.sqrt(norms4))
    else:
        Sigmaloc = np.copy(Sigma)
        G = G / np.sqrt(norms2)

    Gconj = G.conj()
    G2 = np.square(np.abs(G))
    
    obj = np.zeros([niter + 1])
    obj[0] = np.sum(np.abs(Sigmaloc)**2) / 2

    b = proddamastranspose(G, Sigmaloc)
    x = np.zeros(b.shape)
    acols: dict[int, np.ndarray] = {}
    grad = -b
     
    for ni in range(niter):
        improvgrad = np.square(grad) / 2 * (x > grad)
        improv0 = grad * x - np.square(x) / 2
                
        idxgrad = np.argmax(improvgrad)
        idx0 = np.argmax(improv0)
        imp = np.maximum(improvgrad[idxgrad], improv0[idx0])

        if improvgrad[idxgrad] > improv0[idx0]:
            d = grad[idxgrad]
            x[idxgrad] -= d
            
            if idxgrad not in acols:
                if dr:
                    acols[idxgrad] = compute_a_dr(Gconj, G2, G[:, idxgrad],
                                                   np.abs(G[:, idxgrad])**2)
                else:
                    acols[idxgrad] = compute_a(Gconj, G[:, idxgrad])
            grad -= d * acols[idxgrad]
            obj[ni + 1] = obj[ni] - improvgrad[idxgrad]
        else:
            xold = x[idx0]
            x[idx0] = 0

            if idx0 not in acols:
                if dr:
                    acols[idx0] = compute_a_dr(Gconj, G2, G[:, idx0],
                                                np.abs(G[:, idx0])**2)
                else:
                    acols[idx0] = compute_a(Gconj, G[:, idx0])
            grad -= xold * acols[idx0]
            obj[ni + 1] = obj[ni] - improv0[idx0]
                    
        if imp / obj[ni] < rel_eps:
            break
    
    if dr:
        x = x / np.sqrt(norms4)
    else:
        x = x / norms2
        
    return x
