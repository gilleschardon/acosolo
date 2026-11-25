#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Off-grid (gridless) covariance matrix fitting methods.

Iterative greedy methods that jointly optimize source positions and
amplitudes, avoiding grid discretization errors inherent in fixed-grid
beamforming and DAMAS approaches.

Supports three variants:
  - "cmf" : conventional CMF cost
  - "comet1" : inverse-based cost
  - "comet2" : default, second-order inverse cost

Key functions
-------------
gridless_cmf : Off-grid CMF with position+amplitude optimization
gridless_cond : Conditional model — estimates complex amplitudes per snapshot
maximize_nu_cmf / maximize_nu_cond : Source selection criterion
optimize_amplitudes_* : Amplitude-only or joint amplitude/position refinement
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import inv, norm
import scipy.optimize

# --- Named constants ---

SIGMA2_MIN = 1e-3       # Minimum noise variance to avoid singular covariance matrices
BOX_MARGIN = 1e-2       # Margin added to box bounds for optimization
SIGMA2_INIT = 1e-3      # Initial noise power estimate
AMP_THRESHOLD = 1e-4    # Threshold below which amplitude is treated as zero


def gridless_cmf(Sigma: np.ndarray, g, X_init: np.ndarray, Niter: int, box: np.ndarray,
                 G_init: np.ndarray | None = None, mode: str = "comet2",
                 verbose: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Off-grid CMF with position+amplitude optimization.
    
    Parameters
    ----------
    Sigma : ndarray (Nm, Nm)
        Spatial covariance matrix.
    g : callable
        Steering vector function: g(X) returns (Nm, Ns) steering matrix.
    X_init : ndarray (Ngrid, 3)
        3-D search grid.
    Niter : int
        Number of source additions.
    box : ndarray (2, 3)
        Optimization bounds [[lo, lo, lo], [hi, hi, hi]].
    G_init : ndarray (Nm, Ngrid), optional
        Precomputed steering matrix. Generated from g(X_init) if None.
    mode : str, optional
        Cost function: "cmf", "comet1", or "comet2" (default).
    verbose : bool, optional
        If True, print iteration progress (default True).
        
    Returns
    -------
    Xs : ndarray (Ns, 3)
        Optimized source positions.
    Ps : ndarray (Ns,)
        Source powers.
    sigma2 : float
        Estimated noise variance (includes the floor SIGMA2_MIN).
    """
    # generates the dictionary on the grid, if not given as an argument
    if G_init is None:
        G_init = g(X_init)
        
    if mode == 'comet1':
        Sigma_inv = inv(Sigma)
    else:
        Sigma_inv = None
        
    G_init_norm = G_init / np.sqrt(np.real(np.sum(G_init * np.conj(G_init), axis=0)))
        
    dim = X_init.shape[1]

    Xs = np.zeros([0, dim])
    Ps = np.zeros([0])
    sigma2 = 1.0

    for iter in range(Niter):
        
        if verbose:
            print(f"Iter {iter+1}/{Niter}")
        
        Xnew, nu = maximize_nu_cmf(Sigma, Sigma_inv, g, Xs, Ps, sigma2, SIGMA2_MIN,
                                   X_init, G_init_norm, box, mode)
        Xs = np.vstack([Xs, Xnew])
                
        Ps, sigma2 = optimize_amplitudes_cmf(Sigma, Sigma_inv, g, Xs, Ps, sigma2, SIGMA2_MIN, mode)
        Xs, Ps, sigma2 = optimize_amplitudes_positions_cmf(Sigma, Sigma_inv, g, Xs, Ps,
                                                           sigma2, SIGMA2_MIN, box, dim, mode)

    # Internally sigma2 is the excess over the floor SIGMA2_MIN; the model
    # noise variance is their sum.
    return Xs, Ps, sigma2 + SIGMA2_MIN


def maximize_nu_cmf(Sigma: np.ndarray, Sigma_inv: np.ndarray | None, g,
                    Xs: np.ndarray, Ps: np.ndarray, sigma2: float,
                    sigma2l: float, X_init: np.ndarray, G_init: np.ndarray,
                    box: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    """Maximizes the nu criterion for a new source.
    
    Returns
    -------
    Xnew : ndarray (3,)
        Optimized new source position.
    nu : float
        Maximum criterion value.
    """
    Gloc = g(Xs)
    sigma2p = sigma2l + sigma2
    
    if mode == 'cmf':
        Sigma_est = Gloc @ (Gloc * Ps).T.conj() + sigma2p * np.eye(Sigma.shape[0])
        RR = Sigma - Sigma_est
    else:
        # Sources with (near-)zero power are left out, as in
        # amplitudesobj_cmf, to avoid dividing by zero in diag(1/Ps).
        sup = Ps > AMP_THRESHOLD
        if np.any(sup):
            Gs = Gloc[:, sup]
            Sigma_est_inv = (np.eye(Sigma.shape[0]) / sigma2p
                             - Gs @ inv(np.diag(1/Ps[sup]) + Gs.T.conj() @ Gs / (sigma2p))
                               @ Gs.T.conj() / sigma2p**2)
        else:
            Sigma_est_inv = np.eye(Sigma.shape[0]) / sigma2p

        if mode == 'comet1':
            Sigma_est = Gloc @ (Gloc * Ps).T.conj() + sigma2p * np.eye(Sigma.shape[0])
            RR = Sigma_est_inv @ Sigma @ Sigma_est_inv - Sigma_inv
        else:
            RR = Sigma_est_inv @ Sigma @ Sigma @ Sigma_est_inv
    
    nugrid = np.real(np.sum(G_init.conj() * (RR @ G_init), axis=0)
                     - np.sum(np.abs(G_init**2), axis=0))
    idx = np.argmax(nugrid)
    Xgrid = X_init[idx, :]
    
    def nuobj(X):
        gx = g(X)
        gx = gx / np.sqrt(np.sum(gx * gx.conj()))
        return -np.real(gx.T.conj() @ RR @ gx)
   
    boxbounds = scipy.optimize.Bounds(box[0, :] + BOX_MARGIN, box[1, :] - BOX_MARGIN)
    res = scipy.optimize.minimize(nuobj, Xgrid, bounds=boxbounds)

    return res.x, -res.fun


def optimize_amplitudes_cmf(Sigma: np.ndarray, Sigma_inv: np.ndarray | None, g,
                            Xs: np.ndarray, Ps: np.ndarray, sigma2: float,
                            sigma2l: float, mode: str) -> tuple[np.ndarray, float]:
    """Optimize source powers and noise variance.
    
    Returns
    -------
    Ps_new : ndarray
        Optimized source powers.
    sigma2_new : float
        Optimized noise variance.
    """
    Gloc = g(Xs)
    objfunamp = lambda x: amplitudesobj_cmf(Sigma, Sigma_inv, Gloc, x, sigma2l, mode)
    init = np.hstack([Ps, SIGMA2_INIT, sigma2])
    res = scipy.optimize.minimize(objfunamp, init,
                                  bounds=scipy.optimize.Bounds(np.zeros([Ps.shape[0] + 2])))
    return res.x[:-1], res.x[-1]


def amplitudesobj_cmf(Sigma: np.ndarray, Sigma_inv: np.ndarray | None, Gloc: np.ndarray,
                      x: np.ndarray, sigma2l: float, mode: str) -> float:
    """Objective function for amplitude optimization."""
    Ps = x[:-1]
    sigma2p = x[-1]
    
    I = np.eye(Sigma.shape[0])
    Sigma_est = Gloc @ (Gloc * Ps).T.conj() + (sigma2l + sigma2p) * I
    
    if mode == 'cmf':
        return np.sum(np.abs(Sigma - Sigma_est)**2)
    
    sup = Ps > AMP_THRESHOLD
    
    if np.max(Ps) > AMP_THRESHOLD:
        Sigma_est_inv = I / (sigma2l + sigma2p) \
                        - Gloc[:, sup] @ inv(np.diag(1/Ps[sup])
                                             + Gloc[:, sup].T.conj() @ Gloc[:, sup]
                                               / (sigma2l + sigma2p)) \
                          @ Gloc[:, sup].T.conj() / (sigma2l + sigma2p)**2
    else:
        Sigma_est_inv = I / (sigma2l + sigma2p)
        
    if mode == 'comet1':
        return np.real(np.trace(Sigma_est_inv @ Sigma + Sigma_est @ Sigma_inv))
    else:
        return np.real(np.trace(Sigma @ Sigma_est_inv @ Sigma) + np.trace(Sigma_est))


def optimize_amplitudes_positions_cmf(Sigma: np.ndarray, Sigma_inv: np.ndarray | None,
                                      g, Xs: np.ndarray, Ps: np.ndarray, sigma2: float,
                                      sigma2l: float, box: np.ndarray, dim: int,
                                      mode: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimize source positions jointly with powers.
    
    Returns
    -------
    Xs_new : ndarray (Ns, 3)
    Ps_new : ndarray (Ns,)
    sigma2_new : float
    """
    xinit = np.hstack([Xs.ravel(), Ps, sigma2])
    Ns = Ps.shape[0]
    
    lb = np.hstack([np.kron(np.ones([Ns]), box[0, :]), np.zeros([Ns + 1])])
    ub = np.hstack([np.kron(np.ones([Ns]), box[1, :]), np.inf * np.ones([Ns + 1])])
    boxbounds = scipy.optimize.Bounds(lb, ub)

    objfun = lambda x: ampposobj_cmf(Sigma, Sigma_inv, g, x, sigma2l, dim, mode)
    options = {"disp": True, "iprint": 1}
    res = scipy.optimize.minimize(objfun, xinit, bounds=boxbounds, options=options)
    
    Xs = np.reshape(res.x[:dim * Ns], [Ns, dim])
    Ps = res.x[dim * Ns : -1]
    sigma2 = res.x[-1]

    return Xs, Ps, sigma2


def ampposobj_cmf(Sigma: np.ndarray, Sigma_inv: np.ndarray | None, g, x: np.ndarray,
                  sigma2l: float, dim: int, mode: str) -> float:
    """Objective for joint amplitude+position optimization."""
    Ns = (x.shape[0] - 1) // (dim + 1)
    X = np.reshape(x[:dim * Ns], [Ns, dim])
    Psigma2 = x[dim * Ns:]
    Gloc = g(X)
    return amplitudesobj_cmf(Sigma, Sigma_inv, Gloc, Psigma2, sigma2l, mode)


def gridless_cond(Sigs: np.ndarray, g, X_init: np.ndarray, Niter: int, box: np.ndarray,
                  G_init: np.ndarray | None = None, verbose: bool = True) -> tuple:
    """Conditional model — estimates complex amplitudes per snapshot.
    
    Parameters
    ----------
    Sigs : ndarray (Nm, Nsnaps)
        Signal matrix (STFT rows).
    g : callable
        Steering vector function.
    X_init : ndarray (Ngrid, 3)
        3-D search grid.
    Niter : int
        Number of source additions.
    box : ndarray (2, 3)
        Optimization bounds.
    G_init : ndarray, optional
        Precomputed steering matrix.
    verbose : bool, optional
        If True, print iteration progress (default True).
        
    Returns
    -------
    Xs : ndarray (Ns, 3)
    Rs : ndarray (Ns, Nsnaps)
    Is : ndarray (Ns, Nsnaps)
    """
    Nsnaps = Sigs.shape[1]
    
    if G_init is None:
        G_init = g(X_init)
        
    G_init_norm = G_init / np.sqrt(np.real(np.sum(G_init * np.conj(G_init), axis=0)))
        
    dim = X_init.shape[1]

    Xs = np.zeros([0, dim])
    Rs = np.zeros([0, Nsnaps])
    Is = np.zeros([0, Nsnaps])

    for iter in range(Niter):
        
        if verbose:
            print(f"Iter {iter+1}/{Niter}")
        
        Xnew, nu = maximize_nu_cond(Sigs, g, Xs, Rs, Is, X_init, G_init_norm, box)
        Xs = np.vstack([Xs, Xnew])
                
        Rs, Is = optimize_amplitudes_cond(Sigs, g, Xs, Rs, Is)
        Xs, Rs, Is = optimize_amplitudes_positions_cond(Sigs, g, Xs, Rs, Is, box, dim)
        
    return Xs, Rs, Is


def maximize_nu_cond(Sigs: np.ndarray, g, Xs: np.ndarray, Rs: np.ndarray,
                     Is: np.ndarray, X_init: np.ndarray, G_init_norm: np.ndarray,
                     box: np.ndarray) -> tuple[np.ndarray, float]:
    """Maximizes nu for conditional model."""
    Gloc = g(Xs)
    
    A = Rs + 1j * Is
    R = Sigs - Gloc @ A

    nugrid = np.sum(np.abs(G_init_norm.T.conj() @ R)**2, 1)
    idx = np.argmax(nugrid)
    Xgrid = X_init[idx, :]
    
    def nuobj(X):
        gx = g(X)
        gx = gx / np.sqrt(np.sum(gx * gx.conj()))
        return -np.sum(np.abs(gx.T.conj() @ R)**2)
    
    boxbounds = scipy.optimize.Bounds(box[0, :], box[1, :])
    res = scipy.optimize.minimize(nuobj, Xgrid, bounds=boxbounds)

    return res.x, -res.fun


def _ls_amplitudes(Gloc: np.ndarray, Sigs: np.ndarray) -> np.ndarray:
    """Least-squares complex amplitudes: argmin_A ||Sigs - Gloc @ A||_F."""
    return np.linalg.lstsq(Gloc, Sigs, rcond=None)[0]


def optimize_amplitudes_cond(Sigs: np.ndarray, g, Xs: np.ndarray, Rs: np.ndarray,
                             Is: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimize complex amplitudes (amplitude only, no position update).

    For fixed positions the problem is linear, so the amplitudes are the
    exact least-squares solution. ``Rs`` and ``Is`` (the previous
    amplitudes) are not needed and kept only for interface compatibility.
    """
    A = _ls_amplitudes(g(Xs), Sigs)
    return A.real, A.imag


def optimize_amplitudes_positions_cond(Sigs: np.ndarray, g, Xs: np.ndarray, Rs: np.ndarray,
                                       Is: np.ndarray, box: np.ndarray, dim: int
                                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint amplitude+position optimization for conditional model.

    Variable projection: the amplitudes are eliminated by an exact
    least-squares solve for each candidate set of positions, so the
    optimizer only works on the Ns*dim position coordinates. ``Rs`` and
    ``Is`` are kept only for interface compatibility.
    """
    Ns = Xs.shape[0]
    boxbounds = scipy.optimize.Bounds(np.tile(box[0, :], [Ns]),
                                      np.tile(box[1, :], [Ns]))

    objfun = lambda x: positionsobj_cond(x, Sigs, g, Ns, dim)
    res = scipy.optimize.minimize(objfun, Xs.ravel(), bounds=boxbounds)

    Xs = np.reshape(res.x, [Ns, dim])
    A = _ls_amplitudes(g(Xs), Sigs)
    return Xs, A.real, A.imag


def positionsobj_cond(x: np.ndarray, Sigs: np.ndarray, g, Ns: int, dim: int) -> float:
    """Variable-projection objective for the conditional model.

    Squared residual norm after the least-squares amplitudes are fitted for
    the source positions ``x`` (flattened (Ns, dim) array).
    """
    Gloc = g(np.reshape(x, [Ns, dim]))
    return norm(Sigs - Gloc @ _ls_amplitudes(Gloc, Sigs))**2
