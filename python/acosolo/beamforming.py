#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Beamforming maps and Maximum Likelihood estimators.

Supports synchronous single-array beamforming, MLE with analytical
noise/power estimation, and extensions to asynchronous multi-array
configurations (relaxed and strict models).

Also includes adaptive beamforming (MVDR) and MUSIC subspace method.

Key functions
-------------
bmap_unc : Beamforming map (Formulation IV/III)
bmap_cond : Conditional beamforming map using raw data
bmap_cond_sph :Conditional beamforming map using SH subspace
mle_cond : Conditional MLE for single source estimation
MVDR : Minimum Variance Distortionless Response adaptive beamforming
music : MUltiple SIgnal Classification subspace method
bf_unc_crit : Beamforming criterion for MLE
bf_unc_crit_async : Criterion for asynchronous multi-array BF
mle_unc : MLE with optional known noise variance
mle_unc_async_relax : Relaxed MLE for async arrays
mle_unc_async_strict : Strict MLE for async arrays with joint pos/power optimization
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.linalg import norm, inv, eigh
from .utils import normalize


def bmap_unc(Sigma: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Beamforming map with matrix of steering vectors G."""
    G_norm = normalize(G)
    return np.real(np.sum((G_norm.conj().T @ Sigma) * G_norm.T, 1))


def bmap_cond(Data: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Conditional beamforming map using raw measurement data.

    Computes the real-valued beamforming map for the conditional model
    where complex amplitudes are estimated directly from the data.
    Normalizes by the squared norm of each steering vector column,
    matching the convention of bmap_unc.

    Parameters
    ----------
    Data : ndarray (Nm, Nsnaps)
        Measurement (data) matrix.
    G : ndarray (Nm, Ng)
        Steering vector matrix (dictionary), not necessarily normalized.

    Returns
    -------
    map : ndarray (Ng,)
        Conditional beamforming map at each grid point.
    """
    G_norm = normalize(G)
    proj = G_norm.conj().T @ Data
    return np.real(np.sum(proj * proj.conj(), axis=1))


def bmap_cond_sph(Data: np.ndarray, Dict_sph: np.ndarray) -> np.ndarray:
    """Conditional beamforming map using spherical harmonic subspace.

    For each grid point, projects the measurement data onto the
    orthonormal spherical harmonic basis returned by
    ``freefield_sph_dictionary(normalized=True)`` and returns the
    squared Frobenius norm of the projection as the beamforming score.

    Parameters
    ----------
    Data : ndarray (Nm, Nsnaps)
        Measurement (data) matrix.
    Dict_sph : ndarray (Nm, Ng, Z)
        Spherical harmonic dictionary from ``freefield_sph_dictionary``
        with ``normalized=True``.  Columns are orthonormal per grid point.

    Returns
    -------
    map : ndarray (Ng,)
        Conditional beamforming map at each grid point.
    """
    # Dict_sph: (Nm, Ng, Z) -> (Ng, Z, Nm) for conjugate-transpose multiply
    # For each grid point j: proj_j = Dict_sph[:, j, :].conj().T @ Data
    # Dict_sph.transpose(1, 2, 0)[j] = Dict_sph[:, j, :].T, so we need .conj()
    B = Dict_sph.transpose(1, 2, 0).conj()  # (Ng, Z, Nm)
    proj = B @ Data  # (Ng, Z, Nsnaps)
    return np.real(np.sum(np.abs(proj) ** 2, axis=(1, 2)))


def mle_cond(Data: np.ndarray, g, X_init: np.ndarray,
             G_init: np.ndarray | None = None,
             bounds: tuple[np.ndarray, np.ndarray] | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """Conditional MLE for a single source with complex amplitudes per snapshot.

    Estimates the source position and per-snapshot complex amplitudes by
    maximizing the conditional beamforming map criterion (projection of
    the data onto the normalized steering vector), then solving for the
    amplitudes via least squares.

    Parameters
    ----------
    Data : ndarray (Nm, Nsnaps)
        Measurement (data) matrix.
    g : callable
        Steering vector function: g(X) returns steering vector (Nm,) for a
        single position, or (Nm, N) for multiple positions.
    X_init : ndarray (Ngrid, dim)
        3-D search grid for initialization.
    G_init : ndarray, optional
        Precomputed steering matrix from g(X_init).
    bounds : tuple of ndarray, optional
        Optimization bounds [[lo, lo, lo], [hi, hi, hi]]. If None,
        computed from X_init min/max per dimension.

    Returns
    -------
    X : ndarray (dim,)
        Estimated source position.
    A : ndarray (Nsnaps,)
        Estimated per-snapshot complex amplitudes.
    """
    if G_init is None:
        G_init = g(X_init)

    map_init = bmap_cond(Data, G_init)
    idx = np.argmax(map_init)
    Xinitbf = X_init[idx, :]

    if bounds is None:
        lb = X_init.min(axis=0)
        ub = X_init.max(axis=0)
        boxbounds = scipy.optimize.Bounds(lb, ub)
    else:
        boxbounds = scipy.optimize.Bounds(bounds[0], bounds[1])

    def objfun(X):
        gx = g(X)
        if gx.ndim == 1:
            gx = gx[:, np.newaxis]
        ngX2 = np.real(np.sum(gx * gx.conj(), axis=0))
        proj = gx.conj().T @ Data
        val = np.real(np.sum(proj * proj.conj(), axis=0)) / ngX2
        return -val.ravel()[0]

    res = scipy.optimize.minimize(objfun, Xinitbf, bounds=boxbounds)
    X = res.x

    gX = g(X)
    if gX.ndim == 1:
        gX = gX[:, np.newaxis]
    ngX2 = np.real(np.sum(gX * gX.conj(), axis=0))
    A = np.sum(gX.conj() * Data, axis=0) / ngX2

    return X, A


def MVDR(Sigma: np.ndarray, G: np.ndarray, eps: float = 1e-10,
         normalized: bool = True) -> np.ndarray:
    """MVDR adaptive beamforming map.
    
    Parameters
    ----------
    Sigma : ndarray (Nm, Nm)
        Covariance matrix.
    G : ndarray (Nm, Ng)
        Steering vector matrix.
    eps : float
        Regularization factor (default 1e-10).
    normalized : bool
        If True, normalize steering vectors (default True).
    """
    G_norm = normalize(G) if normalized else G
    Sigma_inv = inv(Sigma + eps * np.eye(Sigma.shape[0]))
    return 1 / np.real(np.sum((G_norm.conj().T @ Sigma_inv) * G_norm.T, 1))


def music(Sigma: np.ndarray, G: np.ndarray, nsources: int) -> np.ndarray:
    """MUSIC subspace beamforming map.
    
    Parameters
    ----------
    Sigma : ndarray (Nm, Nm)
        Covariance matrix.
    G : ndarray (Nm, Ng)
        Steering vector matrix (normalized internally).
    nsources : int
        Number of sources for signal subspace estimation
        (1 <= nsources < Nm).
    """
    if not 1 <= nsources < Sigma.shape[0]:
        raise ValueError(
            f"nsources must satisfy 1 <= nsources < Nm={Sigma.shape[0]}, "
            f"got {nsources}")
    lam, v = eigh(Sigma)
    sc = v[:, -nsources:].T.conj() @ normalize(G)
    nproj = np.sum(np.abs(sc)**2, axis=0)
    return 1 / (1 - nproj)


def bf_unc_crit(Sigma: np.ndarray, g, x: np.ndarray) -> float:
    """Beamforming criterion for MLE."""
    gx = normalize(g(x))
    return np.real(gx.conj().T @ Sigma @ gx)


def bf_unc_crit_async(Sigmas: list[np.ndarray], gs: list, Nsnaps: list[int],
                      x: np.ndarray) -> float:
    """Beamforming criterion for asynchronous multi-array MLE."""
    bf = 0.0
    for n in range(len(Sigmas)):
        gx = normalize(gs[n](x))
        bf += np.real(gx.conj().T @ Sigmas[n] @ gx) * Nsnaps[n]
    return bf


def mle_unc(Sigma: np.ndarray, g, X_init: np.ndarray, sigma2: float | None = None,
            G_init: np.ndarray | None = None) -> tuple[np.ndarray, float, float] | tuple[np.ndarray, float]:
    """MLE for a single source with optional known noise variance.
    
    Parameters
    ----------
    Sigma : ndarray (Nm, Nm)
        Covariance matrix.
    g : callable
        Steering vector function: g(X) returns steering vector.
    X_init : ndarray (Ngrid, dim)
        Search grid.
    sigma2 : float, optional
        Known noise variance. If None, estimate analytically.
    G_init : ndarray, optional
        Precomputed steering matrix from g(X_init).
        
    Returns
    -------
    X : ndarray (dim,)
        Estimated source position.
    p : float
        Estimated source power.
    sigma2 : float, optional
        Estimated noise variance (only if sigma2 argument was None).
    """
    if G_init is None:
        G_init = g(X_init)
        
    map_init = bmap_unc(Sigma, G_init)
    idx = np.argmax(map_init)
    Xinitbf = X_init[idx, :]
    
    objfun = lambda x: -bf_unc_crit(Sigma, g, x)
    res = scipy.optimize.minimize(objfun, Xinitbf)

    X = res.x
    gX = g(X)
    ngX2 = np.real(np.sum(gX * gX.conj()))
    B = np.real(gX.conj().T @ Sigma @ gX)
 
    if sigma2 is not None:
        p = (B / ngX2 - sigma2) / ngX2
        return X, p
    
    N = Sigma.shape[0]
    sigma2 = np.real((np.trace(Sigma) - B / ngX2) / (N - 1))
    p = np.real((B / ngX2 - sigma2) / ngX2)
    return X, p, sigma2


def mle_unc_async_relax(Sigmas: list[np.ndarray], gs: list, X_init: np.ndarray,
                        sigma2: float, Nsnaps: list[int],
                        G_init: list[np.ndarray] | None = None) -> tuple[np.ndarray, float]:
    """Relaxed MLE for asynchronous multi-array configuration.
    
    Parameters
    ----------
    Sigmas : list of ndarray
        Covariance matrices for each array.
    gs : list of callable
        Steering vector functions for each array.
    X_init : ndarray
        Search grid.
    sigma2 : float
        Noise variance.
    Nsnaps : list of int
        Number of snapshots per array.
    G_init : list of ndarray, optional
        Precomputed steering matrices.
    """
    if not G_init:
        G_init = [gs[n](X_init) for n in range(len(Sigmas))]
        
    map_init = bmap_unc(Sigmas[0], G_init[0]) * Nsnaps[0]
    for n in range(1, len(Sigmas)):
        map_init += bmap_unc(Sigmas[n], G_init[n]) * Nsnaps[n]
    
    idx = np.argmax(map_init)
    Xinitbf = X_init[idx, :]
    
    objfun = lambda x: -bf_unc_crit_async(Sigmas, gs, Nsnaps, x)
    res = scipy.optimize.minimize(objfun, Xinitbf)
    X = res.x
    
    p = 0.0
    for n in range(len(Sigmas)):
        gX = gs[n](X)
        ngX2 = np.real(np.sum(gX * gX.conj()))
        B = np.real(gX.conj().T @ Sigmas[n] @ gX)
        p += (B / ngX2 - sigma2) / ngX2 * Nsnaps[n]
    p /= np.sum(Nsnaps)
    
    return X, p


def mle_unc_async_strict(Sigmas: list[np.ndarray], gs: list, X_init: np.ndarray,
                         sigma2: float, Nsnaps: list[int],
                         G_init: list[np.ndarray] | None = None,
                         output_relaxed: bool = False
                         ) -> (tuple[np.ndarray, float]
                               | tuple[np.ndarray, float, np.ndarray, float]):
    """Strict MLE for asynchronous multi-array (joint position + power optimization).
    
    Parameters
    ----------
    Sigmas : list of ndarray
    gs : list of callable
    X_init : ndarray
    sigma2 : float
    Nsnaps : list of int
    G_init : list of ndarray, optional
    output_relaxed : bool
        If True, also return the relaxed-model estimate.
        
    Returns
    -------
    X, p, [Xr, pr] : tuple
        Position, power, and optionally relaxed estimates.
    """
    Xr, pr = mle_unc_async_relax(Sigmas, gs, X_init, sigma2, Nsnaps, G_init=G_init)
    
    def obj(Xp):
        X = Xp[:-1]
        p = Xp[-1]
        result = 0.0
        for n in range(len(Sigmas)):
            gx = gs[n](X)
            objl = (-p * np.real(gx.T.conj() @ Sigmas[n] @ gx)
                    / (sigma2 * (sigma2 + p * norm(gx)**2))
                    + np.log(sigma2 + p * norm(gx)**2)) * Nsnaps[n]
            result += objl
        return result
    
    XPinit = np.hstack([Xr, pr])
    dim = np.size(Xr)
    bounds = scipy.optimize.Bounds([-np.inf] * dim + [0],
                                   [+np.inf] * dim + [+np.inf])
    res = scipy.optimize.minimize(obj, XPinit, bounds=bounds)
    X = res.x[:-1]
    p = res.x[-1]
    
    if output_relaxed:
        return X, p, Xr, pr
    return X, p
