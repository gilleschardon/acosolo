#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for array geometry, signal simulation, and matrix ops.

Provides helpers for building microphone arrays, generating simulated
signals (coherent, incoherent, correlated), computing sample covariance
matrices, and linear algebra primitives used by the algorithms.

Key functions
-------------
square_array : Uniform planar microphone grid
grid3D : 3D spatial search grid generation
generate_noise / generate_source / generate_correlated_sources : Signal simulation
scm : Sample covariance matrix estimation
normalize : Unit-norm steering vector normalization
LS_powers : Least-squares power estimation
"""

from __future__ import annotations

import numpy as np
from numpy.random import normal
from scipy.linalg import sqrtm, eigh, pinv


def square_array(aperture: float, N: int, center: list[float] | tuple[float, ...] = None,
                 axis: str = 'z') -> np.ndarray:
    """Uniform planar microphone grid on a square aperture.
    
    Parameters
    ----------
    aperture : float
    N : int
    center : array-like, optional
    axis : str, optional
        Axis normal to the array plane ('x', 'y', or 'z'). Default 'z'
        (array in the xy plane).
        
    Returns
    -------
    coords : ndarray (N*N, 3)
        Microphone positions.
    """
    if center is None:
        center = [0.0, 0.0, 0.0]
    c = np.linspace(-aperture / 2, aperture / 2, N)
    
    if axis == 'x':
        xg, yg, zg = np.meshgrid(center[0], c + center[1], c + center[2])
    elif axis == 'y':
        xg, yg, zg = np.meshgrid(c + center[0], center[1], c + center[2])
    elif axis == 'z':
        xg, yg, zg = np.meshgrid(c + center[0], c + center[1], center[2])
    else:
        raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
        
    return np.vstack([xg.ravel(), yg.ravel(), zg.ravel()]).T


def grid3D(lb: list[float] | tuple[float, ...], ub: list[float] | tuple[float, ...],
           step: float) -> tuple[np.ndarray, list[int]]:
    """3-D spatial search grid.
    
    Parameters
    ----------
    lb : array-like (3,)
        Lower bounds [x, y, z].
    ub : array-like (3,)
        Upper bounds [x, y, z].
    step : float
        Grid spacing.
        
    Returns
    -------
    Xg : ndarray (Npoints, 3)
        Grid points.
    dims : list of int
        Grid dimensions [Nx, Ny, Nz].
    """
    # Tolerance so that e.g. (1 - 0) / 0.1 = 9.999... still gives 11 points.
    def _npts(lo, hi):
        return int(np.floor((hi - lo) / step + 1e-9)) + 1

    xx = np.linspace(lb[0], ub[0], _npts(lb[0], ub[0]))
    yy = np.linspace(lb[1], ub[1], _npts(lb[1], ub[1]))
    zz = np.linspace(lb[2], ub[2], _npts(lb[2], ub[2]))

    dims = [int(xx.size), int(yy.size), int(zz.size)]

    Xg, Yg, Zg = np.meshgrid(xx, yy, zz)
    return np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T, dims


def generate_noise(Nm: int, Ns: int, sigma2: float) -> np.ndarray:
    """Complex Gaussian noise with specified variance.
    
    Parameters
    ----------
    Nm : int
        Number of microphones.
    Ns : int
        Number of snapshots.
    sigma2 : float
        Noise power.
        
    Returns
    -------
    noise : ndarray (Nm, Ns)
    """
    noise = normal(scale=np.sqrt(sigma2 / 2), size=(Nm, Ns)) \
            + 1j * normal(scale=np.sqrt(sigma2 / 2), size=(Nm, Ns))
    return noise


def generate_source(g: np.ndarray, Ns: int, p: float) -> np.ndarray:
    """Generate a signal from a source with specified power.
    
    Parameters
    ----------
    g : ndarray (Nm,) or (Nm, 1)
        Steering vector.
    Ns : int
        Number of snapshots.
    p : float
        Source power.
        
    Returns
    -------
    sig : ndarray (Nm, Ns)
    """
    return np.reshape(g, (-1, 1)) @ (normal(scale=np.sqrt(p / 2), size=(1, Ns))
                               + 1j * normal(scale=np.sqrt(p / 2), size=(1, Ns)))


def generate_correlated_sources(G: np.ndarray, Ns: int, Sigma_source: np.ndarray
                                ) -> np.ndarray:
    """Generate signals from correlated sources.
    
    Parameters
    ----------
    G : ndarray (Nm, Ns)
        Steering matrix for the sources.
    Ns : int
        Number of snapshots.
    Sigma_source : ndarray (Ns, Ns)
        Source covariance matrix.
        
    Returns
    -------
    sig : ndarray (Nm, Ns)
    """
    Ssqrt = sqrtm(Sigma_source)
    return G @ Ssqrt @ ((normal(scale=np.sqrt(1 / 2), size=(G.shape[1], Ns))
                         + 1j * normal(scale=np.sqrt(1 / 2), size=(G.shape[1], Ns))))


def scm(sig: np.ndarray) -> np.ndarray:
    """Sample covariance matrix estimation.
    
    Parameters
    ----------
    sig : ndarray (Nm, Ns)
        Signal matrix.
        
    Returns
    -------
    Sigma : ndarray (Nm, Nm)
        Sample covariance matrix.
    """
    return sig @ sig.conj().T / sig.shape[1]


def normalize(G: np.ndarray) -> np.ndarray:
    """Unit-norm steering vector normalization.
    
    Parameters
    ----------
    G : ndarray (Nm, Ns)
        Steering vector matrix.
        
    Returns
    -------
    G_norm : ndarray (Nm, Ns)
    """
    return G / np.sqrt(np.real(np.sum(G * np.conj(G), axis=0)))


def LS_powers(Sigma: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Least-squares power estimation.
    
    Estimates source powers by projecting the covariance matrix
    onto the null space of the noise eigenvalue.
    
    Parameters
    ----------
    Sigma : ndarray (Nm, Nm)
        Covariance matrix.
    G : ndarray (Nm, Ns)
        Steering matrix.
        
    Returns
    -------
    Pest : ndarray (Ns,)
        Estimated source powers.
    """
    Ns = G.shape[1]
    if Ns >= Sigma.shape[0]:
        raise ValueError(
            f"LS_powers needs fewer sources than microphones to estimate "
            f"the noise level (sources={Ns}, mics={Sigma.shape[0]})")
    lambdas = eigh(Sigma, eigvals_only=True)
    lambda_noise = np.mean(lambdas[:-Ns])
    
    Gpinv = pinv(G)
    PPest = Gpinv @ (Sigma - lambda_noise * np.eye(Sigma.shape[0])) @ Gpinv.T.conj()
    return np.diag(PPest)
