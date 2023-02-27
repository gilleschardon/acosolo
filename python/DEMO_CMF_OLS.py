#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:21:40 2020

@author: gilleschardon
"""

import numpy as np
import matplotlib.pyplot as plt
from acosolo.sourcemodels import freefield2D
from acosolo.beamforming import bmap_unc
from acosolo.utils import square_array, generate_noise, scm, grid3D, generate_correlated_sources
from acosolo.cmf import CMF_OLS_correl


# CMF-OLS for the localization of correlated sources

# Ref.: G. Chardon, F. Ollivier, J. Picheral, Localization of sparse and coherent sources by Orthogonal Least Squares, Journal of the Acoustical Society of America, 2019

k = 20

Array = square_array(1, 5, [0,0,0], axis='z')

Z = 2

g = lambda x : freefield2D(Array, x, Z, k)

Xgrid, dimgrid = grid3D([-1, -1, Z], [1, 1, Z], 0.01)

# coordinates of the sources
XYZs = np.array([ [0.5, 0., Z], [-0.5, 0.1, 2], [0.4, -0.2, Z],[-0.5, -0.5, Z]])

Nsnaps = 100

# covariance matrix of the sources
Sigma_source = np.array([[4, 2, 0, 0], [2, 2, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])

Nsources= 4

sigma2 = 1

sig_source = generate_correlated_sources(g(XYZs), Nsnaps, Sigma_source)
sig_noise = generate_noise(Array.shape[0], Nsnaps, sigma2)

SNR = 10 * np.log10(np.sum(sig_source*sig_source.conj()) / np.sum(sig_noise*sig_noise.conj()))

sig0 = sig_source + sig_noise
    
Sigma0 = scm(sig0)

# source dictionary
A = g(Xgrid)

# estimation of the covariance matrices and indices of the selected sources
S_est, idx = CMF_OLS_correl(Sigma0, A, Nsources)
    
# beamforming map
bmap = bmap_unc(Sigma0, A)
#%%
fig, axs = plt.subplots(2, 2)

axs[0,0].scatter(XYZs[:, 0], XYZs[:, 1])
axs[0,0].scatter(Xgrid[idx, 0], Xgrid[idx, 1])
axs[0,0].set_aspect('equal')
axs[0,0].set_xlim([-1, 1])
axs[0,0].set_ylim([-1, 1])


axs[0, 1].imshow(np.reshape(bmap, [dimgrid[1], dimgrid[0]]), origin="lower")

axs[1,0].imshow(np.abs(Sigma_source))
axs[1,1].imshow(np.abs(S_est))