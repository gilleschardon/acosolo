#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:21:40 2020

@author: gilleschardon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from acosolo.sourcemodels import freefield, freefield2D
from acosolo.gridless import gridless_cmf, gridless_cond
from acosolo.utils import grid3D, scm
from acosolo.beamforming import bmap_unc


# Gridless CMF

# Refs:
# G. Chardon, Gridless covariance matrix fitting methods for three dimensional acoustical source localization, Journal of Sound and Vibration, 2023
# G. Chardon, U. Boureau, Gridless three-dimensional compressive beamforming with the Sliding Frank-Wolfe algorithm, The Journal of the Acoustical Society of America, 2021

# Experimental data
mat = loadmat("data_sfw.mat")

k = mat['k'][0,0] # wavenumber
Sigs = mat['data'] # signals (a row of the STFT, for each microphone)
Sigs = Sigs / 1000 # to avoid numerical problems
Sigma = scm(Sigs) # Spatial covariance matrix
Array = mat['Pmic'] # Array coordinates

#%%
# source model
g = lambda x : freefield(Array, x, k)

# init grid
Xinit, dimgrid = grid3D([-2, -1, 4], [1, 0, 5], 0.05)

# box constraint for the positions
box = np.array([[-2, -1, 4],[1, 0, 5]])

# Conditional model, using 20 snapshots (we estimate the complex amplitudes of the sources for each snapshot)
Xs, Rs, Is = gridless_cond(Sigs[:, :20], g, Xinit, 4, box)

# CMF and COMET2
Xc2, Pc2, sigma2c2 = gridless_cmf(Sigma, g, Xinit, 4, box, mode="comet2")
Xcmf, Pcmf, sigma2cmf = gridless_cmf(Sigma, g, Xinit, 4, box, mode="cmf")

# beamforming in the source plane
Z = 4.6
g2D = lambda x : freefield2D(Array, x, Z, k)

Xgrid, dimgrid = grid3D([-2, -1, Z], [1, 0, Z], 0.01)
A = g2D(Xgrid)
bmap = bmap_unc(Sigma, A)


#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(Xc2[:, 0], Xc2[:, 2],Xc2[:, 1],label="COMET2")
ax.scatter(Xcmf[:, 0], Xcmf[:, 2],Xcmf[:, 1],label="CMF")
ax.scatter(Xs[:, 0], Xs[:, 2],Xs[:, 1],label="cond")

ax.scatter(Array[:, 0], Array[:, 2], Array[:, 1], label="array")

ax.axis('equal')
ax.legend()

plt.figure()
plt.imshow(np.reshape(bmap, [dimgrid[1], dimgrid[0]]), origin="lower")
