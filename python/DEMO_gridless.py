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
from acosolo.gridless import gridless_cmf
from acosolo.utils import grid3D
from acosolo.beamforming import bmap_unc

# Gridless CMF

# Ref: G. Chardon, Gridless covariance matrix fitting methods for three dimensional acoustical source localization, Journal of Sound and Vibration, 2023


# Experimental data
mat = loadmat("damasdata2D.mat")
k = mat['k'][0,0]
Sigma = mat['Data']
Array = mat['Pmic']

# Normalization, for stability
Sigma = Sigma / np.max(Sigma)

#%%
# source model
g = lambda x : freefield(Array, x, k)

# init grid
Xinit, dimgrid = grid3D([-2, -1, 4], [1, 0, 5], 0.05)

# box constraint for the positions
box = np.array([[-2, -1, 4],[1, 0, 5]])

# CMF and COMET2
Xc2, Pc2, sigma2c2 = gridless_cmf(Sigma, g, Xinit, 4, box)
Xcmf, Pcmf, sigma2cmf = gridless_cmf(Sigma, g, Xinit, 4, box, mode="cmf")

# beamforming in the source plane
Z = 4.6
g2D = lambda x : freefield2D(Array, x, Z, k)

Xgrid = grid3D([-2, -1, Z], [1, 0, Z], 0.01)
A = g2D(Xgrid)
bmap = bmap_unc(Sigma, A)


#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(Xc2[:, 0], Xc2[:, 1],Xc2[:, 2],label="COMET2")
ax.scatter(Xcmf[:, 0], Xcmf[:, 1],Xcmf[:, 2],label="CMF")

ax.axis('equal')
ax.legend()

plt.figure()
plt.imshow(np.reshape(bmap, [dimgrid[1], dimgrid[0]]), origin="lower")
