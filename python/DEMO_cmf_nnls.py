#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:21:40 2020

@author: gilleschardon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import acosolo.cmf as cmf
from acosolo.sourcemodels import freefield2D
from acosolo.beamforming import bmap_unc
from acosolo.utils import grid3D


# CMF-NNLS

# Ref.: G. Chardon, J. Picheral, F. Ollivier, Theoretical analysis of the DAMAS algorithm and efficient implementation of the Covariance Matrix Fitting method for large-scale problems, Journal of Sound and Vibration,

mat = loadmat("damasdata2D.mat")
k = mat['k'][0,0]
Sigma = mat['Data']
Array = mat['Pmic']

# Normalization, for stability
Sigma = Sigma / np.max(Sigma)

# Source grid
Z = 4.6

g2D = lambda x : freefield2D(Array, x, Z, k)

Xgrid, dimgrid = grid3D([-2, -1, Z], [1, 0, Z], 0.02)
A = g2D(Xgrid)
bmap = bmap_unc(Sigma, A)

cmf_map = cmf.cmf_nnls_lh(A, Sigma, dr = True)


cmfDB = 10*np.log10(cmf_map)

def plotmap(pmap, name, dynrange):
    mapDB = 10*np.log10(pmap)
    m = np.max(mapDB)    
    plt.figure()
    plt.imshow(mapDB, cmap='hot', vmax=m, vmin=m-dynrange, origin="lower")
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='black')
    plt.axis('image')
    plt.title(name)
    ax = plt.gca()
    ax.set_facecolor('black')



#%%


plotmap(np.reshape(cmf_map, [dimgrid[1], dimgrid[0]]), 'CMF-NNLS', 20)
plotmap(np.reshape(bmap, [dimgrid[1], dimgrid[0]]), 'Beamforming', 20)


