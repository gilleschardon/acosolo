#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:21:40 2020

@author: gilleschardon
"""

import numpy as np
import matplotlib.pyplot as plt
from acosolo.sourcemodels import freefield
from acosolo.beamforming import mle_unc, mle_unc_async_strict
from acosolo.utils import square_array, generate_source, generate_noise, scm
from acosolo.utils import grid3D


# A demo of MLE/beamforming, + MLE for asynchronous arrays

# Refs.:
# G. Chardon, Theoretical analysis of beamforming steering vector formulations for acoustic source localization, Journal of Sound and Vibration, 2022
# G. Chardon, Maximum likelihood estimators and Cramér-Rao bounds for the localization of an acoustical source with asynchronous arrays, under review

# wavenumber
k = 5

# Square regular arrays
Array1 = square_array(0.5, 5, [0,-2,0], axis='y')
Array2 = square_array(0.5, 5, [0,2,0], axis='y')

# Source models
g0 = lambda x : freefield(Array1, x, k)
g1 = lambda x : freefield(Array2, x, k)

gs = [g0, g1]


# Initialisation grid
Xinit, _ = grid3D([-1, -1, -1], [1, 1, 1], 0.1)

# Source
XYZs = np.array([-0.0, 0.1, 0.1])
p = 1

# Noise level and snapshots
Nsnaps = np.array([200, 200])
sigma2 = 0.5

# Here we plot an example, but we could estimate the bias and the variance (and MSE) using Monte-Carlo
Ntest = 1
Xest = np.zeros([Ntest, 3, 4])
Pest = np.zeros([Ntest, 4])

for n in range(Ntest):
    
    # Signals at the arrays
    sig0 = generate_source(g0(XYZs), Nsnaps[0], p) + generate_noise(Array1.shape[0], Nsnaps[0], sigma2)
    sig1 = generate_source(g1(XYZs), Nsnaps[1], p) + generate_noise(Array2.shape[0], Nsnaps[1], sigma2)
    
    # Sample covariance matrices
    Sigma0 = scm(sig0)
    Sigma1 = scm(sig1)
    
    # Estimates: Array1, Array2, Array1+Array2 asynchronous relaxed, and strict
    Xest[n, :, 0], Pest[n, 0] = mle_unc(Sigma0, g0,  Xinit, sigma2)    
    Xest[n, :, 1], Pest[n, 1] = mle_unc(Sigma1, g1,  Xinit, sigma2)   
    Xest[n, :, 3], Pest[n, 3], Xest[n, :, 2], Pest[n, 2] = mle_unc_async_strict([Sigma0, Sigma1], gs, Xinit, sigma2, Nsnaps, output_relaxed=True)

#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

labels = ["BF 1", "BF 2", "relaxed", "strict"]

# the four estimated positions
for n in range(4):
    ax.scatter(Xest[:, 0, n], Xest[:, 1, n], Xest[:, 2, n], label=labels[n])

# the actual position    
ax.scatter(XYZs[0], XYZs[1],XYZs[2],label="source")

# arrays
ax.scatter(Array1[:, 0], Array1[:, 1], Array1[:, 2],label="Array 1")
ax.scatter(Array2[:, 0], Array2[:, 1], Array2[:, 2],label="Array 2")


ax.axis('equal')
ax.legend()