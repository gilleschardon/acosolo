"""acosolo - ACOustical SOurce Localization with Optimization methods.

Implements beamforming, CMF (Covariance Matrix Fitting), gridless methods,
and MLE estimators for acoustical source localization.

Submodules
----------
beamforming : Beamforming maps, MLE estimators for single sources
cmf : CMF-NNLS (Lawson-Hanson), correlated source OLS
gridless : Off-grid CMF with iterative source pick + optimization
sourcemodels : Steering vector models (free-field monopole/dipole)
utils : Array grids, signal generation, SCM, normalization
"""

from . import beamforming, cmf, gridless, sourcemodels, utils

__all__ = [
    "beamforming",
    "cmf",
    "gridless",
    "sourcemodels",
    "utils",
]
