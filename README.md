# acosolo
ACOustical SOurce Localization with Optimization methods

Implementations of some optimization (or sparsity) base methods for source localization:

* Beamforming, Maximum likelihood for one source [1]
* An optimized implementation of CMF [2]
* Gridless methods, for the conditional  [3] (estimation of the powers of random sources) of
unconditional [4] models(estimation of the amplitudes of deterministic sources)
* Greedy localization of correlated sources [5]
* Maximum likelihood for the localization of a source with asynhronous arrays [6]


What you will **not** find here:

* beamforming with formulations I, II, or any combination which is not IV for the position and III for the power, the only efficient one (in the statistical sense)
* DAMAS, which converges very slowly to the solution of CMF
* gridless versions of (HR)-CLEAN-SC, which are not as accurate as COMET2
* methods that assume a particular form of the source model (e.g. far-field sources, etc.)


Demo scripts:

* `DEMO_BF.py`: demonstration of beamforming with synchronous or asynchronous arrays, using simulated data
* `DEMO_cmf_nnls.py`: demonstration of CMF, with experimental data
* `DEMO_gridless.py`: gridless source localization, with experimental data
* `DEMO_CMF_OLS`: localization of correlated sourcesn, with simulated data


Author: G. Chardon

References :

* [1] G. Chardon, Theoretical analysis of beamforming steering vector formulations for acoustic source localization, Journal of Sound and Vibration, 2022
* [2] G. Chardon, J. Picheral, F. Ollivier, Theoretical analysis of the DAMAS algorithm and efficient implementation of the Covariance Matrix Fitting method for large-scale problems, Journal of Sound and Vibration,
* [3] G. Chardon, Gridless covariance matrix fitting methods for three dimensional acoustical source localization, Journal of Sound and Vibration, 2023
* [5] G. Chardon, U. Boureau, Gridless three-dimensional compressive beamforming with the Sliding Frank-Wolfe algorithm, The Journal of the Acoustical Society of America, 2021
* [5] G. Chardon, F. Ollivier, J. Picheral, Localization of sparse and coherent sources by Orthogonal Least Squares, Journal of the Acoustical Society of America, 2019
* [6] G. Chardon, Maximum likelihood estimators and Cramér-Rao bounds for the localization of an acoustical source with asynchronous arrays, under review
