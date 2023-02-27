k = 20;

Array = acosolo.utils.square_array(1, 5, [0,0,0], 'z');

Z = 2;

g = @(x) acosolo.sourcemodels.freefield2D(Array, x, Z, k);

[Xgrid, dimgrid] = acosolo.utils.grid3D([-1, -1, Z], [1, 1, Z], 0.01);

% coordinates of the sources
XYZs = [ 0.5, 0., Z ; -0.5, 0.1, 2 ; 0.4, -0.2, Z  ; -0.5, -0.5, Z];

Nsnaps = 100;

% covariance matrix of the sources
Sigma_source = [4, 2, 0, 0 ; 2, 2, 0, 0 ; 0, 0, 1, 1 ; 0, 0, 1, 1];

Nsources= 4;

sigma2 = 1;

sig_source = acosolo.utils.generate_correlated_sources(g(XYZs), Nsnaps, Sigma_source);
sig_noise = acosolo.utils.generate_noise(size(Array, 1), Nsnaps, sigma2);

sig0 = sig_source + sig_noise;
    
Sigma0 = acosolo.utils.scm(sig0);

% source dictionary
A = g(Xgrid);

% estimation of the covariance matrices and indices of the selected sources
[S_est, idx] = acosolo.cmf.cmf_correl_ols(Sigma0, A, Nsources);
    
% beamforming map
bmap = acosolo.beamforming.bmap_unc(Sigma0, A);
%%

subplot(2, 2, 1) 

scatter(XYZs(:, 1), XYZs(:, 2))
hold on
scatter(Xgrid(idx, 1), Xgrid(idx, 2))
axis('equal')
xlim([-1, 1])
ylim([-1, 1])

subplot(2, 2, 2)
imagesc(reshape(bmap, dimgrid(2), dimgrid(1)))
axis xy

subplot(2, 2, 3)
imagesc(abs(Sigma_source))
subplot(2, 2, 4)

imagesc(abs(S_est))
