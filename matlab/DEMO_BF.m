
k = 5;

% Square regular arrays
Array1 = acosolo.utils.square_array(0.5, 5, [0,-2,0], 'y');
Array2 = acosolo.utils.square_array(0.5, 5, [0,2,0], 'y');

% Source models
g0 = @(x) acosolo.sourcemodels.freefield(Array1, x, k);
g1 = @(x) acosolo.sourcemodels.freefield(Array2, x, k);

gs = {g0, g1};


% Initialisation grid
[Xinit, dims] = acosolo.utils.grid3D([-1, -1, -1], [1, 1, 1], 0.1);

% Source
XYZs = [-0.0, 0.1, 0.1];
p = 1;

% Noise level and snapshots
Nsnaps = [200, 200];
sigma2 = 0.5;

    % Signals at the arrays
    sig0 = acosolo.utils.generate_source(g0(XYZs), Nsnaps(1), p) + acosolo.utils.generate_noise(size(Array1, 1), Nsnaps(1), sigma2);
    sig1 = acosolo.utils.generate_source(g1(XYZs), Nsnaps(2), p) + acosolo.utils.generate_noise(size(Array2, 1), Nsnaps(2), sigma2);
    
    % Sample covariance matrices
    Sigma0 = acosolo.utils.scm(sig0);
    Sigma1 = acosolo.utils.scm(sig1);
    
    % Estimates: Array1, Array2, Array1+Array2 asynchronous relaxed, and strict
    [Xarray1, Parray1] = acosolo.beamforming.mle_unc(Sigma0, g0,  Xinit, sigma2);    
    [Xarray2, Parray2] = acosolo.beamforming.mle_unc(Sigma1, g1,  Xinit, sigma2);   
    [Xrelax, Prelax] = acosolo.beamforming.mle_unc_async_relax({Sigma0, Sigma1}, {g0, g1},  Xinit, sigma2, Nsnaps);
    [Xstrict, Pstrict] = acosolo.beamforming.mle_unc_async_strict({Sigma0, Sigma1}, {g0, g1},  Xinit, sigma2, Nsnaps);

    %Xest[n, :, 3], Pest[n, 3], Xest[n, :, 2], Pest[n, 2] = mle_unc_async_strict([Sigma0, Sigma1], gs, Xinit, sigma2, Nsnaps, output_relaxed=True)

%%
figure()


scatter3(Xarray1(1), Xarray1(2), Xarray1(3))
hold on
scatter3(Xarray2(1), Xarray2(2), Xarray2(3))
scatter3(Xrelax(1), Xrelax(2), Xrelax(3))
scatter3(Xstrict(1), Xstrict(2), Xstrict(3))

% the actual position    
scatter3(XYZs(1), XYZs(2),XYZs(3))

% arrays
scatter3(Array1(:, 1), Array1(:, 2), Array1(:, 3))
scatter3(Array2(:, 1), Array2(:, 2), Array2(:, 3))


axis('equal')
legend("BF 1", "BF 2", "relax", "strict",  "source", "Array 1", "Array 2")