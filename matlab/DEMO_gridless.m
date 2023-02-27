
load data_sfw.mat

g = @(x) acoloso.sourcemodels.freefield(Pmic, x, k);

[Xinit, dimgrid] = acosolo.utils.grid3D([-2, -1, 4], [1, 0, 5], 0.05);

LX = [-2, -1, 4];
UX = [1, 0, 5];

Niter = 4;

[Xc2, Pc2] = acosolo.gridless.sfw_comet2(Pmic, k, Data, Xinit, Niter, LX, UX);
[Xcmf, Pcmf] = acosolo.gridless.sfw_cmf(Pmic, k, Data, Xinit, 0, Niter, LX, UX);
[Xcond, RE, IM] = acosolo.gridless.sfw_multi(Pmic, k, data(:, 1:10), Xinit, 0, Niter, LX, UX);
% beamforming in the source plane
Z = 4.6;
g2D = @(x) acosolo.sourcemodels.freefield2D(Pmic, x, Z, k);

[Xgrid, dimgrid] = acosolo.utils.grid3D([-2, -1, Z], [1, 0, Z], 0.01);
A = g2D(Xgrid);
bmap = acosolo.beamforming.bmap_unc(Data, A);

%%

figure

scatter3(Xc2(:, 1), Xc2(:, 3), Xc2(:, 2))
hold on
scatter3(Xcmf(:, 1), Xcmf(:, 3), Xcmf(:, 2))
scatter3(Xcond(:, 1), Xcond(:, 3), Xcond(:, 2))

scatter3(Pmic(:, 1), Pmic(:, 3), Pmic(:, 2))

axis equal

legend('COMET2', 'CMF', 'cond', 'Array')


figure
imagesc(reshape(bmap, dimgrid(2), dimgrid(1)))
axis xy
axis image