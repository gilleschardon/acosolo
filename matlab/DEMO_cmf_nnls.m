
%% Demo of CMF-NNLS
%
% 2D inversion, grid size = 10800

% loads the data (cov. matrix, frequency, array coordinates)
load damasdemo

% we select the 64 inner microphones
N = Pmic(:, 2).^2 + Pmic(:, 1).^2;
[~, order] = sort(N);
Nm = 64;
Z = order(1:Nm);
Pmic = Pmic(Z, :);
Data = Data(Z, Z);

close all

% source grid
Lx = 180;
Ly = 60;
Lz = 1;
xx = linspace(-2, 1, Lx)';
yy = linspace(-1, 0, Ly)';
zz = 4.4;
[Xg, Yg, Zg] = meshgrid(xx, yy, zz);

% dictionary of sources
D = acosolo.sourcemodels.freefield(Pmic, [Xg(:) Yg(:) Zg(:)], k);

%% Beamforming (no normalization)
% used as input for DAMAS
Cbf = sum(conj(D) .* (Data*D), 1);
Cbf = real(Cbf');
%% Beamforming (normalized)
% used to plot the power map
Dbf = D ./ sum(abs(D).^2, 1);
Cbfn = sum(conj(Dbf) .* (Data*Dbf), 1);
Cbfn = real(Cbfn');
%% optimized NNLS

tic;
xlh = acosolo.cmf.cmf_nnls(D, Data, 1e2);


%% optimized NNLS - diagonal removal

tic;
xlhdr = acosolo.cmf.cmf_nnls_dr(D, Data, 1e2);



%% Maps: CMF-NNLS

figure
set(gcf, 'Position',  [100, 100, 500, 200])

imagesc(xx,yy,reshape(10*log10(xlh), Ly, Lx))
axis xy
axis image
colormap((hot))

title('CMF-NNLS')

xlabel("X")
ylabel("Y")
colorbar

%% Maps: CMF-NNLS diagonal removal

figure
set(gcf, 'Position',  [100, 100, 500, 200])

imagesc(xx,yy,reshape(10*log10(xlhdr), Ly, Lx))
axis xy
axis image
colormap((hot))
title('CMF-NNLS diagonal removal')

xlabel("X")
ylabel("Y")
colorbar



%% Maps: Beamforming

figure
set(gcf, 'Position',  [100, 100, 500, 200])

imagesc(xx,yy,reshape(10*log10(Cbfn), Ly, Lx))
axis xy
axis image
colormap((hot))
title('Beamforming')

colorbar
ax = gca;
xlabel("X")
ylabel("Y")
