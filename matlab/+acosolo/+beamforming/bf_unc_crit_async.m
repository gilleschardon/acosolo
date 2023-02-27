function crit = bf_unc_crit_async(Sigmas, gs, Nsnaps, x)

crit = 0;

for n = 1:length(Sigmas)

    gx = acosolo.utils.normalize(gs{n}(x));       

    crit = crit + Nsnaps(n) * real(gx' * Sigmas{n} * gx);

end
    