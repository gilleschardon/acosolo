function crit = bf_unc_crit(Sigma, g, x)

gx = acosolo.utils.normalize(g(x));       

crit = real(gx' * Sigma * gx);

end
    