function map = bmap_unc(Sigma, A)
    
    A_norm = acosolo.utils.normalize(A);
    map = real(sum((A_norm' * Sigma) .* A_norm.', 2));
end
