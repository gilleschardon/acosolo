function [X, p] = mle_unc_async_relax(Sigmas, gs, X_init, sigma2, Nsnaps)

    map_init = zeros(size(X_init, 1), 1);
    for n = 1:length(Sigmas)
        A_init = gs{n}(X_init);
        
        map_init = map_init + Nsnaps(n) * acosolo.beamforming.bmap_unc(Sigmas{n}, A_init);
    end
    
    [~, idx] = max(map_init);
    Xinitbf = X_init(idx, :);
    
    objfun = @(x) - acosolo.beamforming.bf_unc_crit_async(Sigmas, gs, Nsnaps, x);
    X = fminunc(objfun, Xinitbf);

    p = 0;
    
    for n = 1:length(Sigmas)
        gX = gs{n}(X);
        ngX2 = real(sum(abs(gX).^2));
    
        B = real(gX' * Sigmas{n} * gX);
 

        p = p + Nsnaps(n) * (B / ngX2 - sigma2) / ngX2   ; 
        
    end
    
    p = p / sum(Nsnaps);

end