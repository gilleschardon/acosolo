function [X, p] = mle_unc(Sigma, g, X_init, sigma2)

    A_init = g(X_init);
        
    map_init = acosolo.beamforming.bmap_unc(Sigma, A_init);
    
    [~, idx] = max(map_init);
    Xinitbf = X_init(idx, :);
    
    objfun = @(x) - acosolo.beamforming.bf_unc_crit(Sigma, g, x);
    X = fminunc(objfun, Xinitbf);

    gX = g(X);
    ngX2 = real(sum(abs(gX).^2));
    
    B = real(gX' * Sigma * gX);
 

    p = (B / ngX2 - sigma2) / ngX2   ; 

end