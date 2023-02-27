function [X, p] = mle_unc_async_strict(Sigmas, gs, X_init, sigma2, Nsnaps)

[Xr, pr] = acosolo.beamforming.mle_unc_async_relax(Sigmas, gs, X_init, sigma2, Nsnaps);
    

XPinit = [Xr pr];    
lbounds = [-inf, -inf, -inf, 0];
ubounds = [+inf, +inf, +inf, +inf];
    
    
            
function objj = obj(Xp)
X = Xp(1:end-1);
p = Xp(end);
objj = 0;
for n = 1:length(Sigmas) 
    gx = gs{n}(X);
    objl = (- p * (real(gx' * Sigmas{n} * gx)) / (sigma2 * (sigma2 + p * norm(gx)^2)) + log(sigma2 + p * norm(gx)^2)) * Nsnaps(n);
    objj = objj + objl     ;      
end

end



Xp = fmincon(@obj, XPinit, [], [], [], [], lbounds, ubounds);



    X = Xp(1:end-1);
    p = Xp(end);    

end
      