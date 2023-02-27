function [C, sel] = cmf_correl_ols(Data, A, Ns)


%% OLS for covariance matrix estimation

% orthogonal basis of the identified atoms

Aorth = zeros(size(A, 1), 0);

% selected atoms
sel = [];


for n = 1:Ns
    
    % data projected on the identified atoms
    projdata = (Aorth*Aorth')*Data*(Aorth*Aorth');
    
    % residual
    res = Data - projdata;

    % projections of the atoms on the orthogonal of the identified atoms
    projA = A - Aorth*(Aorth'*A); % L * (  N*K  +  K*N) % N*L
    np = sqrt(sum(abs(projA).^2, 1));
    projA = projA ./ np; % L * N

    % computation of the criterion (see (20))
    DAorth = Data*Aorth; % N^2K  % N*K
    
    nproj1 = abs(sum(projA'*Data .* (projA.'), 2)).^2; % L*N^2
    nproj2 = 2 * sum(abs(projA'*DAorth).^2, 2); % L*N^2
      
    nproj = nproj1+nproj2;

    % we do not test already selected atoms, this is not plain old matching
    % pursuit
    nproj(sel) = NaN;     
        
     % new atom = argmax of the criterion
    [~, idx] = max(nproj);
    % we record the norm of the residual
    nres(n) = norm(res, 'fro');
    % we add the atom to the set
    sel = [sel idx];
    % we add a dimension to the space
    Aorth = [Aorth projA(:, idx)];

end

% final projection
As = A(:, sel);
pAs = pinv(As);
C = pAs * Data * pAs';

end