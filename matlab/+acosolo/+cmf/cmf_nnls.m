function [x, res, Tlh, w] = cmf_nnls(D, Data, epsilon)

%% fast NNLS for CMF-NNLS

n = size(D, 2);

R = true(n, 1);
N = 1:n;

x = zeros(n, 1);


%% w = At (y - Ax)
Ay = acosolo.cmf.proddamastranspose(D, Data);
w = Ay - acosolo.cmf.proddamas(D, x);

iter = 1;
%%
tic

% R : set of active constraints (i.e., zero coefficients)
while (any(R)) && (max(w(R)) > epsilon)
    
    % logs
    obj = norm(D(:, ~R)*spdiags(x(~R),0,sum(~R), sum(~R))*D(:, ~R)' - Data, 'fro')^2;
    Tlh(iter) = toc;

    res(iter) = obj;
    fprintf("T %.2fs Iter %u  Obj. %.8e    Maxw %.4e\n", toc, iter, obj, max(w(R)));


    
    [~, idx] = max(w(R));
    Ridx = N(R);
    idx = Ridx(idx);
    R(idx) = false;
    s = zeros(size(x));
    
    %% small LS problem
    DR = D(:, ~R);
    Gram = abs(DR'*DR).^2;
    
    s(~R) = Gram\ acosolo.cmf.proddamastranspose(D(:, ~R), Data);
       
    
    
    while min(s(~R)) <= 0
        Q = (s <= 0) & (~R);
        alpha = min(x(Q)./(x(Q)-s(Q)));
        
        x = x + alpha * (s - x);
        R = ((x <= eps) & ~R) | R;

        s(:) = 0;
        
        %% small LS problem
        DR = D(:, ~R);

        Gram = abs(DR'*DR).^2;
    
        s(~R) = Gram\ acosolo.cmf.proddamastranspose(D(:, ~R), Data);

    
    end
    x = s;
    %% w = At (y - Ax)

    w = Ay - acosolo.cmf.proddamas_support(D, x, ~R);
    iter = iter + 1;
   
end
    obj = norm(D(:, ~R)*spdiags(x(~R),0,sum(~R), sum(~R))*D(:, ~R)' - Data, 'fro')^2;
    res(iter) = obj;
    Tlh(iter) = toc;
    DR = D(:, ~R);
    Gram = abs(DR'*DR).^2;

    try chol(Gram);
    disp('Solution is unique')
    catch ME
    disp('Solution is not unique')
    end
end
    