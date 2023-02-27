function [Xs, amps, v, nu, Pest] = sfw_comet1(Xm, k, Data, Xgrid, Niter, LX, UX)

%% SFW for COMET1

% Xm microphone positions Mx3
% k wavenumber
% Data covariance matrix, MxM
% Xgrid initialization grid Nx3
% tolpos tolamp tolerance for source fusion and removal (0 recommended for
% greedy version)
% Niter max number of iterations (in greedy version, = number of sources)
% LX UX bounds of the domain

% return
% Xs estimated positions
% amps estimated powers
% v estimated noise variance
% nu stopping criterion
% Pest power reestimated by least-squares

Dgrid = dictionary(Xm, Xgrid, k);
options_nu = optimoptions(@fmincon,'Display', 'off', 'Algorithm','sqp', 'OptimalityTolerance', 1e-12);
options_amps = optimoptions(@fmincon,'Display', 'off', 'Algorithm','sqp', 'SpecifyObjectiveGradient',false, 'CheckGradient', false, 'OptimalityTolerance', 1e-12);
options_all = optimoptions(@fmincon,'Display', 'off', 'Algorithm','sqp',  'OptimalityTolerance', 1e-12);

Xs = zeros(0, 3);
amps = zeros(0, 1);
v = 1;
tolnu = 1e-5;

Datainv = inv(Data);

lbv = 0.01;

for u = 1:Niter
    
    fprintf("Iteration %u\t %u sources\t", u, size(Xs, 1))
    
    Dloc = dictionary(Xm, Xs, k);
    C = Dloc*(amps.*Dloc') + (lbv+v) * eye(size(Data));
    
    Cinv = eye(size(Data))/(lbv + v) - Dloc * inv(diag(1./amps) + Dloc'*Dloc/(lbv + v)) * Dloc' / (lbv+v) ^2;
    
    %% Adding a source

    [xnew, nu] = maximize_nu(Xm, k, Xgrid, Dgrid, Data, Datainv, C, Cinv, LX, UX, options_nu);
    
    fprintf("nu = %.4f\n", nu)
    
    % adding the source to the set
    Xs = [Xs ; xnew];

    %% Optimization of the amplitudes
    Dloc = dictionary(Xm, Xs, k);
    
    
    [amps, v] = optimize_amplitudes(Dloc, Data, Datainv, amps, v, options_amps, lbv);  

    %% Joint optimization of the amplitudes and positions 
    [Xs, amps, v] = optimize_all(Xm, k, Data, Datainv, Xs, amps, v, LX, UX, options_all, lbv);


end

D = dictionary(Xm, Xs, k);
Dpinv = pinv(D);

lambdas = eig(Data);
lambdas = sort(abs(lambdas), 'asc');
lambda0 = mean(lambdas(1:end-Niter));

PPest = Dpinv * (Data - lambda0*eye(size(Data))) * Dpinv';


Pest = max(real(diag(PPest)), 0);
        %% Max iter. reached
        fprintf("Max iter, stopping\n")
    
end


function [Xnu, nu] = maximize_nu(Xm, k, Xgrid, Dgrid, Data, Datainv, C, Cinv, LX, UX, options)

RR = Datainv - Cinv * Data * Cinv;

nugrid = -real(sum(conj(Dgrid).*(RR*Dgrid), 1));

[~, idx] = max(nugrid);
xnewgrid = Xgrid(idx, :);

nuf = @(X) nux_cov(Xm, k, RR, X);
[Xnu, numin] = fmincon(nuf, xnewgrid, [], [], [], [], LX, UX, [], options);
nu = -numin;
%Xnu
end



function [Xs, amps, v] = optimize_all(Xm, k,Data, Datainv, Xs, amps, v, LX, UX, options, lbv)

    xopt = [Xs(:); amps(:) ; v];

    % bounds
    Ns = length(amps);
    lbounds = [ones(Ns,1)*LX(1) ; ones(Ns, 1)*LX(2); ones(Ns, 1)*LX(3); zeros(Ns+1, 1)];
    ubounds = [ones(Ns,1)*UX(1) ; ones(Ns, 1)*UX(2); ones(Ns, 1)*UX(3); Inf(Ns+1, 1)]; % no upper bounds on amplitudes
    
    
    ZZ = fmincon(@(x) obj_amplitudes_positions(Xm, k, Data, Datainv, x, lbv), xopt, [], [], [], [], lbounds, ubounds, [], options);

   
    % extaction of the amplitudes and positions
    Xs = reshape(ZZ(1:3*Ns), Ns, 3);
    amps = ZZ(3*Ns+1:end-1);
    v = ZZ(end);
    
end

function [amps, v] = optimize_amplitudes(Dloc, Data, Datainv, amps, v, options, lbv)

    [ampsv] = fmincon(@(x) obj_amplitudes(Dloc, Data, Datainv, x, lbv), [amps ; eps ; v], [], [], [], [], zeros(size(amps, 1)+2,1),[], [], options);
    amps = ampsv(1:end-1);
    v = ampsv(end);
     
end

function [J] = obj_amplitudes_positions(Xm, k, Data, Datainv, xx, lbv)

Ns = (length(xx)-1)/4;

Xs = reshape(xx(1:3*Ns), Ns, 3);
x = xx(3*Ns+1:end-1);
v = xx(end);

D =  dictionary(Xm, Xs, k);
C = D*(x.*D') + (lbv+v) * eye(size(Data));

sup = (x ~= 0);

Z = diag(1./x(sup)) + D(:, sup)'*D(:, sup)/(lbv + v);


Cinv = eye(size(Data))/(lbv + v) - D(:, sup) * inv(Z) * D(:, sup)' / (lbv+v) ^2;

J = real(trace(Cinv*Data) + trace(C * Datainv));


end



function [J] = obj_amplitudes(D, Data, Datainv, x, lbv)

len = length(x);

C = D*(x(1:end-1).*D') + eye(size(Data)) * (lbv+x(end));

a = x(1:end-1);
v = x(end);

sup = (a > 10*eps);


Z = diag(1./a(sup)) + D(:, sup)'*D(:, sup)/(lbv + x(end));


Cinv = eye(size(Data))/(lbv + x(end)) - D(:, sup) * inv(Z) * D(:, sup)' / (lbv+x(end)) ^2;


J = real(trace(Cinv*Data) + trace(C * Datainv));



end

function [nu] = nux_cov(Xm, k, RR, Xnu)

    d = dictionary(Xm, Xnu, k);
    
    

    nu = real(d'*(RR)*d);

end

