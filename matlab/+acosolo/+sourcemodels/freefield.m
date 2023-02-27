function D = freefield(PX, PS, k)

% PX positions of the sources
% PS positions of the array
% k wavenumber
% source dictionary D

dx = PX(:, 1) - PS(:, 1)';
dz = PX(:, 3) - PS(:, 3)';
dy = PX(:, 2) - PS(:, 2)';


d = sqrt(dx.^2 + dz.^2 + dy.^2);

D = exp(- 1i * k * d)./d;

end
