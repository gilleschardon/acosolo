function D = freefield2D(PX, PS, Z, k)

% PX positions of the sources
% PS positions of the array
% k wavenumber
% source dictionary D

dx = PX(:, 1) - PS(:, 1)';
dz = Z;
dy = PX(:, 2) - PS(:, 2)';


d = sqrt(dx.^2 + dz.^2 + dy.^2);

D = exp(- 1i * k * d)./d;

end
