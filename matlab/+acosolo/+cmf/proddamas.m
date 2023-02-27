function y = proddamas(D, x)

% fast product by Dtilde transpose D (product by the large DAMAS matrix)

z = D*(x.*D');

y = real(sum(conj(D) .* (z*D), 1).');

end