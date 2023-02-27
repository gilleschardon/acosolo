function y = proddamasDR(D, x)

% fast product by Dtilde transpose D (product by the large DAMAS matrix),
% diagonal removal

z = D*(x.*D');
z = z - diag(diag(z));

y = real(sum(conj(D) .* (z*D), 1).');

end