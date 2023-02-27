function y = proddamasDR_support(D, x, S)

% fast product by Dtilde transpose D (product by the large DAMAS matrix),
% diagonal removal
% S support of x

z = D(:, S)*(x(S).* D(:, S)');
z = z - diag(diag(z));

y = real(sum(conj(D) .* (z*D), 1).');

end