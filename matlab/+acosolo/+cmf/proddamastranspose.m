function x = proddamastranspose(D, Data)

% fast product by Dtilde transpose

x = real(sum((D'*Data).*D.', 2));

end