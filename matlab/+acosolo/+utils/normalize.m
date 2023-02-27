function An = normalize(A)

An = A ./ sqrt(sum(abs(A).^2, 1));

end