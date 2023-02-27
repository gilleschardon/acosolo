function sig = generate_correlated_sources(G, Ns, Sigma_source)

Ssqrt = sqrtm(Sigma_source);
sig = G * Ssqrt * (randn(size(G, 2), Ns) + 1i * randn(size(G, 2), Ns));

end