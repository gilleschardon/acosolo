function sig = generate_source(g, Ns, p)

sig = g * ((randn(1, Ns) + 1i * randn(1, Ns)) * sqrt(p/2));
   
end