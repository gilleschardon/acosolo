function sig = generate_noise(Nm, Ns, sigma2)

sig = (randn(Nm, Ns) + 1i * randn(Nm, Ns)) * sqrt(sigma2/2);

end