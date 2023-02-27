function Sigma = scm(sig)

Sigma = sig * sig' / size(sig, 2);

end