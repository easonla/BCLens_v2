# BCLens_v2 

This project is part of my master thesis "Constrain the Bullet Cluster using Strong and Weak lensing". 
It was developed to analyze the cosmological N-body simulation done by Enzo.

key feature including:
- project 3D simulation result into 2D mass (gas and Dark matter) sheet and X-ray intensity (500eV-2000eV,2000eV-5000eV,5000eV-8000eV) and SZ effect.
- derive gravitational strong and weak lensing effect given surface mass distribution using finite differential method and fftw PDE solver
- including two efficient optimizing methods, basinhoping algorithm (Stochastic Metropolitan algorithm) and LSQ method. 

Loss function is defined as chi^2 = chi_mass^2 + chi_xray^2

# Result
![Mass Lensing result](https://github.com/easonla/BCLens_v2/mass_lensing.png)
![Xray result](https://github.com/easonla/BCLens_v2/Xray.png)

