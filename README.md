# BCLens_v2 

This project is part of my master thesis "Constrain the Bullet Cluster using Strong and Weak lensing". 
It was developed to analyze the cosmological N-body simulation done by Gadget2.
key feature including:
- project 3D simulation result into 2D mass (gas and Dark matter) sheet and X-ray observable (500eV-2000eV,2000eV-5000eV,5000eV-8000eV) and SZ effect.
- derive gravitational strong and weak lensing effect given surface mass distribution using finite differential method and multiple grid gauss seidel algorithm.
- determind best timestep, viewing angle and offset (dt,phi,theta,psi,dx,dy) by minimizing chi square function.
- including two efficient optimizing methods, monte carto markov chain algorithm and gradient decent method.

### Development
- ploting routine
