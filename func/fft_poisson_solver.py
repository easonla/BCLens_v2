import numpy as np
from scipy.fftpack import fft2, ifft2


def transform_matrix(rho, h2, N):
    W = np.exp(2. * np.pi * 1j / N)
    Wn = 1.
    Wm = 1.
    for m in range(N):
        for n in range(N):
            domi = (Wm + 1.0 / Wm + Wn + 1.0 / Wn) - 4
            if domi != 0:
                rho[m, n] *= h2 / domi
            Wn *= W
        Wm *= W
    return rho


def fft_poisson_solver(fxy,h2,N):
    """
    solving poisson equation with periodic boundary condition using fast fourier transformation
    (d/dx^2 + d/dy^2)phi = - fxy
    input : fxy np.array(N,N)
    output: phi np.array(N,N)
    """
    rho = fft2(fxy)
    phi = transform_matrix(rho, h2, N)
    return np.real(ifft2(phi))
