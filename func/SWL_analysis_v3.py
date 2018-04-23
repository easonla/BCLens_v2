import traceback
import time
import BulletConstants
from numpy import *
import numpy as np
from yt.mods import *
from yt.visualization.volume_rendering.api import ProjectionCamera

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import _cntr as cntr
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.axes3d import Axes3D

from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy import units as u
import astropy.cosmology as cosmo

Cosmo = cosmo.LambdaCDM(73., 0.270, 0.7299)
sigma_c = 0.3963

from scipy.optimize import fmin_powell as fminp
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import gaussian_filter, convolve
from scipy.ndimage.filters import sobel
from scipy.special import exp1, gamma  # Exponential integral
from scipy.interpolate import griddata

from pylab import *
from subprocess import *
from ctypes import *

import h5py
import sys
import pickle
import math
import random

xmin = ymin = -862.07416614
xmax = ymax = 862.07416614


class Array2d:
    def __init__(self, xmin, xmax, nx, ymin, ymax, ny):
        self.nx = nx
        self.ny = ny

        self.xmin = xmin
        self.ymin = ymin

        self.xmax = xmax
        self.ymax = ymax

        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny

        self.x = linspace(xmin + self.dx / 2, xmax - self.dx / 2, nx)
        self.y = linspace(ymin + self.dy / 2, ymax - self.dy / 2, ny)

        self.data = zeros([nx, ny])


class galaxy:
    def __init__(self):
        self.n = 0
        # initial image position
        self.x0 = []
        self.y0 = []
        # current image position
        self.xc = []
        self.yc = []
        # source position
        self.xs = []
        self.ys = []
        # redshift and redshift factor
        self.z = []
        self.zf = []
        # magnification
        self.mag = []
        self.rms = []
        self.chi = []

    def add(self, x, y, z):
        self.n = self.n + 1
        self.x0.append(x)
        self.y0.append(y)
        self.z.append(z)
        self.zf.append(Zfunc(z))

    def add_source(self, xs, ys):
        self.xs.append(xs)
        self.ys.append(ys)

    def update_image(self, dx, dy, phi):
        "rotation in radian"
        # phi = phi*np.pi/180.
        self.xc = np.array(self.x0) * np.cos(phi) - np.array(self.y0) * np.sin(phi) + dx
        self.yc = np.array(self.x0) * np.sin(phi) + np.array(self.y0) * np.cos(phi) + dy

    def add_mag(self, mag):
        self.mag.append(mag)

    def update_chi(self, sigma):
        x_bar = sum(self.xs) / self.n
        y_bar = sum(self.ys) / self.n
        self.rms = (np.array(self.xs) - x_bar) ** 2 + (np.array(self.ys) - y_bar) ** 2
        self.chi = (np.array(self.xs) - x_bar) ** 2 + (np.array(self.ys) - y_bar) ** 2 / (
                    sigma ** 2 * np.array(self.mag) ** 2)
        return self.chi

    def clean(self):
        self.mag = []
        self.xc = []
        self.yc = []
        self.xs = []
        self.ys = []
        self.rms = []
        self.chi = []


class LensProperties:
    def __init__(self, potential, kappa):
        self.p2r = 2 * potential.xmax / 4.413 / 3600 / 180 * np.pi / potential.nx  # rad /pixels
        self.p2a = 2 * potential.xmax / 4.413 / potential.nx
        self._potential = potential
        self._kappa = kappa
        self._n = potential.nx
        self._alpha = Array2d(potential.xmin, potential.xmax, potential.nx, potential.ymin, potential.ymax,
                              potential.ny)
        self._alpha_x = Array2d(potential.xmin, potential.xmax, potential.nx, potential.ymin, potential.ymax,
                                potential.ny)
        self._alpha_y = Array2d(potential.xmin, potential.xmax, potential.nx, potential.ymin, potential.ymax,
                                potential.ny)
        self._gamma = Array2d(potential.xmin, potential.xmax, potential.nx, potential.ymin, potential.ymax,
                              potential.ny)
        self._gamma1 = Array2d(potential.xmin, potential.xmax, potential.nx, potential.ymin, potential.ymax,
                               potential.ny)
        self._gamma2 = Array2d(potential.xmin, potential.xmax, potential.nx, potential.ymin, potential.ymax,
                               potential.ny)
        self._mag = Array2d(potential.xmin, potential.xmax, potential.nx, potential.ymin, potential.ymax, potential.ny)
        self._calc_div()

    def _calc_div(self):
        self._alpha_x.data, self._alpha_y.data = np.gradient(self._potential.data, self.p2r)
        self._alpha_x.data = -1 * self._alpha_x.data
        self._alpha_y.data = -1 * self._alpha_y.data
        self._alpha.data[:, :] = np.sqrt(self._alpha_x.data[:, :] ** 2 + self._alpha_y.data[:, :] ** 2)
        phi = self._potential.data
        domi = self.p2r
        n, n = np.shape(self._potential.data)
        k = extend_matrix(self._potential.data)
        gamma1 = np.zeros([n, n])
        gamma2 = np.zeros([n, n])
        gamma1[:, :] = 0.5 * ((k[2:, 1:-1] - 2. * k[1:-1, 1:-1] + k[0:-2, 1:-1]) / domi ** 2 - (
                    k[1:-1, 2:] - 2. * k[1:-1, 1:-1] + k[1:-1, 0:-2]) / domi ** 2)
        gamma2[:, :] = (k[2:, 2:] - k[2:, 0:-2] - k[0:-2, 2:] + k[0:-2, 0:-2]) / 4. / domi ** 2
        self._gamma1.data = gamma1 / (1. - self._kappa.data)
        self._gamma2.data = gamma2 / (1. - self._kappa.data)
        self._gamma.data = np.sqrt(self._gamma1.data ** 2 + self._gamma2.data ** 2)
        self._mag.data = ((1 - self._kappa.data) ** 2 - self._gamma.data ** 2)

    def alpha(self):
        return self._alpha

    def alphaX(self):
        return self._alpha_x

    def alphaY(self):
        return self._alpha_y

    def gamma1(self):
        return self._gamma_1

    def gamma2(self):
        return self._gamma_2

    def gamma(self):
        return self._gamma

    def mag(self):
        return self._mag


def findbestfit(pf, w, theta=0, psi=0, phi=0, ngrid=256, listname='B09'):
    try:
        start = time.time()
        simtime = pf.h.parameters['InitialTime'] * BulletConstants.TimeConversion
        nx = ny = ngrid
        mass = Array2d(2 * xmin, 2 * xmax, nx, 2 * ymin, 2 * ymax, ny)
        dmass = Array2d(2 * xmin, 2 * xmax, nx, 2 * ymin, 2 * ymax, ny)
        sumsim = Array2d(2 * xmin, 2 * xmax, nx, 2 * ymin, 2 * ymax, ny)
        kappa_sim = Array2d(2 * xmin, 2 * xmax, nx, 2 * ymin, 2 * ymax, ny)
        [mass, dmass] = ProjectEnzoCompact(pf, mass, phi, theta, psi)
        sumsim.data = gaussian_filter(dmass.data + mass.data, 2.0)
        kappa_sim.data = sumsim.data / sigma_c
        [LensP, err_plot] = MGGS_pot(kappa_sim)
        Lens = LensProperties(LensP, kappa_sim)
        galaxies = read_offset_coordinate1(listname)
        sheardata = Get_Shear()
        xopt = minimize(Lens, galaxies, sheardata, w)
        dx = xopt[0][0]
        dy = xopt[0][1]
        phi = xopt[0][2]
        chi = xopt[1]
        print "finish best minimize chi_sq, simtime = {:.3f} dx={:.3f} dy={:.3f} phi = {:.3f} theta = {:.3f} psi = {:.3f} fom = {:.5f}".format(
            simtime, dx, dy, phi, theta, psi, chi)
        eclaps = time.time() - start
        print "total time = {:.3f}".format(eclaps)
        return (chi, phi)
    except:
        print "findbestfit fail"
        print sys.exc_info()
        return (1E5, 0)


def minimize(Lens, galaxies, sheardata, w):
    start = time.time()
    xbcg, ybcg = findhalocenter(Lens._kappa)
    x0 = np.array([xbcg, ybcg, 0])
    # eps = np.array([Lens.p2a/200,Lens.p2a/200,1e-8])
    try:
        xopt = fminp(chi_sq, x0, (Lens, galaxies, sheardata, w), ftol=0.0001, full_output=1)
        print time.time() - start, 'sec'
        return xopt
    except:
        print "undable to minimize chi_sq"
        print  "sys.exc_info=", sys.exc_info(), "\n"
        traceback.print_exc()
        return 0


def change_coord(sheardata, dx, dy, phi):
    pos = np.array([np.array(sheardata['ra']), np.array(sheardata['dec'])])
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    new_pos = np.dot(R, pos)
    x = new_pos[0, :] + dx
    y = new_pos[1, :] + dy
    return x, y


def chi_strong(x0, Lens, galaxies):
    sigma = 0.6
    chi = 0
    dx, dy, phi = x0[0], x0[1], x0[2]
    for key in galaxies:
        gal = galaxies[key]
        gal.clean()
        gal.update_image(dx, dy, phi)
        for i in range(gal.n):
            xh = gal.xc[i]
            yh = gal.yc[i]
            zf = gal.zf[i]
            xi, yi = ray_trace(xh, yh, zf, Lens._alpha_x, Lens._alpha_y)
            Mi = BiLnrIntrp(xi, yi, Lens._mag)
            gal.add_source(xi, yi)
            gal.add_mag(Mi)
        chi += sum(gal.update_chi(sigma)) / gal.n
    return chi


def chi_weak(x0, Lens, sheardata, take_average=False):
    chi = 0
    sigma = 1.
    sigma_eps_s = 0.2
    sigma_eps_er = 0.1
    dx, dy, phi = x0[0], x0[1], x0[2]
    x, y = change_coord(sheardata, dx, dy, phi)
    Zf = np.array(sheardata['Zf'])
    g1 = np.array(sheardata['g_final[0]'])
    g2 = np.array(sheardata['g_final[0]'])
    if take_average:
        g1 = nearby_avg(x, y, g1)
        g2 = nearby_avg(x, y, g2)
    else:
        for i in range(len(x)):
            # gamma_prime = BiLnrIntrp(x_prime,y_prime,g)
            g1_prime = BiLnrIntrp(x[i], y[i], Lens._gamma1) * Zf[i]
            g2_prime = BiLnrIntrp(x[i], y[i], Lens._gamma1) * Zf[i]
            sigma = (1 - (g1_prime ** 2 + g2_prime ** 2)) ** 2 * sigma_eps_s ** 2 + sigma_eps_er ** 2
            chi += ((g1_prime - g1[i]) ** 2 + (g2_prime - g2[i]) ** 2) / sigma
    chi = chi / len(x)
    return chi


def chi_sq(x0, *arg):
    Lens = arg[0]
    galaxies = arg[1]
    sheardata = arg[2]
    w = arg[3]
    chi_SL = chi_strong(x0, Lens, galaxies)
    chi_WL = chi_weak(x0, Lens, sheardata)
    chi_total = chi_SL + w * chi_WL
    return chi_total


def read_offset_coordinate1(listname='B09'):
    if listname == 'B09':
        filename = 'B09_offset.txt'
    elif listname == 'P12':
        filename = 'P12_offset.txt'
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    galaxies = {}
    for i in range(len(lines)):
        name = lines[i].strip().split()[0][0]
        galaxies[name] = galaxy()
    for i in range(len(lines)):
        name = lines[i].strip().split()[0][0]
        x = float(lines[i].strip().split()[1])
        y = float(lines[i].strip().split()[2])
        z = float(lines[i].strip().split()[3])
        galaxies[name].add(x, y, z)
    return galaxies


def Get_Shear():
    from astropy.io import ascii
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    sheardata = ascii.read("shear_offset.dat")
    return sheardata


def shear_regrid(sheardata, x0):
    g1_grid = Array2d(2 * dataA.xmin, 2 * dataA.xmax, nx, 2 * dataA.ymin, 2 * dataA.ymax, ny)
    g2_grid = Array2d(2 * dataA.xmin, 2 * dataA.xmax, nx, 2 * dataA.ymin, 2 * dataA.ymax, ny)
    g_grid = Array2d(2 * dataA.xmin, 2 * dataA.xmax, nx, 2 * dataA.ymin, 2 * dataA.ymax, ny)
    [dyy, dxx] = meshgrid(g_grid.y / 4.413, g_grid.x / 4.413)
    dx, dy, phi = x0[0], x0[1], x0[2]
    x, y = change_coord(sheardata, dx, dy, phi)
    #     x = np.array(sheardata['ra'])
    #     y = np.array(sheardata['dec'])
    Zf = np.array(sheardata['Zf'])
    g1 = np.array(sheardata['g_final[0]'])
    g2 = np.array(sheardata['g_final[1]'])
    g1_avg, g2_avg, x1, y1 = nearby_avg(x, y, g1, g2)
    coord = np.append(dxx.reshape(nx * ny, 1), dyy.reshape(nx * ny, 1), axis=1)
    point = np.array([x1, y1]).T
    g1_roll = griddata(point, g1_avg, coord, method='cubic')
    g2_roll = griddata(point, g2_avg, coord, method='cubic')
    g1_unroll = g1_roll.reshape((nx, ny))
    g2_unroll = g2_roll.reshape((nx, ny))
    g1_grid.data = g1_unroll
    g2_grid.data = g2_unroll
    g_grid.data = np.sqrt(g1_unroll ** 2 + g2_unroll ** 2)
    return g1_grid, g2_grid, g_grid


def nearby_avg(x, y, g1, g2):
    n0 = 30
    N = 30
    g1_avg, g2_avg, x1, y1 = [], [], [], []
    count = 0
    for i in range(len(x)):
        x_prime = x[i]
        y_prime = y[i]
        D = np.sqrt((x - x_prime) ** 2 + (y - y_prime) ** 2)
        cat = np.argpartition(D, N)
        area = np.pi * D[cat[N - 1]] ** 2 / 3600.
        n = N / area
        if n > n0 and n < 10000:
            g1_avg.append(np.average(g1[cat[:N]]))
            g2_avg.append(np.average(g2[cat[:N]]))
            x1.append(x[i])
            y1.append(y[i])
            count += 1
    return np.array(g1_avg), np.array(g2_avg), np.array(x1), np.array(y1)


def find_index(x, a):
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def findhalocenter(k):
    neighborhood_size = 10
    threshold = 0.01
    data = k.data
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))
    xy = xy.astype(int)
    M = np.zeros(num_objects)
    for i in range(num_objects):
        M[i] = k.data[xy[i, 0], xy[i, 1]]
    j = np.argmax(M)
    x0 = k.x[xy[j, 0]] / 4.413  # in arcsec
    y0 = k.y[xy[j, 1]] / 4.413
    return x0, y0


def ray_trace(x, y, zf, alpha_x, alpha_y):
    ax = BiLnrIntrp(x, y, alpha_x)
    ay = BiLnrIntrp(x, y, alpha_y)
    x0 = x + zf * ax * 3600. * 180. / np.pi
    y0 = y + zf * ay * 3600. * 180. / np.pi
    return x0, y0


def Zfunc(z):
    Dd = Cosmo.angular_diameter_distance(0.296)
    Ds = Cosmo.angular_diameter_distance(z)
    Dds = Cosmo.angular_diameter_distance_z1z2(0.296, z)
    return (Dds) / Ds


def BiLnrIntrp(x, y, data):
    # Bilinear Interpolation
    i = find_index(x, data.x / 4.413)
    j = find_index(y, data.y / 4.413)
    x1 = data.x[i] / 4.413
    x2 = data.x[i + 1] / 4.413
    y1 = data.y[j] / 4.413
    y2 = data.y[j + 1] / 4.413
    f11 = data.data[i, j]
    f12 = data.data[i, j + 1]
    f21 = data.data[i + 1, j]
    f22 = data.data[i + 1, j + 1]
    f = (f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22 * (x - x1) * (
                y - y1)) / ((x2 - x1) * (y2 - y1))
    return f


def extend_matrix(p):
    nx, ny = np.shape(p)
    k = np.zeros([nx + 2, ny + 2])
    k[1:-1, 1:-1] = p[:, :]
    k[0, 1:-1] = p[-1, :]
    k[-1, 1:-1] = p[0, :]
    k[1:-1, 0] = p[:, -1]
    k[1:-1, -1] = p[:, 0]
    return k


def EulerAngles(phi, theta, psi):
    R = zeros([3, 3])
    R[0, 0] = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi)
    R[0, 1] = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi)
    R[0, 2] = sin(psi) * sin(theta)
    R[1, 0] = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi)
    R[1, 1] = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi)
    R[1, 2] = cos(psi) * sin(theta)
    R[2, 0] = sin(theta) * sin(phi)
    R[2, 1] = -sin(theta) * cos(phi)
    R[2, 2] = cos(theta)
    return R


def ProjectEnzoCompact(pf, mass, phi, theta, psi, zmin=-3000, zmax=3000):
    xpixels = mass.nx
    ypixels = mass.ny
    PixelArea = mass.dx * mass.dy

    DM = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)

    pf.field_info['Density'].take_log = False
    pf.field_info['Dark_Matter_Density'].take_log = False
    center = [(mass.xmin + mass.xmax) / 2.0, (mass.ymin + mass.ymax) / 2.0, (zmin + zmax) / 2.0]  # Data Center
    normal_vector = (0.0, 0.0, 1.0)
    north_vector = (0.0, 1.0, 0.0)

    R = EulerAngles(phi, theta, psi)
    normal_vector = dot(R, normal_vector)
    north_vector = dot(R, north_vector)
    width = (mass.xmax - mass.xmin, mass.ymax - mass.ymin, zmax - zmin)
    resolution = (mass.nx, mass.ny)

    MassFactor = BulletConstants.cm_per_kpc ** 2 * PixelArea / (BulletConstants.g_per_Msun * 1E10)  ##output unit
    projcam = ProjectionCamera(center, normal_vector, width, resolution, "Dark_Matter_Density",
                               north_vector=north_vector, pf=pf, interpolated=True)
    DM.data = projcam.snapshot()[:, :]
    projcam = ProjectionCamera(center, normal_vector, width, resolution, "Density", north_vector=north_vector, pf=pf,
                               interpolated=True)
    mass.data = projcam.snapshot()[:, :]
    return [mass, DM]


def residual(ux, fxy, R):
    N, N = np.shape(ux)
    h = R / (N - 1.)
    h2 = h * h
    r = np.zeros((N, N))
    r[1:-1, 1:-1] = -(ux[0:-2, 1:-1] + ux[2:, 1:-1] + ux[1:-1, 0:-2] + ux[1:-1, 2:] - 4 * ux[1:-1, 1:-1]) / h2 + fxy[
                                                                                                                 1:-1,
                                                                                                                 1:-1]
    return r


def ini(L):
    N = 2 ** L + 2
    ui = np.zeros((N, N))
    return ui


def restrict(A, L):
    ##restrict L->L-1
    n = 2 ** (L - 1) + 2
    AA = np.zeros((n, n))
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            c, d = 2 * i - 1, 2 * j - 1
            AA[i, j] = 0.25 * (A[c, d] + A[c + 1, d] + A[c, d + 1] + A[c + 1, d + 1])
    return AA


def relax(A, L):
    ##relax L->L+1
    n = 2 ** (L + 1) + 2
    AA = np.zeros((n, n))
    for i in range(1, n - 1, 2):
        for j in range(1, n - 1, 2):
            AA[i, j] = A[i / 2 + 1, j / 2 + 1]
            AA[i + 1, j] = A[i / 2 + 1, j / 2 + 1]
            AA[i, j + 1] = A[i / 2 + 1, j / 2 + 1]
            AA[i + 1, j + 1] = A[i / 2 + 1, j / 2 + 1]
    return AA


def mgm(u, f, L, w, R, gamma, pre_step, post_step):
    ##Multi-grid method

    #     print"now is running L={:d}".format(L)
    #     print"dimension check"
    #     print u.shape
    #     print f.shape
    ##pre approx
    u_pre = GS_SOR(u, f, R, w, pre_step)
    r = residual(u_pre, f, R)
    if L > 0:
        r = restrict(r, L)
        ui = ini(L - 1)
        for j in range(0, gamma):
            ux = mgm(ui, r, L - 1, w, R, gamma, pre_step, post_step)
        u = u_pre + relax(ux, L - 1)
    ##post approx
    u_post = GS_SOR(u, f, R, w, post_step)
    return u_post


def GS_SOR(ux, fxy, R, w, step):
    ##Gauss Seidel algorithm  with Overrelaxation and periodic boundary condition
    ##Red-Black ordering
    N, N = np.shape(fxy)
    h = R / (N - 1)
    for ii in range(0, step):
        m, n = ux.shape
        h2 = h * h;
        for sweep in ('red', 'black'):
            for i in range(1, m - 1):
                start = 1 + i % 2 if sweep == 'black' else 2 - i % 2
                for j in range(start, n - 1, 2):
                    ux[i, j] = (1 - w) * ux[i, j] + w * (ux[i + 1, j] +
                                                         ux[i - 1, j] +
                                                         ux[i, j + 1] +
                                                         ux[i, j - 1] -
                                                         h2 * fxy[i, j]) * 0.25
    return ux


def MGGS_pot(k, w=1.5, pre_step=2, post_step=3, gamma=1, step=1000, tol=10E-6):
    print "Multi-grid Gauss Seidel Method\n"
    print "w = {:.1f} pre = {:d} post = {:d} gm={:d}".format(w, pre_step, post_step, gamma)
    LensP = Array2d(k.xmin, k.xmax, k.nx, k.ymin, k.ymax, k.ny)
    # design for 440 pixals on each side
    N, N = np.shape(k.data)
    L = log2(N)
    if L % 1 != 0:
        print "number of grid must be 2**L"
        return 0
    L = int(L)
    ui = ini(L)
    fxy = ini(L)
    fxy[1:-1, 1:-1] = 2. * k.data
    # size of image
    R = 2 * k.xmax / 4.313 / 3600 * np.pi / 180.
    err_plot = []
    start = time.time()
    # Main multigrid iteration)
    for t in range(1, step):
        stepstart = time.time()
        ux = mgm(ui, fxy, L, w, R, gamma, pre_step, post_step)
        r = residual(ux, fxy, R)
        err = (1. / (N + 1) ** 2 * np.sum(np.abs(r)))
        err_plot.append(err)
        LensP.data = ux[1:-1, 1:-1]
        ui = ux
        # if t%10 == 1:
        #     end = time.time()-stepstart
        #     alltime = time.time()-start
        #     print "testing step {:d} err = {:.5f} time per step = {:.3f} total time ={:.3f}".format(t,err,end,alltime) 
        #     continue
        if err < tol:
            end = time.time() - start
            print "reach convergence in {:.1f}sec step = {:d} err = {:.8f}".format(end, t, err)
            break
    return LensP, err_plot


def cusp(x, y, alpha_x, alpha_y, z=3.24):
    from scipy.interpolate import splprep, splev
    x0, y0 = [], []
    for i in range(len(x)):
        zf = Zfunc(z)
        xi, yi = ray_trace(x[i], y[i], zf, alpha_x, alpha_y)
        x0.append(xi)
        y0.append(yi)
    x0 = np.array(x0)
    y0 = np.array(y0)
    pts = np.array([x0[::], y0[::]])
    tck, u = splprep(pts, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new


def curve(ax1, x_cr, y_cr, x_cusp, y_cusp):
    cx1 = ax1.plot(np.array(x_cr), np.array(y_cr), color="y")
    cx2 = ax1.plot(np.array(x_cusp), np.array(y_cusp), color="r")
    return ax1


def plot_ccurve_cusp(ax1, Mag, alpha_x, alpha_y):
    [dyy, dxx] = np.meshgrid(Mag.y[10:-10] / 4.413, Mag.x[10:-10] / 4.413)
    levels = [0, 1]
    # c1 = ax1.contour(dxx,dyy,Mag.data[10:-10,10:-10],colors = 'y',levels=levels)
    level = 0
    c1 = cntr.Cntr(dxx, dyy, Mag.data[10:-10, 10:-10])
    nlist = c1.trace(level, level, 0)
    segs = nlist[:len(nlist) // 2]
    x_cr, y_cr, x_cusp, y_cusp = [], [], [], []
    for i in range(len(segs)):
        x = segs[i][:, 0]
        y = segs[i][:, 1]
        x1, y1 = cusp(x, y, alpha_x, alpha_y)
        ax1 = curve(ax1, x, y, x1, y1)
    return ax1


def add_SL_galaxies(ax, galaxies):
    markers = ['bo', 'gv', 'r^', 'c<', 'm>', 'y8', 'ks', 'bp', 'g*', 'rh', 'cH', 'mD', 'yd']
    if galaxies["A"].xs:
        m = 0
        for key in galaxies:
            gal = galaxies[key]
            ax.plot(gal.xc, gal.yc, markers[m], linestyle='None', markersize=6, label=key)
            ax.plot(gal.xs, gal.ys, markers[m], linestyle='None', markersize=3, label=key + 's')
            m += 1
    else:
        m = 0
        for key in galaxies:
            gal = galaxies[key]
            ax.plot(gal.xc, gal.yc, markers[m], linestyle='None', markersize=6, label=key)
            m += 1
    return ax


def kappa_over(Lens, galaxies):
    title = 'critical curve overlap convergence'
    # SkyCoordinate
    alpha = Lens._alpha
    alpha_x = Lens._alpha_x
    alpha_y = Lens._alpha_y
    kappa = Lens._kappa
    Mag = Lens._mag
    fig = plt.figure(figsize=(16, 10.4))
    [dyy, dxx] = np.meshgrid(alpha.y / 4.413, alpha.x / 4.413)
    ax1 = axes([0.1, 0.1, 0.7, 0.8])
    cx1 = ax1.contour(dxx, dyy, alpha.data * 3600. * 180. / np.pi, colors='black', cmap=None, linestyles="solid")
    cax = fig.add_axes([0.95, 0.1, 0.1, 0.8], visible=False)
    plt.colorbar(cx1, cax=cax)
    plt.clabel(cx1, cax=cax, colors="white")
    level = np.linspace(0, 3, 11)
    cx2 = ax1.contourf(dxx, dyy, kappa.data, levels=level)
    for c in cx2.collections:
        c.set_linewidth(0.1)
        c.set_alpha(0.8)
    cax2 = fig.add_axes([0.85, 0.1, 0.02, 0.8])
    plt.colorbar(cx2, cax=cax2)
    x_min = -200
    x_max = 200
    ticksrange = np.linspace(x_min, x_max, 11)
    title = title
    ax1.set_title(title)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([x_min, x_max])
    ax1.set_xticks(ticksrange)
    ax1.set_yticks(ticksrange)
    ax1.set_xlabel("arcsec")
    ax1.set_ylabel("arcsec")
    ax1.set_xlabel("arcsec")
    ax1.set_ylabel("arcsec")
    ax1.grid(True, which="both")
    ax1 = add_SL_galaxies(ax1, galaxies)
    ax1 = plot_ccurve_cusp(ax1, Mag, alpha_x, alpha_y)
    # fig.savefig(fname)
    ax1.legend(loc='upper left')
    fig.show()


def shear_plot1(Lens, galaxies, sheardata, x0, limit=[-200, 200]):
    fig = plt.figure(figsize=(16, 10.4))
    g1_sim, g2_sim, g_sim = Lens._gamma1, Lens._gamma2, Lens._gamma
    g1_grid, g2_grid, g_grid = shear_regrid(sheardata, x0)
    [dyy, dxx] = meshgrid(g_sim.y / 4.413, g_sim.x / 4.413)
    matplotlib.rcParams['contour.negative_linestyle'] = 'dashed'
    ###plot g1
    ax1 = axes([0.15, 0.5, 0.4, 0.4], aspect=1)
    line_levels = np.linspace(-0.5, 0.5, 11)
    cont1A = ax1.contour(dxx, dyy, g1_sim.data, levels=line_levels, cmap=None, colors='k')
    cax = fig.add_axes([0.95, 0.1, 0.1, 0.8], visible=False)
    plt.colorbar(cont1A, cax=cax)
    plt.clabel(cont1A, cax=cax, fmt='%1.1f', fontsize='small')

    filled_levels = np.linspace(-0.5, 0.5, 11)
    cont1B = ax1.contourf(dxx, dyy, g1_grid.data, levels=filled_levels)
    for c in cont1B.collections:
        c.set_linewidth(0.1)
        c.set_alpha(0.8)
    cax2 = fig.add_axes([0.49, 0.5, 0.01, 0.4])
    plt.colorbar(cont1B, cax=cax2)
    ax1.set_title(r"g1 sim v.s. g1 data")
    ax1.set_ylabel('arcsec', labelpad=-15)
    ax1.set_xlim(limit)
    ax1.set_ylim(limit)
    ax1.set_xticks([])
    ax1.grid(True, which="both")

    ###plot g2
    ax2 = axes([0.475, 0.5, 0.4, 0.4], aspect=1)
    line_levels = np.linspace(-0.5, 0.5, 11)
    cont2A = ax2.contour(dxx, dyy, g2_sim.data, levels=line_levels, cmap=None, colors='k')
    cax3 = fig.add_axes([0.95, 0.1, 0.1, 0.8], visible=False)
    plt.colorbar(cont2A, cax=cax3)
    plt.clabel(cont2A, cax=cax3, fmt='%1.1f', fontsize='small')

    filled_levels = np.linspace(-0.5, 0.5, 11)
    cont2B = ax2.contourf(dxx, dyy, g2_grid.data, levels=filled_levels)
    for c in cont2B.collections:
        c.set_linewidth(0.1)
        c.set_alpha(0.8)
    cax4 = fig.add_axes([0.8, 0.5, 0.01, 0.4])
    plt.colorbar(cont2B, cax=cax4)
    ax2.set_title(r"g2 sim v.s. g2 data")
    ax2.set_ylabel('arcsec', labelpad=-15)
    ax2.set_xlim(limit)
    ax2.set_ylim(limit)
    ax2.grid(True, which="both")

    # plot total g
    ax3 = axes([0.15, 0.08, 0.4, 0.4], aspect=1)
    line_levels = np.linspace(0, 0.5, 6)
    cont3A = ax3.contour(dxx, dyy, g_sim.data, levels=line_levels, cmap=None, colors='k')
    cax5 = fig.add_axes([0.95, 0.1, 0.1, 0.8], visible=False)
    plt.colorbar(cont3A, cax=cax5)
    plt.clabel(cont3A, cax=cax5, fmt='%1.1f', fontsize='small')

    filled_levels = np.linspace(0, 0.5, 6)
    cont3B = ax3.contourf(dxx, dyy, g_grid.data, levels=filled_levels)
    for c in cont3B.collections:
        c.set_linewidth(0.1)
        c.set_alpha(0.8)
    cax6 = fig.add_axes([0.49, 0.08, 0.01, 0.4])
    plt.colorbar(cont3B, cax=cax6)
    ax3.set_title('g sim v.s g data')
    ax3.set_ylabel('arcsec', labelpad=-15)
    ax3.set_xlim(limit)
    ax3.set_ylim(limit)
    ax3.grid(True, which="both")
    fig.show()
