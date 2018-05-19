import numpy as np
from scipy.interpolate import griddata

import BulletConstants
from Classes import Array2d
from GravLens import get_shear, get_galaxies


class optfunc:
    # function for optimizing
    def __init__(self, data1list, data2list, shifteddata1list, sigmalist, masklist, align, **kwargs):
        """
        kwargs options
        {'mass':'gravlens' or 'image' ,
        'xray' : 'enable' or 'disable',
        'chi_sq_method': 'overall' or 'reduce'}
        """
        self.kwargs = kwargs
        if kwargs['mass'] == 'gravlens':
            self.lens = Lens(data1list[0], chi_sq_method=kwargs['chi_sq_method'])
            if kwargs['xray'] == 'enable':
                self.images_align = XIA(data1list, data2list, shifteddata1list, sigmalist, masklist, method='Xray',
                                        chi_sq_method=kwargs['chi_sq_method'])
        elif kwargs['mass'] == 'image':
            if kwargs['xray'] == 'enable':
                self.images_align = XIA(data1list, data2list, shifteddata1list, sigmalist, masklist, method='MassX',
                                        chi_sq_method=kwargs['chi_sq_method'])
            elif kwargs['xray'] == 'disable':
                self.images_align = XIA(data1list, data2list, shifteddata1list, sigmalist, masklist, method='Mass',
                                        chi_sq_method=kwargs['chi_sq_method'])
        self.align = align
        self.evals = 0
        self.total_chiq_sq = 0

    def fit(self, x):
        '''
        evaluate chi_sq for image alignment and gravitational lens
        '''

        if self.kwargs['mass'] == 'gravlens':
            if self.kwargs['xray'] == 'enable':
                total_chiq_sq = self.images_align.fom(x) + self.lens.fom(x)
            else:
                total_chiq_sq = self.lens.fom(x)
        else:
            total_chiq_sq = self.images_align.fom(x)
        self.total_chiq_sq = total_chiq_sq
        self._update_align(x)
        self.evals += 1
        return total_chiq_sq

    def _update_align(self, x):
        self.align.d[0] = x[0]
        self.align.d[1] = x[1]
        self.align.d[4] = x[2]


class XIA:
    # Mass/Xray images alignment class
    def __init__(self, data1list, data2list, shifteddata1list, sigmalist, masklist, method='MassX',
                 chi_sq_method='reduce'):
        self.data1list = data1list
        self.data2list = data2list
        self.shifteddata1list = shifteddata1list
        self.sigmalist = sigmalist
        self.masklist = masklist
        self.origin_grid = self._grid(data1list[0])
        self.shifted_grid = self._grid(shifteddata1list[0])
        self.method = method
        self.chi_sq_method = chi_sq_method
        self.chi_sq_list = np.zeros(len(data1list))
        self.chi_sq = 0

    def fom(self, x):
        # This sets the alignment parameters
        # Components 0 and 1 are the x and y aligment offsets.
        # Components 2 and 3 are a shift between the Mass dataset and the others
        # MaxShift gives the max allowed shift in kpc
        # Component 4 is the angular rotation in radians
        self.chi_sq = 0
        dx, dy, phi = x[0], x[1], x[2]
        new_pos = change_coord(self.shifted_grid, dx, dy, phi)
        if self.method == 'Mass':
            nimage = [0]
        elif self.method == 'XayI':
            nimage = [1]
        else:
            nimage = [0, 1]

        for k in nimage:
            mask = self.masklist[k]
            if np.sum(mask.data) == 0:
                pass
            else:
                data1 = self.data1list[k]  ##simulation data, larger
                data2 = self.data2list[k]
                shifteddata1 = self.shifteddata1list[k]
                sigma = self.sigmalist[k]
                data = np.array(data1.data.reshape(data1.nx * data1.ny, 1))
                new_data = griddata(self.origin_grid, data, new_pos, method='linear',
                                    fill_value=0)  # time bottleneck,each eval take 0.16s
                # for faster interpolation method, see
                # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
                shifteddata1.data = new_data.reshape(shifteddata1.nx, shifteddata1.ny)
                self.chi_sq_list[k] = self._fom_cal(data2, shifteddata1, sigma, mask)
                self.chi_sq += self.chi_sq_list[k]
        return self.chi_sq

    def _fom_cal(self, data1, data2, sigma, mask):
        if self.chi_sq_method == 'reduce':
            # For reduce chisq
            return np.sum(np.power((data1.data - data2.data) / sigma.data, 2) * mask.data) / np.sum(mask.data)
        elif self.chi_sq_method == 'overall':
            # For overall chisq
            return np.sum(np.power((data1.data - data2.data) / sigma.data, 2) * mask.data)
        else:
            print 'please specify chi_sq_method [overall] or [reduce]'
            return 0

    def _grid(self, data):
        n = data.nx * data.ny
        [dyy, dxx] = np.meshgrid(data.y, data.x)
        pos = np.append(dxx.reshape(n, 1), dyy.reshape(n, 1), axis=1)
        return pos


class Lens:
    def __init__(self, mass, chi_sq_method='reduce'):
        self.galaxies = get_galaxies()
        self.shear = get_shear()
        self.mass = mass
        self.kappa = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.potiential = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.alpha = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.alphaX = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.alphaY = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.gamma = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.gamma1 = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.gamma2 = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.mag = Array2d(mass.xmin, mass.xmax, mass.nx, mass.ymin, mass.ymax, mass.ny)
        self.grid_arcsec = list()
        self._findvalue_by_potential_fftw(mass)
        self.chisq = 0

    def fom(self, x, w=0.0001):
        self.chisq = self._chi_strong(x) + w * self._chi_weak(x)

    def _chi_strong(self, x):
        sigma = 0.6
        dx, dy, phi = x[0] / 4.413, x[1] / 4.413, x[2]
        pos = self._get_pos()
        zf = self._get_zf()
        new_pos = change_coord(pos, dx, dy, phi)
        data = [self.alphaX, self.alphaY, self.mag]
        [shifted_ax, shifted_ay, shifted_mag] = self._shift(data, x, new_pos)
        source = pos + np.append([zf * shifted_ax.squeeze()], [zf * shifted_ay.squeeze()], axis=0).T
        return self._get_chi_strong(source, shifted_mag)

    def _get_chi_strong(self, source, mag):
        sigma = 0.6
        k = 0
        chi = 0
        for key in self.galaxies:
            gal = self.galaxies[key]
            gal.clean()
            for i in range(gal.n):
                gal.xs.append(source[k, 0])
                gal.ys.append(source[k, 1])
                gal.mag.append(float(mag[k]))
                k += 1
            chi += sum(gal.update_chi(sigma)) / gal.n
        return chi

    def _chi_weak(self, x):
        sigma = 1.
        sigma_eps_s = 0.2
        sigma_eps_er = 0.1
        dx, dy, phi = x[0] / 4.413, x[1] / 4.413, x[2]
        pos = np.array([np.array(self.shear['ra']), np.array(self.shear['dec'])]).T
        new_pos = change_coord(pos, dx, dy, phi)
        data = [self.gamma1, self.gamma2]
        [shifted_g1, shifted_g2] = self._shift(data, x, new_pos)
        Zf = np.array(self.shear['Zf'])
        g1 = np.array(self.shear['g_final[0]'])
        g2 = np.array(self.shear['g_final[0]'])
        sigma = (1 - (shifted_g1 ** 2 + shifted_g2 ** 2)) ** 2 * sigma_eps_s + sigma_eps_er
        chi_sq = np.sum((g1 - shifted_g1) ** 2 + (g2 - shifted_g2) ** 2 / sigma)
        return chi_sq

    def _get_pos(self):
        pos = []
        for key in self.galaxies:
            gal = self.galaxies[key]
            for i in range(gal.n):
                pos.append([gal.x0[i], gal.y0[i]])
        return np.array(pos)  # list of [(x1,y1)...] in arcsec

    def _get_zf(self):
        zf = []
        for key in self.galaxies:
            gal = self.galaxies[key]
            for i in range(gal.n):
                zf.append(gal.zf[i])
        return np.array(zf)

    def _shift(self, dataset, x, new_pos):
        shifteddataset = list()
        for data in dataset:
            data_roll = np.array(data.data.reshape(data.nx * data.ny, 1))
            new_data = griddata(self.grid_arcsec, data_roll, new_pos, method='linear', fill_value=0)
            shifteddataset.append(new_data)
        return shifteddataset  # simulation quantities at position

    def _findvalue_by_potential_integral(self, mass):
        pixal_to_radian = 2 * mass.xmax / BulletConstants.AngularScale / 3600 / 180 * np.pi / mass.nx
        pixal_to_arcsec = 2 * mass.xmax / BulletConstants.AngularScale / mass.nx
        sigmacr = BulletConstants.CriticalSurfaceDensity
        ds = mass.dx * mass.dy
        MassFactor = BulletConstants.cm_per_kpc ** 2 * ds / (BulletConstants.g_per_Msun * 1E10)
        _kappa = mass.data / sigmacr / MassFactor
        # grid position for later operation in arcsec
        [y_grid, x_grid] = np.meshgrid(mass.y, mass.x)
        x_grid = x_grid / BulletConstants.AngularScale
        y_grid = y_grid / BulletConstants.AngularScale
        self.grid_arcsec = np.append(x_grid.reshape(mass.nx * mass.ny, 1), y_grid.reshape(mass.nx * mass.ny, 1), axis=1)

        _potential = np.zeros((mass.nx, mass.ny))
        _gamma1 = np.zeros((mass.nx, mass.ny))
        _gamma2 = np.zeros((mass.nx, mass.ny))
        for i in range(mass.nx):
            for j in range(mass.ny):
                dx = x_grid[i, j] - x_grid
                dy = y_grid[i, j] - y_grid
                _potential[i, j] = (_kappa * np.ma.log(np.sqrt(dx * dx + dy * dy))).sum()

        _potential = _potential * ds / np.pi / BulletConstants.AngularScale / BulletConstants.AngularScale
        _alphaX, _alphaY = np.gradient(_potential, pixal_to_arcsec)  # unit = "
        _alpha = np.sqrt(_alphaX ** 2 + _alphaY ** 2)
        domi = pixal_to_arcsec
        k = self._extend_matrix(_potential)
        _gamma1[:, :] = 0.5 * ((k[2:, 1:-1] - 2. * k[1:-1, 1:-1] + k[0:-2, 1:-1]) / domi ** 2 - (
                k[1:-1, 2:] - 2. * k[1:-1, 1:-1] + k[1:-1, 0:-2]) / domi ** 2)
        _gamma2[:, :] = (k[2:, 2:] - k[2:, 0:-2] - k[0:-2, 2:] + k[0:-2, 0:-2]) / 4. / domi ** 2
        _gamma = np.sqrt(_gamma1 ** 2 + _gamma2 ** 2)
        _mag = (1. - _kappa) ** 2 - _gamma ** 2
        self.kappa.data = _kappa
        self.potiential.data = _potential
        self.alpha.data = _alpha
        self.alphaX.data = _alphaX
        self.alphaY.data = _alphaY
        self.gamma.data = _gamma
        self.gamma1.data = _gamma1
        self.gamma2.data = _gamma2
        self.mag.data = _mag

    def _findvalue_by_potential_fftw(self, mass):
        from fft_poisson_solver import fft_poisson_solver
        # pixal_to_radian = 2 * mass.xmax / BulletConstants.AngularScale / 3600 / 180 * np.pi / mass.nx
        pixal_to_arcsec = 2 * mass.xmax / BulletConstants.AngularScale / mass.nx
        sigmacr = BulletConstants.CriticalSurfaceDensity
        ds = mass.dx * mass.dy
        mass_factor = BulletConstants.cm_per_kpc ** 2 * ds / (BulletConstants.g_per_Msun * 1E10)
        _kappa = mass.data / sigmacr / mass_factor

        # solve poisson equation (d/dx2 + d/dy2)phi = -2 kappa
        h2 = ds / BulletConstants.AngularScale / BulletConstants.AngularScale
        _potential = fft_poisson_solver(- 2 * _kappa, h2, mass.nx)

        # calculate alpha = gradient(potential)
        _alphaX, _alphaY = np.gradient(_potential, pixal_to_arcsec)  # unit = "
        _alpha = np.sqrt(_alphaX ** 2 + _alphaY ** 2)

        # calculate gamma using finite differential method
        _gamma1 = np.zeros((mass.nx, mass.ny))
        _gamma2 = np.zeros((mass.nx, mass.ny))
        domi = pixal_to_arcsec
        k = self._extend_matrix(_potential)
        _gamma1[:, :] = 0.5 * ((k[2:, 1:-1] - 2. * k[1:-1, 1:-1] + k[0:-2, 1:-1]) / domi ** 2 - (
                k[1:-1, 2:] - 2. * k[1:-1, 1:-1] + k[1:-1, 0:-2]) / domi ** 2)
        _gamma2[:, :] = (k[2:, 2:] - k[2:, 0:-2] - k[0:-2, 2:] + k[0:-2, 0:-2]) / 4. / domi ** 2
        _gamma = np.sqrt(_gamma1 ** 2 + _gamma2 ** 2)
        _mag = (1. - _kappa) ** 2 - _gamma ** 2

        # grid position for later operation in arcsec
        [y_grid, x_grid] = np.meshgrid(mass.y, mass.x)
        x_grid = x_grid / BulletConstants.AngularScale
        y_grid = y_grid / BulletConstants.AngularScale
        self.grid_arcsec = np.append(x_grid.reshape(mass.nx * mass.ny, 1), y_grid.reshape(mass.nx * mass.ny, 1), axis=1)
        self.kappa.data = _kappa
        self.potiential.data = _potential
        self.alpha.data = _alpha
        self.alphaX.data = _alphaX
        self.alphaY.data = _alphaY
        self.gamma.data = _gamma
        self.gamma1.data = _gamma1
        self.gamma2.data = _gamma2
        self.mag.data = _mag

    def _extend_matrix(self, p):
        nx, ny = np.shape(p)
        k = np.zeros([nx + 2, ny + 2])
        k[1:-1, 1:-1] = p[:, :]
        k[0, 1:-1] = p[-1, :]
        k[-1, 1:-1] = p[0, :]
        k[1:-1, 0] = p[:, -1]
        k[1:-1, -1] = p[:, 0]
        return k