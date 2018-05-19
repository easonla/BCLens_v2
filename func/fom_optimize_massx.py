import numpy as np
from scipy.interpolate import griddata, interpn
from routines import change_coord
from Classes import Array2d


class optfunc_massx:
    # function for optimizing
    def __init__(self, data1list, data2list, shifteddata1list, sigmalist, masklist, align):
        self.images_align = XIA(data1list, data2list, shifteddata1list, sigmalist, masklist)
        self.align = align
        self.evals = 0
        self.total_chiq_sq = 0
        self.overall_chi_sq = 0

    def fit(self, x):
        '''
        evaluate chi_sq for image alignment
        '''

        self.total_chiq_sq = self.images_align.fom(x)
        self._update_align(x)
        self.evals += 1
        return self.total_chiq_sq

    def fom_compute(self, x):
        self.total_chiq_sq = self.images_align.fom_compute(x)
        self._update_align(x)

    def _update_align(self, x):
        self.align.d[0] = x[0]
        self.align.d[1] = x[1]
        self.align.d[4] = x[2]


class XIA:
    # Mass/Xray images alignment class
    def __init__(self, data1list, data2list, shifteddata1list, sigmalist, masklist):
        self.data1list = data1list
        self.data2list = data2list
        self.shifteddata1list = shifteddata1list
        self.sigmalist = sigmalist
        self.masklist = masklist
        self.origin_gridlist = self._grid(data1list)
        self.shifted_gridlist = self._grid(shifteddata1list)
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

        for k in range(len(self.masklist)):
            mask = self.masklist[k]
            if np.sum(mask.data) == 0:
                pass
            else:
                new_pos = change_coord(self.shifted_gridlist[k], dx, dy, phi)
                data1 = self.data1list[k]  # simulation data, larger
                data2 = self.data2list[k]
                shifteddata1 = self.shifteddata1list[k]
                sigma = self.sigmalist[k]

                #data = np.array(data1.data.reshape(data1.nx * data1.ny, 1))
                # new_data = griddata(self.origin_grid, data, new_pos, method='linear',
                #                    fill_value=0)  # time bottleneck,each eval take ~ 0.16s
                # for faster interpolation method, see
                # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
                new_data = interpn((data1.x, data1.y), data1.data, (new_pos[:, 0], new_pos[:, 1]), method='linear',
                                   fill_value=0)
                shifteddata1.data = new_data.reshape(shifteddata1.nx, shifteddata1.ny)
                # much faster for grid data ~ 10ms

                self.shifteddata1list[k] = shifteddata1
                self.chi_sq_list[k] = self._fom_cal(data2, shifteddata1, sigma, mask)
                self.chi_sq += self.chi_sq_list[k]

        return self.chi_sq

        # else: return 10E5

    def fom_compute(self, x):
        self.chi_sq = 0
        dx, dy, phi = x[0], x[1], x[2]

        for k in range(len(self.masklist)):
            new_pos = change_coord(self.shifted_gridlist[k], dx, dy, phi)
            mask = self.masklist[k]
            data1 = self.data1list[k]  # simulation data, larger
            data2 = self.data2list[k]
            shifteddata1 = self.shifteddata1list[k]
            sigma = self.sigmalist[k]
            new_data = interpn((data1.x, data1.y), data1.data, (new_pos[:, 0], new_pos[:, 1]), method='linear',
                               fill_value=0)
            shifteddata1.data = new_data.reshape(shifteddata1.nx, shifteddata1.ny)
            self.shifteddata1list[k] = shifteddata1
            self.chi_sq_list[k] = self._fom_cal(data2, shifteddata1, sigma, mask)
            if np.sum(mask.data) != 0:
                self.chi_sq += self.chi_sq_list[k]
        return self.chi_sq

    def _fom_cal(self, data1, data2, sigma, mask):
        return np.sum(np.power((data1.data - data2.data) / sigma.data, 2) * mask.data) / np.sum(mask.data)

    def _grid(self, datalist):
        grid_list = []
        for data in datalist:
            n = data.nx * data.ny
            [dyy, dxx] = np.meshgrid(data.y, data.x)
            pos = np.append(dxx.reshape(n, 1), dyy.reshape(n, 1), axis=1)
            grid_list.append(pos)
        return grid_list
