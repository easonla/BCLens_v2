import numpy as np
import sys
from scipy.ndimage import gaussian_filter
from scipy.optimize import basinhopping


from Classes import Array2d, Align
from get_data import ReadLookups
from FetchEnzo import ProjectEnzoData
from fom_optimize_massx import optfunc_massx


def SetAlign(dataB1, phi, theta, psi, ConstrainPhi=False, MaxShift=0.0):
    # This sets the alignment parameters
    # Components 0 and 1 are the x and y aligment offsets.
    # Components 2 and 3 are a shift between the Mass dataset and the others
    # MaxShift gives the max allowed shift in kpc
    # Component 4 is the angular rotation in radians
    align = Align()
    align.dmax[0] = dataB1.xmax
    align.dmax[1] = dataB1.ymax
    align.dmin[0] = -align.dmax[0]
    align.dmin[1] = -align.dmax[1]
    align.dmax[2] = MaxShift
    align.dmin[2] = -align.dmax[2]
    align.dmax[3] = MaxShift
    align.dmin[3] = -align.dmax[3]
    align.dmin[4] = 0.0
    align.dmax[4] = 2.0 * np.pi

    align.d[0] = 0.0
    align.d[1] = 0.0
    align.d[2] = 0.0
    align.d[3] = 0.0

    if ConstrainPhi:
        align.dmin[0] = -250.0
        align.dmax[0] = 250.0
        align.dmin[1] = -250.0
        align.dmax[1] = 250.0

    if np.cos(psi) >= 0.0:
        align.d[4] = 0.22 - np.arctan(np.cos(theta) * np.tan(psi))  # Seed phi close to final result
        if ConstrainPhi:
            align.dmin[4] = 0.12 - np.arctan(np.cos(theta) * np.tan(psi))
            align.dmax[4] = 0.32 - np.arctan(np.cos(theta) * np.tan(psi))
    else:
        align.d[4] = 0.22 + np.pi - np.arctan(np.cos(theta) * np.tan(psi))  # Seed phi close to final result
        if ConstrainPhi:
            align.dmin[4] = 0.12 + np.pi - np.arctan(np.cos(theta) * np.tan(psi))  # Seed psi close to final result
            align.dmax[4] = 0.32 + np.pi - np.arctan(np.cos(theta) * np.tan(psi))  # Seed psi close to final result

    return align


def SimpleFom(pf, data, phi=0.0, theta=0.0, psi=0.0, ConstrainPhi=True, Mask=(1, 0, 0, 0, 0, 0, 0), Z=1.0, TFudge=1.0,
              SpectralIndex=3.8, MaxShift=0.0):
    (dataA, sigmaA, maskA, maskANull, dataB1, sigmaB1, dataB2, sigmaB2, dataB3, sigmaB3, dataC, sigmaC, dataD, sigmaD,
     maskD, maskDNull, dataE, sigmaE) = data
    mask_sum = maskA.data.sum()
    # from save_to_pickle import picklesave, pickleread
    # picklesave(dataA,sigmaA,maskA,maskANull,dataB1,sigmaB1,dataB2,sigmaB2,dataB3,sigmaB3,dataC,sigmaC,dataD,sigmaD,maskD,maskDNull,dataE,sigmaE,sumsim,xraysim1,xraysim2,xraysim3,szsim)
    # pickleread()

    try:
        dmsim = Array2d(2.0 * dataA.xmin, 2.0 * dataA.xmax, 2 * dataA.nx, 2.0 * dataA.ymin, 2.0 * dataA.ymax,
                        2 * dataA.ny)
        masssim = Array2d(2.0 * dataA.xmin, 2.0 * dataA.xmax, 2 * dataA.nx, 2.0 * dataA.ymin, 2.0 * dataA.ymax,
                          2 * dataA.ny)
        sumsim = Array2d(2.0 * dataA.xmin, 2.0 * dataA.xmax, 2 * dataA.nx, 2.0 * dataA.ymin, 2.0 * dataA.ymax,
                         2 * dataA.ny)
        ApecData = ReadLookups(Z)  # Reads the APEC lookup tables.
        for grid in pf.h.grids:
            grid.set_field_parameter('ApecData', ApecData)
            grid.set_field_parameter('TFudge', TFudge)
            grid.set_field_parameter('SpectralIndex', SpectralIndex)
        DMProject = False
        [dmsim, masssim, xraysim1, xraysim2, xraysim3, szsim] = ProjectEnzoData(pf, masssim, phi=phi, theta=theta,
                                                                                psi=psi, DMProject=DMProject)
        sumsim.data = gaussian_filter(dmsim.data + masssim.data, 2.0)
        align = SetAlign(dataB1, phi, theta, psi, ConstrainPhi=ConstrainPhi, MaxShift=MaxShift)

        # The 0.22 is the approximate alignment of the bullet cluster on the sky. 
        tol = 1.0E-7  # Alignment tolerance
        data1list = list((sumsim, xraysim1))
        data2list = list((dataA, dataB1))
        sigmalist = list((sigmaA, sigmaB1))

        # Mask (1,0) = > mass constrain
        # Mask (1,1) = > mass + xray constrain
        # Mask (0,0,1) = > GravLens constrain (S+W)
        # Mask (0,1,1) = > GravLens + xray constrain
        # if Mask[3] == 1:
        #     lens = Lens(sumsim)

        if Mask[1] == 1:
            masklist = list((maskA, maskA))
        else:
            masklist = list((maskA, maskANull))

        [fom, xfom, phi] = FindBestShift(data1list, data2list, sigmalist, masklist, align, tol)
        # phi = align.d[4]
        # shiftedxsim1 = shifteddata1list[1]
        # xray1chisquared = pow((dataB1.data - shiftedxsim1.data) / sigmaB1.data, 2) * maskA.data
        # xfom = xray1chisquared.sum() / mask_sum
        return (fom, xfom, phi)
    except:
        print "Error in SimpleFom routine", sys.exc_info()[0]
        return (1.0E5, 1.0E5, 0.0)  # If there's an error, give it a large FOM


def FindBestShift(data1list, data2list, sigmalist, masklist, align, tol):
    numarrays = len(data1list)
    shifteddata1list = list()
    for i in range(numarrays):
        shifteddata1list.append(
            Array2d(data2list[i].xmin, data2list[i].xmax, data2list[i].nx, data2list[i].ymin, data2list[i].ymax,
                    data2list[i].ny))
    target = optfunc_massx(data1list, data2list, shifteddata1list, sigmalist, masklist, align)
    x0 = [align.d[0], align.d[1], align.d[4]]

    SLSQPopt = {
        'method': 'SLSQP',
        'jac': False,
        'bounds': ((align.dmin[0], align.dmax[0]), (align.dmin[1], align.dmax[1]), (align.dmin[4], align.dmax[4])),
        'tol': 1e-3
    }
    BHopts = {'niter': 1000,
              'T': 20,
              'minimizer_kwargs': SLSQPopt,
              'disp': False,
              'niter_success': 10
              }
    try :
        bestfit = basinhopping(target.fit, x0, **BHopts)
        if bestfit['minimization_failures'] == 0 :
            target.fit(bestfit['x'])
            return [target.images_align.chi_sq, target.images_align.chi_sq_list[1], target.align.d[4]]
        else : return (1.0E5, 1.0E5, 0.0)
    except :
        print "Error in basinhopping routine", sys.exc_info()[0]
        return (1.0E5, 1.0E5, 0.0)  # If there's an error, give it a large FOM