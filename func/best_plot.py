import numpy as np
import pickle
from matplotlib.pyplot import cm
from yt.mods import *
from scipy.ndimage import convolve, gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages

import BulletConstants
from get_data import GetData, GetKernels, GetPF, ReadLookups
from FetchEnzo import _EnzoBMag, ProjectEnzoData, ProjectEnzoTemp, EnclosedMass
from Classes import Array2d
from find_best_fit import SetAlign, FindBestShift
from plot_routines import ComparisonPlot, SetContourLevels, MultiLinePlot, EdgeDetectionPlot, FomLocationPlot


def BestPlot(snap, Z, phi=0.0, theta=0.0, psi=0.0, PlotSuffix='New_Kappa', FomLocate=False, TFudge=1.0,
             ConstrainPhi=False, Mask=(1, 0, 0, 0, 0, 0), SpectralIndex=3.2, MaxShift=0.0):
    (dataA, sigmaA, maskA, maskANull, dataB1, sigmaB1, dataB2, sigmaB2, dataB3, sigmaB3, dataC, sigmaC, dataD, sigmaD,
     maskD, maskDNull, dataE, sigmaE) = GetData()
    (szekernel, kernel60, kernel20, kernel15, kernel5) = GetKernels()
    [dyyA, dxxA] = np.meshgrid(dataA.y, dataA.x)  # Data grid for plots
    [dyyB, dxxB] = np.meshgrid(dataB1.y, dataB1.x)  # Data grid for plots
    [dyyD, dxxD] = np.meshgrid(dataD.y, dataD.x)  # Data grid for plots

    align = SetAlign(dataB1, phi, theta, psi, ConstrainPhi=ConstrainPhi, MaxShift=MaxShift)
    pp = PdfPages('output/Graph_' + PlotSuffix + '.pdf')
    pf = GetPF(snap)
    add_field("BMag", function=_EnzoBMag)
    simtime = pf.h.parameters['InitialTime'] * BulletConstants.TimeConversion  # Put time in Gyears

    dmsim = Array2d(2.0 * dataA.xmin, 2.0 * dataA.xmax, 2 * dataA.nx, 2.0 * dataA.ymin, 2.0 * dataA.ymax, 2 * dataA.ny)
    masssim = Array2d(2.0 * dataA.xmin, 2.0 * dataA.xmax, 2 * dataA.nx, 2.0 * dataA.ymin, 2.0 * dataA.ymax,
                      2 * dataA.ny)
    sumsim = Array2d(2.0 * dataA.xmin, 2.0 * dataA.xmax, 2 * dataA.nx, 2.0 * dataA.ymin, 2.0 * dataA.ymax, 2 * dataA.ny)
    tempmass = Array2d(2.0 * dataD.xmin, 2.0 * dataD.xmax, 2 * dataD.nx, 2.0 * dataD.ymin, 2.0 * dataD.ymax,
                       2 * dataD.ny)
    [syyA, sxxA] = np.meshgrid(dmsim.y, dmsim.x)  # Sim grid for plots
    [syyD, sxxD] = np.meshgrid(tempmass.y, tempmass.x)  # Sim grid for plots

    ApecData = ReadLookups(Z)  # Reads the APEC lookup tables.
    for grid in pf.h.grids:
        grid.set_field_parameter('ApecData', ApecData)
        grid.set_field_parameter('TFudge', TFudge)
        grid.set_field_parameter('SpectralIndex', SpectralIndex)

    DMProject = False
    [dmsim, masssim, xraysim1, xraysim2, xraysim3, szsim] = ProjectEnzoData(pf, masssim, phi=phi, theta=theta, psi=psi,
                                                                            DMProject=DMProject)
    sumsim.data = gaussian_filter(dmsim.data + masssim.data, 2.0)
    szsim.data = convolve(abs(szsim.data * 1E6), szekernel.data, mode='constant',
                          cval=0.0)  # Smooth simulation with given transfer function
    # xraysim1.data=convolve(xraysim1.data,kernel5.data,mode='constant',cval=0.0)#Smooth simulation with given transfer function
    # xraysim2.data=convolve(xraysim2.data,kernel5.data,mode='constant',cval=0.0)#Smooth simulation with given transfer function
    # xraysim3.data=convolve(xraysim3.data,kernel5.data,mode='constant',cval=0.0)#Smooth simulation with given transfer function

    [tempsim, bmagsim, synchsim] = ProjectEnzoTemp(pf, tempmass, phi=phi, theta=theta, psi=psi)
    synchsim.data = convolve(synchsim.data, kernel20.data, mode='constant',
                             cval=0.0)  # Smooth simulation with given transfer function
    BMax = pf.h.sphere((0, 0, 0), (1000.0, "kpc")).quantities["MaxLocation"]("BMag")[0] * 1.0E6
    BMin = pf.h.sphere((0, 0, 0), (1000.0, "kpc")).quantities["MinLocation"]("BMag")[0] * 1.0E6
    print "Bmax = %.4g, Bmin = %.4g\n" % (BMax, BMin)
    sys.stdout.flush()
    # sys.exit()
    # Projected Enzo sim data

    tol = 1.0E-7  # Alignment tolerance

    data1list = list((sumsim, xraysim1, xraysim2, xraysim3, szsim, tempsim, synchsim, bmagsim))
    data2list = list((dataA, dataB1, dataB2, dataB3, dataC, dataD, dataE, dataE))
    sigmalist = list((sigmaA, sigmaB1, sigmaB2, sigmaB3, sigmaC, sigmaD, sigmaE, sigmaE))
    masklist = list()
    # This code appends the necessary masks. If Mask[i]==0, this data is not included in the FOM
    for i in range(5):
        if Mask[i] == 0:
            masklist.append((maskANull))
            print "i = %d, Mask[i] = %d, appending maskAnull\n" % (i, Mask[i])
        else:
            masklist.append((maskA))
            print "i = %d, Mask[i] = %d, appending maskA\n" % (i, Mask[i])

    if Mask[5] == 0:
        masklist.append((maskDNull))
        print "i = 5, Mask[i] = %d, appending maskDNull\n" % Mask[i]
    else:
        masklist.append(maskD)
        print "i = 5, Mask[i] = %d, appending maskD\n" % Mask[i]
    masklist.append(maskANull)  # Always mask the radio
    masklist.append(maskANull)  # Always mask the radio bmag
    sys.stdout.flush()

    def get_item(optfunc):
        if optfunc is None:
            print "Optimization Fail, Try again"
            exit()
        else:
            return optfunc.images_align.shifteddata1list, optfunc.align, optfunc.total_chiq_sq

    optfunc = FindBestShift(data1list, data2list, sigmalist, masklist, align, tol)
    pickle_output = open('output/optfunc.pkl', 'wb')
    pickle.dump(optfunc, pickle_output)
    pickle_output.close()
    print "Done with pickle"
    sys.stdout.flush()

    (shifteddata1list, align, fom) = get_item(optfunc)
    shiftedmsim = shifteddata1list[0]
    shiftedxsim1 = shifteddata1list[1]
    shiftedxsim2 = shifteddata1list[2]
    shiftedxsim3 = shifteddata1list[3]
    shiftedszsim = shifteddata1list[4]
    shiftedtempsim = shifteddata1list[5]
    shiftedsynchsim = shifteddata1list[6]
    shiftedbmagsim = shifteddata1list[7]
    MassWithin250 = EnclosedMass(dataA, 250)
    print "Main Data Mass within250 = %f, Bullet Data Mass Within250 = %f\n" % (MassWithin250[0], MassWithin250[1])
    MassWithin250 = EnclosedMass(shiftedmsim, 250)
    print "Main Sim Mass within250 = %f, Bullet Sim Mass Within250 = %f\n" % (MassWithin250[0], MassWithin250[1])

    try:
        [filled_levels, line_levels] = SetContourLevels(0.0, 70.0)
        fig = ComparisonPlot(dataA, shiftedmsim, "Mass Lensing", filled_levels=filled_levels, line_levels=line_levels,
                             simtime=simtime, fom=fom, line=[80, 53, 0.26],
                             legend_location=[0.80, 0.3], line_plot_yticks=[0.0, 20.0, 40.0, 60.0])
        pp.savefig(fig)
        fig.savefig('output/mass_lensing.png')

        [filled_levels, line_levels] = SetContourLevels(-10.0, -4.0)
        fig = ComparisonPlot(dataB1, shiftedxsim1, "X-ray Flux - 500-2000eV", filled_levels=filled_levels,
                             line_levels=line_levels, cmap=cm.spectral, simtime=simtime,
                             fom=fom, line=[70, 55, 0.24], legend_location=[0.50, 1.0],
                             line_plot_yticks=[0.0, 1.0, 2.0, 3.0], line_plot_multiplier=1.0E6, take_log=True)
        pp.savefig(fig)
        fig.savefig('output/mass_lensing.png')

        fig = ComparisonPlot(dataB2, shiftedxsim2, "X-ray Flux - 2000-5000eV", filled_levels=filled_levels,
                             line_levels=line_levels, cmap=cm.spectral, simtime=simtime,
                             fom=fom, line=[70, 55, 0.24], legend_location=[0.50, 1.0],
                             line_plot_yticks=[0.0, 0.5, 1.0], line_plot_multiplier=1.0E6, take_log=True)
        pp.savefig(fig)
        fig.savefig('output/xray_bin1.png')

        fig = ComparisonPlot(dataB3, shiftedxsim3, "X-ray Flux - 5000-8000eV", filled_levels=filled_levels,
                             line_levels=line_levels, cmap=cm.spectral, simtime=simtime,
                             fom=fom, line=[70, 55, 0.24], legend_location=[0.50, 1.0],
                             line_plot_yticks=[0.0, 1.0, 2.0, 3.0], line_plot_multiplier=1.0E7, take_log=True)
        pp.savefig(fig)
        fig.savefig('output/xray_bin2.png')

        [filled_levels, line_levels] = SetContourLevels(-10.0, -4.0)
        fig = MultiLinePlot(dataB1, shiftedxsim1, "X-ray Flux - 500-2000eV", filled_levels=filled_levels,
                            line_levels=line_levels, cmap=cm.spectral, simtime=simtime,
                            fom=fom, line1=[78, 56, -0.24], line2=[78, 56, 0.24], line3=[78, 56, 1.02],
                            legend_location1=[0.70, 0.30], legend_location2=[0.70, 0.30], legend_location3=[0.70, 0.30],
                            line_plot_yticks=[-8.0, -7.0, -6.0], take_log=True)
        pp.savefig(fig)
        fig.savefig('output/xray_bin3.png')


        [filled_levels, line_levels] = SetContourLevels(-10.0, -4.0)
        ED_filled_levels = np.linspace(0, 10.0, 11)
        ED_line_levels = np.linspace(1.0, 5.0, 5)
        (fig, efom) = EdgeDetectionPlot(dataB1, shiftedxsim1, "X-ray Shock Edge Detection", filled_levels=filled_levels,
                                        ED_filled_levels=ED_filled_levels,
                                        ED_line_levels=ED_line_levels, ED_multiplier=1.0E7, cmap=cm.spectral,
                                        simtime=simtime, fom=fom, take_log=True)
        pp.savefig(fig)
        fig.savefig('output/xray_edge_detect.png')

        [filled_levels, line_levels] = SetContourLevels(0.0, 20.0)
        fig = ComparisonPlot(dataD, shiftedtempsim, "Gas Temperature (keV)", filled_levels=filled_levels,
                             line_levels=line_levels, cmap=cm.spectral, simtime=simtime,
                             fom=fom, line=[70, 52, 0.24], legend_location=[0.60, 1.0],
                             line_plot_yticks=[5.0, 10.0, 15.0])
        pp.savefig(fig)
        fig.savefig('output/gas_temperature.png')

        [filled_levels, line_levels] = SetContourLevels(0.0, 500.0)
        fig = ComparisonPlot(dataC, shiftedszsim, "SZ Temperature Decrement ($\mu K$)", filled_levels=filled_levels,
                             line_levels=line_levels, simtime=simtime,
                             fom=fom, line=[70, 59, 0.24], legend_location=[0.70, 0.32],
                             line_plot_yticks=[0.0, 100.0, 200.0, 300.0])
        pp.savefig(fig)
        fig.savefig('output/sz_temperature.png')

        filled_levels = [0.0, 6.0, 12.0, 24.0, 36.0, 48.0, 96.0]
        line_levels = [0.0, 6.0, 12.0, 24.0, 36.0, 48.0, 96.0]
        fig = ComparisonPlot(dataE, shiftedsynchsim, "Radio Flux - 1.3 GHz ($\mu$ Jy / pixel)",
                             filled_levels=filled_levels, line_levels=line_levels, cmap=cm.spectral,
                             simtime=simtime, fom=fom, line=[70, 59, 0.24], legend_location=[1.10, 1.0],
                             line_plot_yticks=[0.0, 25.0, 50.0, 75.0])
        pp.savefig(fig)
        fig.savefig('output/radioflux.png')

        if FomLocate:
            numpixels = dataA.nx * dataA.ny
            masschisquared = pow((dataA.data - shiftedmsim.data) / sigmaA.data, 2) * maskA.data
            xray1chisquared = pow((dataB1.data - shiftedxsim1.data) / sigmaB1.data, 2) * maskA.data
            xray2chisquared = pow((dataB2.data - shiftedxsim2.data) / sigmaB2.data, 2) * maskA.data
            xray3chisquared = pow((dataB3.data - shiftedxsim3.data) / sigmaB3.data, 2) * maskA.data
            szechisquared = pow((dataC.data - shiftedszsim.data) / sigmaC.data, 2) * maskA.data
            tempchisquared = pow((dataD.data - shiftedtempsim.data) / sigmaD.data, 2) * maskD.data

            mask_sum = maskA.data.sum()
            datasets = [masschisquared, xray1chisquared, xray2chisquared, xray3chisquared, tempchisquared,
                        szechisquared]
            plotsets = [dataA, dataB1, dataB2, dataB3, dataD, dataC]
            subtitles = ["Mass", "X-ray 1", "X-ray 2", "X-ray 3", "Temp", "SZE"]
            fig = FomLocationPlot(datasets, subtitles, plotsets, mask_sum, cmap=cm.spectral, simtime=simtime, fom=fom)
            pp.savefig(fig)
            fig.savefig('output/fomlocal.png')

        print "Finished plots for snapshot file", snap, "\n"
        sys.stdout.flush()
    except:
        print "Unexpected error in plot routines.\n"
        print"sys.exc_info=", sys.exc_info(), "\n"
        pp.close()
        return 0
    pp.close()
    return efom