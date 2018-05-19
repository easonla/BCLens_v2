#Author: Yi-Hsuan Hsu, NYU;
#Date: March-2-18
from func import get_data
from func import best_plot


def main():
    """
    This is a faster, pure python code to constraint best simulation time and viewing angle
    Available for mass, mass-Xray, Gravlens, Gravlens-Xray constraints
    :return:
    """

    # read config file
    parameters, mask = get_data.read_input("fominput")
    with open('output/newfom_massx1.out', 'rb') as f:
        file = f.read().split(",")
    result = [float(s) for s in file]

    print "Run Plotting"
    (Z, snapmin, snapmax, theta, psi, TFudge, MaxShift, ConstrainPhi, SpectralIndex) = parameters
    bestsnap = int(result[2])
    bestphi = result[3]
    besttheta = result[4]
    bestpsi = result[5]
    efom = best_plot.BestPlot(bestsnap, Z, phi=bestphi, theta=besttheta, psi=bestpsi,
                              PlotSuffix='New_MassX', FomLocate=False, TFudge=TFudge,
                              ConstrainPhi=ConstrainPhi, Mask=mask, SpectralIndex=SpectralIndex, MaxShift=MaxShift)
    print "Finished Plot!"
    print efom


if __name__ == "__main__":
    main()
