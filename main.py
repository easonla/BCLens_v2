#Author: Yi-Hsuan Hsu, NYU;
#Date: March-2-18
from func import get_data
from func import find_fom
from func import best_plot


def main():
    """
    This is a faster, pure python code to constraint best simulation time and viewing angle
    Available for mass, mass-Xray, Gravlens, Gravlens-Xray constraints
    :return:
    """

    # read config file
    parameters, mask = get_data.read_input("fominput")
    # result = (fom, xfom, snap, phi, Theta , Psi , simtime, counter)
    result = find_fom.FindFom(parameters, mask)
    print "Best FOM Found: "
    print ("fom = {:f} xfom = {:f} snap = {:d} phi = {:f} theta = {:f} psi = {:f} simtime = {:f} counter = {:d}"
           .format(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]))

    with open('output/newfom_massx1.out', 'w') as f:
        f.write(', '.join(str(s) for s in result))
    print "Finished FindFom!"

    print "Run Plotting"
    (Z, snapmin, snapmax, theta, psi, TFudge, MaxShift, ConstrainPhi, SpectralIndex) = parameters
    bestsnap = result[2]
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
