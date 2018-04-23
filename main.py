__author__ = "yhhsu"
from func import get_data
from func import find_fom


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

    with open('newfom_massx1.out','w') as f:
        f.write(result)
    print "Finished!"


if __name__ == "__main__":
    main()
