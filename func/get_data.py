import BulletConstants
import Config
import numpy as np
from Classes import Array2d
from yt.mods import *


def read_input(filename):
    toppath = Config.toppath
    file = open(toppath + filename, 'r')
    lines = file.readlines()
    file.close()
    Z = float(lines[0].strip().split()[1])
    snapmin = int(lines[1].strip().split()[1])
    snapmax = int(lines[2].strip().split()[1])
    theta = float(lines[3].strip().split()[1])
    psi = float(lines[4].strip().split()[1])
    TFudge = float(lines[8].strip().split()[1])
    MaxShift = float(lines[9].strip().split()[1])
    SpectralIndex = float(lines[10].strip().split()[1])
    ConstrainPhi = bool(lines[11].strip().split()[1])
    Mass_Constrain = int(lines[12].strip().split()[1])
    Xray1_Constrain = int(lines[13].strip().split()[1])
    Xray2_Constrain = int(lines[14].strip().split()[1])
    Xray3_Constrain = int(lines[15].strip().split()[1])
    SZ_Constrain = int(lines[16].strip().split()[1])
    parameters = (
        Z, snapmin, snapmax,
        theta, psi, TFudge,
        MaxShift, ConstrainPhi, SpectralIndex
    )
    mask = (Mass_Constrain, Xray1_Constrain, Xray2_Constrain, Xray3_Constrain, SZ_Constrain)
    return parameters, mask


def GetPF(snap):
    # Retrieves a yt - pf file given a snapshot number
    datapath = Config.datapath
    if snap < 10:
        filename = datapath + "DD000" + str(snap) + "/output_000" + str(snap)
    elif snap < 100:
        filename = datapath + "DD00" + str(snap) + "/output_00" + str(snap)
    elif snap < 1000:
        filename = datapath + "DD0" + str(snap) + "/output_0" + str(snap)
    else:
        filename = datapath + "DD" + str(snap) + "/output_" + str(snap)

    pf = load(filename)
    return pf


def GetData():
    # This reads in the image data
    toppath = Config.toppath
    datapathA = toppath + 'data/kappa_25Apr12.dat'
    datapathB1 = toppath + 'data/xray_500_2000_29Jun11.dat'
    datapathB2 = toppath + 'data/xray_2000_5000_29Jun11.dat'
    datapathB3 = toppath + 'data/xray_5000_8000_29Jun11.dat'
    datapathC = toppath + 'data/sze_data_13Mar13.dat'
    datapathD = toppath + 'data/xray_temp_23Jun11.dat'
    datapathE = toppath + 'data/radio_24Apr13.dat'

    datapathsA = toppath + 'data/mass_sigma_11Feb13.dat'
    datapathsB1 = toppath + 'data/xray_sigma_500_2000_18Jul11.dat'
    datapathsB2 = toppath + 'data/xray_sigma_2000_5000_18Jul11.dat'
    datapathsB3 = toppath + 'data/xray_sigma_5000_8000_18Jul11.dat'
    datapathsC = toppath + 'data/sze_sigma_13Mar13.dat'
    datapathsD = toppath + 'data/xray_temp_sigma_23Jun11.dat'
    datapathsE = toppath + 'data/radio_sigma_5Oct11.dat'

    datapathmA = toppath + 'data/mask_3May12.dat'
    datapathmANull = toppath + 'data/mask_23Jun11_null.dat'

    datapathmD = toppath + 'data/xtemp_mask_23Jun11.dat'
    datapathmDNull = toppath + 'data/xtemp_mask_null_29Jun11.dat'  # Ignore temp in this case.

    # First, the Mass data - designated as A
    xscaleA = yscaleA = BulletConstants.AngularScale * 3600 * BulletConstants.MassDataScale
    # Pixel size in kpc. 4.413 kpc/" is the angular scale at the bullet cluster
    # 3600 "/degree, 9.86E-4 degrees/pixel is the data scale

    xpixelsA = ypixelsA = 110
    sxpixelsA = sypixelsA = 220
    dxmaxA = xscaleA * xpixelsA / 2
    dymaxA = yscaleA * ypixelsA / 2
    dxminA = -xscaleA * xpixelsA / 2
    dyminA = -yscaleA * ypixelsA / 2
    sxmaxA = xscaleA * xpixelsA
    symaxA = yscaleA * ypixelsA

    dataA = Array2d(dxminA, dxmaxA, xpixelsA, dyminA, dymaxA, ypixelsA)
    dataA = GetBulletData(datapathA, dataA)  # Mass density measured data
    sigmaA = Array2d(dxminA, dxmaxA, xpixelsA, dyminA, dymaxA, ypixelsA)
    sigmaA = GetBulletData(datapathsA, sigmaA)  # Mass sigma

    # sigmaA.data = sigmaA.data / 2.0  # Arbitrarily reduce mass sigma to improve mass fit.

    maskA = Array2d(dxminA, dxmaxA, xpixelsA, dyminA, dymaxA, ypixelsA)
    maskA = GetBulletData(datapathmA, maskA)  # Mask
    maskANull = Array2d(dxminA, dxmaxA, xpixelsA, dyminA, dymaxA, ypixelsA)
    maskANull = GetBulletData(datapathmANull, maskANull)  # Null Mask

    xscaleB = yscaleB = BulletConstants.AngularScale * 3600 * BulletConstants.XRayDataScale
    # Pixel size in kpc. 4.413 kpc/" is the angular scale at the bullet cluster
    # 3600 "/degree, 9.86E-4 degrees/pixel is the data scale

    xpixelsB = ypixelsB = 110
    sxpixelsB = sypixelsB = 220
    dxmaxB = xscaleB * xpixelsB / 2
    dymaxB = yscaleB * ypixelsB / 2
    sxmaxB = dxmaxB * 2
    symaxB = dymaxB * 2
    dataB1 = Array2d(-dxmaxB, dxmaxB, xpixelsB, -dymaxB, dymaxB, ypixelsB)
    dataB1 = GetBulletData(datapathB1, dataB1)  # XRay measured data
    sigmaB1 = Array2d(-dxmaxB, dxmaxB, xpixelsB, -dymaxB, dymaxB, ypixelsB)
    sigmaB1 = GetBulletData(datapathsB1, sigmaB1)  # XRay sigma
    dataB2 = Array2d(-dxmaxB, dxmaxB, xpixelsB, -dymaxB, dymaxB, ypixelsB)
    dataB2 = GetBulletData(datapathB2, dataB2)  # XRay measured data
    sigmaB2 = Array2d(-dxmaxB, dxmaxB, xpixelsB, -dymaxB, dymaxB, ypixelsB)
    sigmaB2 = GetBulletData(datapathsB2, sigmaB2)  # XRay sigma
    dataB3 = Array2d(-dxmaxB, dxmaxB, xpixelsB, -dymaxB, dymaxB, ypixelsB)
    dataB3 = GetBulletData(datapathB3, dataB3)  # XRay measured data
    sigmaB3 = Array2d(-dxmaxB, dxmaxB, xpixelsB, -dymaxB, dymaxB, ypixelsB)
    sigmaB3 = GetBulletData(datapathsB3, sigmaB3)  # XRay sigma

    # dataB.data=gaussian_filter(dataB.data,0.5)
    # 0.5-sigma smoothing of X-Ray data
    dataC = Array2d(dxminA, dxmaxA, xpixelsA, dyminA, dymaxA, ypixelsA)
    dataC = GetBulletData(datapathC, dataC)  # Measured SZE data
    sigmaC = Array2d(dxminA, dxmaxA, xpixelsA, dyminA, dymaxA, ypixelsA)
    # sigmaC=GetBulletData(datapathsC,sigmaC)# SZE Sigma
    sigmaC.data = sigmaC.data + 25.0E-6  # Set to 25 microKelvin based on Plagge et.al.

    dataC.data = abs(dataC.data * 1E6)  # convert to micro Kelvin
    sigmaC.data = sigmaC.data * 1E6  # convert to micro Kelvin

    xscaleD = yscaleD = BulletConstants.AngularScale * 3600 * BulletConstants.XRayTempDataScale
    # Pixel size in kpc. 4.413 kpc/" is the angular scale at the bullet cluster
    # 3600 "/degree, 1.09333 degrees/pixel is the data scale

    xpixelsD = ypixelsD = 100
    sxpixelsD = sypixelsD = 200
    dxmaxD = xscaleD * xpixelsD / 2
    dymaxD = yscaleD * ypixelsD / 2
    sxmaxD = dxmaxD * 2
    symaxD = dymaxD * 2
    dataD = Array2d(-dxmaxD, dxmaxD, xpixelsD, -dymaxD, dymaxD, ypixelsD)
    dataD = GetBulletData(datapathD, dataD)  # XRay measured data

    sigmaD = Array2d(-dxmaxD, dxmaxD, xpixelsD, -dymaxD, dymaxD, ypixelsD)
    sigmaD = GetBulletData(datapathsD, sigmaD)  # Xray Temp Sigma
    maskD = Array2d(-dxmaxD, dxmaxD, xpixelsD, -dymaxD, dymaxD, ypixelsD)
    maskD = GetBulletData(datapathmD, maskD)  # Mask
    maskDNull = Array2d(-dxmaxD, dxmaxD, xpixelsD, -dymaxD, dymaxD, ypixelsD)
    maskDNull = GetBulletData(datapathmDNull, maskDNull)  # Null Mask

    dataE = Array2d(-dxmaxA, dxmaxA, xpixelsA, -dymaxA, dymaxA, ypixelsA)
    sigmaE = Array2d(-dxmaxA, dxmaxA, xpixelsA, -dymaxA, dymaxA, ypixelsA)
    dataE = GetBulletData(datapathE,
                          dataE)  # Radio measured data
    sigmaE = GetBulletData(datapathsE, sigmaE)  # Radio Sigma

    return (
    dataA, sigmaA, maskA, maskANull, dataB1, sigmaB1, dataB2, sigmaB2, dataB3, sigmaB3, dataC, sigmaC, dataD, sigmaD,
    maskD, maskDNull, dataE, sigmaE)


def GetBulletData(filename, data):
    infile = open(filename, 'r')
    oneddata = infile.readlines()
    for i in range(data.nx):
        for j in range(data.ny):
            data.data[i, j] = float(oneddata[i + j * data.nx])
    return data


def GenerateMatrix(snapmin, snapmax, psi, theta):
    FomMatrix = {}
    MinVz = BulletConstants.BulletVz - 3.0 * BulletConstants.BulletSigVz  # Minimum Bullet radial velocity consistent with observations
    MaxVz = BulletConstants.BulletVz + 3.0 * BulletConstants.BulletSigVz  # Maximum Bullet radial velocity consistent with observations
    DeltaPsi = 3  # Using Psi and Theta 100* psi and theta, so I can use integers
    DeltaTheta = 3  # Using DeltaPsi and Delta theta of 0.03 radians or about 2 degrees
    PsiMax = int(100 * psi) + 18
    PsiMin = int(100 * psi) - 18
    ThetaMax = int(100 * theta) + 18
    ThetaMin = int(100 * theta) - 18
    NPsi = int((PsiMax - PsiMin) / DeltaPsi) + 3
    NTheta = int((ThetaMax - ThetaMin) / DeltaTheta) + 3
    FomMatrix = {}  # Dictionary with entries (fom, xfom, phi, BulletDMPos, MainDMPos, simtime)
    for snap in range(snapmin - 1, snapmax + 2):  # First fill the whole array with 1E6 - this IDs the boundaries
        for Psi in range(PsiMin - DeltaPsi, PsiMax + 2 * DeltaPsi, DeltaPsi):
            for Theta in range(ThetaMin - DeltaTheta, ThetaMax + 2 * DeltaTheta, DeltaTheta):
                FomMatrix[snap, Psi, Theta] = (1.0E6, 1.0E6, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
    for snap in range(snapmin,
                      snapmax + 1):  # Next put -1.0 everywhere except at the boundaries - this IDs conditions not yet run
        for Psi in range(PsiMin, PsiMax + DeltaPsi, DeltaPsi):
            for Theta in range(ThetaMin, ThetaMax + DeltaTheta, DeltaTheta):
                FomMatrix[snap, Psi, Theta] = (-1.0, -1.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
    try:  # Load positions into left (snapmin - 1) boundary
        pf = GetPF(snapmin - 1)
        simtime = pf.h.parameters['InitialTime'] * BulletConstants.TimeConversion  # Put time in Gyears
        CenOut = FindEnzoCentroids(pf)  # CenOut = [NumPart, Masses, Centroids, MassWithin250K]
        BulletDMPos = CenOut[2][0]
        MainDMPos = CenOut[2][1]
    except:
        simtime = 0.0
        BulletDMPos = [0.0, 0.0, 0.0]
        MainDMPos = [0.0, 0.0, 0.0]
    for Psi in range(PsiMin, PsiMax + DeltaPsi, DeltaPsi):
        for Theta in range(ThetaMin, ThetaMax + DeltaTheta, DeltaTheta):
            FomMatrix[snapmin - 1, Psi, Theta] = (1.0E6, 1.0E6, 0.0, BulletDMPos, MainDMPos, simtime)
    return FomMatrix


def ReadLookups(Z):
    # This subroutine reads in the data from the APEC lookup tables and places it in an array
    # The data is interpolated to get the data for the required Z (metallicity)
    ApecData = np.zeros([281, 4])  # Array to hold the data
    for bin in range(3):  # bin 0 is 0.5-2kev, bin 1 is 2-5kev, bin 2 is 5-8kev, bin 3 is 0.5-8kev
        if bin == 0:
            infile = open(Config.toppath + 'data/apec/apec_xray_0.5_2.0.txt', 'r')
        elif bin == 1:
            infile = open(Config.toppath + 'data/apec/apec_xray_2.0_5.0.txt', 'r')
        elif bin == 2:
            infile = open(Config.toppath + 'data/apec/apec_xray_5.0_8.0.txt', 'r')
        lines = infile.readlines()
        infile.close()
        counter = 0
        for line in lines:
            if line.strip().split()[0] == 'LogT':  # Skips header line
                continue
            minZ = max(0, int(round((Z * 10))))
            maxZ = minZ + 1
            if maxZ > 10:
                maxZ = 10
                minZ = 9
            f = (maxZ - 10.0 * Z) * float(line.strip().split()[minZ + 1]) + (10.0 * Z - minZ) * float(
                line.strip().split()[maxZ + 1])
            ApecData[counter, bin] = f
            if bin == 0:
                ApecData[counter, 3] = f  # This bin is just the sum of the other three
            else:
                ApecData[counter, 3] = ApecData[counter, 3] + f
            counter = counter + 1
    return ApecData
