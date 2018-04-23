import numpy as np
import sys
import BulletConstants
from routines import euler_angles
from find_best_fit import SimpleFom
# from MonteCarloFom_Origin import SimpleFomFom  ##Use old c# routine, varified
from get_data import GetData, GetPF
from FetchEnzo import FindEnzoCentroids


def FindFom(Parameters, Mask):
    (Z, snapmin, snapmax, theta, psi, TFudge, MaxShift, ConstrainPhi, SpectralIndex) = Parameters
    # This is a new, faster Fom finding routine that uses only mass and X-ray, and searches for a
    # minimum in time, psi, and theta.
    # This version first scans through a Theta,Psi matrix to determine the optimum.

    data = (
    dataA, sigmaA, maskA, maskANull, dataB1, sigmaB1, dataB2, sigmaB2, dataB3, sigmaB3, dataC, sigmaC, dataD, sigmaD,
    maskD, maskDNull, dataE, sigmaE) = GetData()
    mask_sum = maskA.data.sum()

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

        # Next, run the Time Stripe
    Theta = int(100 * theta)
    Psi = int(100 * psi)
    bestfom = 1.0E6
    bestsnap = snapmin
    for snap in range(snapmin, snapmax + 1):
        phi = 0.0
        try:
            pf = GetPF(snap)
            simtime = pf.h.parameters['InitialTime'] * BulletConstants.TimeConversion  # Put time in Gyears
            CenOut = FindEnzoCentroids(pf)  # CenOut = [NumPart, Masses, Centroids, MassWithin250K]
            BulletDMPos = CenOut[2][0]
            MainDMPos = CenOut[2][1]
            dt = simtime - FomMatrix[snap - 1, Psi, Theta][5]
            BulletDMVel = (BulletDMPos - FomMatrix[snap - 1, Psi, Theta][3]) / dt
            MainDMVel = (MainDMPos - FomMatrix[snap - 1, Psi, Theta][4]) / dt
            RelVel = BulletDMVel - MainDMVel  # Relative cluster velocities
            for PsiTest in range(PsiMin, PsiMax + DeltaPsi,
                                 DeltaPsi):  # Since we know RelVel, run Vz test on entire matrix
                for ThetaTest in range(ThetaMin, ThetaMax + DeltaTheta, DeltaTheta):
                    psitest = PsiTest / 100.0
                    thetatest = ThetaTest / 100.0
                    R = euler_angles(-psitest, -thetatest, 0.0)
                    Vz = abs(np.dot(R, RelVel)[2])  # This is the observed Z-Velocity of the bullet relative to the CM
                    if Vz > MaxVz or Vz < MinVz:  # If Vz is outside of observed Mean +/- 3 Sigma
                        print 'Outside allowed Vz, snap = %d, Psi = %.3f, Theta = %.3f, Vz = %.2f, MinVz = %.2f, MaxVz = %.2f\n' % (
                        snap, psitest, thetatest, Vz, MinVz, MaxVz)
                        sys.stdout.flush()
                        FomMatrix[snap, PsiTest, ThetaTest] = (1.0E5, 1.0E5, 0.0, BulletDMPos, MainDMPos,
                                                               simtime)  # If outside allowed Vz, give it a large FOM
                        fom = 1.0E5
                    else:
                        print 'Within allowed Vz, snap = %d, Psi = %.3f, Theta = %.3f, Vz = %.2f, MinVz = %.2f, MaxVz = %.2f\n' % (
                        snap, psitest, thetatest, Vz, MinVz, MaxVz)
                        sys.stdout.flush()
                        FomMatrix[snap, PsiTest, ThetaTest] = (-1.0, -1.0, 0.0, BulletDMPos, MainDMPos, simtime)
            (fom, xfom, phi, BulletDMPos, MainDMPos, simtime) = FomMatrix[snap, Psi, Theta]
            if fom > 9.9E4:  # Failed Vz test
                print 'In Time Stripe, outside allowed Vz, snap = %d, Psi = %.3f, Theta = %.3f\n' % (snap, psi, theta)
                sys.stdout.flush()
                continue
            else:
                print 'In Time Stripe, within allowed Vz, snap = %d, Psi = %.3f, Theta = %.3f\n' % (snap, psi, theta)
                (fom, xfom, phi) = SimpleFom(pf, data, phi=phi, theta=theta, psi=psi, ConstrainPhi=ConstrainPhi,
                                             Mask=Mask, Z=Z, TFudge=TFudge, SpectralIndex=SpectralIndex,
                                             MaxShift=MaxShift)
                FomMatrix[snap, Psi, Theta] = (fom, xfom, phi, BulletDMPos, MainDMPos, simtime)
                print "In Time stripe, fom = %f, snap = %d, Psi = %.3f, Theta = %.3f" % (fom, snap, psi, theta)
                sys.stdout.flush()
            if fom < bestfom:
                bestfom = fom
                bestsnap = snap
        except:
            continue
    print "Finished Time stripe, bestfom = %f, bestsnap = %d, Psi = %.3f, Theta = %.3f" % (
    bestfom, bestsnap, psi, theta)
    sys.stdout.flush()
    # Next, run the Theta Stripe
    snap = bestsnap
    besttheta = theta
    for Theta in range(ThetaMin, ThetaMax + DeltaTheta, DeltaTheta):
        phi = 0.0
        theta = Theta / 100.0
        (fom, xfom, phi, BulletDMPos, MainDMPos, simtime) = FomMatrix[snap, Psi, Theta]
        if fom > 9.9E4:  # Failed Vz test
            print 'In ThetaStripe, outside allowed Vz, snap = %d, Psi = %.3f, Theta = %.3f\n' % (snap, psi, theta)
            sys.stdout.flush()
            continue
        else:
            try:
                print 'In Theta Stripe, within allowed Vz, snap = %d, Psi = %.3f, Theta = %.3f\n' % (snap, psi, theta)
                pf = GetPF(snap)
                (fom, xfom, phi) = SimpleFom(pf, data, phi=phi, theta=theta, psi=psi, ConstrainPhi=ConstrainPhi,
                                             Mask=Mask, Z=Z, TFudge=TFudge, SpectralIndex=SpectralIndex,
                                             MaxShift=MaxShift)
                FomMatrix[snap, Psi, Theta] = (fom, xfom, phi, BulletDMPos, MainDMPos, simtime)
                print "In Theta stripe, fom = %f, snap = %d, Psi = %.3f, Theta = %.3f" % (fom, snap, psi, theta)
                sys.stdout.flush()
            except:
                FomMatrix[snap, Psi, Theta] = (fom, xfom, phi, BulletDMPos, MainDMPos, simtime)
        if fom < bestfom:
            bestfom = fom
            bestsnap = snap
            besttheta = theta

        # Next, run the Minimization search
    counter = 0
    MaxCounter = (snapmax - snapmin + 1) * NPsi * NTheta
    while counter < MaxCounter:
        counter = counter + 1
        print "%d times through Minimization Routine - Snap = %d, Psi = %.2f, Theta = %.2f" % (
        counter, snap, psi, theta)
        sys.stdout.flush()
        # After completing both stripes, find the best fom so far and go to this point:
        fom = 1.0E6
        for sn in range(snapmin - 1, snapmax + 1):
            for Ps in range(PsiMin - DeltaPsi, PsiMax + DeltaPsi, DeltaPsi):
                for Th in range(ThetaMin - DeltaTheta, ThetaMax + DeltaTheta, DeltaTheta):
                    if FomMatrix[sn, Ps, Th][0] < fom and FomMatrix[sn, Ps, Th][0] > 0:
                        (fom, xfom, phi, BulletDMPos, MainDMPos, simtime) = FomMatrix[sn, Ps, Th]
                        snap = sn
                        Theta = Th
                        Psi = Ps

        # If this point is better than all points around it (within tolerance), we're done
        print "Finished both stripes, fom = %f, snap = %d, Psi = %.3f, Theta = %.3f" % (fom, snap, psi, theta)
        sys.stdout.flush()
        FomMatTol = FomMatrix[snap, Psi, Theta][0] * 0.999
        if FomMatTol < FomMatrix[snap + 1, Psi, Theta][0] and FomMatTol < FomMatrix[snap - 1, Psi, Theta][0] \
                and FomMatTol < FomMatrix[snap, Psi + DeltaPsi, Theta][0] and FomMatTol < \
                FomMatrix[snap, Psi - DeltaPsi, Theta][0] \
                and FomMatTol < FomMatrix[snap, Psi, Theta + DeltaTheta][0] and FomMatTol < \
                FomMatrix[snap, Psi, Theta - DeltaTheta][0]:
            return (fom, xfom, snap, phi, Theta / 100.0, Psi / 100.0, simtime, counter)

        # If not, find a surrounding point not yet done and run it.
        for (sn, Ps, Th) in [(snap - 1, Psi, Theta), (snap + 1, Psi, Theta), (snap, Psi + DeltaPsi, Theta),
                             (snap, Psi - DeltaPsi, Theta), \
                             (snap, Psi, Theta + DeltaTheta), (snap, Psi, Theta - DeltaTheta)]:
            if FomMatrix[sn, Ps, Th][0] > 999.0:  # This means it's a boundary or a failed Vz test
                continue
            elif FomMatrix[sn, Ps, Th][0] < 0.0:  # This means it hasn't yet been run, so move there and run it.
                (fom, xfom, phi, BulletDMPos, MainDMPos, simtime) = FomMatrix[sn, Ps, Th]
                snap = sn
                phi = 0.0
                Theta = Th
                theta = Theta / 100.0
                Psi = Ps
                psi = Psi / 100.0
                try:
                    pf = GetPF(snap)
                    (fom, xfom, phi) = SimpleFom(pf, data, phi=phi, theta=theta, psi=psi, ConstrainPhi=ConstrainPhi,
                                                 Mask=Mask, Z=Z, TFudge=TFudge, SpectralIndex=SpectralIndex,
                                                 MaxShift=MaxShift)
                    FomMatrix[snap, Psi, Theta] = (fom, xfom, phi, BulletDMPos, MainDMPos, simtime)
                except:
                    continue

        return (1.0E5, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0)  # Returns garbage if it fails to find an optimum
