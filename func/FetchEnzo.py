import numpy as np
import time
from yt.mods import *

import BulletConstants
from Numerical_Routine import EulerAngles


def _EnzoTXRay(field,data):
    ApecData = data.get_field_parameter('ApecData')
    TFudge = data.get_field_parameter('TFudge')
    T = data['Temperature'] * BulletConstants.DegreesK_to_keV * TFudge
    logT = np.log10(T) * 100.0
    np.putmask(logT,logT>179.0,179.0)
    np.putmask(logT,logT<-100.0,-100.0)
    minT = logT.astype(int)
    flux = (minT+1.0-logT) * ApecData[minT+100,3] + (logT-minT) * ApecData[minT+101,3]
    return T * flux * data['Density']**2 / BulletConstants.mp**2 * 1E-14

def _EnzoXRay(field,data):
    ApecData = data.get_field_parameter('ApecData')
    TFudge = data.get_field_parameter('TFudge')
    logT = np.log10(data['Temperature'] * BulletConstants.DegreesK_to_keV * TFudge) * 100.0
    np.putmask(logT,logT>179.0,179.0)
    np.putmask(logT,logT<-100.0,-100.0)
    minT = logT.astype(int)
    flux = (minT+1.0-logT) * ApecData[minT+100,3] + (logT-minT) * ApecData[minT+101,3]
    return flux * data['Density']**2 / BulletConstants.mp**2 * 1E-14

def _EnzoXRay1(field,data):
    ApecData = data.get_field_parameter('ApecData')
    TFudge = data.get_field_parameter('TFudge')
    logT = np.log10(data['Temperature'] * BulletConstants.DegreesK_to_keV * TFudge) * 100.0
    np.putmask(logT,logT>179.0,179.0)
    np.putmask(logT,logT<-100.0,-100.0)
    minT = logT.astype(int)
    flux = (minT+1.0-logT) * ApecData[minT+100,0] + (logT-minT) * ApecData[minT+101,0]
    return flux * data['Density']**2 / BulletConstants.mp**2 * 1E-14 

def _EnzoXRay2(field,data):
    ApecData = data.get_field_parameter('ApecData')
    TFudge = data.get_field_parameter('TFudge')
    logT = np.log10(data['Temperature'] * BulletConstants.DegreesK_to_keV * TFudge) * 100.0
    np.putmask(logT,logT>179.0,179.0)
    np.putmask(logT,logT<-100.0,-100.0)
    minT = logT.astype(int)
    flux = (minT+1.0-logT) * ApecData[minT+100,1] + (logT-minT) * ApecData[minT+101,1]
    return flux * data['Density']**2 / BulletConstants.mp**2 * 1E-14

def _EnzoXRay3(field,data):
    ApecData = data.get_field_parameter('ApecData')
    TFudge = data.get_field_parameter('TFudge')
    logT = np.log10(data['Temperature'] * BulletConstants.DegreesK_to_keV * TFudge) * 100.0
    np.putmask(logT,logT>179.0,179.0)
    np.putmask(logT,logT<-100.0,-100.0)
    minT = logT.astype(int)
    flux = (minT+1.0-logT) * ApecData[minT+100,2] + (logT-minT) * ApecData[minT+101,2]
    return flux * data['Density']**2 / BulletConstants.mp**2 * 1E-14

def _EnzoSZ(field,data):
    TFudge = data.get_field_parameter('TFudge')
    T = data['Temperature'] * BulletConstants.DegreesK_to_eV * TFudge
    SZ = BulletConstants.EnzoSZFactor * data['Density'] * T 
    return SZ

def _EnzoBMag(field, data):
    return np.sqrt(data["Bx"] * data["Bx"] + data["By"] * data["By"] + data["Bz"] * data["Bz"])

def _EnzoSynch(field,data):
    P = data.get_field_parameter('SpectralIndex')
    Eta = 1.00 # Fudge Factor
    PFactor = (P-2.0)/(P+1.0)*gamma(P/4.0+19.0/12.0)*gamma(P/4.0-1.0/12.0)*gamma((P+5.0)/4.0)/gamma((P+7.0)/4.0)
    B = sqrt(data['Bx']**2 + data['By']**2 + data['Bz']**2)
    omegac = BulletConstants.OmegaCPreFactor * B
    omega =  2.0 * pi * BulletConstants.RadioFrequency * (1 + BulletConstants.redshift)
    return  Eta * BulletConstants.RadioPreFactor * PFactor * pow(B,3.0) * pow(omega/omegac,-(P-1.0)/2.0)

def FindEnzoCentroids(pf): 
    start = time.time()
    #print 'Starting Centroid finding\n'
    mass=list()
    x=list()
    y=list()
    z=list()
    NPart = 0 # Particle counter	
    for i in range(pf.h.num_grids): # Read in all of the particle masses and positions
        PixelVolume = pf.h.grids[i]['dx'] * pf.h.grids[i]['dy'] * pf.h.grids[i]['dz']
            for j in range(int(pf.h.grid_particle_count[i])):
            #if NPart%100000 == 0:
            #print '*'
            mass.append(pf.h.grids[i]['particle_mass'][j] * PixelVolume)	
            x.append(pf.h.grids[i]['particle_position_x'][j])
            y.append(pf.h.grids[i]['particle_position_y'][j])
            z.append(pf.h.grids[i]['particle_position_z'][j])
            NPart = NPart + 1

    Masses=np.zeros([2])
    NumPart=np.zeros([2])
    Centroids=np.zeros([2,3])
    # Finding the 2 different masses
    Masses[0] = mass[0]
    #print 'Mass of particle type 0  = %.4f'%Masses[0]
    while (Masses[1] == 0):
        pc = int(NPart * rand()) # Keep randomly choosing particles until I have found 2 different masses
        #print 'Trying another mass = %.3f'%mass[pc]
        if abs(mass[pc] - Masses[0]) > 1E-8:
            Masses[1] = mass[pc]
			#print 'Mass of particle type 1  = %.4f'%Masses[1]
    for n in range(NPart): # n cycles through the number of particles of this type
    #if n%100000 == 0:
    #print '*'
        if abs(mass[n] - Masses[0]) < 1E-8:	
            NumPart[0] = NumPart[0] + 1
            Centroids[0,0] = Centroids[0,0] + x[n]
            Centroids[0,1] = Centroids[0,1] + y[n]
            Centroids[0,2] = Centroids[0,2] + z[n]
        else:	
            NumPart[1] = NumPart[1] + 1
            Centroids[1,0] = Centroids[1,0] + x[n]
            Centroids[1,1] = Centroids[1,1] + y[n]
            Centroids[1,2] = Centroids[1,2] + z[n]

    for k in range(2): # k denotes the bullet or main particles
        if NumPart[0] > NumPart[1]: # Swap everything to put the bullet particles first
            TempNum = NumPart[0]
            NumPart[0] = NumPart[1]
            NumPart[1] = TempNum
            TempMass = Masses[0]
            Masses[0] = Masses[1]
            Masses[1] = TempMass
            for m in range(3):
                TempCentroid = Centroids[0,m]
                Centroids[0,m] = Centroids[1,m]
                Centroids[1,m] = TempCentroid

        for m in range(3):
            Centroids[k,m] = Centroids[k,m] / NumPart[k]

    elapsed = (time.time()-start)
    print "Elapsed time to locate centroids = "+str(elapsed)

    return [NumPart, Masses, Centroids]

def ProjectEnzoData(pf,mass,phi=0.0,theta=0.0,psi=0.0,zmin=-3000.0,zmax=3000.0,DMProject=False):
    # Projects 3D Enzo data onto a 2D grid
    # Returns DM mass, Baryon mass, Three Xray intensities, and SZ data.
    # DMProject = True runs a separate DM Projection.
    # DMProject = False uses the yt raycasting.

    #print 'PATH = ',os.environ['PATH']
    #print 'PYTHONPATH = ',os.environ['PYTHONPATH']
    #print 'LD_LIBRARY_PATH = ',os.environ['LD_LIBRARY_PATH']
    #envir = Popen('env',shell=True, stdout = PIPE).communicate()[0].split('\n')
    #print envir

    DM = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny) 
    if DMProject:
	print "Doing Separate DM Projection"
    	DM = ProjectEnzoDM(pf,mass,1,zmin,zmax,-psi,-theta,-phi)
    else:
	print "Skipping Separate DM Projection"
    start = time.time()
    xpixels=mass.nx
    ypixels=mass.ny
    PixelArea = mass.dx * mass.dy
    # Mass density will go in the input 2d array. 
    # Will create additional 2d arrays for the Xray and SZ data
    Xray1=Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
    Xray2=Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
    Xray3=Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
    SZ=Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)

    add_field('XRay1', function = _EnzoXRay1, units=r"\rm{cm}^{-3}\rm{s}^{-1}", projected_units=r"\rm{cm}^{-2}\rm{s}^{-1}", validators=[ValidateParameter('ApecData')], take_log=False)
    add_field('XRay2', function = _EnzoXRay2, units=r"\rm{cm}^{-3}\rm{s}^{-1}", projected_units=r"\rm{cm}^{-2}\rm{s}^{-1}", validators=[ValidateParameter('ApecData')], take_log=False)
    add_field('XRay3', function = _EnzoXRay3, units=r"\rm{cm}^{-3}\rm{s}^{-1}", projected_units=r"\rm{cm}^{-2}\rm{s}^{-1}", validators=[ValidateParameter('ApecData')], take_log=False)
    add_field('SZ', function =   _EnzoSZ, units =r"\rm{K}\rm{cm}^{-1}" , projected_units=r"\rm{K}",take_log=False)

    pf.field_info['Density'].take_log=False
    pf.field_info['XRay1'].take_log=False
    pf.field_info['XRay2'].take_log=False
    pf.field_info['XRay3'].take_log=False
    pf.field_info['SZ'].take_log=False
    pf.field_info['Dark_Matter_Density'].take_log=False

    center = [(mass.xmin+mass.xmax)/2.0,(mass.ymin+mass.ymax)/2.0,(zmin+zmax)/2.0] # Data Center
    normal_vector=(0.0,0.0,1.0)
    north_vector = (0.0,1.0,0.0)

    R = EulerAngles(phi,theta,psi)
    normal_vector = np.dot(R,normal_vector)
    north_vector = np.dot(R,north_vector)
    width = (mass.xmax - mass.xmin, mass.ymax - mass.ymin, zmax - zmin)
    resolution = (mass.nx,mass.ny)

    MassFactor = BulletConstants.cm_per_kpc**2 * PixelArea / (BulletConstants.g_per_Msun * 1E10)
    XFactor    = BulletConstants.cm_per_kpc**2 * PixelArea * BulletConstants.AreaFactor

    if not DMProject:
	print "Using yt Ray casting for DM"
        projcam=ProjectionCamera(center,normal_vector,width,resolution,"Dark_Matter_Density",north_vector=north_vector,pf=pf, interpolated=True)
        DM.data =  projcam.snapshot()[:,:,0] * MassFactor 

    projcam=ProjectionCamera(center,normal_vector,width,resolution,"Density",north_vector=north_vector,pf=pf, interpolated=True)
    mass.data =  projcam.snapshot()[:,:] * MassFactor
    projcam=ProjectionCamera(center,normal_vector,width,resolution,"XRay1",north_vector=north_vector,pf=pf, interpolated=True)
    Xray1.data = projcam.snapshot()[:,:] * XFactor
    projcam=ProjectionCamera(center,normal_vector,width,resolution,"XRay2",north_vector=north_vector,pf=pf, interpolated=True)
    Xray2.data = projcam.snapshot()[:,:] * XFactor
    projcam=ProjectionCamera(center,normal_vector,width,resolution,"XRay3",north_vector=north_vector,pf=pf, interpolated=True)
    Xray3.data = projcam.snapshot()[:,:] * XFactor
    projcam=ProjectionCamera(center,normal_vector,width,resolution,"SZ",north_vector=north_vector,pf=pf, interpolated=True)
    SZ.data   = projcam.snapshot()[:,:]
    elapsed = (time.time()-start)
    print "Elapsed time to run Enzo gas projection = "+str(elapsed)

    return [DM, mass,Xray1,Xray2,Xray3,SZ]