# Craig Lage

# NYU

# 7-Jan-11

#Constants used in the simulations of the Bullet Cluster

import numpy as np

#****************MAIN PROGRAM*****************

# The following are constants used in the subroutines:
Tconv = 4.25E-3 # Convert U into temperature in eV. Basically k*Mean molecular weight*2/3 ; (2/3=gamma-1)
nconv = 1/1.08E-67 # Convert rho into particles/L^3 in internal units - basically mstar in internal units
mp = 1.673E-24 # Proton mass in CGS
mstar = 2.15E-24 # Average ion mass in CGS
cGadget = 3.0E5 # Speed of light in internal units
reGadget = 9.13E-35 # Classical electron radius in internal units
alpha = 1/137.036 # Fine structure constant - dimensionless
c = 2.9978E10 # Speed of light in CGS
re = 2.8179E-13 # Classical electron radius in CGS
me = 511000.0 # Electron mass in eV
esu = 4.8032E-10 # Electron charge in esu
erg_per_eV = 1.602E-12
joule_per_eV = 1.602E-19
redshift = 0.296 # redshift to bullet cluster
Emin = 500.0 * (1.0 + redshift) # Minimum detector energy in eV, increased by redshift factor
Emax = 6000.0 * (1.0 + redshift) # Maximum detector energy in eV, increased by redshift factor
Emin1 = 500.0 * (1.0 + redshift) # Minimum detector energy in eV, increased by redshift factor
Emax1 = 2000.0 * (1.0 + redshift) # Maximum detector energy in eV, increased by redshift factor
Emin2 = 2000.0 * (1.0 + redshift) # Minimum detector energy in eV, increased by redshift factor
Emax2 = 5000.0 * (1.0 + redshift) # Maximum detector energy in eV, increased by redshift factor
Emin3 = 5000.0 * (1.0 + redshift) # Minimum detector energy in eV, increased by redshift factor
Emax3 = 8000.0 * (1.0 + redshift) # Maximum detector energy in eV, increased by redshift factor
Ze2Ne = 1.84 # Average Z^2 for plasma * Ne/Ni factor
NeNi = 1.10 # Ratio of electron to ion density
GauntFactor = 1.4 # To get better fit to APEC plasma emission 
PreFactor = 16/3 * np.sqrt(2*np.pi/3) * alpha * reGadget * reGadget * cGadget  * Ze2Ne * GauntFactor
TimeFactor = 1.0 / 3.09E16 # One second exposure time in internal units
AreaFactor = 1E-4 / (4*np.pi*(1.529E6*3.09E19)**2) # Detector area/(4*pi*R^2) - Ratioing to 1cm2 area unit
EnzoPreFactor = 16/3 * np.sqrt(2*np.pi/3) * alpha * re * re * c  * Ze2Ne * GauntFactor
GammaMinus1 = 0.667
#RadioPreFactor = np.sqrt(3)/(24 * np.pi) * esu**3 / (me * erg_per_eV)**2 # Use for equipartition model
#RadioPreFactor = np.sqrt(3)/(2) * esu**3 / (me * erg_per_eV) / mp # density model
RadioPreFactor = np.sqrt(3 * np.pi)/(32 * np.pi**2) * esu**3 / (me * erg_per_eV)**2
RadioFrequency = 1.3E9 # Frequency of radio measurements
microJanskys_per_CGS = 1E29 # MicroJanskys in CGS
OmegaCPreFactor = 3 * esu * c/ (me * erg_per_eV)
# Have double checked luminosity distance of 1.52 GPc to Bullet Cluster with Hogg et.al.  
# Data after using Chandra ciao expmap correction is in photons/cm2/sec

TCMB = 2.725 # CMB Temperature in K
SZFactor = 8.0/3.0 * np.pi * reGadget * reGadget  * NeNi * nconv / me  * 2 * 0.47 *  TCMB # SZ Pre-factor
EnzoSZFactor = 8.0/3.0 * np.pi * re * re  * NeNi / mstar / me  * 2 * 0.47 *  TCMB # SZ Pre-factor
# The 0.47 is a factor for calculating the brightness temperature at 150 GHz, and needs more work.
DataShift = 5.0 # Maximum allowed data shift between Mass data and other sets in arseconds
AngularScale = 4.413 # Angular scale at Bullet Cluster in kps/"
MassDataScale = 0.00098661 # Mass data scale in degrees/pixel
XRayDataScale = 0.0009864848 # XRay data scale in degrees/pixel
XRayTempDataScale = 0.00109333 # XRay Temp data scale in degrees/pixel

BulletVz = 616 # Motion of bullet away from us in km/sec , as per Barrena, et.al.
BulletSigVz = 80 # Motion of bullet away from us in km/sec , as per Barrena, et.al.

TimeConversion = 0.97777 # Puts time from internal units into Gyears

Nsph = 50 # Number of SPH nearest neighbors
HsmlMax = 1000.0  # Maximum SPH radius

kBoltzmann = 1.38065E-16
DegreesK_to_eV = 1.38065E-23 / 1.6022E-19
DegreesK_to_keV = 1.38065E-23 / 1.6022E-19 * 1.0E-3
cm_per_kpc = 3.0857E21
km_per_kpc = 3.0857E16
sec_per_Gy = 86400 * 365.25 * 1E9
g_per_Msun = 1.9891E33

LogRhoMin = -12.0
LogPressureMin = -3.0
LogXrayMin=-12.0

CriticalSurfaceDensity = 0.3963

#************END MAIN PROGRAM*************************

