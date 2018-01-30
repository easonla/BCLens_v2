import numpy as np
import Config
import BulletConstants
from Classes import Array2d
from scipy.interpolate import griddata
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.cosmology as cosmo
Cosmo = cosmo.LambdaCDM(73.,0.270,0.7299)

class Lens:
    def __init__(self,mass):
        self.mass = mass
        self.kappa = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self.potiential = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self.alpha = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self.alphaX = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self.alphaY = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self.gamma = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self.gamma1 = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self.gamma2 = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self.mag = Array2d(mass.xmin,mass.xmax,mass.nx,mass.ymin,mass.ymax,mass.ny)
        self._findvalue_by_potential(mass)
    def _findvalue_by_potential(self,mass):
        # faster ~47s for 220x220
        sigmacr = BulletConstants.CriticalSurfaceDensity
        ds = mass.dx * mass.dy
        MassFactor = BulletConstants.cm_per_kpc**2 * ds / (BulletConstants.g_per_Msun * 1E10)
        _kappa = mass.data/sigmacr/MassFactor
        [y_grid,x_grid] = np.meshgrid(mass.y,mass.x)
        x_grid = x_grid/4.413
        y_grid = y_grid/4.413
        _potential = np.zeros((mass.nx,mass.ny))
        _gamma1 = np.zeros((mass.nx,mass.ny))
        _gamma2 = np.zeros((mass.nx,mass.ny))
        for i in range(mass.nx):
            for j in range(mass.ny):
                dx = x_grid[i,j] - x_grid
                dy = y_grid[i,j] - y_grid
                _potential[i,j] = _kappa[i,j]*np.ma.log(np.sqrt(dx*dx+dy*dy)).sum()
        _potential = _potential/np.pi
        _alphaX, _alphaY = np.gradient(_potential) #unit = "
        _alpha = np.sqrt(_alphaX**2 + _alphaY**2)
        domi = 1.
        k = self._extend_matrix(_potential)
        _gamma1[:,:] = 0.5*((k[2:,1:-1]-2.*k[1:-1,1:-1]+k[0:-2,1:-1])/domi**2 - (k[1:-1,2:]-2.*k[1:-1,1:-1]+k[1:-1,0:-2])/domi**2)
        _gamma2[:,:] =(k[2:,2:]-k[2:,0:-2]-k[0:-2,2:]+k[0:-2,0:-2])/4./domi**2
        _gamma = np.sqrt(_gamma1**2 + _gamma2**2)
        _mag = (1.-_kappa)**2 - _gamma**2
        self.kappa.data = _kappa
        self.potiential.data = _potential
        self.alpha.data = _alpha
        self.alphaX.data = _alphaX
        self.alphaY.data = _alphaY
        self.gamma.data = _gamma
        self.gamma1.data = _gamma1
        self.gamma2.data = _gamma2
        self.mag.data = _mag
    def _findvalue_by_mass(self,mass):
        # ~70s for 220x220
        sigmacr = 0.3963
        ds = mass.dx * mass.dy
        MassFactor = BulletConstants.cm_per_kpc**2 * ds / (BulletConstants.g_per_Msun * 1E10)
        _kappa = mass.data/sigmacr/MassFactor
        [y_grid,x_grid] = np.meshgrid(mass.y,mass.x)
        #_potiential = np.zeros((mass.nx,mass.ny))
        _alphaX = np.zeros((mass.nx,mass.ny))
        _alphaY = np.zeros((mass.nx,mass.ny))
        _gamma1 = np.zeros((mass.nx,mass.ny))
        _gamma2 = np.zeros((mass.nx,mass.ny))
        for i in range(mass.nx):
            for j in range(mass.ny):
                dx = x_grid[i,j] - x_grid
                dy = y_grid[i,j] - y_grid
                domi = (dy*dy + dx*dx)
                domi_sq = domi * domi
                d1 = (dy*dy - dx*dx)/ domi_sq
                d2 = (-2.*dx*dy)/ domi_sq
                _alphaX[i,j] = np.nansum(_kappa * dx/ domi) #unit
                _alphaY[i,j] = np.nansum(_kappa * dy/ domi)
                _gamma1[i,j] = np.nansum(_kappa*d1)
                _gamma2[i,j] = np.nansum(_kappa*d2)
        _alpha = np.sqrt(_alphaX*_alphaX + _alphaY*_alphaY)
        _gamma = np.sqrt(_gamma1**2 + _gamma2**2)
        _mag = ((1-_kappa)**2-_gamma**2)
        self.kappa.data = _kappa
        self.alpha.data = _alpha*ds/4.413/3600./180.
        self.alphaX.data = _alphaX*ds/4.413/3600./180.
        self.alphaY.data = _alphaY*ds/4.413/3600./180.
        self.gamma.data = _gamma*ds/np.pi
        self.gamma1.data = _gamma1*ds/np.pi
        self.gamma2.data = _gamma2*ds/np.pi
        _mag = ((1-_kappa)**2-_gamma**2)
        self.mag.data = _mag
    def _extend_matrix(self,p):
		nx,ny=np.shape(p)
		k=np.zeros([nx+2,ny+2])
		k[1:-1,1:-1]=p[:,:]
		k[0,1:-1]=p[-1,:]
		k[-1,1:-1]=p[0,:]
		k[1:-1,0]=p[:,-1]
		k[1:-1,-1]=p[:,0]
		return k

class galaxy:
    def __init__(self):
        self.n=0
        #initial image position
        self.x0=[]
        self.y0=[]
        #current image position
        self.xc=[]
        self.yc=[]
        #source position
        self.xs=[]
        self.ys=[]
        #redshift and redshift factor
        self.z=[]
        self.zf=[]
        #magnification
        self.mag=[]
        self.rms=[]
        self.chi=[]
    def add(self,x,y,z):
        self.n=self.n+1
        self.x0.append(x)
        self.y0.append(y)
        self.z.append(z)
        self.zf.append(Zfunc(z))
    def add_source(self,xs,ys):
        self.xs.append(xs)
        self.ys.append(ys)
    def update_image(self,dx,dy,phi):
        "rotation in radian"
        #phi = phi*np.pi/180.
        self.xc = np.array(self.x0)*np.cos(phi)-np.array(self.y0)*np.sin(phi)+dx
        self.yc = np.array(self.x0)*np.sin(phi)+np.array(self.y0)*np.cos(phi)+dy
    def add_mag(self,mag):
        self.mag.append(mag)
    def update_chi(self,sigma):
        x_bar=sum(self.xs)/self.n
        y_bar=sum(self.ys)/self.n
        self.rms = (np.array(self.xs)-x_bar)**2+(np.array(self.ys)-y_bar)**2
        self.chi = (np.array(self.xs)-x_bar)**2+(np.array(self.ys)-y_bar)**2/(sigma**2*np.array(self.mag)**2)
        return self.chi
    def clean(self):
        self.mag = []
        self.xc = []
        self.yc = []
        self.xs = []
        self.ys = []
        self.rms = []
        self.chi = []

def get_galaxies(listname='B09'):
    if listname == 'B09':
        filename = 'B09_offset.txt'
    elif listname == 'P12':
        filename = 'P12_offset.txt'
    filename = Config.toppath + 'data/' + filename
    file = open(filename,'r')
    lines = file.readlines()
    file.close()
    galaxies = {}
    for i in range(len(lines)):
        name=lines[i].strip().split()[0][0]
        galaxies[name]=galaxy()
    for i in range(len(lines)):  
        name=lines[i].strip().split()[0][0]
        x=float(lines[i].strip().split()[1])
        y=float(lines[i].strip().split()[2])
        z=float(lines[i].strip().split()[3])
        galaxies[name].add(x,y,z)
    return galaxies

def get_shear():
	filename = Config.toppath + 'data/shear_offset.dat'
	return ascii.read(filename)

def chi_strong(x0,Lens,galaxies):
    sigma = 0.6
    chi = 0
    dx,dy,phi = x0[0]/4.413,x0[1]/4.413,x0[2]/4.413
    for key in galaxies:
        gal = galaxies[key]
        gal.clean()
        gal.update_image(dx,dy,phi) 
        for i in range(gal.n):
            xh = gal.xc[i]
            yh = gal.yc[i]
            zf = gal.zf[i]
            xi,yi = ray_trace(xh,yh,zf,Lens.alphaX,Lens.alphaY)
            Mi = BiLnrIntrp(xi,yi,Lens.mag)
            gal.add_source(xi,yi)
            gal.add_mag(Mi)
        chi += sum(gal.update_chi(sigma))/gal.n
    return chi

def BiLnrIntrp(x,y,data):
    #Bilinear Interpolation 
    i = find_index(x,data.x/4.413)
    j = find_index(y,data.y/4.413)
    x1 = data.x[i]/4.413
    x2 = data.x[i+1]/4.413
    y1 = data.y[j]/4.413
    y2 = data.y[j+1]/4.413
    f11 = data.data[i,j]
    f12 = data.data[i,j+1]
    f21 = data.data[i+1,j]
    f22 = data.data[i+1,j+1]
    f = (f11*(x2-x)*(y2-y)+f21*(x-x1)*(y2-y)+f12*(x2-x)*(y-y1)+f22*(x-x1)*(y-y1))/((x2-x1)*(y2-y1))
    return f

def find_index(x,a):
    lo = 0
    hi = len(a)
    while lo<hi:
        mid = (lo+hi)//2
        if a[mid]<x:lo=mid+1
        else: hi=mid
    return lo

def ray_trace(x,y,zf,alpha_x,alpha_y):
    ax = BiLnrIntrp(x,y,alpha_x)
    ay = BiLnrIntrp(x,y,alpha_y)
    x0 = x + zf*ax
    y0 = y + zf*ay
    return x0,y0

def chi_weak(x0,Lens,sheardata,take_average=False):
    chi=0
    sigma = 1.
    sigma_eps_s = 0.2
    sigma_eps_er = 0.1
    dx,dy,phi = x0[0],x0[1],x0[2]
    x,y = change_coord(sheardata,dx,dy,phi)
    Zf = np.array(sheardata['Zf'])
    g1 = np.array(sheardata['g_final[0]'])
    g2 = np.array(sheardata['g_final[0]'])
    if take_average:
        g1 = nearby_avg(x,y,g1)
        g2 = nearby_avg(x,y,g2)
    else:
        for i in range(len(x)):
            #gamma_prime = BiLnrIntrp(x_prime,y_prime,g)
            g1_prime = BiLnrIntrp(x[i],y[i],Lens._gamma1)*Zf[i]
            g2_prime = BiLnrIntrp(x[i],y[i],Lens._gamma1)*Zf[i]
            sigma = (1-(g1_prime**2+g2_prime**2))**2*sigma_eps_s**2+sigma_eps_er**2
            chi += ((g1_prime-g1[i])**2+(g2_prime-g2[i])**2)/sigma
    chi = chi/len(x)
    return chi

def change_coord(sheardata,dx,dy,phi):
    pos = np.array([np.array(sheardata['ra']),np.array(sheardata['dec'])])
    c,s=np.cos(phi),np.sin(phi)
    R = np.array([[c,-s],[s,c]])
    new_pos = np.dot(R,pos)
    x = new_pos[0,:] +dx
    y = new_pos[1,:] +dy
    return x,y

def nearby_avg(x,y,g1,g2):
    n0 = 30
    N=30
    g1_avg,g2_avg,x1,y1=[],[],[],[]
    count = 0
    for i in range(len(x)):
        x_prime = x[i]
        y_prime = y[i]
        D = np.sqrt((x-x_prime)**2+(y-y_prime)**2)
        cat = np.argpartition(D, N)
        area = np.pi*D[cat[N-1]]**2/3600.
        n = N/area
        if n>n0 and n<10000:
            g1_avg.append(np.average(g1[cat[:N]]))
            g2_avg.append(np.average(g2[cat[:N]]))
            x1.append(x[i])
            y1.append(y[i])
            count+=1
    return np.array(g1_avg),np.array(g2_avg),np.array(x1),np.array(y1)


def Zfunc(z):
    Dd=Cosmo.angular_diameter_distance(0.296)
    Ds=Cosmo.angular_diameter_distance(z)
    Dds=Cosmo.angular_diameter_distance_z1z2(0.296,z)
    return (Dds)/Ds
