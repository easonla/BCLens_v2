import time
import numpy as np
import math
import sys
import pickle
from scipy.ndimage import gaussian_filter, convolve
from scipy.interpolate import griddata

from Classes import Array2d
from GetData import ReadLookups
from FetchEnzo import ProjectEnzoData
from MonteCarloFom_Origin import SetAlign


class optfunc:
	#function for optimizing
	def __init__(self, data1list, data2list, shifteddata1list, sigmalist, masklist, align):
		self.images_align=XIA(data1list, data2list, shifteddata1list, sigmalist, masklist)
		self.align = align
		self.EPS = 0 
		self.evals = 0
		self.f = 0
	def find_new_fom(self,x):
		self.images_align.xfom(x)
		self.lens.chi_sq
		self.f = self.images_align.f + self.lens.chi_sq
		self.evals += 1

class XIA:
	#Xray images alignment class
	def __init__(self, data1list, data2list, shifteddata1list, sigmalist, masklist):
		self.data1list = data1list
		self.data2list = data2list
		self.shifteddata1list = shifteddata1list
		self.sigmalist = sigmalist
		self.masklist  = masklist
		self.n     = len(data1list)
		self.fom   = np.zeros(len(data1list))
		self.f = 0

	def xfom(self,x):
		# This sets the alignment parameters
		# Components 0 and 1 are the x and y aligment offsets.
		# Components 2 and 3 are a shift between the Mass dataset and the others
		# MaxShift gives the max allowed shift in kpc
		# Component 4 is the angular rotation in radians
		self.f = 0
		dx, dy, phi = x[0],x[1],x[4]
		shifteddata = self.shifteddata2list[0]
		n = shifteddata.nx * shifteddata.ny
		[dyy,dxx] = np.meshgrid(shifteddata.y,shifteddata.x)
		pos = np.append(dxx.reshape(n,1),dyy.reshape(n,1),axis=1)
		new_pos = change_coord(pos,dx,dy,phi)

		for k in range(self.n):
			mask = self.masklist[k]
			if np.sum(mask.data) == 0 : pass
			else : 
				data1 = self.data1list[k] ##simulation data, larger
				data2 = self.data2list[k]
				shifteddata1 = self.shifteddata1list[k]
				sigma = self.sigmalist[k]
				data = np.array(data1.data.reshape(n,1))
				new_data = griddata(new_pos,data,pos,method='linear',fill_value=0) ## time bottleneck,each eval take 0.16s
				## for faster interpolation method, see
				## https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
				shifteddata1.data = new_data.reshape(shifteddata1.nx,shifteddata1.ny)
				self.fom[k] = self._fom_cal(data2,shifteddata1,sigma,mask)
				self.f += self.fom[k]

	def _fom_cal(self,data1,data2,sigma,mask):
		d1 = data1.data 
		d2 = data2.data
		chisq = np.sum(np.power((d1-d2)/sigma.data,2) * mask.data) / np.sum(mask.data)
		return chisq

class Lens:
    def __init__(self,mass,galaxies,shear):
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
        sefl.grid_arcsec = list()
        self._findvalue_by_potential(mass)
        self.galaxies = galaxies
        self.shear = shear
        self.chisq = 0

    def chi_sq(self,x,w):
    	self.chisq = _chi_strong(x) + w * _chi_weak(x)
    def _chi_strong(self,x):
    	sigma = 0.6
		dx,dy,phi = x0[0],x0[1],x0[2]
		pos = _get_pos(self.galaxies)
		zf = _get_zf(self.galaxies)
		new_pos = change_coord(pos,dx,dy,phi)
		data = [self.alphaX,self.alphaY,self.mag]
		[shifted_ax,shifted_ay,shifted_mag] = _shift(data,x,new_pos)
		source = pos + [zf * shifted_ax, zf * shifted_ay]
		return _get_chi_strong(source,mag)
	def _get_chi_strong(self,source,mag):
		k = 0
		for key in galaxies:
			gal.clean()
			gal = galaxies[key]
			for i in range(gal.n):
				gal.xs.append(source[0,k])
				gal.ys.append(source[1,k])
				gal.mag.append(mag[k])
				k += 1
			chi += sum(gal.update_chi(sigma))/gal.n
		return chi 
	def _chi_weak(self,x):
		sigma = 1.
		sigma_eps_s = 0.2
		sigma_eps_er = 0.1
		dx,dy,phi = x0[0]/4.413,x0[1]/4.413,x0[2]/4.413
		pos = np.array([np.array(sheardata['ra']),np.array(sheardata['dec'])])
		new_pos = change_coord(pos,dx,dy,phi)
		data = [self.gamma1,self.gamma2]
		[shifted_g1, shifted_g2] = _shift(data,x,new_pos)
		Zf = np.array(sheardata['Zf'])
		g1 = np.array(sheardata['g_final[0]'])
		g2 = np.array(sheardata['g_final[0]'])
		sigma = (1 - (shifted_g1**2 + shifted_g2**2))**2*sigma_eps_s+sigma_eps_er
		chi_sq = np.sum( (g1 - shifted_g1)**2 + (g2 - shifted_g2)**2)/sigma
		return chi_sq
	def _get_pos(self,galaxies):
		pos = []
		for key in galaxies:
			gal = galaxies[key]
			for i in range(gal.n):
				pos.append([gal.x0,gal.y0])
		return np.array(zf) #list of [(x1,y1)...] in arcsec
	def _get_zf(self,zf):
		zf = []
		for key in galaxies:
			gal = galaxies[key]
			for i in range(gal.n):
				zf.append(gal.zf)
		return np.array(zf) 

	def _shift(self,data,x,new_pos):
		for data in dataset:
			data_roll = np.array(data.data.reshape(data.nx*data.ny,1))
			new_data = griddata(new_pos,data_roll,self.grid_arcsec,method='linear',fill_value=0)
			shifteddataset.append(new_data)
		return shifteddataset #simulation quantities at position

    def _findvalue_by_potential(self,mass):
        # faster ~47s for 220x220
        sigmacr = 0.3963
        ds = mass.dx * mass.dy
        MassFactor = BulletConstants.cm_per_kpc**2 * ds / (BulletConstants.g_per_Msun * 1E10)
        _kappa = mass.data/sigmacr/MassFactor
        
        #grid position for later operation in arcsec
        [y_grid,x_grid] = np.meshgrid(mass.y,mass.x)
        x_grid = x_grid/4.413
        y_grid = y_grid/4.413
        self.grid_arcsec = np.append(self.x_grid.reshape(mass.nx*mass.ny,1),self.y_grid.reshape(mass.nx*mass.ny,1),axis=1)
        

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

def change_coord(pos,dx,dy,phi):
    c,s=np.cos(phi),np.sin(phi)
    R = np.array([[c,-s],[s,c]])
    new_pos = np.dot(R,pos.T).T + np.array([[dx,dy]])
    return new_pos

def findbestshift(data1list, data2list, shifteddata1list,sigmalist, masklist, align, tol, method = "MCMC"):
	#initializing target function for optimizing
	kmax = 50
	tol = 10E-7
	func = optfunc(data1list, data2list, shifteddata1list, sigmalist, masklist, align)
	if method == "MCMC":
		func = mcmc_optimizer(func,kmax,tol)
	elif method == "Gradient": func = grad_optimizer(func,tol)
	return func


def grad_optimizer(func,tol):



	return func

'''
def minimize(Lens,galaxies,sheardata,w):
    start = time.time()
    xbcg,ybcg=findhalocenter(Lens._kappa)
    x0 = np.array([xbcg,ybcg,0])
    #eps = np.array([Lens.p2a/200,Lens.p2a/200,1e-8])
    try:
        xopt = fminp(chi_sq,x0,(Lens,galaxies,sheardata,w),ftol=0.0001,full_output=1)
        print time.time()-start,'sec'
        return xopt
    except:
        print "undable to minimize chi_sq"
        print  "sys.exc_info=",sys.exc_info(),"\n"
        traceback.print_exc()
        return 0
'''


def mcmc_optimizer(func,kmax,tol):
	#total fom = gravitational lensing + X-ray1 + (X-ray2 + X-ray3 + SZ)
	start = time.time()
	align = func.align
	p = align.d[:]
	pmin = align.dmin[:]
	pmax = align.dmax[:]
	bestp = np.zeros([5])
	radius = np.zeros([5])
	#Temperature
	T0 = 1.0
	#Initial Point
	func.find_new_fom(p)
	f = func.f
	bestf = f
	## MCMC efficiency test
	f_evals = []
	f_evals.append(f)
	accept = 0 

	while func.evals < kmax:
		T = T0 * (kmax - func.evals) / kmax # Decrease the temperature as evaluation proceeds
		radius [:] = (pmax[:] - pmin[:]) * (kmax - func.evals) / kmax
		newp = newpoint(p,pmin,pmax,radius)
		print radius
		func.find_new_fom(newp)
		newf = func.f
		randy = np.random.random()
		pr = prob(f,newf,T)
		#print "Eval No.{:f} T = {:f}  pr = {:f} randy = {:f} f={:f}".format(func.evals,T,pr,randy,newf) 
		if pr > randy:
			f = newf
			f_evals.append(f)
			accept += 1
			p[:] = newp[:]
		if np.abs(f-bestf)<tol:break
		elif f < bestf :
			bestf = f
			bestp[:]=p[:]
		func.align.d[:]=bestp[:]
		fom = bestf
	elapsed = (time.time()-start)
	print "Elapsed time to find optimal alignment = "+str(elapsed)+"\n"
	accept = float(accept)/kmax
	return func, f_evals, accept

def newpoint(x,xmin,xmax,r):
	newx = np.zeros([5])
	newx[:] = xmin[:] - 0.01 
	for i in range(5):
		while (newx[i] < xmin[i]) and (newx[i] > xmax[i]):
			randy = np.random.random()
			newx[i] = (x[i] - r[i]) + 2 * r[i] * randy
	return newx

def prob(f, newf, T):
	return np.exp((f-newf/T))


