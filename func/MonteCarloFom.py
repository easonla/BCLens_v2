import numpy as np
import math
from scipy.ndimage import gaussian_filter, convolve

from classed import Array2d
from GetDate import ReadLookups
from FetchEnzo import ProjectEnzoData, 

class Func:
    def __init__(self,data1list, data2list, shifteddata1list,sigmalist, masklist, align):
        self.data1 = data1list
        self.data2 = data2list
        self.shifteddata = shifteddata1list
        self.sigma = sigmalist
        self.mask  = masklist
        self.align = align
        self.n     = len(data1list)
        self.EPS = 0 
        self.evals = 0
        self.f = 0

    def operator():
        for i in range(self.n): self.align.d[i] = x[i]
        self.evals += 1 
        self.f = _shift()

    def _shift():
        fom = np.zeros([self.n])
        for k in range(self.n):
            for i in range(self.data2[k].nx):
                x = self.data2[k].nx[i]
                for j in range(self.data2[k].ny):
                    y = self.data2[k].ny[j]
                    xprime = x*np.cos(self.align.d[4])+y*np.sin(self.align.d[4])+self.align.d[0]
                    yprime = x*np.sin(self.align.d[4])+y*np.cos(self.align.d[4])+self.align.d[1]
                    if k==0: ## Shift Mass data relative to Other
                        xprime += self.align.d[2]
                        yprime += self.align.d[3]
                    self.shifteddata[k].data[i+j*self.data2[k].nx]=_interp(self.data1[k],xprime,yprime)


    def _interp(data,xprime,yprime):
        d = 0
        i = int(math.floor(xprime -  data.xmin)/data.dx)
        j = int(math.floor(yprime -  data.ymin)/data.dy)
        for m in range(i-1,i+2):
            deltax = math.fabs((xprime - data.x[m])/data.dx)
            for n in range(j-1,j+2):
                deltay = math.fabs((yprime - data.y[n])/data.dy)
                d = d + kernel(deltax,deltay) * data.data[m+n*data.nx]
        return d

    def _kernel(deltax,deltay):
        if (deltax >= 1.) or (deltay >= 1.):
            return 0
        else: return (1.0-deltax)*(1.0-deltay)

def newpoint(x,xmin,xmax,r):
    newx[:] = xmin[:] - 0.01 
    for i in range(5):
        while (newx[i] < xmin[i]) and (newx[i] > xmax[i]):
            randy = rand()
            newx[i] = (x[i] - r[i]) + 2 * r[i] * randy
    return newx

def prob(f, newf, T):
    return np.exp((f-newf/T))


def SimpleFom(pf,data,phi=0.0,theta=0.0,psi=0.0,ConstrainPhi=True,Mask=(1,0,0,0,0,0,0),Z=1.0, TFudge=1.0, SpectralIndex=3.8, MaxShift=0.0):
    (dataA,sigmaA,maskA,maskANull,dataB1,sigmaB1,dataB2,sigmaB2,dataB3,sigmaB3,dataC,sigmaC,dataD,sigmaD,maskD,maskDNull,dataE,sigmaE) = data
    mask_sum = maskA.data.sum()
    try:
        dmsim=Array2d(2.0*dataA.xmin,2.0*dataA.xmax,2*dataA.nx,2.0*dataA.ymin,2.0*dataA.ymax,2*dataA.ny)
        masssim=Array2d(2.0*dataA.xmin,2.0*dataA.xmax,2*dataA.nx,2.0*dataA.ymin,2.0*dataA.ymax,2*dataA.ny)
        sumsim=Array2d(2.0*dataA.xmin,2.0*dataA.xmax,2*dataA.nx,2.0*dataA.ymin,2.0*dataA.ymax,2*dataA.ny)
        ApecData = ReadLookups(Z) # Reads the APEC lookup tables.
        for grid in pf.h.grids:
            grid.set_field_parameter('ApecData',ApecData)
            grid.set_field_parameter('TFudge',TFudge)
            grid.set_field_parameter('SpectralIndex',SpectralIndex)
        DMProject = False
        [dmsim,masssim,xraysim1,xraysim2,xraysim3,szsim]=ProjectEnzoData(pf,masssim,phi=phi,theta=theta,psi=psi,DMProject=DMProject)
        sumsim.data=gaussian_filter(dmsim.data+masssim.data,2.0)
        align = SetAlign(dataB1, phi, theta, psi, ConstrainPhi=ConstrainPhi, MaxShift=MaxShift)

        # The 0.22 is the approximate alignment of the bullet cluster on the sky. 
        tol=1.0E-7 # Alignment tolerance
        data1list=list((sumsim,xraysim1))
        data2list=list((dataA,dataB1))
        sigmalist=list((sigmaA,sigmaB1))
        if Mask[1] == 1:
            masklist=list((maskA,maskA))
        else:
            masklist=list((maskA,maskANull))

        [shifteddata1list,align,fom]=FindBestShift(data1list, data2list, sigmalist, masklist, align, tol)
        phi = align.d[4]
        shiftedxsim1=shifteddata1list[1]
        xray1chisquared = pow((dataB1.data-shiftedxsim1.data)/sigmaB1.data,2) * maskA.data
        xfom = xray1chisquared.sum() / mask_sum
        return (fom, xfom, phi)
    except:
        print "Error in SimpleFom routine",sys.exc_info()[0]
        return (1.0E5, 1.0E5, 0.0) # If there's an error, give it a large FOM


def FindBestShift_Python(data1list,data2list,sigmalist,masklist,align,tol):
    # This routine finds the best alignment between n sets of 2D
    # arrays.  data1 is a list of the larger arrays, and data2 is a list of the smaller,
    # unshifted arrays. sigma is a list of the data standard deviation

    start = time.time()

    numarrays = len(data1list)
    shifteddata1list=list()

	for i in range(numarrays):
		shifteddata1list.append(Array2d(data2list[i].xmin,data2list[i].xmax,data2list[i].nx,data2list[i].ymin,data2list[i].ymax,data2list[i].ny))

	diff=0
	gtol=tol
    func = Func(data1list, data2list, shifteddata1list, sigmalist, masklist, align)
    p = align.d[:]
    pmin = align.dmin[:]
    pmax = align.dmax[:]
    radius = np.zeros([5])
    kmax = 10000
    #Temperature
    T0 = 1.0
    while func.evals < kmax:
        T = T0 * (kmax - Func.evals) / kmax # Decrease the temperature as evaluation proceeds
        radius [:] = (pmax[:] - pmin[:]) * (kmax - Func.evals) / kmax
        newp = newpoint(p,pmin,pmax,r)
        func.operator()
        newf = func.f
        randy = rand()
        pr = prob(f,newf,T)
        if pr > randy:
            for i in range(5):
                p[i] = newp[i]
        if f < bestf :
            bestf = f
            for i in range(5):
                bestp[i]=p[i]
        for i in range(5):
            func.align.d = bestp[i]
        fom = bestf

    elapsed = (time.time()-start)
    print "Elapsed time to find optimal alignment = "+str(elapsed)+"\n"
    return [shifteddata1list,align,fom]

