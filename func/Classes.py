from ctypes import *
import numpy as np

class Array2d:
    def __init__(self,xmin,xmax,nx,ymin,ymax,ny):
        self.nx=nx
        self.ny=ny

        self.xmin=xmin
        self.ymin=ymin
        
        self.xmax=xmax
        self.ymax=ymax
        
        self.dx=(xmax-xmin)/nx
        self.dy=(ymax-ymin)/ny
        
        self.x=np.linspace(xmin+self.dx/2,xmax-self.dx/2,nx)
        self.y=np.linspace(ymin+self.dy/2,ymax-self.dy/2,ny)

        self.data=np.zeros([nx,ny])


class CArray2d(Structure):
    # This is for calling c subroutines
    _fields_ = [("nx", c_int),("ny", c_int),("xmin",c_double),("xmax",c_double),("ymin",c_double),("ymax",c_double),("dx",c_double),("dy",c_double),("x",POINTER(c_double)),("y",POINTER(c_double)),("data",POINTER(c_double))]

    def __init__(self,array):
    # Copies from Python to c
        self.nx=array.nx
        self.ny=array.ny
        self.dx=array.dx
        self.dy=array.dy
        self.xmin=array.xmin
        self.xmax=array.xmax
        self.ymin=array.ymin
        self.ymax=array.ymax
        size=array.nx*array.ny  
        self.data=(c_double*size)()
        self.x=(c_double*array.nx)()
        self.y=(c_double*array.ny)()

        for j in range(array.ny):
            self.y[j]=array.y[j]

        for i in range(array.nx):
            self.x[i]=array.x[i]
            for j in range(array.ny):
                self.data[i+j*array.nx]=array.data[i,j]

class ArraySet(Structure):
    # This carries a set of c arrays for shipping to the alignment subroutines
    _fields_ = [("numarrays", c_int), ("data1",POINTER(CArray2d)), ("data2",POINTER(CArray2d)), ("shifteddata1",POINTER(CArray2d)), ("sigma",POINTER(CArray2d)), ("mask",POINTER(CArray2d))]
    def __init__(self,numarrays,data1list, data2list, shifteddata1list, sigmalist, masklist):
        self.numarrays=numarrays
        self.data1=(CArray2d*numarrays)()
        self.data2=(CArray2d*numarrays)()
        self.shifteddata1=(CArray2d*numarrays)()
        self.sigma=(CArray2d*numarrays)()
        self.mask=(CArray2d*numarrays)()
        for i in range(numarrays):
            self.data1[i]=CArray2d(data1list[i])
            self.data2[i]=CArray2d(data2list[i])
            self.shifteddata1[i]=CArray2d(shifteddata1list[i])
            self.sigma[i]=CArray2d(sigmalist[i])
            self.mask[i]=CArray2d(masklist[i])

class Align:
    def __init__(self):
        self.d=np.zeros([5])
        self.dmin=np.zeros([5])
        self.dmax=np.zeros([5])

class CAlign(Structure):
    # This is for calling c subroutines
    _fields_ = [("d", c_double*5),("dmin", c_double*5),("dmax", c_double*5)]
    def __init__(self,align):
    # Copies from Python to c
        for i in range(5):
            self.d[i]=align.d[i]
            self.dmin[i]=align.dmin[i]
            self.dmax[i]=align.dmax[i]