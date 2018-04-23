import numpy as np


def euler_angles(phi=0.0, theta=0.0, psi=0.0):
    R=np.zeros([3,3])
    R[0,0]=np.cos(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.sin(psi)
    R[0,1]=np.cos(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.sin(psi)
    R[0,2]=np.sin(psi)*np.sin(theta)
    R[1,0]=-np.sin(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.cos(psi)
    R[1,1]=-np.sin(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.cos(psi)
    R[1,2]=np.cos(psi)*np.sin(theta)
    R[2,0]=np.sin(theta)*np.sin(phi)
    R[2,1]=-np.sin(theta)*np.cos(phi)
    R[2,2]=np.cos(theta)
    return R


def change_coord(pos, dx, dy, phi):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    new_pos = np.dot(R, pos.T).T + np.array([[dx, dy]])
    return new_pos