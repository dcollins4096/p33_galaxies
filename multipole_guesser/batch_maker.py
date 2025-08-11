
from importlib import reload
import matplotlib.pyplot as plt
import plot_multipole
reload(plot_multipole)
import make_multipole
import pdb
import os
import voxelpuller
reload(voxelpuller)
import numpy as np
import h5py
import time
import datetime

Ntheta_phi = 5000
Nsph = 1000
N_ell = 3
Nzones = 32
L_max = 2
center = np.array([Nzones//2,Nzones//2,Nzones//2])
rho = np.ones([Nzones]*3)
fname = 'clm_take2.h5'


N_ell_em = ((np.arange(N_ell)+1)*2+1).sum()

Clm_all=np.zeros([Nsph,N_ell_em])
RM_all=np.zeros([Nsph,Ntheta_phi])

np.random.seed(8675309)
theta_all = np.zeros([Nsph,Ntheta_phi])
phi_all = np.zeros([Nsph,Ntheta_phi])

t0 = time.time()
for nnn in np.arange(Nsph):


    this_theta = np.random.random(Ntheta_phi)*np.pi
    this_phi   = np.random.random(Ntheta_phi)*np.pi*2
    theta_all[nnn,:]=this_theta
    phi_all[nnn,:]=this_phi
    Clm_all[nnn,:]= np.random.random(N_ell_em)
    #Clm_all[nnn,5]=1
    count=0
    Clmd = {}
    for ell in np.arange(N_ell)+1:
        for em in np.arange(-ell,ell+1):
            Clmd[ (ell,em)]=Clm_all[nnn,count]
            count+=1
    X, Y, Z, Bx, By, Bz = make_multipole.multipole_B_field(Nzones, L_max, Clmd, grid_extent=1.0)

    
    if 1:
        counter=0
        for theta, phi in zip(theta_all[nnn],phi_all[nnn]):
            dx = 1/Nzones
            vox, lengths = voxelpuller.voxels_and_path_lengths2(center, theta, phi, Bx.shape)
            lengths *= dx
            nhat = voxelpuller.direction_from_angles(theta,phi)
            RM_all[nnn,counter] = ((Bx[vox[:,0],vox[:,1],vox[:,2]]*nhat[0]+
                                    By[vox[:,0],vox[:,1],vox[:,2]]*nhat[1]+
                                    Bz[vox[:,0],vox[:,1],vox[:,2]]*nhat[2])*lengths*rho[vox[:,0],vox[:,1],vox[:,2]]).sum()
            counter+=1
    #print( "saving",nnn)
    #plot_multipole.plot_stream_and_rm(X,Y,Z,Bx,By,Bz,theta_all[nnn,:],phi_all[nnn,:],RM_all[nnn,:], fname='sph_%04d'%nnn)
    tnow = time.time()
    telap = tnow-t0
    t_per = telap/(nnn+1)
    tleft = (Nsph-nnn)*t_per
    eta = tnow+tleft
    etab = datetime.datetime.fromtimestamp(eta)
    eta = "%0.2d:%0.2d:%0.2d"%(etab.hour, etab.minute, int(etab.second))
    print('SPHERE %d/%d eta %s'%(nnn,Nsph,eta))

fptr=h5py.File(fname,'w')
fptr['Clm']=Clm_all
fptr['Rm']=RM_all
fptr['theta']=theta_all
fptr['phi']=phi_all
fptr.close()
