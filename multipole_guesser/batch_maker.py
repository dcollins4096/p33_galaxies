
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
import torch

Ntheta_phi = 1000
Nsph = 3000
N_ell = 2
Nzones = 32
L_max = 2
center = np.array([Nzones//2,Nzones//2,Nzones//2])
rho = np.ones([Nzones]*3)
fname = 'clm_take6.h5'


N_ell_em = ((np.arange(N_ell)+1)*2+1).sum()

RM_all=np.zeros([Nsph,Ntheta_phi])

np.random.seed(8675309)
theta_all = np.zeros([Nsph,Ntheta_phi])
phi_all = np.zeros([Nsph,Ntheta_phi])

this_theta = np.random.random(Ntheta_phi)*np.pi
this_phi   = np.random.random(Ntheta_phi)*np.pi*2

t0 = time.time()
for nnn in np.arange(Nsph):


    this_clm = np.random.random(N_ell_em)
    count=0
    Clmd = {}

    for ell in np.arange(N_ell)+1:
        for em in np.arange(-ell,ell+1):
            Clmd[ (ell,em)]=this_clm[count]
            count+=1

    X, Y, Z, Bx, By, Bz = make_multipole.multipole_B_field(Nzones, L_max, Clmd, grid_extent=1.0)


    if 1:
        counter=0
        this_rm = np.zeros(Ntheta_phi)
        for theta, phi in zip(this_theta, this_phi):
            dx = 1/Nzones
            vox, lengths = voxelpuller.voxels_and_path_lengths2(center, theta, phi, Bx.shape)
            lengths *= dx
            nhat = voxelpuller.direction_from_angles(theta,phi)
            this_rm[counter] = ((Bx[vox[:,0],vox[:,1],vox[:,2]]*nhat[0]+
                                 By[vox[:,0],vox[:,1],vox[:,2]]*nhat[1]+
                                 Bz[vox[:,0],vox[:,1],vox[:,2]]*nhat[2])*lengths*rho[vox[:,0],vox[:,1],vox[:,2]]).sum()
            counter+=1


    stuff={'Clm':this_clm,'Rm':this_rm, 'theta':this_theta,'phi':this_phi}
    if not os.path.exists(fname):
        # Create file and dataset with unlimited rows
        with h5py.File(fname, "w") as f:
            for setname in ['Clm','Rm','theta','phi']:
                size = {'Clm':N_ell_em}.get(setname,Ntheta_phi)
                dset = f.create_dataset(
                    setname,
                    shape = (0,size),
                    maxshape=(None, size),  # allow unlimited rows
                    chunks=(1, size),       # chunking needed for resizing
                    dtype='float64'
                )

    with h5py.File(fname, "r+") as f:
        for setname in ['Clm','Rm','theta','phi']:
            size = {'Clm':N_ell_em}.get(setname,Ntheta_phi)
            arr = stuff[setname]
            dset = f[setname]
            dset.resize((dset.shape[0] + 1, size))   # increase row count by 1
            dset[-1, :] = arr    

    tnow = time.time()
    telap = tnow-t0
    t_per = telap/(nnn+1)
    tleft = (Nsph-nnn)*t_per
    eta = tnow+tleft
    etab = datetime.datetime.fromtimestamp(eta)
    now = datetime.datetime.fromtimestamp(tnow)
    maybe_tomorrow = ""
    if etab.day != now.day:
        maybe_tomorrow = " %04d-%02d-%02d "%(etab.year,etab.month,etab.day)
    eta = "%s%0.2d:%0.2d:%0.2d"%(maybe_tomorrow,etab.hour, etab.minute, int(etab.second))
    print('SPHERE %d/%d eta %s'%(nnn,Nsph,eta))

