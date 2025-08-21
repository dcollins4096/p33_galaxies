
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

Ntheta_phi = 5000
Nsph = 1
N_ell = 4
Nzones = 16
center = np.array([Nzones//2,Nzones//2,Nzones//2])
rho = np.ones([Nzones]*3)
fname = 'clm_take3_L=4.h5'
CLOBBER = True
simple_test=True
make_plots=False

N_ell_em = ((np.arange(N_ell)+1)*2+1).sum()
ell_list=np.arange(N_ell)+2
N_positive = (ell_list).sum()
print(ell_list)
pdb.set_trace()

RM_all=np.zeros([Nsph,Ntheta_phi])

theta_all = np.zeros([Nsph,Ntheta_phi])
phi_all = np.zeros([Nsph,Ntheta_phi])

this_theta = np.random.random(Ntheta_phi)*np.pi
this_phi   = np.random.random(Ntheta_phi)*np.pi*2

t0 = time.time()
for nnn in np.arange(Nsph):


    phase = np.exp(1j*np.random.random(N_ell_em)*2*np.pi)
    amp = np.random.random(N_ell_em)
    positive_mask = np.zeros(N_ell_em, dtype='bool')
    this_clm = amp*phase
    all_index = np.arange(N_ell_em)
    if simple_test:
        #this_clm[:]=0
        this_clm[0]=-1 #m-1
        #this_clm[1]=1. #m0
        #this_clm[2]=1. #m1
        #this_clm[3]=0. #m-2
        #this_clm[4]=0. #m-1
        #this_clm[5]=0. #m0
        #this_clm[6]=0. #m1
        #this_clm[7]=0. #m2
        #this_clm[8]=0.
    count=0
    Clmd = {"N_ell":N_ell}

    for ell in np.arange(N_ell)+1:
        index_pm = ell+ell**2-1
        Clmd[(ell,0)] = this_clm[index_pm].real
        this_clm[index_pm] = this_clm[index_pm].real
        positive_mask[index_pm]=True
        for em in np.arange(1,ell+1):
            index_pm = em+ell+ell**2-1
            index_mm = -em+ell+ell**2-1
            conj = (-1)**em*np.conj(this_clm[index_pm])
            Clmd[ (ell,em)]=this_clm[index_pm]
            Clmd[ (ell,-em)]=conj
            this_clm[index_mm]=conj
            positive_mask[index_pm]=True
    if 0:
        for ell in np.arange(N_ell)+1:
            for em in np.arange(-ell,ell+1):
                clm_x = Clmd[(ell,em)]
                print("Clmd[(%d,%d)]= %0.2f + i %0.2f"%(ell,em,clm_x.real,clm_x.imag))

    #pdb.set_trace()

    X, Y, Z, Bx, By, Bz = make_multipole.multipole_B_field(Nzones, N_ell, Clmd, grid_extent=1.0)

    #Bx[:,:,:]=0.3
    #By[:,:,:]=0.3
    #Bz[:,:,:]=0.3
    #mask out the center, which explodes and gives blocks with huge values, and the corners, which add a funny signal.
    lin = np.arange(Nzones) - Nzones//2
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r0 = 5.0  # mask radius (in voxel units)
    mask = (r <= r0)+(r>Nzones//2)
    Bx[mask]=0
    By[mask]=0
    Bz[mask]=0

    if 1:
        counter=0
        this_rm = np.zeros(Ntheta_phi)
        for theta, phi in zip(this_theta, this_phi):
            dx = 1/Nzones
            vox, lengths = voxelpuller.voxels_and_path_lengths2(center, theta, phi, Bx.shape)
            lengths *= dx
            nhat = voxelpuller.direction_from_angles(theta,phi)
            voxline=vox[:,0],vox[:,1],vox[:,2]
            this_rm[counter] = ((Bx[voxline]*nhat[0]+
                                 By[vox[:,0],vox[:,1],vox[:,2]]*nhat[1]+
                                 Bz[vox[:,0],vox[:,1],vox[:,2]]*nhat[2])*lengths*rho[vox[:,0],vox[:,1],vox[:,2]]).sum()
            #this_rm[counter] = Bx[voxline].sum()
            counter+=1

    if np.random.random() >0:
        plot_multipole.plot_stream_and_rm(X,Y,Z,Bx,By,Bz,this_theta,this_phi,this_rm,fname='image_Nell_%d_%04d'%(N_ell,nnn), clm=Clmd)
        print(this_theta[:10])


    stuff={'Clm':this_clm[positive_mask],'Rm':this_rm, 'theta':this_theta,'phi':this_phi}
    #print('Clmd',Clmd)
    #print('Clm',stuff['Clm'])
    if not os.path.exists(fname) or CLOBBER:

        # Create file and dataset with unlimited rows
        with h5py.File(fname, "w") as f:
            for setname in ['Clm','Rm','theta','phi']:
                dtype = {'Clm':'complex64'}.get(setname, 'float64')
                size = {'Clm':N_positive}.get(setname,Ntheta_phi)
                dset = f.create_dataset(
                    setname,
                    shape = (0,size),
                    maxshape=(None, size),  # allow unlimited rows
                    chunks=(1, size),       # chunking needed for resizing
                    dtype=dtype
                )
            f['Ntheta_phi']=   Ntheta_phi 
            f['Nsph']=         Nsph 
            f['N_ell']=        N_ell 
            f['Nzones']=       Nzones 
            f['N_positive']=   N_positive

    with h5py.File(fname, "r+") as f:
        for setname in ['Clm','Rm','theta','phi']:

            size = {'Clm':N_positive}.get(setname,Ntheta_phi)
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
    print('SPHERE %d/%d eta %s Nell %d'%(nnn,Nsph,eta,N_ell))

