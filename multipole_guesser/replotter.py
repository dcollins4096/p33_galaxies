
from importlib import reload
import matplotlib.pyplot as plt
import plot_multipole
reload(plot_multipole)
import make_multipole
import h5py
import numpy as np

fname = 'clm_take1_L=3.h5'

pull_list = [0]

fptr = h5py.File(fname,'r')
try:
    Clm = fptr['Clm'][()]
    Rm  = fptr['Rm'][()]
    phi = fptr['phi'][()]
    theta =fptr['theta'][()]
    N_ell  = fptr['N_ell'][()]
    Nzones = fptr['Nzones'][()]
except:
    raise
finally:
    fptr.close()
for nnn in pull_list:
    this_clm = Clm[nnn]
    this_rm  = Rm[nnn]
    this_th  = theta[nnn]
    this_ph  = phi[nnn]
    Clmd={'N_ell':N_ell}


    counter = 0
    for ell in np.arange(N_ell)+1:
        print('ell=======',ell)
        for m in range(0, ell + 1):
            print('m',m,counter)
            c_lm = this_clm[counter]
            if m == 0:
                Clmd[(ell, 0)] = np.real(c_lm)          # drop tiny imag
            else:
                Clmd[(ell,  m)] = c_lm
                Clmd[(ell, -m)] = ((-1)**m) * np.conj(c_lm)
            counter += 1
    #print('this_clm',this_clm)
    #print('Clmd', Clmd)
    X, Y, Z, Bx, By, Bz = make_multipole.multipole_B_field(Nzones, N_ell, Clmd, grid_extent=1.0)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r0 = 5.0  # mask radius (in voxel units)
    mask = (r <= r0)+(r>Nzones//2)
    Bx[mask]=0
    By[mask]=0
    Bz[mask]=0
    plot_multipole.plot_stream_and_rm(X,Y,Z,Bx,By,Bz,this_theta,this_phi,this_rm,clm=Clmd,fname='reimage_N_ell_%d_%04d'%(N_ell,nnn))
    print(this_theta[:10])

