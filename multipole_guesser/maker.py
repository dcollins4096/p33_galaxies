
from importlib import reload
import plot_multipole
reload(plot_multipole)
import make_multipole
import pdb
import os
import voxelpuller
reload(voxelpuller)

def go( N = 32, L_max = 2, coeffs = {(1, 0): 1e-7, (2, 0): 5e-8}):
    X, Y, Z, Bx, By, Bz = make_multipole.multipole_B_field(N, L_max, coeffs, grid_extent=1.0)
    #return X,Y,Z,Bx,By,Bz
    

    fig,ax=plt.subplots(1,3, figsize=(12,4))
    plot_multipole.plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='x',ax_in=ax[0])
    plot_multipole.plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='y',ax_in=ax[1])
    plot_multipole.plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='z',ax_in=ax[2])
    fig.savefig('%s/plots/test'%(os.environ['HOME']))

#x,y,z,bx,by,bz=go()
#go( coeffs={(1,-1):1e-7})

N=32
#cube = np.arange(N**3).reshape(N,N,N)
cube = np.zeros(N**3).reshape(N,N,N)
center = np.array([N//2,N//2,N//2])
center = np.array([1,1,16])
theta= -np.pi/3
theta = 0
phi = np.pi/4
vox, lengths = voxelpuller.voxels_and_path_lengths2(center, theta, phi, cube.shape)
print('lengths',lengths.sum(), lengths.size)
#print(cube[vox[:,0], vox[:,1], vox[:,2]])
cube[vox[:,0], vox[:,1], vox[:,2]] = 1

fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax in [0,1,2]:
    axes[ax].imshow( cube.sum(axis=ax).reshape(N,N))
fig.savefig('%s/plots/rays'%os.environ['HOME'])




