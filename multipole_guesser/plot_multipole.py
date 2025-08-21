import numpy as np
import matplotlib.pyplot as plt
import os
import healpy as hp
import pdb


def rmplot( sky, pooled,clm_model=None, clm_real=None,fname='ploot'):
    plt.close('all')

    theta, phi, rm = sky

    #hp.mollview(pooled,   title="Pooled RM",   unit="RM", cmap="coolwarm", fig=fig, sub=(1,2,2))

    plt.clf()
    plt.figure(figsize=(12, 4))

    # Left: bscatter
    #plt.subplot(2, 2, 1, projection = 'mollweide')
    #plt.scatter(theta, phi, c=rm, s=10, cmap='coolwarm')
    #hp.graticule()
    #plt.title("Original points (scatter)")
    #lon = phi - np.pi       # shift to [-π, π], center at 0
    lat = np.pi/2 - theta   # colatitude -> latitude

    # Flip longitude to match healpy's east-to-west orientation
    #lon = -lon
    #lon = np.remainder(lon + np.pi, 2*np.pi)# - np.pi

    lon = phi
    lon = np.pi - ( (lon + np.pi) % (2*np.pi) ) 

    fig = plt.figure(figsize=(12, 4))

# Left: scatter in mollweide projection
    #ax1 = fig.add_subplot(2, 2, 1, projection='mollweide')
    ax1 = fig.add_subplot(2, 2, 1)#, projection='mollweide')
    sc = ax1.scatter(lon, lat, c=rm, s=10, cmap='coolwarm')

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.grid(False)
    ax1.set_facecolor('w')
    ax1.set_title("Original points (scatter)")

# Colorbar for only the scatter plot
    cbar = fig.colorbar(sc, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.07)
    cbar.set_label("Rotation Measure")

# Right: healpy map
    hp.mollview(pooled, title="Pooled Healpix map",
                cmap='coolwarm', sub=(2, 2, 2), notext=True)

    plt.show()


    plt.subplot(2,2,3)
    plt.plot(clm_model.detach().numpy().real)
    plt.plot(clm_real.detach().numpy().real)
    plt.title('real')
    plt.subplot(2,2,4)
    plt.plot(clm_model.detach().numpy().imag)
    plt.plot(clm_real.detach().numpy().imag)
    plt.title('imag')
    plt.tight_layout()
    plt.savefig("%s/plots/%s"%(os.environ['HOME'],fname))

def rmplot2d(theta2d,phi2d,pooled, sky,clm_model=None, clm_real=None,fname='ploot'):
    plt.close('all')

    theta, phi, rm = sky
    plt.clf()
    plt.figure(figsize=(12, 4))
    lat = np.pi/2 - theta   # colatitude -> latitude
    lon = phi
    lon = np.pi - ( (lon + np.pi) % (2*np.pi) ) 

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(2, 2, 1)#, projection='mollweide')
    sc = ax1.scatter(theta,phi, c=rm, s=10, cmap='coolwarm')

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.grid(False)
    ax1.set_facecolor('w')
    ax1.set_title("Original points (scatter)")

# Colorbar for only the scatter plot
    cbar = fig.colorbar(sc, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.07)
    cbar.set_label("Rotation Measure")

# Right: healpy map
    #hp.mollview(pooled, title="Pooled Healpix map",
    #            cmap='coolwarm', sub=(2, 2, 2), notext=True)
    plt.subplot(2,2,2)
    plt.pcolormesh(theta2d, phi2d, pooled)

    plt.show()


    plt.subplot(2,2,3)
    a2 = clm_model.detach().numpy().real
    a1 = clm_real.detach().numpy().real
    plt.plot(a1, a2)
    plt.title('real')
    plt.xlabel('target'); plt.ylabel('model')
    plt.subplot(2,2,4)
    a4=clm_model.detach().numpy().imag
    a3=clm_real.detach().numpy().imag
    #pdb.set_trace()
    plt.plot(a3,a4)
    plt.title('imag')
    plt.xlabel('target'); plt.ylabel('model')
    plt.tight_layout()
    plt.savefig("%s/plots/%s"%(os.environ['HOME'],fname))


def plot_stream_and_rm(X,Y,Z,Bx,By,Bz,theta,phi,rm,fname='image', clm=None):

    fig,ax=plt.subplots(2,3,figsize=(12,8))
    ax0,ax1=ax
    plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='x',ax_in=ax0[0])
    plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='y',ax_in=ax0[1])
    plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='z',ax_in=ax0[2])

    clm_mag = []
    clm_phase = []
    if clm is not None:
        for ell in np.arange(clm['N_ell'])+1:
            for em in np.arange(-ell,ell+1):
                clm_mag.append( np.abs(clm[(ell,em)] ) )
                clm_phase.append( np.angle(clm[(ell,em)]))
        ax1[2].plot(clm_mag)
        tw = ax1[2].twinx()
        tw.plot(clm_phase)

    scat=ax1[0].scatter(theta,phi,c=rm, cmap='viridis')
    fig.colorbar(scat,ax=ax1[0])
    ax1[1].hist(rm)
    fig.savefig('%s/plots/%s'%(os.environ['HOME'],fname))
    plt.close(fig)
    fig,ax=plt.subplots(3,3,figsize=(12,8))
    for i in range(3):
        Bi = [Bx,By,Bz][i]
        for j in range(3):
            pl=ax[i][j].imshow( np.abs(Bi).sum(axis=j))
            fig.colorbar(pl,ax=ax[i][j])

    fig.savefig('%s/plots/%s_field'%(os.environ['HOME'],fname))
    



def plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='z', slice_index=None, ax_in=None):
    """
    Plot streamlines of the B-field in a plane perpendicular to `axis`.
    axis can be 'x', 'y', or 'z'.
    """
    N = X.shape[0]
    if slice_index is None:
        slice_index = N // 2

    if ax_in is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        ax=ax_in

    if axis == 'z':
        # XY plane at fixed z
        xvals = X[:, 0, 0]
        yvals = Y[0, :, 0]
        U = Bx[:, :, slice_index]
        V = By[:, :, slice_index]
        X2, Y2 = np.meshgrid(xvals, yvals, indexing='ij')

    elif axis == 'y':
        # XZ plane at fixed y
        xvals = X[:, 0, 0]
        zvals = Z[0, 0, :]
        U = Bx[:, slice_index, :]
        V = Bz[:, slice_index, :]
        X2, Y2 = np.meshgrid(xvals, zvals, indexing='ij')

    elif axis == 'x':
        # YZ plane at fixed x
        yvals = Y[0, :, 0]
        zvals = Z[0, 0, :]
        U = By[slice_index, :, :]
        V = Bz[slice_index, :, :]
        X2, Y2 = np.meshgrid(yvals, zvals, indexing='ij')

    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


    #pdb.set_trace()
    speed = np.sqrt(U**2 + V**2)
    strm = ax.streamplot(Y2,X2, V, U, color=speed, cmap='plasma',
                         linewidth=1, density=1.5)

    ax.set_aspect('equal')
    if ax_in is None:
        cbar = fig.colorbar(strm.lines, ax=ax)
        cbar.set_label('|B| [T]')
        plt.title(f"B-field streamlines in {axis}-slice at index {slice_index}")
        plt.show()

