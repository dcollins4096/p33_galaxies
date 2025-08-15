import numpy as np
import matplotlib.pyplot as plt
import os
import healpy as hp


def rmplot( sky, pooled,fname='ploot'):

    theta, phi, rm = sky

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    #hp.mollview(pooled,   title="Pooled RM",   unit="RM", cmap="coolwarm", fig=fig, sub=(1,2,2))

    plt.clf()
    plt.figure(figsize=(12, 5))

    # Left: scatter
    plt.subplot(1, 2, 1, projection = 'mollweide')
    #plt.scatter(theta, phi, c=rm, s=10, cmap='coolwarm')
    #hp.graticule()
    #plt.title("Original points (scatter)")
    lon = -1*(phi - np.pi)       # shift so -pi..pi
    lat = np.pi/2 - theta   # convert colatitude -> latitude

    x = 2 * np.sqrt(2) / np.pi * lon * np.cos(lat/2)
    y = np.sqrt(2) * np.sin(lat/2)

    plt.scatter(x, y, c=rm, s=10, cmap='coolwarm')
    plt.xticks([])            # remove x tick labels
    plt.yticks([])            # remove y tick labels
    plt.grid(False)           # remove grid
    plt.gca().set_facecolor('w')  # set background color like healpy

    # Right: mollview using sub argument
    hp.mollview(pooled, title="Pooled Healpix map", cmap='coolwarm', sub=(1, 2, 2))
    plt.savefig("%s/plots/%s"%(os.environ['HOME'],fname))



def plot_stream_and_rm(X,Y,Z,Bx,By,Bz,theta,phi,rm,axis='z',fname='image'):

    fig,ax=plt.subplots(2,3,figsize=(12,8))
    ax0,ax1=ax
    plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='x',ax_in=ax0[0])
    plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='y',ax_in=ax0[1])
    plot_B_streamlines(X, Y, Z, Bx, By, Bz, axis='z',ax_in=ax0[2])

    scat=ax1[0].scatter(theta,phi,c=rm, cmap='viridis')
    fig.colorbar(scat,ax=ax1[0])
    ax1[1].hist(rm)
    fig.savefig('%s/plots/%s'%(os.environ['HOME'],fname))
    plt.close(fig)



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

