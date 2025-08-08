import numpy as np
import matplotlib.pyplot as plt

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

