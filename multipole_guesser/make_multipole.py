import numpy as np
from scipy.special import sph_harm

def multipole_B_field(N, L_max, coeffs, grid_extent=1.0):
    """
    Compute B-field on an (N,N,N) grid from multipole expansion coefficients.

    Parameters
    ----------
    N : int
        Number of grid points in each dimension.
    L_max : int
        Maximum multipole order.
    coeffs : dict
        Dictionary of {(l, m): C_lm} coefficients for expansion.
        C_lm in Tesla*m^(l+2) units for magnetic scalar potential expansion.
    grid_extent : float
        Physical half-size of the grid (in meters).
    
    Returns
    -------
    X, Y, Z : 3D ndarrays of shape (N, N, N)
    Bx, By, Bz : 3D ndarrays of shape (N, N, N)
    """

    # Coordinate grid
    lin = np.linspace(-grid_extent, grid_extent, N)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(np.divide(Z, r, out=np.zeros_like(Z), where=r>0))
    phi = np.arctan2(Y, X)

    # Scalar potential Phi
    Phi = np.zeros_like(r, dtype=np.complex128)

    for l in range(1, L_max+1):  # l starts at 1 for magnetic fields
        for m in range(-l, l+1):
            if (l, m) in coeffs:
                Ylm = sph_harm(m, l, phi, theta)  # SciPy uses (m, l) ordering
                Phi += coeffs[(l, m)] * Ylm / (r**(l+1) + (r==0))

    # Real part only (physical field)
    imag = np.abs(Phi.imag).sum()/np.abs(Phi.real).sum()
    if imag > 1e-8:
        print('Problematic imaginary part')
    Phi = Phi.real

    # Magnetic field B = -grad(Phi)
    # Use np.gradient with spacing = grid spacing
    spacing = (2*grid_extent)/(N-1)
    dPhidx, dPhidy, dPhidz = np.gradient(Phi, spacing, edge_order=2)
    Bx = -dPhidx
    By = -dPhidy
    Bz = -dPhidz

    return X, Y, Z, Bx, By, Bz


# Example usage
#if __name__ == "__main__":
if 0:
    N = 32
    L_max = 2
    coeffs = {
        (1, 0): 1e-7,   # Dipole aligned with z
        (2, 0): 5e-8,   # Quadrupole term
    }
    X, Y, Z, Bx, By, Bz = multipole_B_field(N, L_max, coeffs, grid_extent=1.0)

    # Magnitude for visualization
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    print("B-field magnitude range:", B_mag.min(), B_mag.max())

