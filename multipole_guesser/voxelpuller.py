import numpy as np
import pdb

def direction_from_angles(theta, phi):
    """
    Convert spherical angles (in radians) to a 3D unit vector.
    theta = polar angle from +z axis
    phi = azimuthal angle from +x axis in xy-plane
    """
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=float)

def ray_box_exit_point(p0, direction, shape):
    """
    Given start point p0 and unit direction, find where the ray exits the box.
    Box is from (0,0,0) to (shape[0], shape[1], shape[2]).
    Returns exit point coordinates.
    """
    t_vals = []
    for axis in range(3):
        if direction[axis] > 0:
            t_vals.append((shape[axis] - p0[axis]) / direction[axis])
        elif direction[axis] < 0:
            t_vals.append(-p0[axis] / direction[axis])
        else:
            t_vals.append(np.inf)
    # Take the smallest positive distance to a boundary
    t_exit = min([t for t in t_vals if t > 0])
    return p0 + direction * t_exit

def voxels_on_ray(p0, theta, phi, shape):
    """
    Return all voxels from p0 to the box edge in direction (theta, phi).
    """
    # Convert angles to direction vector
    direction = direction_from_angles(theta, phi)

    # Find end point at box edge
    p1 = ray_box_exit_point(np.array(p0, dtype=float), direction, shape)

    # Traverse voxels
    return voxels_on_line_3d(p0, p1, shape)

# --- Amanatides–Woo traversal with bounds check ---
def voxels_on_line_3d(p0, p1, shape):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    d = p1 - p0
    voxel = np.floor(p0).astype(int)
    step = np.sign(d).astype(int)
    inv_d = np.where(d != 0, 1.0 / np.abs(d), np.inf)
    t_max = np.where(
        step > 0,
        (np.floor(p0) + 1 - p0) * inv_d,
        (p0 - np.floor(p0)) * inv_d
    )
    t_delta = inv_d
    voxels = []
    def in_bounds(v):
        return all(0 <= v[i] < shape[i] for i in range(3))
    while in_bounds(voxel):
        voxels.append(tuple(voxel))
        if np.all(voxel == np.floor(p1).astype(int)):
            break
        axis = np.argmin(t_max)
        t_max[axis] += t_delta[axis]
        voxel[axis] += step[axis]
    return np.array(voxels, dtype=int)

import numpy as np

import numpy as np

def voxels_and_path_lengths2(p0, theta, phi, shape):
    """
    Traverse voxels from starting point p0 along a ray defined by spherical angles (theta, phi),
    returning all voxel indices traversed and the path length inside each voxel.

    Parameters
    ----------
    p0 : array-like of shape (3,)
        Starting point coordinates (floats).
    theta : float
        Polar angle in radians (angle from +z axis).
    phi : float
        Azimuthal angle in radians (angle in xy-plane from +x axis).
    shape : tuple of ints (Nz, Ny, Nx)
        Shape of the 3D grid.

    Returns
    -------
    voxels : np.ndarray of shape (M, 3), dtype=int
        Indices of the visited voxels along the ray.
    lengths : np.ndarray of shape (M,), dtype=float
        Lengths of the path inside each voxel.
    """

    p0 = np.array(p0, dtype=float)
    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=float)

    # Calculate exit point parameter t_exit for the box boundaries
    t_vals = []
    for i in range(3):
        if direction[i] > 0:
            t_vals.append((shape[i] - p0[i]) / direction[i])
        elif direction[i] < 0:
            t_vals.append(-p0[i] / direction[i])
        else:
            t_vals.append(np.inf)
    t_exit = min(t for t in t_vals if t > 0)
    p1 = p0 + direction * t_exit

    # Compute total physical length inside the box
    total_length = np.linalg.norm(p1 - p0)

    # Normalize direction to unit vector for stepping calculations
    direction_unit = direction / np.linalg.norm(direction)

    # Initialize voxel traversal
    voxel = np.floor(p0).astype(int)
    step = np.sign(direction_unit).astype(int)

    inv_d = np.empty(3)
    for i in range(3):
        if direction_unit[i] != 0:
            inv_d[i] = 1.0 / abs(direction_unit[i])
        else:
            inv_d[i] = np.inf

    t_max = np.empty(3)
    for i in range(3):
        if step[i] > 0:
            t_max[i] = (np.floor(p0[i]) + 1 - p0[i]) * inv_d[i]
        elif step[i] < 0:
            t_max[i] = (p0[i] - np.floor(p0[i])) * inv_d[i]
        else:
            t_max[i] = np.inf

    t_delta = inv_d

    voxels = []
    lengths = []

    def in_bounds(v):
        return all(0 <= v[i] < shape[i] for i in range(3))

    t_prev = 0.0
    while in_bounds(voxel):
        voxels.append(tuple(voxel))
        if np.all(voxel == np.floor(p1).astype(int)):
            lengths.append((t_exit - t_prev) )
            break

        axis = np.argmin(t_max)
        t_next = t_max[axis]
        path_len = (t_next - t_prev) #* total_length
        #print(".",t_next,t_prev,path_len)
        #pdb.set_trace()
        lengths.append(path_len)

        t_max[axis] += t_delta[axis]
        voxel[axis] += step[axis]
        t_prev = t_next

    return np.array(voxels, dtype=int), np.array(lengths)


