from starter2 import *
import numpy as np
import healpy as hp
from scipy.spatial import ConvexHull

import projector.proj as proj
reload(proj)


def _normalize(v, axis=0, eps=1e-30):
    v = np.asarray(v, dtype=np.float64)
    n = np.sqrt((v * v).sum(axis=axis, keepdims=True))
    return v / np.maximum(n, eps)


def _healpix_boundary_vecs_and_centers(nside):
    npix = hp.nside2npix(nside)
    boundaries = np.zeros((npix, 3, 4), dtype=np.float64)
    centers = np.zeros((npix, 3), dtype=np.float64)

    for ipix in range(npix):
        boundaries[ipix] = hp.boundaries(nside, ipix, step=1)
        centers[ipix] = np.array(hp.pix2vec(nside, ipix), dtype=np.float64)

    return boundaries, centers


def _order_boundary_vecs_ccw(ray_dirs, center_vec):
    """
    Order 4 boundary vectors in cyclic CCW order around the pixel center.
    Input/output shape: (4,3)
    """
    ray_dirs = np.asarray(ray_dirs, dtype=np.float64)
    c = np.asarray(center_vec, dtype=np.float64)
    c = c / np.maximum(np.linalg.norm(c), 1e-30)

    if abs(c[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    e1 = np.cross(ref, c)
    e1 /= np.maximum(np.linalg.norm(e1), 1e-30)
    e2 = np.cross(c, e1)
    e2 /= np.maximum(np.linalg.norm(e2), 1e-30)

    x = ray_dirs @ e1
    y = ray_dirs @ e2
    ang = np.arctan2(y, x)
    order = np.argsort(ang)
    return ray_dirs[order]


# ============================================================
# 3D cube ∩ cone geometry
# ============================================================

def _unique_points(points, tol=1e-10):
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    key = np.round(arr / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return arr[np.sort(idx)]


def _infer_cube_edges(corners, tol=1e-10):
    """
    corners: (8,3)
    Return the 12 edge index pairs.
    """
    from itertools import combinations

    corners = np.asarray(corners, dtype=np.float64)
    ds = []
    for i, j in combinations(range(8), 2):
        d = np.linalg.norm(corners[j] - corners[i])
        if d > tol:
            ds.append(d)

    edge_len = min(ds)
    edges = []
    for i, j in combinations(range(8), 2):
        d = np.linalg.norm(corners[j] - corners[i])
        if abs(d - edge_len) <= max(tol, 1e-7 * edge_len):
            edges.append((i, j))

    if len(edges) != 12:
        raise ValueError("Could not infer 12 cube edges from corners.")
    return edges


def _point_in_hull(pt, hull, tol=1e-10):
    """
    scipy ConvexHull equations satisfy A x + b <= 0 inside hull.
    """
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    return np.all(A @ pt + b <= tol)


def _unique_face_planes_from_hull(hull, tol=1e-10):
    """
    Deduplicate triangulated hull planes into actual cube face planes.
    Returns list of (n, c) for plane n·x + c = 0.
    """
    planes = []
    for eq in hull.equations:
        n = eq[:-1].astype(np.float64)
        c = float(eq[-1])

        nn = np.linalg.norm(n)
        n /= np.maximum(nn, 1e-30)
        c /= np.maximum(nn, 1e-30)

        k = np.argmax(np.abs(n))
        if n[k] < 0:
            n = -n
            c = -c

        found = False
        for n0, c0 in planes:
            if np.linalg.norm(n - n0) < tol and abs(c - c0) < tol:
                found = True
                break
        if not found:
            planes.append((n, c))
    return planes


def _cone_side_normals(ray_dirs):
    """
    ray_dirs: (4,3), cyclic order around the healpix pixel boundary.
    Returns inward normals n_i such that inside cone means n_i·x >= 0.
    """
    U = np.asarray(ray_dirs, dtype=np.float64)
    U = np.array([u / np.maximum(np.linalg.norm(u), 1e-30) for u in U])

    interior = U.mean(axis=0)
    interior /= np.maximum(np.linalg.norm(interior), 1e-30)

    normals = []
    for i in range(4):
        u0 = U[i]
        u1 = U[(i + 1) % 4]
        n = np.cross(u0, u1)
        nn = np.linalg.norm(n)
        if nn < 1e-14:
            raise ValueError("Adjacent cone rays are collinear.")
        n /= nn
        if np.dot(n, interior) < 0:
            n = -n
        normals.append(n)
    return np.asarray(normals), U


def _point_in_cone(pt, side_normals, tol=1e-10):
    return np.all(side_normals @ pt >= -tol)


def _segment_plane_intersection(p0, p1, n, tol=1e-14):
    """
    Segment p(t)=p0+t*(p1-p0), t in [0,1], with plane n·x = 0.
    """
    d = p1 - p0
    denom = np.dot(n, d)
    if abs(denom) < tol:
        return None
    t = -np.dot(n, p0) / denom
    if -tol <= t <= 1.0 + tol:
        t = min(1.0, max(0.0, t))
        return p0 + t * d
    return None


def _ray_plane_intersection(u, n, c, tol=1e-14):
    """
    Ray x = t u, t>=0, with plane n·x + c = 0.
    """
    denom = np.dot(n, u)
    if abs(denom) < tol:
        return None
    t = -c / denom
    if t >= -tol:
        t = max(0.0, t)
        return t * u
    return None


def _cube_fully_inside_cone(corners, ray_dirs, tol=1e-10):
    side_normals, _ = _cone_side_normals(ray_dirs)
    vals = side_normals @ corners.T
    return np.all(vals >= -tol)


def cube_cone_intersection_vertices(corners, ray_dirs, tol=1e-10):
    """
    corners : (8,3) cube corners
    ray_dirs: (4,3) healpix pixel boundary rays in cyclic order

    Returns vertices of the convex polyhedron cube ∩ cone.
    Cone apex is at the origin.
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError("corners must have shape (8,3)")

    side_normals, U = _cone_side_normals(ray_dirs)
    hull = ConvexHull(corners)
    edges = _infer_cube_edges(corners, tol=tol)
    face_planes = _unique_face_planes_from_hull(hull, tol=tol)

    pts = []

    # 1) cube corners inside cone
    for p in corners:
        if _point_in_cone(p, side_normals, tol=tol):
            pts.append(p)

    # 2) cone apex inside cube
    origin = np.zeros(3, dtype=np.float64)
    if _point_in_hull(origin, hull, tol=tol):
        pts.append(origin)

    # 3) cube-edge / cone-side-plane intersections
    for i, j in edges:
        p0 = corners[i]
        p1 = corners[j]
        for n in side_normals:
            p = _segment_plane_intersection(p0, p1, n, tol=tol)
            if p is None:
                continue
            if _point_in_cone(p, side_normals, tol=tol) and _point_in_hull(p, hull, tol=tol):
                pts.append(p)

    # 4) cone-edge-ray / cube-face intersections
    for u in U:
        for n, c in face_planes:
            p = _ray_plane_intersection(u, n, c, tol=tol)
            if p is None:
                continue
            if _point_in_cone(p, side_normals, tol=tol) and _point_in_hull(p, hull, tol=tol):
                pts.append(p)

    return _unique_points(pts, tol=tol)


def cube_cone_intersection_volume(corners, ray_dirs, tol=1e-10):
    """
    corners : (8,3)
    ray_dirs: (4,3)

    Volume of intersection between cube and infinite 4-sided cone.
    """
    verts = cube_cone_intersection_vertices(corners, ray_dirs, tol=tol)

    if verts.shape[0] < 4:
        return 0.0

    centered = verts - verts.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=tol) < 3:
        return 0.0

    hull = ConvexHull(verts)
    return hull.volume


# ============================================================
# Projector
# ============================================================

def project(cube, xyz, dxyz, proj_center, proj_axis,
            bucket=None, molplot=False, moreplots=False,
            NSIDE=4, exclude=1, verbose=False):
    """
    Project Cartesian cell emission onto a healpix sky using exact 3D
    cube∩pixel-cone volumes.

    Notes
    -----
    - count_map now stores accumulated overlap volume, not integer hit count.
    - Includes sanity checks:
        * overlap volume must be >= 0
        * overlap volume must be <= cell volume
        * per-zone summed overlap volume can be checked against cell volume
    """

    cube = np.asarray(cube, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    dxyz = np.asarray(dxyz, dtype=np.float64)
    proj_center = np.asarray(proj_center, dtype=np.float64).reshape(3, 1)

    # shift origin to projection center
    xyz_shift = xyz - proj_center

    # exclude close cells
    rrr = np.sqrt((xyz_shift ** 2).sum(axis=0))
    cell_scale = dxyz.mean(axis=0)
    nxyz = rrr / np.maximum(cell_scale, 1e-30)
    mask = nxyz > exclude

    cube = cube[mask]
    xyz_shift = xyz_shift[:, mask]
    dxyz = dxyz[:, mask]

    Nz = cube.size
    NPIX = hp.nside2npix(NSIDE)

    final_map = np.zeros(NPIX, dtype=np.float64)
    count_map = np.zeros(NPIX, dtype=np.float64)   # overlap volume, not hit count

    if Nz == 0:
        return final_map, count_map

    # cube corner offsets
    shifter = np.array([
        [[-0.5, -0.5, +0.5, +0.5, +0.5, +0.5, -0.5, -0.5]],
        [[-0.5, -0.5, -0.5, -0.5, +0.5, +0.5, +0.5, +0.5]],
        [[-0.5, +0.5, +0.5, -0.5, -0.5, +0.5, +0.5, -0.5]]
    ], dtype=np.float64)

    xyz_z = xyz_shift.reshape(3, Nz, 1)
    dxyz_z = dxyz.reshape(3, Nz, 1)
    corners = xyz_z + shifter * dxyz_z  # (3, Nz, 8)

    # rotate into projector frame
    corners_rot, phi_rot, theta_rot = proj.make_phi_theta(corners, proj_axis)
    xyz_p = proj.rotate(xyz_z, proj_axis)
    xyz_p = np.asarray(xyz_p, dtype=np.float64).reshape(3, Nz)
    corners_rot = np.asarray(corners_rot, dtype=np.float64)

    # preserve your emission convention
    rrr2 = (xyz_p ** 2).sum(axis=0)
    zone_volume = dxyz.prod(axis=0)
    zone_emission = cube * zone_volume# / np.maximum(rrr2, 1e-30)

    # unit vectors for centers and corners
    center_vecs = _normalize(xyz_p, axis=0)                # (3, Nz)
    corner_vecs = _normalize(corners_rot, axis=0)          # (3, Nz, 8)

    # angular search radius
    center_T = center_vecs.T                               # (Nz, 3)
    corner_T = np.transpose(corner_vecs, (1, 2, 0))       # (Nz, 8, 3)

    dots = np.sum(corner_T * center_T[:, None, :], axis=2)
    dots = np.clip(dots, -1.0, 1.0)
    circle_radius = np.arccos(dots).max(axis=1)

    # precompute healpix boundaries and centers
    pix_boundary_vecs, pix_center_vecs = _healpix_boundary_vecs_and_centers(NSIDE)

    if molplot:
        plt.clf()
        m = np.arange(NPIX)
        from healpy.newvisufunc import projview
        projview(m, title='ones', cmap='Reds', projection_type='polar')

    for izone in range(Nz):
        if 1:
            print("s8p Izone/Nzone %d/%d = %0.2f" % (izone, Nz, izone / max(Nz, 1)))

        quantity = zone_emission[izone]
        zone_center_vec = center_vecs[:, izone]
        zone_corners = corners_rot[:, izone, :].T   # (8,3)
        this_zone_volume = zone_volume[izone]

        my_pix = hp.query_disc(
            nside=NSIDE,
            vec=zone_center_vec,
            radius=float(circle_radius[izone]),
            inclusive=True
        )

        if verbose:
            print("N pixels %d" % len(my_pix))

        vsum = 0.0

        for ipix in my_pix:
            # hp.boundaries returns (3,4); transpose to (4,3)
            ray_dirs = pix_boundary_vecs[ipix].T
            ray_dirs = _order_boundary_vecs_ccw(ray_dirs, pix_center_vecs[ipix])

            try:
                if _cube_fully_inside_cone(zone_corners, ray_dirs):
                    intersect_volume = this_zone_volume
                else:
                    intersect_volume = cube_cone_intersection_volume(zone_corners, ray_dirs)
            except Exception:
                print("Failed on izone=%d ipix=%d" % (izone, ipix))
                print("zone_center_vec =", zone_center_vec)
                print("pix_center      =", pix_center_vecs[ipix])
                print("zone_corners    =\n", zone_corners)
                print("ray_dirs        =\n", ray_dirs)
                raise

            # sanity checks
            if intersect_volume < -1e-12:
                print("NEGATIVE VOLUME on izone=%d ipix=%d" % (izone, ipix))
                print("intersect_volume =", intersect_volume)
                print("zone_volume      =", this_zone_volume)
                print("zone_corners     =\n", zone_corners)
                print("ray_dirs         =\n", ray_dirs)
                raise RuntimeError("Negative intersection volume")

            if intersect_volume > this_zone_volume * (1.0 + 1e-8):
                print("TOO-LARGE VOLUME on izone=%d ipix=%d" % (izone, ipix))
                print("intersect_volume =", intersect_volume)
                print("zone_volume      =", this_zone_volume)
                print("zone_center_vec  =", zone_center_vec)
                print("pix_center       =", pix_center_vecs[ipix])
                print("zone_corners     =\n", zone_corners)
                print("ray_dirs         =\n", ray_dirs)
                raise RuntimeError("Intersection volume exceeds zone volume")

            if intersect_volume <= 0.0:
                continue

            overlap_fraction = intersect_volume / np.maximum(this_zone_volume, 1e-30)
            net_light = quantity * overlap_fraction

            final_map[ipix] += net_light
            count_map[ipix] += intersect_volume
            vsum += intersect_volume

            if moreplots:
                print("izone %d ipix %d vol %0.3e frac %0.3e" %
                      (izone, ipix, intersect_volume, overlap_fraction))

        if verbose:
            frac = vsum / np.maximum(this_zone_volume, 1e-30)
            print("zone %d summed_overlap/zone_volume = %0.6e" % (izone, frac))

    if molplot:
        plt.show()

    return final_map, count_map
