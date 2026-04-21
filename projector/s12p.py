from starter2 import *
import numpy as np
import healpy as hp
from scipy.spatial import ConvexHull

import projector.proj as proj
reload(proj)


# ============================================================
# Global cache
# ============================================================

_HEALPIX_CACHE = {}


# ============================================================
# Basic helpers
# ============================================================

def _normalize(v, axis=0, eps=1e-30):
    v = np.asarray(v, dtype=np.float64)
    n = np.sqrt((v * v).sum(axis=axis, keepdims=True))
    return v / np.maximum(n, eps)


def _unique_points(points, tol=1e-10):
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    key = np.round(arr / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return arr[np.sort(idx)]


# ============================================================
# Fixed cube topology
# ============================================================

_CUBE_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 7], [1, 6], [2, 5], [3, 4],
], dtype=np.int64)


# ============================================================
# HEALPix geometry
# ============================================================

def _order_boundary_vecs_ccw(ray_dirs, center_vec):
    """
    ray_dirs: (4,3)
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


def _cone_side_normals(ray_dirs):
    """
    ray_dirs: (4,3), cyclic order
    Returns inward normals n_i such that inside cone means n_i·x >= 0.
    """
    U = np.asarray(ray_dirs, dtype=np.float64)
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-30)

    interior = U.mean(axis=0)
    interior /= np.maximum(np.linalg.norm(interior), 1e-30)

    u0 = U
    u1 = np.roll(U, -1, axis=0)
    normals = np.cross(u0, u1)
    nn = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(nn[:, 0] < 1e-14):
        raise ValueError("Adjacent cone rays are collinear.")
    normals = normals / nn

    flip = (normals @ interior) < 0.0
    normals[flip] *= -1.0
    return normals, U


def _precompute_healpix_geometry(nside):
    npix = hp.nside2npix(nside)

    pix_centers = np.zeros((npix, 3), dtype=np.float64)
    pix_rays = np.zeros((npix, 4, 3), dtype=np.float64)
    pix_side_normals = np.zeros((npix, 4, 3), dtype=np.float64)

    for ipix in range(npix):
        center = np.array(hp.pix2vec(nside, ipix), dtype=np.float64)
        rays = hp.boundaries(nside, ipix, step=1).T
        rays = _order_boundary_vecs_ccw(rays, center)
        side_normals, rays = _cone_side_normals(rays)

        pix_centers[ipix] = center
        pix_rays[ipix] = rays
        pix_side_normals[ipix] = side_normals

    return pix_centers, pix_rays, pix_side_normals


def _get_healpix_geometry(nside):
    if nside not in _HEALPIX_CACHE:
        _HEALPIX_CACHE[nside] = _precompute_healpix_geometry(nside)
    return _HEALPIX_CACHE[nside]


# ============================================================
# Projector-frame box geometry
# ============================================================

def _projector_box_axes_and_face_normals(proj_axis):
    """
    All zones in one projector call share the same rotated box axes.
    Returns:
        box_axes:     (3,3) = [ex_r, ey_r, ez_r]
        face_normals: (6,3) = [-ex,+ex,-ey,+ey,-ez,+ez]
    """
    ex = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    ey = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    ez = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)

    ex_r = np.asarray(proj.rotate(ex, proj_axis), dtype=np.float64).reshape(3)
    ey_r = np.asarray(proj.rotate(ey, proj_axis), dtype=np.float64).reshape(3)
    ez_r = np.asarray(proj.rotate(ez, proj_axis), dtype=np.float64).reshape(3)

    ex_r /= np.maximum(np.linalg.norm(ex_r), 1e-30)
    ey_r /= np.maximum(np.linalg.norm(ey_r), 1e-30)
    ez_r /= np.maximum(np.linalg.norm(ez_r), 1e-30)

    box_axes = np.array([ex_r, ey_r, ez_r], dtype=np.float64)

    face_normals = np.array([
        -ex_r, +ex_r,
        -ey_r, +ey_r,
        -ez_r, +ez_r,
    ], dtype=np.float64)

    return box_axes, face_normals


def _cube_face_offsets_from_center_size(zone_center, dxyz_zone, face_normals, box_axes):
    """
    zone_center:   (3,)
    dxyz_zone:     (3,)
    face_normals:  (6,3)
    box_axes:      (3,3) = [ex_r, ey_r, ez_r]

    Returns face_c for planes n·x + c = 0,
    with inside box satisfying n·x + c <= tol.
    """
    ex_r, ey_r, ez_r = box_axes
    hx, hy, hz = 0.5 * np.asarray(dxyz_zone, dtype=np.float64)

    face_points = np.array([
        zone_center - hx * ex_r,   # -x face
        zone_center + hx * ex_r,   # +x face
        zone_center - hy * ey_r,   # -y face
        zone_center + hy * ey_r,   # +y face
        zone_center - hz * ez_r,   # -z face
        zone_center + hz * ez_r,   # +z face
    ], dtype=np.float64)

    face_c = -np.sum(face_normals * face_points, axis=1)
    return face_c


def _points_in_cube(pts, face_normals, face_c, tol=1e-10):
    vals = pts @ face_normals.T + face_c[None, :]
    return np.all(vals <= tol, axis=1)


def _origin_in_cube(face_c, tol=1e-10):
    # origin is inside iff n·0 + c = c <= tol for all faces
    return np.all(face_c <= tol)


# ============================================================
# Cone / cube screening
# ============================================================

def _cube_fully_inside_cone(corners, side_normals, tol=1e-10):
    vals = corners @ side_normals.T   # (8,4)
    return np.all(vals >= -tol)


def _cube_fully_outside_cone(corners, side_normals, tol=1e-10):
    """
    If there exists a cone side plane such that all cube corners lie
    strictly outside that half-space, cube and cone do not intersect.
    """
    vals = corners @ side_normals.T   # (8,4)
    return np.any(np.all(vals < -tol, axis=0))


def _classify_pixels_for_zone(zone_corners, cand_side_normals, tol=1e-10):
    """
    zone_corners:       (8,3)
    cand_side_normals:  (Np,4,3)

    Returns:
        full_inside: (Np,)
        full_outside:(Np,)
        need_exact:  (Np,)
    """
    vals = np.einsum('ic,pjc->pij', zone_corners, cand_side_normals)  # (Np,8,4)

    full_inside = np.all(vals >= -tol, axis=(1, 2))
    full_outside = np.any(np.all(vals < -tol, axis=1), axis=1)
    need_exact = ~(full_inside | full_outside)

    return full_inside, full_outside, need_exact


# ============================================================
# Exact intersection helpers
# ============================================================

def _points_in_cone(pts, side_normals, tol=1e-10):
    vals = pts @ side_normals.T
    return np.all(vals >= -tol, axis=1)


def _segment_plane_intersections_batch(p0, p1, plane_normals, tol=1e-14):
    """
    p0, p1: (M,3)
    plane_normals: (K,3), planes n·x = 0
    """
    d = p1 - p0
    denom = d @ plane_normals.T
    numer = -(p0 @ plane_normals.T)

    valid = np.abs(denom) > tol
    t = np.zeros_like(denom)
    np.divide(numer, denom, out=t, where=valid)

    valid &= (t >= -tol) & (t <= 1.0 + tol)
    t = np.clip(t, 0.0, 1.0)

    pts = p0[:, None, :] + t[:, :, None] * d[:, None, :]
    return pts, valid


def _ray_plane_intersections_batch(ray_dirs, face_normals, face_c, tol=1e-14):
    """
    ray_dirs:     (R,3), x = t u, t >= 0
    face_normals: (F,3), plane n·x + c = 0
    """
    ray_dirs = np.asarray(ray_dirs, dtype=np.float64)
    face_normals = np.asarray(face_normals, dtype=np.float64)
    face_c = np.asarray(face_c, dtype=np.float64)

    denom = ray_dirs @ face_normals.T           # (R,F)
    numer = -face_c[None, :]                    # (1,F), broadcasts over R

    t = np.zeros_like(denom)
    valid = np.abs(denom) > tol
    np.divide(numer, denom, out=t, where=valid)

    valid &= (t >= -tol)
    t = np.maximum(t, 0.0)

    pts = ray_dirs[:, None, :] * t[:, :, None]
    return pts, valid


def cube_cone_intersection_vertices_precomputed(
    corners,
    edge_p0, edge_p1,
    face_normals, face_c,
    ray_dirs, side_normals,
    tol=1e-10
):
    """
    corners      : (8,3)
    edge_p0/p1   : (12,3)
    face_normals : (6,3), precomputed once per projector call
    face_c       : (6,), computed once per zone
    ray_dirs     : (4,3)
    side_normals : (4,3)
    """
    pts_list = []

    # 1) cube corners inside cone
    mask_corners = _points_in_cone(corners, side_normals, tol=tol)
    if np.any(mask_corners):
        pts_list.append(corners[mask_corners])

    # 2) cone apex inside cube
    if _origin_in_cube(face_c, tol=tol):
        pts_list.append(np.zeros((1, 3), dtype=np.float64))

    # 3) cube-edge / cone-side-plane intersections
    pts_ep, valid_ep = _segment_plane_intersections_batch(
        edge_p0, edge_p1, side_normals, tol=tol
    )
    pts_ep = pts_ep.reshape(-1, 3)
    valid_ep = valid_ep.reshape(-1)

    if np.any(valid_ep):
        cand = pts_ep[valid_ep]
        mask = _points_in_cone(cand, side_normals, tol=tol) & \
               _points_in_cube(cand, face_normals, face_c, tol=tol)
        if np.any(mask):
            pts_list.append(cand[mask])

    # 4) cone-edge-ray / cube-face intersections
    pts_rf, valid_rf = _ray_plane_intersections_batch(
        ray_dirs, face_normals, face_c, tol=tol
    )
    pts_rf = pts_rf.reshape(-1, 3)
    valid_rf = valid_rf.reshape(-1)

    if np.any(valid_rf):
        cand = pts_rf[valid_rf]
        mask = _points_in_cone(cand, side_normals, tol=tol) & \
               _points_in_cube(cand, face_normals, face_c, tol=tol)
        if np.any(mask):
            pts_list.append(cand[mask])

    if len(pts_list) == 0:
        return np.empty((0, 3), dtype=np.float64)

    pts = np.concatenate(pts_list, axis=0)
    return _unique_points(pts, tol=tol)


def cube_cone_intersection_volume_precomputed(
    corners,
    edge_p0, edge_p1,
    face_normals, face_c,
    ray_dirs, side_normals,
    tol=1e-10
):
    verts = cube_cone_intersection_vertices_precomputed(
        corners,
        edge_p0, edge_p1,
        face_normals, face_c,
        ray_dirs, side_normals,
        tol=tol
    )

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

    Speedups included:
      - cached HEALPix geometry by NSIDE
      - fixed projector-frame box axes and face normals
      - per-zone face offsets from center + cell size
      - per-zone cube-edge precompute
      - cheap full-inside / full-outside screening
      - batched pixel classification before exact geometry
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
    count_map = np.zeros(NPIX, dtype=np.float64)

    if Nz == 0:
        return final_map, count_map

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

    zone_volume = dxyz.prod(axis=0)
    zone_emission = cube * zone_volume

    center_vecs = _normalize(xyz_p, axis=0)                   # (3, Nz)
    corner_vecs = _normalize(corners_rot, axis=0)             # (3, Nz, 8)

    center_T = center_vecs.T                                  # (Nz, 3)
    corner_T = np.transpose(corner_vecs, (1, 2, 0))          # (Nz, 8, 3)
    zone_corners_all = np.transpose(corners_rot, (1, 2, 0))  # (Nz, 8, 3)
    dxyz_T = dxyz.T                                           # (Nz, 3)

    dots = np.sum(corner_T * center_T[:, None, :], axis=2)
    dots = np.clip(dots, -1.0, 1.0)
    circle_radius = np.arccos(dots).max(axis=1)

    # cached per-NSIDE geometry
    pix_center_vecs, pix_boundary_vecs, pix_side_normals = _get_healpix_geometry(NSIDE)

    # fixed projector-frame box axes and face normals
    box_axes, face_normals = _projector_box_axes_and_face_normals(proj_axis)

    if molplot:
        plt.clf()
        m = np.arange(NPIX)
        from healpy.newvisufunc import projview
        projview(m, title='ones', cmap='Reds', projection_type='polar')

    for izone in range(Nz):
        if 1:
            print("s12p Izone/Nzone %d/%d = %0.2f" % (izone, Nz, izone / max(Nz, 1)))

        quantity = zone_emission[izone]
        zone_center_vec = center_vecs[:, izone]
        zone_center = xyz_p[:, izone]
        zone_corners = zone_corners_all[izone]   # (8,3)
        this_zone_volume = zone_volume[izone]

        # per-zone geometry precompute
        edge_p0 = zone_corners[_CUBE_EDGES[:, 0]]
        edge_p1 = zone_corners[_CUBE_EDGES[:, 1]]
        face_c = _cube_face_offsets_from_center_size(
            zone_center, dxyz_T[izone], face_normals, box_axes
        )

        my_pix = hp.query_disc(
            nside=NSIDE,
            vec=zone_center_vec,
            radius=float(circle_radius[izone]),
            inclusive=True
        )

        if my_pix.size == 0:
            continue

        cand_side_normals = pix_side_normals[my_pix]   # (Np,4,3)

        # batched screening
        full_inside, full_outside, need_exact = _classify_pixels_for_zone(
            zone_corners, cand_side_normals
        )

        if np.any(full_inside):
            pix_in = my_pix[full_inside]
            final_map[pix_in] += quantity
            count_map[pix_in] += this_zone_volume

        pix_exact = my_pix[need_exact]
        vsum = float(np.count_nonzero(full_inside)) * this_zone_volume

        for ipix in pix_exact:
            ray_dirs = pix_boundary_vecs[ipix]
            side_normals = pix_side_normals[ipix]

            try:
                # Keep these for robustness near tolerance boundaries
                if _cube_fully_inside_cone(zone_corners, side_normals):
                    intersect_volume = this_zone_volume
                elif _cube_fully_outside_cone(zone_corners, side_normals):
                    intersect_volume = 0.0
                else:
                    intersect_volume = cube_cone_intersection_volume_precomputed(
                        zone_corners,
                        edge_p0, edge_p1,
                        face_normals, face_c,
                        ray_dirs, side_normals
                    )
            except Exception:
                print("Failed on izone=%d ipix=%d" % (izone, ipix))
                print("zone_center_vec =", zone_center_vec)
                print("pix_center      =", pix_center_vecs[ipix])
                print("zone_center     =", zone_center)
                print("zone_corners    =\n", zone_corners)
                print("ray_dirs        =\n", ray_dirs)
                print("face_normals    =\n", face_normals)
                print("face_c          =\n", face_c)
                raise

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

