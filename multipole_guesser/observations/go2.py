# pip install astroquery astropy healpy matplotlib pandas
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# -------------------------------
# 1) Fetch full NVSS RM catalog (~37k sources)
# -------------------------------
if 'nvss' not in dir():
    nvss_list = Vizier.get_catalogs('J/ApJ/702/1230')
    nvss = nvss_list[0]  # get the main table
    print(len(nvss))

# -------------------------------
# 2) Fetch SDSS DR16Q quasar catalog
# -------------------------------
if not os.path.exists('dr16q.fits'):
    print('getting sdss, will take a minute')
    v = Vizier(columns=['RA_ICRS','DE_ICRS'])
    v.ROW_LIMIT = -1  # fetch all rows
    tables = v.get_catalogs('VII/289/dr16q')
    dr16q = tables[0]
    #dr16q.write('dr16q.hdf5')
    dr16q_small = dr16q['RAJ2000', 'DEJ2000']
    dr16q_small.write('dr16q.fits', overwrite=True)
else:
    from astropy.table import Table
    dr16q = Table.read('dr16q.fits')

ra = nvss['RAJ2000'].data.data # .data gives a plain ndarray
dec = nvss['DEJ2000'].data.data
#mask = ~np.isnan(ra) & ~np.isnan(dec)
#ra = ra[mask]
#dec = dec[mask]


# -------------------------------
# 3) Cross-match NVSS sources to SDSS quasars (<=15 arcsec)
# -------------------------------
if 1:
    if 'nvss_coords' not in dir():
        nvss_coords = SkyCoord(nvss['RAJ2000'], nvss['DEJ2000'], unit=(u.hourangle, u.deg))
    if 'qso_coords' not in dir():
        qso_coords  = SkyCoord(dr16q['RAJ2000'].data, dr16q['DEJ2000'].data, unit(u.deg,u.deg))

    if 1:
        delta = 120*u.arcsec
        delta = 2*u.arcmin
        idx_qso, idx_nvss, sep2d, _ = qso_coords.search_around_sky(nvss_coords, delta)
        print(f"Found {len(idx_qso)} matches")
    if 0:
        idx, sep2d, _ = qso_coords.match_to_catalog_sky(nvss_coords)
        mask = sep2d <= 90*u.arcsec
        qso_nvss = nvss[idx[mask]]
        qso_nvss = qso_nvss[~qso_nvss['RM'].mask]  # drop missing RM
        print(f"Matched quasar RMs: {len(qso_nvss)}")

if 1:
    qso_to_nvss = defaultdict(list)
    nvss_to_qso = defaultdict(list)
    nmultq=0
    nmultr=0

    for q_idx, n_idx in zip(idx_qso, idx_nvss):
        qso_to_nvss[q_idx].append(n_idx)
        nvss_to_qso[n_idx].append(q_idx)
        if len(qso_to_nvss[q_idx])>1:
            nmultq+=1
        if len(nvss_to_qso[q_idx])>1:
            nmultr+=1
    print("overlaps",nmultq, nmultr)

# -------------------------------
# 4) Prepare coordinates for Mollweide plot
# -------------------------------

    to_plot = nvss
    q_coords = SkyCoord(to_plot['RAJ2000'], to_plot['DEJ2000'], unit=(u.hourangle, u.deg), frame='icrs').galactic
    l = np.remainder(q_coords.l.wrap_at(180*u.deg).radian, 2*np.pi) - np.pi  # [-pi, pi]
    b = q_coords.b.radian
    rm = to_plot['RM'].astype(float)  # rad/m^2

# -------------------------------
# 5) Plot
# -------------------------------
if 0:
    plt.figure(figsize=(10,5))
    ax = plt.subplot(111, projection='mollweide')
    sc = ax.scatter(l, b, c=rm, s=6, alpha=0.8, cmap='seismic', vmin=-300, vmax=300)
    ax.grid(True)
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.07, fraction=0.05)
    cb.set_label('Rotation measure (rad m$^{-2}$)')
    plt.title('Quasar Rotation Measures (NVSS RM × SDSS DR16Q)')
    plt.tight_layout()
    import os
    plt.savefig('%s/plots/fig'%os.environ['HOME'])
if 0:
    plt.clf()
    plt.hist(rm, bins=100)
    plt.tight_layout()
    plt.savefig('%s/plots/rm_hist'%os.environ['HOME'])
if 0:
    b = sorted(rm)
    y = np.arange(len(b))/len(b)
    plt.clf()
    plt.plot(b,y,marker='*')
    plt.savefig('%s/plots/rm_chist'%os.environ['HOME'])
    

