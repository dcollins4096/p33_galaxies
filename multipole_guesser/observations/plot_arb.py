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

surveys = [['J/ApJ/702/1230', "Taylor et al. (2009).37,543 " , "Taylor09"],
['J/ApJ/728/97'  , "194 Van Eck et al.  (2011); ", "VanEck22"],
['J/ApJS/145/213' ,   "380 RMs from the Canadian Galactic Plane Survey (Brown et al. 2003); " , "Brown03"],
['J/ApJ/663/258' , "148 RMs from the Southern Galactic Plane Survey (Brown et al. 2007); ", "Brown07"],
['J/ApJ/755/21'  ,   "200 RMs near the Large Magellanic Cloud (Gaensler et al. 2005; S. A. Mao 2012, in preparation); ", "Mao2012"],
['J/ApJ/714/1170',   "813 high-latitude RMs (Mao et al.  2010), " , "Mao10"],
['J/ApJ/688/1029',   "60 RMs near the Small Magellanic Cloud (Mao et al.  2008) ", "Mao08"],
['J/ApJ/707/114'             ,   "160 RMs near Centaurus A (Feain et al. 2009); ", "Feain09"],
['J/ApJS/45/97'             ,   "(Simard-Normandin et al. 1981;", "Simard-Normandin81"],
['x'             ,   "Clegg et al. 1992; ", "Clegg92"],
['x'             ,   "Oren & Wolfe 1995; ", "Oren95"],
['x'             ,   "Minter & Spangler 1996; ", "Minter96"],
['x'             ,   "Gaensler et al. 2001", "Gaensler01"]]
#                  "905 RMs from various other observational efforts ", "XXX"
#                  "#extragalactic RMs we use is 40,403."
#doesnt' work :(
#['J/Science/307/610',   "200 RMs near the Large Magellanic Cloud (Gaensler et al. 2005; S. A. Mao 2012, in preparation); ", "Gaensler05"],
#['J/Ap&SS/141/303'             ,   "Broten et al. 1988; ", "Broten88"],


#nvss_list = Vizier.get_catalogs('J/ApJ/702/1230')
#nvss = nvss_list[0]  # get the main table
#print(len(nvss))
v = Vizier()
v.ROW_LIMIT=-1
nvss_list = v.get_catalogs(surveys[9][0])
nvss = nvss_list[0]
print(len(nvss), len(nvss_list))
    



# -------------------------------
# 3) Cross-match NVSS sources to SDSS quasars (<=15 arcsec)
# -------------------------------
if 1:
    #if 'nvss_coords' not in dir():
    #    nvss_coords = SkyCoord(nvss['RAJ2000'], nvss['DEJ2000'], unit=(u.hourangle, u.deg))

# -------------------------------
# 4) Prepare coordinates for Mollweide plot
# -------------------------------

    to_plot = nvss
    if 'RAJ2000' in to_plot.columns:
        q_coords = SkyCoord(to_plot['RAJ2000'], to_plot['DEJ2000'], unit=(u.hourangle, u.deg), frame='icrs').galactic
    if 'GLON' in to_plot.columns:
        q_coords = SkyCoord(to_plot['GLON'], to_plot['GLAT'], unit=(u.deg, u.deg), frame='galactic')
    
    l = np.remainder(q_coords.l.wrap_at(180*u.deg).radian, 2*np.pi) - np.pi  # [-pi, pi]
    b = q_coords.b.radian
    rm = to_plot['RM'].astype(float)  # rad/m^2

if 1:
    #being dumb.
    plt.clf()
    plt.hist(l, histtype='step',color='r')
    plt.hist(b, histtype='step',color='g')
    plt.savefig('%s/plots/EL_B'%os.environ['HOME'])

# -------------------------------
# 5) Plot
# -------------------------------
if 1:
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
if 1:
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
    

