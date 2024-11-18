
from starter2 import *
from scipy.ndimage import gaussian_filter
import pdb
import projector.proj as proj
reload(proj)
import healpy as hp
import astropy
import dtools.davetools as dt
import shapely
from shapely.geometry import Polygon


def project(cube, xyz, dxyz, proj_center, proj_axis,bucket=None, molplot=False, moreplots=False, NSIDE = 4, exclude=1):

    
    verbose=False
    #shift center.
    #mask out closest zones
    proj_center.shape=3,1
    xyz= xyz - proj_center
    rrr = np.sqrt((xyz**2).sum(axis=0))
    nxyz = rrr/dxyz.mean(axis=0)
    mask = nxyz > exclude
    cube = cube[mask]
    xyz= xyz[:,mask]
    dxyz=dxyz[:,mask]
    Nz = cube.size

    #create destination map
    NPIX = hp.nside2npix(NSIDE)
    final_map = np.zeros(NPIX)
    count_map = np.zeros(NPIX)
    

    #this shift puts them in an order that makes sense to draw lines
    shifter = np.array([[[-0.5, -0.5, +0.5, +0.5,  0.5, +0.5, -0.5, -0.5]],
                        [[-0.5, -0.5, -0.5, -0.5,  0.5, +0.5, +0.5, +0.5]],
                        [[-0.5,  0.5, +0.5, -0.5, -0.5, +0.5, +0.5, -0.5]]])

    dxyz.shape = 3,Nz,1
    xyz.shape = 3,Nz,1

    #get the corners of each zone.
    #Get projected theta and phi for each corner.
    #Get the 6 furthest from the center of projection
    #sort them in a clockwise fashion.
    if verbose: print('corners')
    corners = xyz+shifter*dxyz


    #this can be streamlined, taken out of make_phi_theta
    if verbose: print('rotate')
    xyz_p = proj.rotate(xyz,proj_axis)

    #the thing to accumulate
    rrr2 =  (xyz_p**2).sum(axis=0).flatten()
    zone_volume = dxyz.prod(axis=0).flatten()
    zone_emission = cube/rrr2*zone_volume
    zone_emission = cube

    if verbose: print('make phi theta')
    corners_persp,phi_persp,theta_persp=proj.make_phi_theta(corners, proj_axis)
   
    #take out the periodic wrap in phi
    phi_min = phi_persp.min(axis=1)
    phi_max = phi_persp.max(axis=1)
    dphi = phi_max-phi_min
    the_seam = dphi > np.pi
    shift = the_seam.astype('float')
    shift.shape = shift.size,1
    shift = (shift * (phi_persp<np.pi)) * 2*np.pi
    phi_persp = phi_persp + shift

    #draw circles around each zone.
    xyz_p.shape = 3,Nz
    phi_cen=phi_persp.mean(axis=1)
    theta_cen=theta_persp.mean(axis=1)
    phi_cen.shape=phi_cen.size,1
    theta_cen.shape=theta_cen.size,1

    distance_persp = np.sqrt((phi_persp-phi_cen)**2 + (theta_persp-theta_cen)**2)
    distance = distance_persp

    asrt_distance = np.argsort(distance,axis=1)
    circle_radius = np.max(distance,axis=1)


    if molplot:
        plt.clf()
        m = np.arange(NPIX)
        hp.mollview(m, title="Mollview image RING")
    for izone in range(Nz):
        print("Izone/Nzone %d/%d = %0.2f"%(izone,Nz, izone/Nz) )

        quantity = zone_emission[izone] #q*V/r^2
        all_theta = theta_persp[izone]
        all_phi   = phi_persp[izone]


        all_corners = np.stack([all_theta,all_phi])
        corner_poly = Polygon(all_corners.T)
        hull = shapely.convex_hull(corner_poly)
        edge_theta, edge_phi = hull.exterior.xy
        #the polygon comes out closed.
        edge_theta = np.array(edge_theta)[:-1]
        edge_phi = np.array(edge_phi)[:-1]

        #the 2d polygon for intersecting
        zone_poly = np.stack([edge_theta,edge_phi])
        q = Polygon(zone_poly.T)
        zone_area = q.area
        zone_column_density = quantity/zone_area

        #compute the zone center in cartesian
        #We can streamline this, can be moved out of the zone loop.
        this_cen_theta= theta_cen[izone]
        this_cen_phi= phi_cen[izone]
        center_x = np.sin(this_cen_theta)*np.cos(this_cen_phi)
        center_y = np.sin(this_cen_theta)*np.sin(this_cen_phi)
        center_z = np.cos(this_cen_theta)
        center = np.stack([center_x,center_y,center_z])

        if molplot:
            hp.projscatter( this_cen_theta, this_cen_phi, c='r')
            hp.projscatter(edge_theta, edge_phi, c='orange')


        #the all important pixel query
        my_pix = hp.query_disc(nside=NSIDE,vec=center,radius=circle_radius[izone], inclusive=True)

        #code that works to find boundaries.
        print("N pixels %d"%len(my_pix))
        for ipix in my_pix:
            xyz = hp.boundaries(NSIDE, ipix, step=1)
            theta, phi = hp.vec2ang(xyz.T)
            #shift.  Not perfect yet.
            dphi = phi.max()-phi.min()
            phi += ((dphi > np.pi/2)+the_seam[izone])*(phi < np.pi/2)*2*np.pi

            #intersect.
            #We can probably move the poly outside the zone loop, make an array of polygons.
            #faster but more memory.
            ray_poly = np.stack([theta,phi])
            p = Polygon(ray_poly.T)
            intersection = p.intersection(q)
            area = intersection.area

            #fill the final array
            #the final quantity = q*V/A*intersection/r^2
            count_map[ipix] += 1
            net_light=area*zone_column_density
            final_map[ipix] += net_light
            #final_map[ipix] += area
            #final_map[ipix] = max(final_map[ipix],p.area)
            #final_map[ipix] = max(final_map[ipix],q.area)
            #final_map[ipix] += q.area
            #final_map[ipix] = max( final_map[ipix], area)
            #final_map[ipix] = 1


            if moreplots:
                #save this plotting until the caps are perfect.
                print(izone)
                fig,ax=plt.subplots(1,1)
                ax.plot(*q.exterior.xy)
                ax.plot(*p.exterior.xy)
                if hasattr(intersection, 'exterior'):
                    ax.plot(*intersection.exterior.xy)
                    ax.set(title="%d %0.2e"%(ipix,intersection.area))
                else:
                    ax.set(title='empty')
                ax.set(xlim=[-np.pi,2*np.pi], ylim=[-np.pi,3*np.pi])
                ax.plot([0,0,np.pi,np.pi,0],[0,2*np.pi,2*np.pi,0,0])
                #ax.plot(theta,phi)
                ##ax.plot(edge_theta,edge_phi)
                #pdb.set_trace()
                prefix = '%s/intersector_'%plot_dir
                nplot = len(glob.glob(prefix+"*"))
                fig.savefig(prefix+"%03d"%nplot)
                plt.close(fig)

    if molplot:
        #finish plotting
        prefix='%s/moltest'%plot_dir
        nplots = len(glob.glob(prefix+"*"))
        plt.savefig(prefix+"%03d"%nplots)
    return final_map, count_map
        
