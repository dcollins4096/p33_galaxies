
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
    if verbose: print('corners')
    corners = xyz+shifter*dxyz

    if verbose: print('make phi theta')
    corners_persp,phi_persp,theta_persp=proj.make_phi_theta(corners, proj_axis)

    #if verbose: print('rotate')
    xyz_p = proj.rotate(xyz,proj_axis)

    #the thing to accumulate
    rrr2 =  (xyz_p**2).sum(axis=0).flatten()
    zone_volume = dxyz.prod(axis=0).flatten()
    zone_emission = cube/rrr2*zone_volume

    #take out the periodic wrap in phi
    phi_min = phi_persp.min(axis=1)
    phi_max = phi_persp.max(axis=1)
    dphi = phi_max-phi_min
    the_seam = (dphi > np.pi/2)
    shift = the_seam.astype('float')
    shift.shape = shift.size,1
    shift = (shift * (phi_persp<np.pi/2)) * 2*np.pi
    phi_persp = phi_persp + shift

    #THE POLE.  Might not be right still.
    #the pole zones have the x-axis passing through them, and y and z corners
    #must cross zero.
    x_min = corners_persp[0,...].min(axis=1)
    print(x_min.shape)
    x_max = corners_persp[0,...].max(axis=1)
    y_min = corners_persp[1,...].min(axis=1)
    y_max = corners_persp[1,...].max(axis=1)
    z_min = corners_persp[2,...].min(axis=1)
    z_max = corners_persp[2,...].max(axis=1)

    pole_zone = ( (y_min <= 0 ) * ( y_max >= 0 ) * (z_min <= 0 ) * (z_max >= 0 ) ) 
    z = np.cos(theta_persp)
    pole_zone_also = (np.abs(np.cos(theta_persp))>2/3).any(axis=1)
    pole_zone += pole_zone_also
    north_pole = pole_zone * (x_min > 0)
    south_pole = pole_zone * (x_min < 0)

    #draw circles around each zone.
    phi_cen=phi_persp.mean(axis=1)
    theta_cen=theta_persp.mean(axis=1)
    phi_cen.shape=phi_cen.size,1
    theta_cen.shape=theta_cen.size,1

    distance = np.sqrt((phi_persp-phi_cen)**2 + (theta_persp-theta_cen)**2)
    circle_radius = np.max(distance,axis=1)


    #not quite working right
    pix_theta_all = np.zeros([NPIX,4])
    pix_phi_all = np.zeros([NPIX,4])
    for ipix in np.arange(NPIX):
        this_xyz = hp.boundaries(NSIDE, ipix, step=1)
        this_theta, this_phi = hp.vec2ang(this_xyz.T)
        pix_theta_all[ipix,...]=this_theta
        pix_phi_all[ipix,...]=this_phi
        #xyz_pix_boundaries[ipix,...] = this_xyz.T
    #I bet we can speed it up.
    #pix_theta_all, pix_phi_all = hp.vec2ang(xyz_pix_boundaries.T)


    if molplot:
        plt.clf()
        m = np.arange(NPIX)
        from healpy.newvisufunc import projview, newprojplot
        projview(m, title='ones',  cmap='Reds', projection_type='polar')
    verbose2=True
    for izone in range(Nz):
        if verbose2:
            print("Izone/Nzone %d/%d = %0.2f"%(izone,Nz, izone/Nz), )

        quantity = zone_emission[izone] #q*V/r^2
        all_theta = theta_persp[izone]
        all_phi   = phi_persp[izone]

        if pole_zone[izone]:
            if south_pole[izone]:
                this_r = np.pi-all_theta
            else:
                this_r = all_theta

            all_horz = this_r*np.cos(all_phi)
            all_vert = this_r*np.sin(all_phi)
        else:
            all_horz = all_theta
            all_vert = all_phi


        all_corners = np.stack([all_horz,all_vert])
        corner_poly = Polygon(all_corners.T)
        hull = shapely.convex_hull(corner_poly)
        edge_theta, edge_phi = hull.exterior.xy
        #the polygon comes out closed.
        edge_theta = np.array(edge_theta)[:-1]
        edge_phi = np.array(edge_phi)[:-1]



        #the 2d polygon for intersecting
        #How much of this can we vectorize?
        #zone_poly = np.stack([edge_theta,edge_phi])
        zone_poly_points = np.stack([edge_theta,edge_phi])
        zone_poly = Polygon(zone_poly_points.T)
        zone_area = zone_poly.area
        zone_column_density = quantity/zone_area

        #compute the zone center in cartesian
        #This can be vectorized, move out of the loop.
        this_cen_theta= theta_cen[izone]
        this_cen_phi= phi_cen[izone]
        center_x = np.sin(this_cen_theta)*np.cos(this_cen_phi)
        center_y = np.sin(this_cen_theta)*np.sin(this_cen_phi)
        center_z = np.cos(this_cen_theta)
        center = np.stack([center_x,center_y,center_z])


        if pole_zone[izone]:
            edge_theta_theta=np.sqrt(edge_theta**2+edge_phi**2)
            if south_pole[izone]:
                edge_theta_theta=np.pi-edge_theta_theta
            edge_theta_phi = np.arctan2(edge_phi,edge_theta)
        else:
            edge_theta_theta=edge_theta
            edge_theta_phi=edge_phi

#       if molplot and south_pole[izone] and False:
#           if 0:
#               print('-----')
#               frmat = " %8.5f"*8
#               print(frmat%tuple(corners_persp[0,izone,:]))
#               print(frmat%tuple(corners_persp[1,izone,:]))
#               print(frmat%tuple(corners_persp[2,izone,:]))

#           #hp.projscatter( this_cen_theta, this_cen_phi, c='r')
#           #hp.projscatter(edge_theta, edge_phi, c='orange')
#           #newprojplot(this_cen_theta,this_cen_phi, marker='o')
#           #newprojplot(theta=this_cen_theta,phi=this_cen_phi, marker="o", color="r", markersize=10)
#           #newprojplot(theta=all_theta,phi=all_phi,marker="o",color="g",markersize=10)
#           edge_theta_theta=np.sqrt(edge_theta**2+edge_phi**2)
#           if south_pole[izone]:
#               edge_theta_theta=np.pi-edge_theta_theta
#           edge_theta_phi = np.arctan2(edge_phi,edge_theta)
#           #newprojplot(theta=edge_theta_theta,phi=edge_theta_phi,marker='o',color='b',markersize=10)

#           #pdb.set_trace()


        #the all important pixel query
        my_pix = hp.query_disc(nside=NSIDE,vec=center,radius=circle_radius[izone], inclusive=True)

        #code that works to find boundaries.
        if verbose2:
            print("N pixels %d"%len(my_pix))
        for ipix in my_pix:
            xyz = hp.boundaries(NSIDE, ipix, step=1)
            pix_theta, pix_phi = hp.vec2ang(xyz.T)
            #almost.
            #pix_theta = pix_theta_all[ipix]
            #pix_phi   = pix_phi_all[ipix]

            if False:
                newprojplot(theta=edge_theta_theta,phi=edge_theta_phi,marker='o',color='b',markersize=1)
                newprojplot(theta=pix_theta,phi=pix_phi,marker='o',color='g',markersize=1)


            #picking up some strange things that get returned.
            if pix_theta.min() > all_theta.max():
                continue
            if pix_theta.max() < all_theta.min():
                continue
            #check for convexity
            #We can do a rough cut on phi to see if we actually need to do this.
            #https://stackoverflow.com/questions/471962/how-do-i-efficiently-determine-if-a-polygon-is-convex-non-convex-or-complex
            zxprod = np.zeros(pix_phi.size)
            x=pix_theta
            y=pix_phi
            #print('theta',theta)
            #print('phi',phi)
            #if zprod changes sign, its not convex
            #print('ipix',ipix)
            if (ipix > 3 and ipix < NPIX - 4) and not pole_zone[izone]:
                #don't have to do this, only when phi is within a healpix zone of 0 or 2 pi.
                #This can be vectorized at this level, get rid of this k loop.
                for k in np.arange(zxprod.size)-2:
                    dx1 = x[k+1]-x[k]
                    dy1 = y[k+1]-y[k]
                    dx2 = x[k+2]-x[k+1]
                    dy2 = y[k+2]-y[k+1]
                    zxprod[k] = dx1*dy2 - dy1*dx2

                if np.abs(zxprod).sum() - np.abs(zxprod.sum()) > 0:
                    #Healpix pixels always have positive area, unless they straddle \phi=0,2pi.
                    #Then they're always negative except one point, and by jumping the first point before 
                    #the positive point by 2pi fixes the convexity.

                    #FIGURE OUT HOW TO AVOID np.where
                    nneg=(zxprod<0).sum()
                    if nneg == 3:
                        positive=np.where(zxprod>0)[0][0]
                        shifter = positive-1
                    if nneg == 1:
                        shifter = np.argmin(pix_phi)
                        
                    if pix_phi[shifter]  > np.pi:
                        pix_phi[shifter] -= 2*np.pi
                        if edge_phi.min()>np.pi:
                            pix_phi += 2*np.pi
                    else:
                        pix_phi[shifter] += 2*np.pi
                        if edge_phi.min()<np.pi:
                            pix_phi -= 2*np.pi

                #sew up the seam.
                if edge_phi.min() > pix_phi.max():
                    pix_phi += 2*np.pi

            #if ipix== 764:
            #    pdb.set_trace()
            if pole_zone[izone]:
                if south_pole[izone]:
                    this_r = np.pi-pix_theta
                else:
                    this_r = pix_theta

                pix_horz = this_r*np.cos(pix_phi)
                pix_vert = this_r*np.sin(pix_phi)
            else:
                pix_horz = pix_theta
                pix_vert = pix_phi
            #if ipix== 764 and pole_zone[izone]:
            #    pdb.set_trace()


            #intersect.
            #We can probably move the poly outside the zone loop, make an array of polygons.
            #faster but more memory.
            ray_poly_points = np.stack([pix_horz,pix_vert])
            ray_poly = Polygon(ray_poly_points.T)
            try:
                intersection = ray_poly.intersection(zone_poly)
            except:
                fig,ax=plt.subplots(1,1)
                ax.plot(edge_theta_theta, edge_theta_phi, c='r')
                ax.plot(pix_theta, pix_phi, c='g')
                #ax.plot(*zone_poly.exterior.xy, c='r')
                #ax.plot(*ray_poly.exterior.xy, c='g')
                fig.savefig('%s/waste'%plot_dir)
                raise
            area = intersection.area

            #fill the final array
            #the final quantity = q*V/A*intersection/r^2
            count_map[ipix] += 1
            net_light=area*zone_column_density
            final_map[ipix] += net_light

            if moreplots:
                #save this plotting until the caps are perfect.
                fig,ax=plt.subplots(1,1)
                ax.plot(*zone_poly.exterior.xy)
                ax.plot(*ray_poly.exterior.xy)
                if hasattr(intersection, 'exterior'):
                    ax.plot(*intersection.exterior.xy)
                    ax.set(title="izone %d ipix %d %0.2e"%(izone,ipix,intersection.area))
                else:
                    ax.set(title='empty')
                ttt,ppp=ray_poly.exterior.xy
                for i in range(len(zxprod)):
                    ax.text(ttt[i],ppp[i],'%0.2e'%zxprod[i])

                ax.set(xlim=[-np.pi,2*np.pi], ylim=[-np.pi,3*np.pi])
                ax.plot([0,0,np.pi,np.pi,0],[0,2*np.pi,2*np.pi,0,0])
                #ax.plot(theta,phi)
                ##ax.plot(edge_theta,edge_phi)
                #pdb.set_trace()
                prefix = '%s/intersector_ipix_%d_'%(plot_dir,ipix)
                nplot = len(glob.glob(prefix+"*"))
                fig.savefig(prefix+"%03d"%nplot)
                plt.close(fig)

    if molplot:
        #finish plotting
        prefix='%s/moltest'%plot_dir
        nplots = len(glob.glob(prefix+"*"))
        plt.savefig(prefix+"%03d"%nplots)
    return final_map, count_map
        
