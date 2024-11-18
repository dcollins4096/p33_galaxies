
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
    #mask out closes zones
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
    

    #
    # Get corners, project
    #
    #this shift puts them in an order that makes sense to draw lines
    shifter = np.array([[[-0.5, -0.5, +0.5, +0.5,  0.5, +0.5, -0.5, -0.5]],
                        [[-0.5, -0.5, -0.5, -0.5,  0.5, +0.5, +0.5, +0.5]],
                        [[-0.5,  0.5, +0.5, -0.5, -0.5, +0.5, +0.5, -0.5]]])

    dxyz.shape = 3,Nz,1
    xyz.shape = 3,Nz,1

    if verbose: print('corners')
    corners = xyz+shifter*dxyz

    #we'll need this later.
    #Check for the zones whose projections will be squares.
    #Zones with only 4 corners in projection are those that contain the origin
    #along one coordiate. Thus, for zones whose unrotated coordinates have a mix of
    #positive and negative values will only have 4 corners as seen from the origin at 0.
    #four_corners = (np.abs(corners).sum(axis=2) - np.abs(corners.sum(axis=2)) >0).any(axis=0)
    four_corners = (np.abs(corners).sum(axis=2) - np.abs(corners.sum(axis=2)) >0).sum(axis=0) > 1

    #this can be streamlined, taken out of make_phi_theta
    if verbose: print('rotate')
    xyz_p = proj.rotate(xyz,proj_axis)

    #the thing to accumulate
    rrr2 =  (xyz_p**2).sum(axis=0).flatten()
    zone_volume = dxyz.prod(axis=0).flatten()
    zone_emission = cube/rrr2*zone_volume
    zone_emission = cube



    #the orthographic projection is used to determine the exterior corners.
    if verbose: print('work')
    #cor_p = proj.rotate(corners, proj_axis)
    #corners_oblique, phi_oblique, theta_oblique = proj.obliqueproj(xyz_p, cor_p)
    corners_persp,phi_persp,theta_persp=proj.make_phi_theta(corners, proj_axis)
    if 1:
        phi_min = phi_persp.min(axis=1)
        phi_max = phi_persp.max(axis=1)
        dphi = phi_max-phi_min
        the_seam = dphi > np.pi
        shift = the_seam.astype('float')
        shift.shape = shift.size,1
        shift = (shift * (phi_persp<np.pi)) * 2*np.pi
        phi_persp = phi_persp + shift

    if 1:
        xyz_p.shape = 3,Nz
        phi_cen=phi_persp.mean(axis=1)
        theta_cen=theta_persp.mean(axis=1)
        phi_cen.shape=phi_cen.size,1
        theta_cen.shape=theta_cen.size,1

        #Decide if we're using perspective or orth projections
        if 0:
            theta_use=theta_oblique
            phi_use = phi_oblique
            distance_oblique = (phi_oblique-phi_cen)**2 + (theta_oblique-theta_cen)**2
            distance = distance_oblique
        else:
            theta_use=theta_persp
            phi_use = phi_persp
            distance_persp = np.sqrt((phi_persp-phi_cen)**2 + (theta_persp-theta_cen)**2)
            distance = distance_persp
        #distance = np.maximum(distance_oblique, distance_persp)

        if verbose: print('more sort')
        asrt_distance = np.argsort(distance,axis=1)
        circle_radius = np.max(distance,axis=1)


    if molplot:
        #temp plot stuff
        plt.clf()
        m = np.arange(NPIX)
        hp.mollview(m, title="Mollview image RING")
        #hp.projscatter(theta_persp.flatten(),phi_persp.flatten(), c='r')
    #From here, this needs to be in cython.
    #import s2p_loop 
    #return s2p_loop.zone_loop(Nz, zone_emission, theta_persp, phi_persp, NPIX,NSIDE, final_map)
    nnn=0
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


        #healpy can't deal if there are colinear points.
        #Compute the area for every set of three points.  
        #If there are colinear points, reject the middle one.
        if 0:
            ntheta=len(edge_theta)
            arrr=np.zeros(ntheta)
            for it in np.arange(ntheta+1)-3:
                arrr[it+1] = edge_theta[it  ]*(edge_phi[it+2]-edge_phi[it+1])+\
                             edge_theta[it+1]*(edge_phi[it+0]-edge_phi[it+2])+\
                             edge_theta[it+2]*(edge_phi[it+1]-edge_phi[it+0])
            #print("ARRR", arrr)
            keep = np.abs(arrr)>np.abs(arrr).sum()*0.05
            edge_theta=edge_theta[keep]
            edge_phi = edge_phi[keep]

        if 0:
            #check for convexity
            #https://stackoverflow.com/questions/471962/how-do-i-efficiently-determine-if-a-polygon-is-convex-non-convex-or-complex
            zxprod = np.zeros(edge_phi.size)
            x=edge_theta
            y=edge_phi
            #if zprod changes sign, its not convex
            for k in np.arange(zxprod.size)-2:
                dx1 = x[k+1]-x[k]
                dy1 = y[k+1]-y[k]
                dx2 = x[k+2]-x[k+1]
                dy2 = y[k+2]-y[k+1]
                zxprod[k] = dx1*dy2 - dy1*dx2
            if np.abs(zxprod).sum() - np.abs(zxprod.sum()) > 0:
                print("Convex Structure, FIX ME VERY BAD")
                #continue

        if 0:
            #compute the polygon in 3d
            xyzpoly = astropy.coordinates.spherical_to_cartesian(1,np.pi/2-edge_theta,edge_phi)
            xyzpoly = np.array(xyzpoly)
            poly = np.array(xyzpoly).T
        if 1:
            #poly = hp.ang2vec(edge_theta,edge_phi)
            polyx = np.sin(edge_theta)*np.cos(edge_phi)
            polyy = np.sin(edge_theta)*np.sin(edge_phi)
            polyz = np.cos(edge_theta)
            poly = np.stack([polyx,polyy,polyz]).T
        if 1:

            this_cen_theta= theta_cen[izone]
            this_cen_phi= phi_cen[izone]
            center_x = np.sin(this_cen_theta)*np.cos(this_cen_phi)
            center_y = np.sin(this_cen_theta)*np.sin(this_cen_phi)
            center_z = np.cos(this_cen_theta)
            center = np.stack([center_x,center_y,center_z])

        if molplot:
            hp.projscatter( this_cen_theta, this_cen_phi, c='r')
            hp.projscatter(edge_theta, edge_phi, c='orange')



        #check for degenerate corners.  
        #for some reason I'm still getting a degenerate corner when I don't think I should.
        #https://healpix.sourceforge.io/html/Healpix_cxx/healpix__base_8cc_source.html line 1000
        if 0:
            degenerate=False
            for i in np.arange(poly.shape[0])-2:
                normal = np.cross(poly[i], poly[i+1])
                hnd = np.dot(normal, poly[i+2])
                #print('hnd',hnd)
                if np.abs(hnd) < 1e-10:
                    print("Degenerate Corner FIX ME",i)
                    degenerate=True
            #if degenerate:
            #    continue

        #the all important pixel query

        zone_column_density=-1e6
        try:
            #my_pix = hp.query_polygon(NSIDE,poly, inclusive=True)
            my_pix = hp.query_disc(nside=NSIDE,vec=center,radius=circle_radius[izone], inclusive=True)
            #we'll also need this.  Can be streamlined.
            zone_poly = np.stack([edge_theta,edge_phi])
            q = Polygon(zone_poly.T)
            zone_area = q.area
            zone_column_density = quantity/zone_area
        except:
            if molplot:
                hp.projscatter(edge_theta,edge_phi, c='r')
            pdb.set_trace()
            print("MISSED A BAD CORNER")


        #plot the points we'll use.
        if moreplots:
            fig,axes=plt.subplots(1,2,figsize=(12,8))
            ax1=axes[0];ax2=axes[1]#yeah that's right.
            #ax1.plot(phi_persp[izone],theta_persp[izone])
            #ax1.plot(edge_phi, edge_theta,c='r')
            #ax1.set(ylim=[np.pi,0], xlim=[np.pi, -np.pi])
            #ax1.plot(edge_phi, edge_theta,c='orange')
            ax1.scatter(edge_phi, edge_theta,c='orange')
            ax1.set(title='pix = %d'%izone)
            ax1.set(xlabel='phi',ylabel='theta')



            ntheta=len(edge_theta)
            if 1:
                for it in np.arange(ntheta)-2:
                    area = edge_theta[it  ]*(edge_phi[it+1]-edge_phi[it+2])+\
                           edge_theta[it+1]*(edge_phi[it+2]-edge_phi[it+0])+\
                           edge_theta[it+2]*(edge_phi[it+0]-edge_phi[it+1])
                    ttt=nar([edge_theta[it],edge_theta[it+1],edge_theta[it+2],edge_theta[it]])
                    ppp=nar([edge_phi[it],edge_phi[it+1],edge_phi[it+2],edge_phi[it]])
                    ax1.plot( ppp,ttt)
                    ax1.text(ppp[:-1].mean(), ttt[:-1].mean(), "%d %0.2f"%(it,area))
                    if np.abs(area) < 0.02:
                        print('fu2')
                    print('fu', area)
                #print(area)
            #ax1.scatter(phi_cen[izone],theta_cen[izone],c='r')
            #d = distance[izone]
            #for nd, dd in enumerate(d):
            #    ax1.text(phi_use[izone][nd], theta_use[izone][nd],"%0.3e"%dd)

            prefix = '%s/cubeproj'%plot_dir
            nplot = len(glob.glob(prefix+"*"))
            fig.savefig(prefix+"%03d"%nplot)
            plt.close(fig)
            #pdb.set_trace()
        
            #plot the polygon
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(*poly.T)
            ax.scatter3D(*poly.T)
            for p in poly:
                b = np.stack([[0,0,0],p])
                ax.plot3D(*b.T)
            for psi2 in np.linspace(0,90,5):
                ax.view_init(elev=27,azim=psi2)
                prefix = '%s/3d'%plot_dir
                nplot = len(glob.glob(prefix+"*"))
                fig.savefig(prefix+"%03d"%nplot)
            plt.close(fig)


        if molplot and False:
            #temp plot stuff
            plt.clf()
            m = np.arange(NPIX)
            m[my_pix]=m.max()
            hp.mollview(m, title="Mollview image RING")
            hp.projscatter(edge_theta,edge_phi, c='r')

        #code that works to find boundaries.
        print(len(my_pix))
        for ipix in my_pix:
            ######yes this works
            xyz = hp.boundaries(NSIDE, ipix, step=1)
            theta, phi = hp.vec2ang(xyz.T)
            dphi = phi.max()-phi.min()
            #shift
            phi += ((dphi > np.pi/2)+the_seam[izone])*(phi < np.pi/2)*2*np.pi

            ray_poly = np.stack([theta,phi])
            p = Polygon(ray_poly.T)
            intersection = p.intersection(q)
            area = intersection.area

            #the final quantity = q*V/A*intersection/r^2
            net_light=area*zone_column_density
            final_map[ipix] += net_light
            #final_map[ipix] += area
            #final_map[ipix] = max(final_map[ipix],p.area)
            #final_map[ipix] = max(final_map[ipix],q.area)
            #final_map[ipix] += q.area
            #final_map[ipix] = max( final_map[ipix], area)
            count_map[ipix] += 1
            #final_map[ipix] = 1

            #
            # plotting 
            #
            if molplot and False:
                print('w')

                hp.projscatter(theta,phi,c='k')

            if moreplots or ipix == 0:
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

            #if I decide to roll my own:
            #2.) InZone
            #3.) CrossInPixel
            #4.) CrossInZone
            #5.) Sort both
            #6.) Intersection 1
            #7.) Intersection 2
            #8.) Collect interior points
            #9.) sort interior points clockwise
            #10.) area of interior points
            #11.) Add the right thing to the destination plot.
            ######


    if molplot:
        #finish plotting
        prefix='%s/moltest'%plot_dir
        nplots = len(glob.glob(prefix+"*"))
        plt.savefig(prefix+"%03d"%nplots)
    if 0:
        plt.clf()
        image = final_map+0
        image[image==0]=np.nan
        hp.mollview(image, title="Ones",badcolor='w', bgcolor=[0.5]*3,norm='log')
        for izone in range(Nz):
            color = ['r','orange','yellow','green','blue','violet','cyan','magenta']
            hp.projscatter(theta_persp[izone,:], phi_persp[izone,:], c=color[izone])
        hp.projscatter(theta_cen.flatten(), phi_cen.flatten(),marker='^',c='k')
        prefix='%s/mmage'%plot_dir
        nplots = len(glob.glob(prefix+"*"))
        plt.savefig(prefix+"%03d"%nplots)
    #final_map[1]=0
    return final_map, count_map
        
