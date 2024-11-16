
from starter2 import *
plt.close('all')
import scipy.stats
import dtools.math.equal_probability_binner as epb
reload(epb)
from scipy.ndimage import gaussian_filter

def make_wire(N):
    dx = 1./N
    x,y,z = np.mgrid[0:1:dx,0:1:dx,0:1:dx]
    out = np.zeros_like(x)
    #out[:,-1,:]=1
    #out[0,-1,:]=1

    if 1:
        out[0,-1,:]=1
        out[-1,-1,:]=1
        out[:,-1,0]=1
        out[:,-1,-1]=1
    out[0,:,0]=1
    out[-1,:,0]=1
    out[-1,:,-1]=1
    out[0,:,-1]=1
    out[:,0,0]=1
    out[:,0,-1]=1
    out[0,0,:]=1
    out[-1,0,:]=1
    if 0:
        for nx in [0,-1]:
            for ny in [0,-1]:
                out[nx,:,ny]=1
        for nx in [0,-1]:
            for ny in [0,-1]:
                out[:,nx,ny]=1
    return out

def make_xyz(cube,flat=False):
    N = cube.shape[0]
    print("cube shape",cube.size)
    dx = 1./N
    x,y,z = np.mgrid[0.5*dx:1+0.5*dx:dx,0.5*dx:1+0.5*dx:dx,0.5*dx:1+0.5*dx:dx]
    if flat:
        x=x.flatten()
        y=y.flatten()
        z=z.flatten()
    xyz = np.stack([x,y,z])
    return xyz, dx*np.ones_like(xyz)

def make_sphere(N):
    dx = 1./N
    x,y,z = np.mgrid[0.5*dx:1+0.5*dx:dx,0.5*dx:1+0.5*dx:dx,0.5*dx:1+0.5*dx:dx]
    cube = np.zeros_like(x)
    r2 = x**2+y**2+z**2
    cube[ r2<0.5**2]=1
    return cube
def make_cube(N):
    dx = 1./N
    x,y,z = np.mgrid[0.5*dx:1+0.5*dx:dx,0.5*dx:1+0.5*dx:dx,0.5*dx:1+0.5*dx:dx]

    cube = np.zeros_like(x)
    c = nar([0.2,0.8,0.4])
    c1 = nar([0.2,0.8,0.4])
    c2 = nar([0.8,0.1,0.4])
    r = 0.1
    #ok = (x-c[0])**2+(y-c[1])**2+(z-c[2])**2 < r**2
    ok1 = (x-c1[0])**2+(y-c1[1])**2 < r**2
    ok2 = (x-c2[0])**2+(y-c2[1])**2 < r**2
    #ok = (y-c[1])**2+(z-c[2])**2 < r**2
    #cube[ok1+ok2] = 1
    cube[ok1]=1
    cube[ok2]=2
    #cube += 10*make_wire(N)
    return cube

def make_cube_full(N, stick_or_sphere=0):

    if stick_or_sphere==0:
        cube = make_cube(N)
    else:
        cube = make_sphere(N)
    xyz, dxyz = make_xyz(cube,flat=True)
    return cube.flatten(), xyz, dxyz

def unit_maker(v):
    return v/np.sqrt((v**2).sum(axis=0))
def checky_monkey(v1,v2, tolerance=1e-8):
    diff = v1-v2
    mag = (diff**2).sum(axis=0)
    aligned = mag < tolerance
    return aligned

def check_aligned(corners,xyz):
    #this does not work.
    c0 = corners[...,0]
    c1 = corners[...,1]
    c2 = corners[...,2]
    c3 = corners[...,3]
    c4 = corners[...,4]
    c01 = unit_maker(c1-c0)
    c12 = unit_maker(c2-c1)
    c34 = unit_maker(c4-c3)
    nhat = unit_maker(xyz)
    aligned = checky_monkey( c01, nhat)
    aligned += checky_monkey( c12, nhat)
    aligned += checky_monkey( c34, nhat)
    pdb.set_trace()
    return aligned


def make_phi_theta(xyz,projax,center=None):
    if center is not None:
        center.shape=(3,1,1,1)
        xyz_p  = xyz-center
    else:
        xyz_p = xyz
    #ensure unit vector
    r_proj = np.sqrt(projax[0]**2 + projax[1]**2 + projax[2]**2)
    projax /= r_proj
    theta = np.arccos(projax[2]) 
    phi = np.arctan2(projax[1],projax[0])
    #now get the rotated coordinate axes note that r hat is unit magnitude by definition here since projax is normalized  
    z_p = projax
    y_p = [-1*np.sin(phi),np.cos(phi),0] 
    x_p = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -1*np.sin(theta)]
    x_new = xyz_p[0]*x_p[0] + xyz_p[1]*x_p[1] + xyz_p[2]*x_p[2]
    y_new = xyz_p[0]*y_p[0] + xyz_p[1]*y_p[1] + xyz_p[2]*y_p[2]
    z_new = xyz_p[0]*z_p[0] + xyz_p[1]*z_p[1] + xyz_p[2]*z_p[2]
    r_new = np.sqrt(x_new**2+y_new**2+z_new**2)
    #theta_new = np.arctan2(np.sqrt(y_new**2 + x_new**2),z_new)
    #phi_new = np.arctan2(y_new, x_new) + np.pi
    ##phi_new = np.arccos(x_new)
    theta_new = np.arccos(x_new/r_new)
    phi_new = np.arctan2(z_new,y_new) + np.pi
    xyz_new = np.stack([x_new,y_new,z_new])
    return xyz_new, phi_new, theta_new

def rotate(xyz,projax,center=None):
    if center is not None:
        center.shape=(3,1,1,1)
        xyz_p  = xyz-center
    else:
        xyz_p = xyz
    #ensure unit vector
    r_proj = np.sqrt(projax[0]**2 + projax[1]**2 + projax[2]**2)
    projax /= r_proj
    theta = np.arccos(projax[2]) 
    phi = np.arctan2(projax[1],projax[0])
    #now get the rotated coordinate axes note that r hat is unit magnitude by definition here since projax is normalized  
    z_p = projax
    y_p = [-1*np.sin(phi),np.cos(phi),0] 
    x_p = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -1*np.sin(theta)]
    x_new = xyz_p[0]*x_p[0] + xyz_p[1]*x_p[1] + xyz_p[2]*x_p[2]
    y_new = xyz_p[0]*y_p[0] + xyz_p[1]*y_p[1] + xyz_p[2]*y_p[2]
    z_new = xyz_p[0]*z_p[0] + xyz_p[1]*z_p[1] + xyz_p[2]*z_p[2]
    r_new = np.sqrt(x_new**2+y_new**2+z_new**2)
    xyz_new = np.stack([x_new,y_new,z_new])
    return xyz_new

def obliqueproj(xyz_p, corner_p):
    N = xyz_p.shape[1]
    r_p = np.sqrt((xyz_p**2).sum(axis=0))
    nhat = xyz_p/r_p
    c_dot_n = (corner_p*nhat).sum(axis=0)
    corner_pp = corner_p - c_dot_n*nhat + xyz_p
    x_new,y_new,z_new = corner_pp
    r_new = np.sqrt((corner_pp**2).sum(axis=0))
    theta_new = np.arccos(x_new/r_new)
    phi_new = np.arctan2(z_new,y_new)
    return corner_pp, phi_new, theta_new

def test1():
    cube = make_cube(128)
    fig,ax=plt.subplots(1,1)
    ax.imshow(cube.sum(axis=0))
    fig.savefig('%s/cube'%plot_dir)

def test2():

    N = 128
    dx = 1./N
    x,y,z = np.mgrid[0:1:dx,0:1:dx,0:1:dx]
    cube = make_cube(N)
    den, bx, by, count= scipy.stats.binned_statistic_2d(x.flatten(),y.flatten(),cube.flatten(),bins=128)
    plt.clf()
    plt.imshow(den)
    plt.savefig('%s/stat'%plot_dir)

def test3():

    projax=1
    N = 128
    Nbins = N*8
    dx = 1./N
    xyz = np.stack(np.mgrid[0:1:dx,0:1:dx,0:1:dx])
    center = nar([0.5,-0.2,0.5])
    center.shape=(3,1,1,1)
    xyz_p = xyz-center
    x2 = xyz_p[0].mean(axis=projax)
    y2 = xyz_p[1].mean(axis=projax)
    z2 = xyz_p[2].mean(axis=projax)
    r = np.sqrt((xyz_p**2).sum(axis=0))
    sl = tuple([slice(None),slice(0,1),slice(None)])
    if 1:
        zt = xyz_p[2]
        rt = r
        yt = xyz_p[1]
        xt = xyz_p[0]
    theta = np.arccos( zt/rt )
    phi   = np.arctan2( yt, xt)

    cube = make_cube(N)
    wire = make_wire(N)
    www, bx, by, count= scipy.stats.binned_statistic_2d(phi.flatten(),theta.flatten(),wire.flatten(),bins=Nbins)
    den, bx, by, count= scipy.stats.binned_statistic_2d(phi.flatten(),theta.flatten(),cube.flatten(),bins=Nbins)
    #den, bx, by, count= scipy.stats.binned_statistic_2d(phi.flatten(),theta.flatten(),phi.flatten(),bins=128)
    #den, bx, by, count= scipy.stats.binned_statistic_2d(xt.flatten(),zt.flatten(),cube.flatten(),bins=128)
    xcen = 0.5*(bx[1:]+bx[:-1])
    ycen = 0.5*(by[1:]+by[:-1])
    xx,yy = np.meshgrid(xcen,ycen,indexing='ij')

    if 1:
        okboth = (~np.isnan(www))*(~np.isnan(den))
        den[okboth] = np.maximum(www[okboth],den[okboth])
        fig,ax=plt.subplots(2,2)
        ax0=ax[0][0];ax1=ax[0][1];ax2=ax[1][0];ax3=ax[1][1]
        ok = ~np.isnan(den)
        nrm = mpl.colors.Normalize(vmin=den[ok][den[ok]>0].min(),vmax=den[ok].max())
        cmap = copy.copy(mpl.cm.get_cmap("Reds"))
        cmap.set_under('w')
        cmap.set_bad('k')
        ax0.pcolormesh(x2,z2,cube.sum(axis=projax),cmap=cmap,norm=nrm)
        ax1.pcolormesh(xx,yy,den, cmap=cmap,norm=nrm)
        #epb.equal_prob(den[ok][den[ok]>0].flatten(),16,ax=ax2)



        fig.tight_layout()
        for a in ax.flatten()[:2]:
            a.set_aspect('equal')

        fig.savefig('%s/prj'%plot_dir)


    if 0:
        #theta tests for sanity
        thetaP = theta.mean(axis=projax)
        phiP   = phi.mean(axis=projax)
        fig,ax=plt.subplots(2,2)
        ax0=ax[0][0];ax1=ax[0][1];ax2=ax[1][0];ax3=ax[1][1]
        p=ax0.pcolormesh(x2,z2,thetaP/np.pi)
        fig.colorbar(p,ax=ax0)
        ax0.set(title=r'$\theta$')
        p=ax1.pcolormesh(x2,z2,phiP/np.pi)
        fig.colorbar(p,ax=ax1)
        ax1.set(title=r'$\phi$')
        p=ax2.pcolormesh(x2,z2,x2)
        ax2.set(title='x')
        fig.colorbar(p,ax=ax2)
        p=ax3.pcolormesh(x2,z2,z2)
        fig.colorbar(p,ax=ax3)
        ax3.set(title='z')
        fig.tight_layout()
        fig.savefig('%s/theta'%plot_dir)

def test5():
    projax=1
    N = 128
    Nbins = N//4
    dx_v = 1./N
    xyz = np.stack(np.mgrid[0:1:dx_v,0:1:dx_v,0:1:dx_v])
    center = nar([0.5,-0.2,0.5])
    center.shape=(3,1,1,1)
    xyz_p = xyz-center
    r = np.sqrt((xyz_p**2).sum(axis=0))
    theta = np.arccos( xyz_p[2]/r )
    phi   = np.arctan2( xyz_p[1], xyz_p[0])
    cube=make_cube(N)
    if 1:
        fig6,ax6=plt.subplots(2,2)
        ax6[0][0].imshow(phi.sum(axis=0))
        ax6[0][1].imshow(phi.sum(axis=1))
        ax6[1][0].imshow(phi.sum(axis=2))
        ax6[1][1].imshow(r.sum(axis=projax))
        fig6.savefig('%s/arg'%plot_dir)


    eps = 1e-15
    minmin_theta = theta.min()-eps
    minmin_phi = phi.min() - eps
    maxmax_theta = theta.max() +eps
    maxmax_phi = phi.max() + eps
    dtheta = (maxmax_theta-minmin_theta)/(Nbins)
    dphi   = (maxmax_phi - minmin_phi)/(Nbins)
    min_theta_bin =((theta - minmin_theta)//dtheta).astype('int')
    min_phi_bin =  ((phi - minmin_phi)//dphi).astype('int')
    i=0;j=0


    H = np.zeros(Nbins**2)
    F = np.zeros(Nbins**2)
    F[:]=np.nan
    index  = (min_theta_bin+i + Nbins*(min_phi_bin +j)).flatten()
    #H[index] += cube.flatten()
    #H[index] += xyz_p[2].flatten()
    cf = cube.flatten()
    print('slooooow')
    for ni,i in enumerate(index):
        H[i]+=cf[ni]
    print('done')
    H.shape=Nbins,Nbins
    not_ok = np.isnan(F)
    #H[not_ok]=F[not_ok]

    theta_bins = np.linspace( minmin_theta, maxmax_theta, Nbins)
    phi_bins = np.linspace( minmin_phi, maxmax_phi, Nbins)
    theta_cen = 0.5*(theta_bins)
    phi_cen = 0.5*(phi_bins)
    coordPhi, coordTheta = np.meshgrid(phi_cen,theta_cen, indexing='ij')


    ok = ~np.isnan(H)
    nrm = mpl.colors.Normalize( vmin=H[ok].min(), vmax=H[ok].max())
    #nrm = mpl.colors.Normalize(vmin=0,vmax=4)
    cmap = copy.copy(mpl.cm.get_cmap("Reds"))
    cmap.set_under('w')
    cmap.set_bad('k')
    fig,axes=plt.subplots(1,2)
    ax0=axes[0]; ax1=axes[1]
    p=ax0.pcolormesh(coordPhi, coordTheta, H, cmap=cmap,norm=nrm )
    fig.colorbar(p,ax=ax0)
    fig.savefig('%s/take5'%plot_dir)

def test4():
    projax=1
    N = 128
    Nbins = N*4
    dx_v = 1./N

    cube=make_cube(N)

    xyz = np.stack(np.mgrid[0:1:dx_v,0:1:dx_v,0:1:dx_v])
    center = nar([0.5,-0.2,0.5])
    center.shape=(3,1,1,1)
    xyz_p = xyz-center
    r = np.sqrt((xyz_p**2).sum(axis=0))
    theta = np.arccos( xyz_p[2]/r )
    phi   = np.arctan2( xyz_p[1], xyz_p[0])

    #get the corners
    got=False
    for sX in [0,-0.5,0.5]:
        for sY in [0,-0.5,0.5]:
            for sZ in [0,-0.5,0.5]:
                shift = np.stack([sX,sY,sZ])*dx_v
                shift.shape=(3,1,1,1)
                xyz_shift = xyz_p+shift
                r_shift = np.sqrt((xyz_shift**2).sum(axis=0))
                this_theta = np.arccos(xyz_shift[2]/r_shift)
                this_phi = np.arctan2(xyz_shift[1],xyz_shift[0])
                if not got:
                    max_theta = this_theta
                    min_theta = this_theta
                    max_phi = this_phi
                    min_phi = this_phi
                    got=True
                else:
                    max_theta = np.maximum(max_theta, this_theta)
                    min_theta = np.minimum(min_theta, this_theta)
                    max_phi   = np.maximum(max_phi, this_phi)
                    min_phi   = np.minimum(min_phi, this_phi)


    #get the bins
    eps = 1e-15
    minmin_theta = min_theta.min()-eps
    minmin_phi = min_phi.min() - eps
    maxmax_theta = max_theta.max() +eps
    maxmax_phi = max_phi.max() + eps
    dtheta = (maxmax_theta-minmin_theta)/(Nbins)
    dphi   = (maxmax_phi - minmin_phi)/(Nbins)
    min_theta_bin =((min_theta - minmin_theta)//dtheta).astype('int')
    min_phi_bin =  ((min_phi - minmin_phi)//dphi).astype('int')
    max_theta_bin =((max_theta - minmin_theta)//dtheta).astype('int')
    max_phi_bin =  ((max_phi - minmin_phi)//dphi).astype('int')
    Ntheta = max_theta_bin-min_theta_bin
    Nphi   = max_phi_bin - min_phi_bin
    Ntheta_max = Ntheta.max()
    Nphi_max = Nphi.max()
    print("Ntheta across, Npphi",Ntheta_max, Nphi_max)
    print("max theta bin",max_theta_bin.max())
    Ntheta_bins=Nbins
    Nphi_bins = Nbins


    H = np.zeros(Nphi_bins*Ntheta_bins)
    F = np.zeros(Nphi_bins*Ntheta_bins)
    F[:]=np.nan
    print("angles", minmin_theta, maxmax_theta, minmin_phi, maxmax_phi)
    theta_bins = np.linspace( minmin_theta, maxmax_theta, Ntheta_bins)
    phi_bins = np.linspace( minmin_phi, maxmax_phi, Nphi_bins)
    theta_cen = 0.5*(theta_bins)
    phi_cen = 0.5*(phi_bins)
    coordPhi, coordTheta = np.meshgrid(phi_cen,theta_cen, indexing='ij')

    r1 = range(Nphi_max)
    r2 = range(Ntheta_max)
    
    cube_ind = np.arange(cube.size)
    cube_flat = cube.flatten()
    #cube_flat = phi.flatten()
    for i in r1:
        for j in r2:
            mask = ((Ntheta >= i)*(Nphi >= j)).flatten()
            index  = (min_theta_bin+i + Ntheta_bins*(min_phi_bin +j)).flatten()
            cube_ind_mask = cube_ind[mask]
            #yes a loop.  Direct indexing fails.  Sorry.
            print('loop',i,j,Ntheta_max)
            mask_ind=index[mask]
            for Hi, Ii in enumerate(cube_ind[mask]):
                H[mask_ind[Hi]] += cube_flat[Ii]
            #loop in one direction
            #for Ii, Hi in enumerate(index[mask]):
            #    H[Hi] += cube_flat[cube_ind[Ii]]
            F[index[mask]]=0
    not_ok = np.isnan(F)
    H[not_ok]=F[not_ok]
    H.shape = Nphi_bins,Ntheta_bins

    H = gaussian_filter(H,2)
    #nrm = mpl.colors.Normalize(vmin=den[ok][den[ok]>0].min(),vmax=den[ok].max())
    ok = ~np.isnan(H)
    nrm = mpl.colors.Normalize( vmin=H[ok].min(), vmax=H[ok].max())
    #nrm = mpl.colors.Normalize(vmin=0,vmax=4)
    cmap = copy.copy(mpl.cm.get_cmap("Reds"))
    cmap.set_under('g')
    cmap.set_bad('k')


    fig,axes=plt.subplots(1,1)
    ax0=axes
    p=ax0.pcolormesh(coordPhi, coordTheta, H, cmap=cmap,norm=nrm )
    fig.colorbar(p,ax=ax0)
    
    fig.savefig('%s/test2'%plot_dir)



if 0:
    #cube = make_cube(N)
    #wire = make_wire(N)
    #www, bx, by, count= scipy.stats.binned_statistic_2d(phi.flatten(),theta.flatten(),wire.flatten(),bins=Nbins)
    #den, bx, by, count= scipy.stats.binned_statistic_2d(phi.flatten(),theta.flatten(),cube.flatten(),bins=Nbins)
    ##den, bx, by, count= scipy.stats.binned_statistic_2d(phi.flatten(),theta.flatten(),phi.flatten(),bins=128)
    ##den, bx, by, count= scipy.stats.binned_statistic_2d(xt.flatten(),zt.flatten(),cube.flatten(),bins=128)
    xcen = 0.5*(bx[1:]+bx[:-1])
    ycen = 0.5*(by[1:]+by[:-1])
    xx,yy = np.meshgrid(xcen,ycen,indexing='ij')

    if 1:
        okboth = (~np.isnan(www))*(~np.isnan(den))
        den[okboth] = np.maximum(www[okboth],den[okboth])
        fig,ax=plt.subplots(2,2)
        ax0=ax[0][0];ax1=ax[0][1];ax2=ax[1][0];ax3=ax[1][1]
        ok = ~np.isnan(den)
        nrm = mpl.colors.Normalize(vmin=den[ok][den[ok]>0].min(),vmax=den[ok].max())
        cmap = copy.copy(mpl.cm.get_cmap("Reds"))
        cmap.set_under('w')
        cmap.set_bad('k')
        ax0.pcolormesh(x2,z2,cube.sum(axis=projax),cmap=cmap,norm=nrm)
        ax1.pcolormesh(xx,yy,den, cmap=cmap,norm=nrm)
        import dtools.math.equal_probability_binner as epb
        #epb.equal_prob(den[ok][den[ok]>0].flatten(),16,ax=ax2)



        fig.tight_layout()
        for a in ax.flatten()[:2]:
            a.set_aspect('equal')

        fig.savefig('%s/prj'%plot_dir)
