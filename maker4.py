
import yt
from starter2 import *
import pdb
import projector.s2p as s2p
import projector.proj as proj
import healpy as hp
reload(s2p)
reload(proj)

if 0:
    #all sky
    cube, xyz, dxyz = proj.make_cube_full(8, stick_or_sphere=0)
    cube += 0.01*cube.mean()
    cube = np.ones_like(cube)
    Nz = cube.size
    NSIDE = 4
    NPIX = hp.nside2npix(NSIDE)
    image = np.arange(NPIX)
    xyz_center = xyz.mean(axis=1)
    #xyz_center.shape=xyz_center.size,1
    #xyz -= 0.99*xyz_center #not exactly at zero.
    proj_axis=nar([1.,0,0])

    molplot=True
    image=s2p.project(cube,xyz,dxyz,xyz_center,proj_axis,molplot=molplot, NSIDE=128)

    if 0:
        plt.clf()
        hp.mollview(np.log(image), title="Two Sticks")
        prefix='%s/sky'%plot_dir
        nplots = len(glob.glob(prefix+"*"))
        plt.savefig(prefix+"%03d"%nplots)

if 1:
    #horse around
    cube, xyz, dxyz = proj.make_cube_full(8, stick_or_sphere=0)
    #cube += 0.01*cube.mean()
    cube = np.ones_like(cube)
    molplot=False
    cube_center = xyz.mean(axis=1)

    proj_center = cube_center-nar([1e-5,0,0])
    #proj_center = nar([xyz[0].min()-1, cube_center[1],cube_center[2]])
    proj_axis = nar([1.,0.,0])
    image, counts=s2p.project(cube,xyz,dxyz,proj_center,proj_axis,molplot=molplot, NSIDE=16,exclude=0)


    plt.clf()
    image[image==0]=np.nan
    hp.mollview(image, title="Ones",badcolor='w', bgcolor=[0.5]*3,norm='log')
    prefix='%s/image'%plot_dir
    nplots = len(glob.glob(prefix+"*"))
    plt.savefig(prefix+"%03d"%nplots)
    plt.clf()
    hp.mollview(counts, title="counts")
    prefix='%s/counts'%plot_dir
    nplots = len(glob.glob(prefix+"*"))
    plt.savefig(prefix+"%03d"%nplots)

if 0:
    #strafe the camera around the zone
    cube, xyz, dxyz = proj.make_cube_full(8, stick_or_sphere=0)
    cube += 0.01*cube.min()
    cube = np.ones_like(cube)
    molplot=False
    theta = np.linspace(np.pi/7,np.pi*5/6,10)-np.pi/2
    phi   = np.linspace(-np.pi,np.pi,10)
    #theta = theta[9:10] #broken
    theta = theta[0:1]
    phi = phi[0:1]
    cube_center = xyz.mean(axis=1)
    cube_center.shape=cube_center.size,1
    xyz -= cube_center

    r=0.1
    for nph,ph in enumerate(phi):
        for nth,th in enumerate(theta):
            print("Nth, Nph",nph,nth)
            x=r*np.sin(th)*np.cos(ph)
            y=r*np.sin(th)*np.sin(ph)
            z=r*np.cos(th)
            proj_center=nar([x,y,z])
            dcenter = proj_center-cube_center.flatten()
            proj_axis = -dcenter/(dcenter**2).sum()
            bucket={'theta':th,'phi':ph}

            image=s2p.project(cube,xyz,dxyz,proj_center,proj_axis, bucket=bucket,molplot=molplot, NSIDE=128)

            plt.clf()
            hp.mollview(np.log(image), title="Two Sticks")
            prefix='%s/proj_sticks'%plot_dir
            nplots = len(glob.glob(prefix+"*"))
            plt.savefig(prefix+"%03d"%nplots)


if 0:
    ds = yt.load('datasets/IsolatedGalaxy/galaxy0030/galaxy0030')
    #ad = ds.all_data()
    c = ds.arr([0.5]*3, 'code_length')
    r = ds.quan(1./16, 'code_length')
    L = ds.arr([0.5-1./32]*3,'code_length')
    R = ds.arr([0.5+1./32]*3,'code_length')
    dx = ds.index.get_smallest_dx()*32
    Nz = (R-L)/dx
    cg = ds.covering_grid(3,L,Nz)
    ad=cg
    #sphere = ds.sphere(c, r)
    cube_cube = ad['density'].in_units('code_density').v
    cube=cube_cube.flatten()
    space_units='code_length'
    xyz = np.stack([ad['x'].in_units(space_units).v.flatten(),
                    ad['y'].in_units(space_units).v.flatten(),
                    ad['z'].in_units(space_units).v.flatten()])
    dxyz = np.stack([ad['dx'].in_units(space_units).flatten().v,
                     ad['dy'].in_units(space_units).flatten().v,
                     ad['dz'].in_units(space_units).flatten().v])
    if 0:
        density=cg['density']
        print("Nz %0.2e"%density.size)
        fig,ax=plt.subplots(1,1)
        ax.imshow( np.log(density.sum(axis=2)))
        fig.savefig("%s/galaxy_cg"%plot_dir)

    if 0:
        proj=ds.proj('density',2, data_source=sphere)
        pw=proj.to_pw()
        #pw.zoom(16)
        pw.save('%s/galaxy'%plot_dir)

    dx_min = dxyz.min()
    fig,ax=plt.subplots(1,1)
    ax.imshow(np.log(cube_cube.sum(axis=0)))
    fig.savefig('%s/cube'%plot_dir)

    shift = ds.quan(8,'kpc').in_units('code_length').v
    proj_center = np.array([0.5]*3)+np.array([shift,0,0]) + dx_min/128
    proj_axis = np.array([-1.,0,0])
    image=s2p.project(cube,xyz,dxyz,proj_center,proj_axis, NSIDE=128)

    plt.clf()
    hp.mollview(np.log(image), title="Isolated Galaxy")
    prefix='%s/galaxy'%plot_dir
    nplots = len(glob.glob(prefix+"*"))
    plt.savefig(prefix+"%03d"%nplots)


if 0:
    cube, xyz, dxyz = proj.make_cube_full(2)
    #dxyz/=8
    old_center = nar([0.5]*3)
    old_center.shape = old_center.size,1
    new_center = nar([0.0,0.0,0.0])
    new_center.shape=new_center.size,1
    #xyz += new_center-old_center
    proj_center = nar([0.5,0.7,0.6])
    if 0:
        #proj_center = nar([0.5]*3)
        #proj_center = nar([0,0,0])
        #proj_axis   = nar([0,1,0],dtype='float')
        dcenter = proj_center-new_center.flatten()
        proj_axis_tmp= -dcenter/(dcenter**2).sum()
        proj_axis = np.cross(proj_axis_tmp,[0,0,1])
        proj_axis=proj_axis_tmp

    #proj_axis=nar([0,1,1],dtype='float')
    #proj_axis/=(proj_axis**2).sum()



    if 0:
        #rotate the camera, fixed in space
        theta = np.linspace(np.pi/7,np.pi*5/6,10)-np.pi/2
        phi   = np.linspace(-np.pi,np.pi,10)
        for ph in phi:
            for th in theta:
                proj_axis=nar(astropy.coordinates.spherical_to_cartesian(1,th,ph))
                print(proj_axis)
                bucket={'theta':th,'phi':ph}
                corners=s2p.project(cube,xyz,dxyz,proj_center,proj_axis, bucket=bucket)
