import jinja2
import numpy as np
nar = np.array
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pdb
import unyt

def flatten(array):
    out = "%s "*len(array)%tuple(array)
    return out

agora = True

Nroot = 64
Nlevels = 8
#For the root grid, the number of actual grids will be lower than Nblocks[0]

#estimate block count
if 1:
    #r = 0.11*unyt.Mpc
    r = 0.04*unyt.Mpc
    dx_finest = 80*unyt.pc
    Nside = 4
    Nblock_finest = r/(dx_finest*Nside)
    print(Nblock_finest)

#Nblocks = [[16,16,16],[16,16,16],[30,30,10],[52,52,4],[100,100,1]]
Nblocks = [[24,24,8],[36,36,6],[68,68,4],[128,128,1]]
#Nblocks=[[128,128,1]]
#Nblocks=[]
for level in range(Nlevels-len(Nblocks)+1):
    Nblocks = [[16,16,16]]+Nblocks
#work = [np.prod(bl) for bl in Nblocks]
#print('work',work)
GalaxyDiameter = 0.05 #Mpc

cm_per_mpc = 3.08567758e+24
if 1:
    Nside = 4

    LengthUnits                = 4.04446e24    # 1.31072 mpc (chosen to get dx = 80 pc on level 8)
    dx_finest_mpc = 80e-6
    dx_root_mpc = dx_finest_mpc*2**Nlevels
    dx_finest_code = 1./16384
if 0:
    Nside = Nroot/Nblocks[0][0] #assume cubic top grid
    dx_finest_mpc = GalaxyDiameter/(Nblocks[-1][0]*Nside)
    dx_root_mpc = dx_finest_mpc*2**Nlevels
    LengthUnits = Nroot*dx_root_mpc*cm_per_mpc
    dx_finest_code = dx_finest_mpc*cm_per_mpc/LengthUnits

L = (LengthUnits*unyt.cm).in_units('kpc')
print("LengthUnits %0.2e %s"%(L,L.units))

kwargs={}
radius={}





SRLeft = np.zeros([Nlevels+1,3])
SRRight = np.zeros([Nlevels+1,3])
#print(this)
#relative to zero being the center.  We'll shift to [0.5,0.5,0.5] as the center later.
block_size = [dx_finest_code*Nside*2**level for level in np.arange(Nlevels,-1,-1)]
print(block_size)
SRLeft[-1] = [-0.5*Nblocks[-1][0]*block_size[-1], -0.5*Nblocks[-1][1]*block_size[-1], 0]
SRRight[-1] = [0.5*Nblocks[-1][0]*block_size[-1],  0.5*Nblocks[-1][1]*block_size[-1], Nblocks[-1][2]*block_size[-1]]
#SRLeft[-2] = [-0.5*Nblocks[-2][0]*block_size[-2], -0.5*Nblocks[-2][1]*block_size[-2],  -1*block_size[-2]]
#SRRight[-2] = [0.5*Nblocks[-2][0]*block_size[-2],  0.5*Nblocks[-2][1]*block_size[-2], 1.5*block_size[-2]]

#SRLeft[-3] = [-0.5*Nblocks[-3][0]*block_size[-3], -0.5*Nblocks[-3][1]*block_size[-3], -0.5*Nblocks[-3][2]*block_size[-3]]
#SRRight[-3] = [0.5*Nblocks[-3][0]*block_size[-3],  0.5*Nblocks[-3][1]*block_size[-3],  0.5*Nblocks[-3][2]*block_size[-3]]

#Nblocks_coarse=[16,16,16]

for level in range(0,Nlevels-0):
    print('poot',level)
    SRLeft[level] = [-0.5*Nblocks[level][0]*block_size[level], -0.5*Nblocks[level][1]*block_size[level], -0.5*Nblocks[level][2]*block_size[level]]
    SRRight[level] = [0.5*Nblocks[level][0]*block_size[level],  0.5*Nblocks[level][1]*block_size[level],  0.5*Nblocks[level][2]*block_size[level]]

SRLeft+=0.5
SRRight+=0.5

Center = 0.5*(SRRight[-1]+SRLeft[-1])



fig,ax=plt.subplots(1,1, figsize=(8,8))
ax.set_aspect('equal')
c=['r','g','b','c','m','y']
allwidth = np.zeros(Nlevels+1)
for level in range(1,Nlevels+1):
    print("====",level)
    print(Nblocks[level])
    print(SRLeft[level] )
    print(SRRight[level])
    width = SRRight[level][0]-SRLeft[level][0]
    height =SRRight[level][2]-SRLeft[level][2]
    allwidth[level]=height
    print(height,width)
    color=c[level%len(c)]
    #color = np.random.random(3)
    print("COLOR",level,color)
    rect = plt.Rectangle( [SRLeft[level][0], SRLeft[level][2]], width, height, fc=color)
    ax.add_patch(rect)
ax.axhline(0.5, c='k',linewidth=0.1)
ax.set(xlim=[0.4,0.6],ylim=[0.4,0.6])
fig.savefig("%s/plots/rectangle.pdf"%(os.environ['HOME']))

kwargs['StaticRefineRegionLeftEdge'] = [flatten(N) for N in SRLeft]
kwargs['StaticRefineRegionRightEdge'] = [flatten(N) for N in SRRight]
kwargs['LengthUnits']=LengthUnits
if agora:

    kwargs['AgoraRestartCenterPosition'] = flatten(Center)
    kwargs['MaximumRefinementLevel']=Nlevels
    fname = 'galaxy_3_agora.enzo'
    template = 'galaxy_template_agora.enzo'
if not agora:
    kwargs['DiskGravityPosition'] = flatten(Center)
    kwargs['GalaxySimulationDiskPosition'] = flatten(Center)
    kwargs['TopGridDimensions'] = flatten( (nar(Nblocks[0])*Nside).astype('int'))
    kwargs['DomainLeftEdge'] = flatten( SRLeft[0])
    kwargs['DomainRightEdge'] = flatten( SRRight[0])
    kwargs['MaximumRefinementLevel']=Nlevels
    fname = 'galaxy_3_isogal.enzo'
    template = 'galaxy_template_isogal.enzo'

    radius["DiskGravityStellarDiskScaleHeightR"]    = 3.5e-3    ## Mpc
    radius["DiskGravityStellarDiskScaleHeightz"]    = 0.325e-3  ## Mpc
    radius["DiskGravityStellarBulgeR"]              = 3.066e-3  ## Mpc
    radius["DiskGravityDarkMatterR"]             = 2.5E-3       ## Mpc
    radius["GalaxySimulationDiskScaleHeightR"]      = 3.5e-3    ## Mpc
    radius["GalaxySimulationDiskScaleHeightz"]      = 0.325e-3  ## Mpc
    radius["GalaxySimulationDiskRadius"]            = 0.2       ## code units; > TruncRadius
    radius["GalaxySimulationTruncationRadius"]      = 0.0312    ## Mpc
    radius["GalaxySimulationGasHaloRotationScaleRadius"]   = 10.0  # kpc
#for feature in radius:
#    radius[feature] = radius[feature]*cm_per_mpc/LengthUnits
kwargs.update(radius)
kwargs['levels']=list(range(Nlevels+1))
#StaticRefineRegionRightEdge
#StaticRefineRegionLevel

kwargs['UserDefinedRootGrid']="2 2 2"


loader=jinja2.FileSystemLoader('.')
env = jinja2.Environment(loader=loader)
template = env.get_template(template)
fname = fname
foutptr = open(fname,'w')
foutptr.write( template.render(**kwargs))
foutptr.close()
