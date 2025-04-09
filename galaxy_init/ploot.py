

import yt
import numpy as np
import yt
import sys
import os
import matplotlib.pyplot as plt
import pdb
if 'frame' not in dir():
    frame=0
    if len(sys.argv) > 1:
        frame=int(sys.argv[1])
field = 'density'
cwd = os.getcwd()
basename = cwd.split('/')[-1]
def do(frame):
    #ds = yt.load('DD%04d/DD%04d'%(frame,frame))
    ds = yt.load('TT%04d/time%04d'%(frame,frame))
    print(ds)
    ds.index.print_stats()
    for nax,axis in enumerate([0,2]):
        plot_dir = "%s/plots"%os.environ['HOME']
        print('doit')
        proj = ds.proj(field,axis,method='max')
        print('otherstuff')
        pw = proj.to_pw()

        c=[ds.arr([0.5,0.500244140625],'code_length'),ds.arr([0.5, 0.5], 'code_length')][nax]
        pw.set_center(c)
        #pw.annotate_grids()
        name = "%s_nog_%04d"%(basename,frame)
        #pw.save('%s/%s_'%(plot_dir,name))
        pw.zoom(8)
        pw.save('%s/%s_8'%(plot_dir,name))
        #pw.zoom(8)
        #pw.save('%s/%s_64'%(plot_dir,name))
#print(frame)
do(frame) 


