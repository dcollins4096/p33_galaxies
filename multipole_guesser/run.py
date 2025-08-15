from importlib import reload
import sys
import os
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import loader
import plot_multipole
reload(plot_multipole)
reload(loader)

new_model = 1
load_model = 0
train_model = 1
save_model = 0
plot_models = 1

if new_model:
    import networks.net0015 as net
    reload(net)
    model = net.thisnet()

#sky, clm = loader.loader('clm_take6.h5',ntrain=512, nvalid=3)
#sky, clm = loader.loader('clm_take6.h5',ntrain=4, nvalid=3)
sky, clm = loader.loader('clm_take7_partial.h5',ntrain=20, nvalid=3)

if load_model:
    model.load_state_dict(torch.load("models/test%d.pth"%net.idd))

if train_model:

    t0 = time.time()

    net.train(model,sky['train'],clm['train'], sky['valid'],clm['valid'])

    if save_model:
        oname = "models/test%d.pth"%model.idd
        torch.save(model.state_dict(), oname)
        print("model saved ",oname)

    t1 = time.time() - t0
    hrs = t1//3600
    minute = (t1-hrs*3600)//60
    sec = (t1 - hrs*3600-minute*60)#//60
    total_time="%02d:%02d:%02d"%(hrs,minute,sec)


model.idd = net.idd
if plot_models:
    ds_train = net.SphericalDataset(sky['train'], clm['train'], fit_stats=True, nside=model.nside)
    ds_val   = net.SphericalDataset(sky['valid'], clm['valid'], rm_mean=ds_train.rm_mean, rm_std=ds_train.rm_std, fit_stats=False, nside=model.nside) 
    print('ploot')
    err=[]
    delta = []
    fig,ax=plt.subplots(1,3, figsize=(12,4))
    for n in range(len(sky['train'])):
        if 1:
            rm, clm = ds_train[n]
            moo = model(rm.unsqueeze(0))
            err.append( model.criterion(moo,clm.unsqueeze(0)))
            delta = torch.abs( 1-moo/clm).detach().numpy()[0]
            ax[1].plot(delta, marker='*')
            c1, m1 =  clm.detach().numpy(),moo[0].detach().numpy()
            args = np.argsort(c1)
            ax[2].plot(c1[args],m1[args])

        #plot_multipole.rmplot( sky['train'][n], rm, fname = "rm_and_sampled_%04d"%n)

    err = torch.tensor(err).detach().numpy()
    ax[0].hist(err)
    ax[0].set(xlabel='Err', ylabel='P(err)')
    ax[1].set(xlabel='Clm',ylabel='Rel Err', yscale='log')
    ax[2].set(xlabel='actual',ylabel='guess')
    fig.tight_layout()
    oname = '%s/plots/errhist_%d'%(os.environ['HOME'],model.idd)
    print(oname)
    fig.savefig(oname)


