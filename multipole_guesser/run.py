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
    import networks.net0029 as net
    reload(net)
    sky, clm, Nell = net.load_data()
    model = net.thisnet(Nell)
    model.idd = net.idd

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


if plot_models:
    if hasattr(net,'sampler2d'):
        size_args = {'ntheta':32, 'nphi':32}
    else:
        size_args = {'nside':model.nside}
    ds_train = net.SphericalDataset(sky['train'], clm['train'], fit_stats=True, **size_args)
    ds_val   = net.SphericalDataset(sky['valid'], clm['valid'], rm_mean=ds_train.rm_mean, rm_std=ds_train.rm_std, fit_stats=False, **size_args)
    ds_tst   = net.SphericalDataset(sky['test'], clm['test'], rm_mean=ds_train.rm_mean, rm_std=ds_train.rm_std, fit_stats=False, **size_args)
    print('ploot')
    err=[]
    delta = []
    fig,ax=plt.subplots(1,3, figsize=(12,4))
    subset = 'train'
    mmin=20
    mmax=-20
    for n in range(len(sky[subset])):
        rm, this_clm = ds_train[n]
        if 1:
            moo = model(rm.unsqueeze(0))
            err.append( model.criterion(moo,this_clm.unsqueeze(0)))
            #delta = torch.abs( 1-moo/this_clm).detach().numpy()[0]
            #ax[1].plot(delta, marker='*')
            c1, m1 =  this_clm.detach().numpy(),moo[0].detach().numpy()
            args = np.argsort(np.abs(c1.real))
            ax[1].plot(c1.real[args],m1.real[args])
            ax[1].set(title='real')
            args = np.argsort(np.abs(c1.imag))
            ax[2].plot(c1.imag[args],m1.imag[args])
            ax[2].set(title='imag')
            mmin = min([c1.imag.min(), m1.imag.min(), c1.real.min(), m1.real.min()])
            mmax = max([c1.imag.max(), m1.imag.max(), c1.real.max(), m1.real.max()])
        if 0:
            moo = model(rm.unsqueeze(0))
            err.append( model.criterion(moo,this_clm.unsqueeze(0)))
            delta = torch.abs( 1-moo/this_clm).detach().numpy()[0]
            ax[1].plot(delta, marker='*')
            c1, m1 =  this_clm.detach().numpy(),moo[0].detach().numpy()
            args = np.argsort(c1)
            ax[2].plot(c1[args],m1[args])

        #plot_multipole.rmplot( sky[subset][n], rm, clm_model = moo, clm_real = clm, fname = "rm_and_sampled_%04d"%n)
        if 0:
            plot_multipole.rmplot2d(rm[0], rm[1], rm[2], sky[subset][n], clm_model = moo[0], clm_real = this_clm, fname = "rm_and_sampled_%04d"%n)


    err = torch.tensor(err).detach().numpy()
    ax[1].plot([mmin,mmax],[mmin,mmax],c='k')
    ax[2].plot([mmin,mmax],[mmin,mmax],c='k')
    ax[0].hist(err)
    ax[0].set(xlabel='Err', ylabel='P(err)')
    ax[1].set(title="Clm real",xlabel='actual',ylabel='guess')
    ax[2].set(title="Clm imag",xlabel='actual',ylabel='guess')
    fig.tight_layout()
    oname = '%s/plots/errhist_%d'%(os.environ['HOME'],model.idd)
    print(oname)
    fig.savefig(oname)


