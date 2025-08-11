from importlib import reload
import sys
import os
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import loader
reload(loader)

new_model = 0
load_model = 0
train_model = 0
save_model = 0
plot_models = 1

if new_model:
    import networks.net0001 as net
    reload(net)
    model = net.thisnet()

sky, clm = loader.loader('clm_take1.h5',ntrain=80, nvalid=10)

if load_model:
    model.load_state_dict(torch.load("models/test%d.pth"%net.idd))

if train_model:

    t0 = time.time()

    net.train(model,sky['train'],clm['train'], sky['valid'],clm['valid'])

    if save_model:
        oname = "models/test%d.pth"%testnum
        torch.save(model.state_dict(), oname)
        print("model saved ",oname)

    t1 = time.time() - t0
    hrs = t1//3600
    minute = (t1-hrs*3600)//60
    sec = (t1 - hrs*3600-minute*60)#//60
    total_time="%02d:%02d:%02d"%(hrs,minute,sec)


if plot_models:
    err=[]
    delta = []
    fig,ax=plt.subplots(1,3, figsize=(12,4))
    for sss,ccc in zip( sky['test'],clm['test']):
        moo = model(sss.unsqueeze(0))
        err.append( model.criterion(moo,ccc.unsqueeze(0)))
        delta = torch.abs( 1-moo/ccc).detach().numpy()[0]
        ax[1].plot(delta, marker='*')
        ax[2].plot( ccc.detach().numpy(),moo[0].detach().numpy())

    err = torch.tensor(err).detach().numpy()
    ax[0].hist(err)
    ax[0].set(xlabel='Err', ylabel='P(err)')
    ax[1].set(xlabel='Clm',ylabel='Rel Err', yscale='log')
    ax[2].set(xlabel='actual',ylabel='guess')
    fig.tight_layout()
    fig.savefig('%s/plots/errhist_%d'%(os.environ['HOME'],model.idd))


