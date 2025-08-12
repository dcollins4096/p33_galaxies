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

new_model = 1
load_model = 0
train_model = 1
save_model = 1
plot_models = 1

if new_model:
    import networks.net0003 as net
    reload(net)
    model = net.thisnet()

sky, clm = loader.loader('clm_take4.h5',ntrain=900, nvalid=3)

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
    print('ploot')
    err=[]
    delta = []
    fig,ax=plt.subplots(1,3, figsize=(12,4))
    for sss,ccc in zip( sky['test'],clm['test']):
        moo = model(sss.unsqueeze(0))
        err.append( model.criterion(moo,ccc.unsqueeze(0)))
        delta = torch.abs( 1-moo/ccc).detach().numpy()[0]
        ax[1].plot(delta, marker='*')
        c1, m1 =  ccc.detach().numpy(),moo[0].detach().numpy()
        args = np.argsort(c1)
        ax[2].plot(c1[args],m1[args])

    err = torch.tensor(err).detach().numpy()
    ax[0].hist(err)
    ax[0].set(xlabel='Err', ylabel='P(err)')
    ax[1].set(xlabel='Clm',ylabel='Rel Err', yscale='log')
    ax[2].set(xlabel='actual',ylabel='guess')
    fig.tight_layout()
    oname = '%s/plots/errhist_%d'%(os.environ['HOME'],model.idd)
    print(oname)
    fig.savefig(oname)


