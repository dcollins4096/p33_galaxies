import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from importlib import reload
import pdb
import random
import matplotlib.pyplot as plt
import numpy as np
import plot
import datetime
plot_dir = "%s/plots"%os.environ['HOME']

idd = 501
what = "501 but with out tubes that leave"
time_data=True
next_frame=True

def init_weights_constant(m):
    if isinstance(m, nn.Linear):
        #nn.init.constant_(m.weight, 0.5)
        nn.init.constant_(m.bias, 0.1)

def thisnet():

    hidden_dims = 256,
    conv_channels = 32
    model = main_net(hidden_dims=hidden_dims, conv_channels=conv_channels)
    return model

def train(model,data,parameters, validatedata, validateparams):
    #epochs = 300000
    #epochs  = 50000
    epochs  = 60000
    #epochs = 300
    lr = 1e-3
    batch_size=3
    trainer(model,data,parameters,validatedata,validateparams,epochs=epochs,lr=lr,batch_size=batch_size)

def trainer(model, data,parameters, validatedata,validateparams,epochs=1, lr=1e-3, batch_size=10, test_num=0, weight_decay=None):
    optimizer = optim.AdamW( model.parameters(), lr=lr)
    from torch.optim.lr_scheduler import CyclicLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=epochs
    )
    losses=[]
    a = torch.arange(len(data))
    N = len(data)
    seed = 8675309
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    t0 = time.time()
    minlist=[];meanlist=[];maxlist=[];stdlist=[]
    nframes = 11
    ntubes = data.shape[0]//nframes

    all_pairs = [(tube, frame) for tube in range(ntubes) for frame in range(nframes-1)]
    all_pairs = torch.tensor(all_pairs)
    num_samples = len(all_pairs)

    nsubcycle = 0
    for epoch in range(epochs):
        idxs = torch.randint(0, num_samples, (batch_size,))
        batch_pairs = all_pairs[idxs]
        subset_n  = batch_pairs[:,0]*nframes + batch_pairs[:,1]
        subset_np1 = subset_n + 1

        data_n =  data[subset_n]
        param_n = parameters[subset_n]
        data_np1 =  data[subset_np1]
        param_np1 = parameters[subset_np1]
        optimizer.zero_grad()
        output1=model(data_n, param_n, param_np1)
        loss = model.criterion(output1, data_np1)
        loss.backward()
        optimizer.step()
        scheduler.step()
        tnow = time.time()
        tel = tnow-t0
        if (epoch>0 and epoch%100==0) or epoch==10:
            model.eval()
            all_loss = plot.compute_losses_next(model, validatedata, validateparams)
            validate_losses = all_loss['mean']
            model.train()

            time_per_epoch = tel/epoch
            epoch_remaining = epochs-epoch
            time_remaining_s = time_per_epoch*epoch_remaining
            eta = tnow+time_remaining_s
            etab = datetime.datetime.fromtimestamp(eta)

            if 1:
                hrs = time_remaining_s//3600
                minute = (time_remaining_s-hrs*3600)//60
                sec = (time_remaining_s - hrs*3600-minute*60)#//60
                time_remaining="%02d:%02d:%02d"%(hrs,minute,sec)
            if 1:
                eta = "%0.2d:%0.2d:%0.2d"%(etab.hour, etab.minute, int(etab.second))

            mean = validate_losses.mean()
            std = validate_losses.std()
            mmin = validate_losses.min()
            mmax = validate_losses.max()
            minlist.append(mmin)
            maxlist.append(mmax)
            meanlist.append(mean)
            stdlist.append(std)
           # print("test%d Epoch %d loss %0.2e LR %0.2e time left %8s loss mean %0.2e var %0.2e min %0.2e max %0.2e"%
           #       (idd,epoch,loss, optimizer.param_groups[0]['lr'], time_remaining, mean, std, mmin, mmax))
            print("test%d %d L %0.2e LR %0.2e left %8s  eta %8s loss mean %0.2e var %0.2e min %0.2e max %0.2e"%
                  (idd,epoch,loss, optimizer.param_groups[0]['lr'],time_remaining, eta, mean, std, mmin, mmax))
            loss_batch=[]
    print("Run time", tel)
    plt.clf()
    plt.plot(meanlist,c='k')
    plt.plot(np.array(meanlist)+np.array(stdlist),c='b')
    plt.plot(np.array(meanlist)-np.array(stdlist),c='b')
    plt.plot(minlist,c='r')
    plt.plot(maxlist,c='r')
    plt.yscale('log')
    plt.savefig('%s/errortime_test%d'%(plot_dir,idd))


class main_net(nn.Module):
    def __init__(self, output_length=1000, hidden_dims=(128, 128), conv_channels=32, characteristic=False):
        super().__init__()
        self.idd = idd
        self.next_frame = next_frame
        self.time_data = time_data
        self.output_length = output_length

        # Project 6 input values to a pseudo-spatial format (3 channels)
        #self.fc1 = nn.Linear(5*output_length, 5 * output_length)
        #self.relu1 = nn.ReLU()

        # Conv block 1 (acts on the "3 x output_length" format)
        dil = 1
        kern = 5
        padding = dil*(kern-1)//2
        dil2 = 2
        padding2 = dil2*(kern-1)//2
        dil3 = 4
        padding3 = dil3*(kern-1)//2
        dil4 = 8
        padding4 = dil4*(kern-1)//2
        self.conv1 = nn.Sequential(
            nn.Conv1d(5, conv_channels, kernel_size=kern, padding=padding, dilation=dil),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 5, kernel_size=kern, padding=padding, dilation=dil),
            nn.ReLU()
        )

        # FC block 2: merge spatial info
        if 0:
            self.fc2 = nn.Sequential(
                nn.Linear(3 * output_length, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], 3 * output_length)
            )
        if 0:
            in_dim = 3*output_length
            out_dim = 3*output_length
            layers=[]
            dims = [in_dim] + list(hidden_dims) + [out_dim]
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
                    print('relu')
            self.fc2 = nn.Sequential(*layers)
        if 1:
            in_dim = 5*output_length
            out_dim = 5*output_length
            layers=[]
            dims = [in_dim] + list(hidden_dims) + [out_dim]
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.fc2 = nn.Sequential(*layers)

        # Conv block 2
        dil = 1
        kern = 3
        padding = dil*(kern-1)//2
        self.conv2 = nn.Sequential(
            nn.Conv1d(5, conv_channels, kernel_size=kern, padding=padding, dilation=dil),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 5, kernel_size=kern, padding=padding, dilation=dil)
        )
        dil1 = 2
        kern1 = 3
        padding1 = dil1*(kern1-1)//2
        dil2 = 2
        kern2 = 3
        padding2 = dil2*(kern2-1)//2
        if 1:
            self.conv3a = nn.Sequential(
                nn.Conv1d(5, conv_channels, kernel_size=kern1, padding=padding1, dilation=dil1),
                nn.ReLU())
            self.conv3b = nn.Sequential(
                nn.Conv1d(conv_channels, 2*conv_channels, kernel_size=kern2, padding=padding2, dilation=dil2),
                nn.ReLU())
            self.conv3e = nn.Sequential(
                nn.Conv1d(2*conv_channels, conv_channels, kernel_size=kern2, padding=padding2, dilation=dil2),
                nn.ReLU())
            self.convdone = nn.Sequential(
                nn.Conv1d(conv_channels, 3, kernel_size=kern1, padding=padding1, dilation=dil1)
            )
        self.conv1.apply(init_weights_constant)
        self.conv2.apply(init_weights_constant)
        self.conv3a.apply(init_weights_constant)
        self.conv3b.apply(init_weights_constant)
        self.conv3e.apply(init_weights_constant)
        self.convdone.apply(init_weights_constant)
        self.fc2.apply(init_weights_constant)
        #self.fc1.apply(init_weights_constant)
        self.T = nn.Parameter(torch.eye(3) + 0.01 * torch.randn(3, 3)) 

        self.mse=nn.MSELoss()
        self.l1 = nn.L1Loss()

    def criterion(self,guess,target, initial=None):
        L1 = self.l1(target,guess)
        return L1

    def forward(self, datum_n, p_n, p_np1):
        if len(p_n.shape) == 1:
            p_n = p_n.view(1,p_n.shape[0])
            p_np1 = p_np1.view(1,p_np1.shape[0])
            datum_n=datum_n.view(1,3,self.output_length)
        tn = p_n[:,6].view(-1,1,1)
        tna = tn.expand(-1,1,1000)
        tnp1 = p_np1[:,6].view(-1,1,1)
        tnb = tnp1.expand(-1,1,1000)
        batch_size=p_n.shape[0]
        x = torch.cat([datum_n, tna,tnb], dim=1)  # shape: [5, 1000]
        # FC1 to expand global features into spatial representation
        #x_flat = x.view(x.size(0),-1)
        #x = self.fc1(x_flat)  # (batch_size, 3*output_length)
        #x = self.relu1(x)
        #x = x.view(batch_size, 5, self.output_length)  # shape (B, 3, L)

        # Conv block 1: local patterns
        x = x + self.conv1(x)  # Residual connection

        # FC2 block: reprocess globally
        x_flat = x.view(batch_size, -1)
        x_flat = self.fc2(x_flat)
        x = x_flat.view(batch_size,5, self.output_length)
        #x = self.attn(x)

        # Conv block 2: refine locally again
        x = x + self.conv2(x)
        #conv 3
        #x = x + self.conv3(x)
        x1 = self.conv3a(x)
        x2 = self.conv3b(x1)
        x5 =x1+ self.conv3e(x2)
        z = x[:,:3,:]+self.convdone(x5)
        
        return z  # shape (B, 3, output_length)

