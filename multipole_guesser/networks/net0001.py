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
import datetime
plot_dir = "%s/plots"%os.environ['HOME']

idd = 1
what = "First draft"

def init_weights_constant(m):
    if isinstance(m, nn.Linear):
        #nn.init.constant_(m.weight, 0.5)
        nn.init.constant_(m.bias, 0.1)

def thisnet():

    hidden_dims = 256,
    conv_channels = 32
    model = main_net()
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs  = 1000
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
    losslist=[]

    nsubcycle = 0
    num_samples=len(data)
    for epoch in range(epochs):
        subset_n  = torch.randint(0, num_samples, (batch_size,))

        data_n =  data[subset_n]
        param_n = parameters[subset_n]
        optimizer.zero_grad()
        output1=model(data_n)
        loss = model.criterion(output1, param_n)
        loss.backward()
        optimizer.step()
        scheduler.step()
        tnow = time.time()
        tel = tnow-t0
        if (epoch>0 and epoch%100==0) or epoch==10:
            model.eval()
            mod1 = model(validatedata)
            this_loss = model.criterion(mod1, validateparams)
            losslist.append(this_loss)
            
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

           # print("test%d Epoch %d loss %0.2e LR %0.2e time left %8s loss mean %0.2e var %0.2e min %0.2e max %0.2e"%
           #       (idd,epoch,loss, optimizer.param_groups[0]['lr'], time_remaining, mean, std, mmin, mmax))
            print("test%d %d L %0.2e LR %0.2e left %8s  eta %8s validate loss %0.2e"%
                  (idd,epoch,loss, optimizer.param_groups[0]['lr'],time_remaining, eta, this_loss))
            loss_batch=[]
    print("Run time", tel)
    plt.clf()
    LLL = torch.tensor(losslist).detach().numpy()
    plt.plot(LLL,c='k')
    plt.yscale('log')
    plt.savefig('%s/errortime_test%d'%(plot_dir,idd))


import torch
import torch.nn as nn

class main_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.idd = idd
        
        self.features = nn.Sequential(
            # input: [batch, 3, 5000]
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3),  # -> [batch, 32, 2500]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # -> [batch, 64, 1250]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # -> [batch, 128, 625]
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # -> [batch, 256, 313]
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> [batch, 256, 1]
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),          # -> [batch, 256]
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 15)     # final output
        )
        self.l1 = nn.L1Loss()

    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    def criterion(self,guess,actual):
        return self.l1(guess,actual)

# Example usage
if __name__ == "__main__":
    model = ConvNet3to15()
    example_input = torch.randn(8, 3, 5000)  # batch of 8
    output = model(example_input)
    print(output.shape)  # should be [8, 15]

