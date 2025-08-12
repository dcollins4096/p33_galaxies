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

idd = 3
what = "Spherical Transformer but actually runs.  Local attention block"


def thisnet():
    hidden_dims = 256,
    conv_channels = 32
    model = main_net()
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs  = 50000
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
        if epoch > 0: #(epoch>0 and epoch%100==0) or epoch==10:
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
import torch.nn.functional as F

def knn(xyz, k):
    """
    xyz: [B, N, 3]
    Returns idx: [B, N, k] with indices of k nearest neighbors
    """
    dist = torch.cdist(xyz, xyz)  # [B, N, N]
    idx = dist.topk(k, largest=False)[1]  # [B, N, k]
    return idx

class LocalSphereAttention(nn.Module):
    def __init__(self, dim, n_heads, k):
        super().__init__()
        self.n_heads = n_heads
        self.k = k
        self.head_dim = dim // n_heads

        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.bias_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, n_heads)
        )

    def gather_neighbor_xyz(self, xyz, idx):
        """
        Gathers neighbor xyz coordinates using advanced indexing.

        xyz: [B, N, 3]
        idx: [B, N, k]

        Returns:
            neighbor_xyz: [B, N, k, 3]
        """
        B, N, _ = xyz.shape
        k = idx.size(-1)

        batch_indices = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, N, k)  # [B, N, k]
        neighbor_xyz = xyz[batch_indices, idx, :]  # [B, N, k, 3]
        return neighbor_xyz

    def forward(self, x, xyz):
        """
        x: [B, N, C] input features
        xyz: [B, N, 3] input xyz coordinates on sphere

        Returns:
            out: [B, N, C] output features after local attention
        """
        B, N, C = x.shape
        k = self.k

        idx = knn(xyz, k)  # [B, N, k]

        # Project features
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim)  # [B, N, H, D]
        k_feat = self.k_proj(x).view(B, N, self.n_heads, self.head_dim)
        v_feat = self.v_proj(x).view(B, N, self.n_heads, self.head_dim)

        # Gather neighbor k and v features
        # Expand idx dims for gathering
        idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.n_heads, self.head_dim)  # [B, N, k, H, D]

        k_neighbors = torch.gather(
            k_feat.unsqueeze(1).expand(-1, N, -1, -1, -1),  # [B, N, N, H, D]
            2,
            idx_expanded
        )  # [B, N, k, H, D]

        v_neighbors = torch.gather(
            v_feat.unsqueeze(1).expand(-1, N, -1, -1, -1),  # [B, N, N, H, D]
            2,
            idx_expanded
        )  # [B, N, k, H, D]

        # Permute dims for multihead attention computation
        q = q.permute(0, 2, 1, 3)  # [B, H, N, D]
        k_neighbors = k_neighbors.permute(0, 3, 1, 2, 4)  # [B, H, N, k, D]
        v_neighbors = v_neighbors.permute(0, 3, 1, 2, 4)  # [B, H, N, k, D]

        # Gather neighbor xyz coords
        neighbor_xyz = self.gather_neighbor_xyz(xyz, idx)  # [B, N, k, 3]

        # Compute relative positional encodings (bias)
        rel_pos = xyz.unsqueeze(2) - neighbor_xyz  # [B, N, k, 3]
        bias_out = self.bias_mlp(rel_pos.view(-1, 3))  # [(B*N*k), H]
        bias_out = bias_out.view(B, N, k, self.n_heads).permute(0, 3, 1, 2)  # [B, H, N, k]

        # Compute attention scores
        attn_scores = torch.einsum('bhid,bhikd->bhik', q, k_neighbors) / (self.head_dim ** 0.5)
        attn_scores = attn_scores + bias_out
        attn = F.softmax(attn_scores, dim=-1)

        # Aggregate values
        out = torch.einsum('bhik,bhikd->bhid', attn, v_neighbors)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # [B, N, C]

        out = self.o_proj(out)
        return out


class SphereformerBlock(nn.Module):
    def __init__(self, dim, n_heads, k):
        super().__init__()
        self.attn = LocalSphereAttention(dim, n_heads, k)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, xyz):
        x = x + self.attn(self.norm1(x), xyz)
        x = x + self.mlp(self.norm2(x))
        return x

class main_net(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64, depth=4, n_heads=8, k=15, num_outputs=15):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([
            SphereformerBlock(embed_dim, n_heads, k) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, num_outputs)

    def forward(self, sky):
        """
        sky: [B, 3, N] where channels = (theta, phi, rm)
        """
        B, C, N = sky.shape
        assert C == 3, "sky should have 3 channels (theta, phi, rm)"

        theta = sky[:, 0, :]
        phi = sky[:, 1, :]
        rm = sky[:, 2, :]

        xyz = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=-1)  # [B, N, 3]

        feats = torch.stack([theta, phi, rm], dim=-1)  # [B, N, 3]
        x = self.embed(feats)  # [B, N, embed_dim]

        for blk in self.blocks:
            x = blk(x, xyz)

        x = self.norm(x)
        x = x.mean(dim=1)  # global pooling
        out = self.fc_out(x)  # [B, num_outputs]
        return out

    def criterion(self, pred, target):
        return F.l1_loss(pred, target)

