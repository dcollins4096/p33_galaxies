import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import pdb
plot_dir = "%s/plots"%os.environ['HOME']

idd = 4
what = "Sphereformer take 3"



def thisnet():
    return main_net()

def train(model, data, parameters, validatedata, validateparams):
    epochs = 10000
    lr = 1e-3
    batch_size = 3
    trainer(model, data, parameters, validatedata, validateparams,
            epochs=epochs, lr=lr, batch_size=batch_size)

lr = 1e-3

import bucket

def trainer(model, data, parameters, validatedata, validateparams,
            epochs=1, lr=1e-3, batch_size=10):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs
    )
    seed = 8675309
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_samples = len(data)

    # Precompute caches for training and validation sets
    if not hasattr(bucket, 'xyz_cache_train'):
        print("Precomputing train cache...")
        xyz_cache_train, idx_cache_train = precompute_cache(data, model.k)
        print("Precomputing validation cache...")
        xyz_cache_val, idx_cache_val = precompute_cache(validatedata, model.k)
        bucket.xyz_cache_train, bucket.idx_cache_train = xyz_cache_train, idx_cache_train 
        bucket.xyz_cache_val, bucket.idx_cache_val = xyz_cache_val, idx_cache_val 
    else:
        xyz_cache_train, idx_cache_train = bucket.xyz_cache_train, bucket.idx_cache_train
        xyz_cache_val, idx_cache_val = bucket.xyz_cache_val, bucket.idx_cache_val 

    losslist = []
    t0 = time.time()

    verbose = False
    for epoch in range(epochs):
        if verbose:
            print('Epoch %d'%epoch)
        model.train()
        subset_n = torch.randint(0, num_samples, (batch_size,))

        data_n = data[subset_n]
        param_n = parameters[subset_n]
        xyz_n = xyz_cache_train[subset_n]
        idx_n = idx_cache_train[subset_n]

        if verbose:
            print(' model')
        optimizer.zero_grad()
        output = model(data_n, xyz=xyz_n, idx=idx_n)
        if verbose:
            print(' crit')
        loss = model.criterion(output, param_n)
        if verbose:
            print(' back')
        loss.backward()
        if verbose:
            print(' step')
        optimizer.step()
        scheduler.step()
        tnow = time.time()
        tel = tnow-t0

        if epoch > 0:
            model.eval()
            with torch.no_grad():
                val_output = model(validatedata, xyz=xyz_cache_val, idx=idx_cache_val)
                val_loss = model.criterion(val_output, validateparams)
                losslist.append(val_loss.item())

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
                  (idd,epoch,loss, optimizer.param_groups[0]['lr'],time_remaining, eta, val_loss))
            loss_batch=[]


    # Plot loss curve
    plt.clf()
    plt.plot(losslist, 'k-')
    plt.yscale('log')
    plt.xlabel('Validation Checkpoints')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Training')
    plt.savefig(f'{plot_dir}/errortime_test_cached.png')


def knn(xyz, k):
    """
    xyz: [B, N, 3]
    Returns idx: [B, N, k] with indices of k nearest neighbors
    """
    dist = torch.cdist(xyz, xyz)  # [B, N, N]
    idx = dist.topk(k, largest=False)[1]  # [B, N, k]
    return idx

def precompute_cache(sky, k):
    """
    Precompute xyz and knn indices for the dataset

    sky: [dataset_size, 3, N]
    k: number of neighbors

    Returns:
        xyz_cache: [dataset_size, N, 3]
        idx_cache: [dataset_size, N, k]
    """
    theta = sky[:, 0, :]
    phi = sky[:, 1, :]
    xyz = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=-1)  # [dataset_size, N, 3]

    idx = knn(xyz, k)  # [dataset_size, N, k]
    return xyz, idx

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

    def forward(self, x, xyz, idx):
        """
        x: [B, N, C]
        xyz: [B, N, 3]
        idx: [B, N, k] (precomputed neighbor indices)
        """
        B, N, C = x.shape
        k = self.k

        # Project features
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim)
        k_feat = self.k_proj(x).view(B, N, self.n_heads, self.head_dim)
        v_feat = self.v_proj(x).view(B, N, self.n_heads, self.head_dim)

        # Prepare idx for gathering neighbors with head_dim
        idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.n_heads, self.head_dim)  # [B, N, k, H, D]

        # Gather neighbors for k and v
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

        # Rearrange dims for multihead attention
        q = q.permute(0, 2, 1, 3)               # [B, H, N, D]
        k_neighbors = k_neighbors.permute(0, 3, 1, 2, 4)  # [B, H, N, k, D]
        v_neighbors = v_neighbors.permute(0, 3, 1, 2, 4)  # [B, H, N, k, D]

        # Gather neighbor xyz coordinates
        neighbor_xyz = self.gather_neighbor_xyz(xyz, idx)  # [B, N, k, 3]

        # Relative position encoding bias
        rel_pos = xyz.unsqueeze(2) - neighbor_xyz  # [B, N, k, 3]
        bias_out = self.bias_mlp(rel_pos.view(-1, 3))  # [(B*N*k), H]
        bias_out = bias_out.view(B, N, k, self.n_heads).permute(0, 3, 1, 2)  # [B, H, N, k]

        # Attention scores
        attn_scores = torch.einsum('bhid,bhikd->bhik', q, k_neighbors) / (self.head_dim ** 0.5)
        attn_scores = attn_scores + bias_out
        attn = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        out = torch.einsum('bhik,bhikd->bhid', attn, v_neighbors)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)            # [B, N, C]

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

    def forward(self, x, xyz, idx):
        x = x + self.attn(self.norm1(x), xyz, idx)
        x = x + self.mlp(self.norm2(x))
        return x

class main_net(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64, depth=4, n_heads=8, k=15, num_outputs=15):
        super().__init__()
        self.idd = idd
        self.embed = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([
            SphereformerBlock(embed_dim, n_heads, k) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, num_outputs)
        self.k = k

    def forward(self, sky, xyz=None, idx=None):
        """
        sky: [B, 3, N]
        xyz: precomputed xyz [B, N, 3] or None
        idx: precomputed neighbor idx [B, N, k] or None
        """
        B, C, N = sky.shape
        assert C == 3, "sky should have 3 channels (theta, phi, rm)"

        theta = sky[:, 0, :]
        phi = sky[:, 1, :]
        rm = sky[:, 2, :]

        if xyz is None:
            xyz = torch.stack([
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta)
            ], dim=-1)  # [B, N, 3]

        if idx is None:
            idx = knn(xyz, self.k)  # compute neighbors if no cache

        feats = torch.stack([theta, phi, rm], dim=-1)  # [B, N, 3]
        x = self.embed(feats)

        for blk in self.blocks:
            x = blk(x, xyz, idx)

        x = self.norm(x)
        x = x.mean(dim=1)  # global mean pooling
        out = self.fc_out(x)
        return out

    def criterion(self, pred, target):
        return F.l1_loss(pred, target)
