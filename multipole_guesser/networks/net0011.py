import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
import datetime
import healpy as hp
import random
import os
import matplotlib.pyplot as plt
idd = 11
what = "New reduced net for memorizing"

def thisnet():
# create a healpix grid of nside_out
    nside_out = 8
    M = 12 * nside_out**2
    pix_idx = np.arange(M)
    theta_pix, phi_pix = hp.pix2ang(nside_out, pix_idx)  # in radians
    pix_theta_phi = torch.tensor(np.vstack([theta_pix, phi_pix]).T, dtype=torch.float32)  # [M,2]

# toy inputs
    B = 2
    N = 1000   # your large input
    Fe = 1

    model = main_net(nside=nside_out, output_dim=8)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs  = 8000
    lr = 1e-4
    batch_size=4 #net 8
    trainer(model,data,parameters,validatedata,validateparams,epochs=epochs,lr=lr,batch_size=batch_size)

# ---------------------------
# Dataset with input normalization
# ---------------------------
class SphericalDataset(Dataset):
    """
    data:  (N, 3, T)  with rows [theta, phi, rm]
    label: (N, D_out)
    We normalize:
      - theta in [0,pi]  -> scaled to [-1,1] via (theta/pi)*2-1
      - phi   in [0,2pi] -> scaled to [-1,1] via (phi/pi)-1
      - rm    -> standardized by train mean/std (computed here across full tensor)
    """
    def __init__(self, data, targets, rm_mean=None, rm_std=None, fit_stats=False):
        assert data.ndim == 3 and data.size(1) == 3
        self.data = data.clone()
        self.targets = targets.clone()

        theta = self.data[:, 0, :]
        phi   = self.data[:, 1, :]
        rm    = self.data[:, 2, :]

        # rm standardize
        if rm_mean is None or rm_std is None or fit_stats:
            rm_mean = rm.mean()
            rm_std  = rm.std().clamp_min(1e-6)
        rm_scaled = (rm - rm_mean) / rm_std

        self.data[:, 0, :] = theta
        self.data[:, 1, :] = phi
        self.data[:, 2, :] = rm

        self.rm_mean = rm_mean
        self.rm_std  = rm_std

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed=8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lrs.append(base_lr * (step / max(1, self.warmup_steps)))
            else:
                p = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lrs.append(base_lr * 0.5 * (1.0 + math.cos(math.pi * min(p,1.0))))
        return lrs

# ---------------------------
# Train / Eval
# ---------------------------
def trainer(
    model,
    train_data, train_targets,
    val_data,   val_targets,
    *,
    epochs=200,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    warmup_frac=0.05,
    device=None,
    plot_path=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()



    # Fit normalization on train set only
    ds_train = SphericalDataset(train_data, train_targets, fit_stats=True)
    ds_val   = SphericalDataset(val_data,   val_targets,
                                rm_mean=ds_train.rm_mean, rm_std=ds_train.rm_std, fit_stats=False)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=max(64, batch_size), shuffle=False, drop_last=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    print("Total Steps", total_steps)
    warmup_steps = int(warmup_frac * total_steps)
    #scheduler = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_val = float("inf")
    best_state = None
    patience = 25
    bad_epochs = 0

    train_curve, val_curve = [], []
    t0 = time.time()
    verbose=False

    for epoch in range(1, epochs+1):
        model.train()
        if verbose:
            print("Epoch %d"%epoch)
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            if verbose:
                print("  model")
            #with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            if 1:
                preds = model(xb)
                if verbose:
                    print("  crit")
                loss  = model.criterion(preds, yb)

            if verbose:
                print("  scale backward")
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if verbose:
                print("  steps")
            scaler.step(optimizer)
            scaler.update()
            #scheduler.step()

            running += loss.item() * xb.size(0)

        train_loss = running / len(ds_train)
        train_curve.append(train_loss)

        # validate
        if verbose:
            print("  valid")
        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                vloss = model.criterion(preds, yb)
                vtotal += vloss.item() * xb.size(0)
            val_loss = vtotal / len(ds_val)
            val_curve.append(val_loss)

        # early stopping
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # progress line
        now = time.time()
        time_per_epoch = (now - t0) / epoch
        secs_left = time_per_epoch * (epochs - epoch)
        etad = datetime.datetime.fromtimestamp(now + secs_left)
        eta = etad.strftime("%H:%M:%S")
        nowdate = datetime.datetime.fromtimestamp(now)
        #lr = scheduler.get_last_lr()[0]
        lr = lr

        print(f"[{epoch:3d}/{epochs}] train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr:.2e} | bad {bad_epochs:02d} | ETA {eta}")
        if nowdate.day - etad.day != 0:
            print('tomorrow')

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best val {best_val:.4f}.")
            print('disabled')
            #break

    # restore best
    #if best_state is not None:
    #    model.load_state_dict(best_state)

    # quick plot (optional)
    if plot_path:
        plt.clf()
        plt.plot(train_curve, label="train")
        plt.plot(val_curve,   label="val")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)

    return model, {"train": train_curve, "val": val_curve, "best_val": best_val}

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# PyG / scatter imports
from torch_geometric.nn import GraphConv, global_mean_pool, knn_graph
from torch_scatter import scatter_add

import torch
import torch.nn as nn
import torch.nn.functional as F
import healpy as hp

# ---- Healpix pooling layer ----
class HealpixSampler(nn.Module):
    def __init__(self, nside, in_features):
        super().__init__()
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.in_features = in_features

    def forward(self, x):
        # x: [B, N, in_features] where in_features includes theta, phi, RM
        B,  F, N= x.shape
        theta = x[:, 0, :]
        phi = x[:,  1, :]
        vals = x[:,  2:, :]

        # Map each (theta, phi) to healpix pixel index
        pix = hp.ang2pix(self.nside, theta.cpu().numpy(), phi.cpu().numpy())

        # Mean pool per pixel
        pooled = torch.zeros(B, self.npix, F - 2, device=x.device)
        counts = torch.zeros(B, self.npix, device=x.device)

        for b in range(B):
            for i in range(N):
                p = pix[b, i]
                pooled[b, p] += vals[b,0, i]
                counts[b, p] += 1

        counts[counts == 0] = 1  # avoid div by zero
        pooled /= counts.unsqueeze(-1)

        return pooled  # [B, npix, in_features - 2]

# ---- Simple GNN block ----
class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        # x: [B, N, F]
        # adj: [N, N] adjacency matrix
        agg = torch.matmul(adj, x)  # [B, N, F]
        out = self.lin(agg)
        out = self.norm(out)
        return F.relu(out)

# ---- Main model ----
class main_net(nn.Module):
    def __init__(self, nside, input_dim=3, hidden_dim=64, output_dim=100):
        super().__init__()
        self.idd = idd
        self.sampler = HealpixSampler(nside, input_dim)
        self.gnn1 = GraphConv(input_dim - 2, hidden_dim)
        self.gnn2 = GraphConv(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

        # Learnable output scale
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

        # Build Healpix adjacency
        self.adj = self._build_healpix_adj(nside)
        self.l1 = nn.L1Loss()

    def _build_healpix_adj(self, nside):
        npix = hp.nside2npix(nside)
        adj = torch.zeros(npix, npix)
        for pix in range(npix):
            neighbors = hp.get_all_neighbours(nside, pix)
            for nb in neighbors:
                if nb >= 0:
                    adj[pix, nb] = 1
        # Normalize adjacency
        deg = adj.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1
        adj = adj / deg
        return adj

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, N, 3]  (theta, phi, RM)
        pooled = self.sampler(x)  # [B, npix, 1]
        h = self.gnn1(pooled, self.adj.to(x.device))
        h = self.gnn2(h, self.adj.to(x.device))
        out = self.readout(h.mean(dim=1))  # global mean pooling
        return out * self.output_scale
    def criterion(self,guess,target):
        return self.l1(guess,target)

