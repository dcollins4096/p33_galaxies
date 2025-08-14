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
idd = 10
what = "Net 7 but rewritten for awesome by gpt"

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

    model = main_net(nside_out=nside_out,
                               pix_theta_phi=pix_theta_phi,
                               in_channels=Fe,
                               sampler_hidden=32,
                               sampler_K=64,
                               gnn_hidden=64,
                               gnn_layers=3,
                               graph_k=8,
                               num_coeffs=8)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs  = 8000
    lr = 1e-6
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

# ---------- Learnable sparse Healpix Sampler ----------
class SparseHealpixSampler(nn.Module):
    """
    Sparse learnable sampler that maps LOS -> HEALPix pixels
    by using the K nearest LOS for each pixel and computing
    learned attention weights over those K.
    """
    def __init__(self, in_channels=1, hidden_dim=32, K=64):
        """
        in_channels: number of features per LOS (e.g., RM)
        hidden_dim: hidden dimension for the attention MLP
        K: number of nearest LOS to attend per pixel
        """
        super().__init__()
        self.K = K
        # small MLP to produce a logit per (pixel,LOS) based on (feature, angular_dist)
        self.att_mlp = nn.Sequential(
            nn.Linear(in_channels + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # single logit
        )

    @staticmethod
    def sph_to_cart(theta, phi):
        # theta (colatitude) in [0, pi], phi in [0, 2pi]
        # theta, phi can be (...,) shapes
        st = torch.sin(theta)
        x = st * torch.cos(phi)
        y = st * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)  # (..., 3)

    def forward(self, x, los_theta_phi, pix_theta_phi):
        """
        Args:
            x: [B, N, F] LOS features (F == in_channels)
            los_theta_phi: [B, N, 2] (theta, phi) in radians
            pix_theta_phi: [M, 2] HEALPix pixel centers (theta, phi) in radians

        Returns:
            pooled: [B, M, F] pooled features on HEALPix grid
            neighbor_idx: [B, M, K] indices of chosen LOS per pixel (long)
        """
        B, N, Fe = x.shape
        M = pix_theta_phi.shape[0]
        K = min(self.K, N)

        device = x.device
        # 1) compute cartesian unit vectors
        r_los = self.sph_to_cart(los_theta_phi[...,0], los_theta_phi[...,1])  # [B, N, 3]
        r_pix = self.sph_to_cart(pix_theta_phi[:,0].to(device), pix_theta_phi[:,1].to(device))  # [M, 3]

        # 2) compute pairwise cosines r_los · r_pix for distance; but we do this in a memory-efficient way:
        #    we want, for each batch and each pixel, top-K closest LOS.
        # Compute cos = [B, N, M] via einsum
        # Note: memory for B*N*M can be large if M is large; M is typically small (e.g., nside_out=8 -> M=768).
        cos = torch.einsum('bnc,mc->bnm', r_los, r_pix)  # [B, N, M]
        cos = cos.clamp(-1.0, 1.0)
        # Angular distance d = arccos(cos) -- monotonic wrt cos, so we can use cos for nearest
        # For nearest neighbors, highest cos == smallest angle.
        # 3) For each (B, M) find top-K along N: we need [B, M, K] indices -> permute then topk
        cos_permuted = cos.permute(0, 2, 1)  # [B, M, N]
        topk_vals, topk_idx = torch.topk(cos_permuted, k=K, dim=2)  # both [B, M, K]

        # 4) gather features and distances for chosen neighbors
        # Gather LOS features -> [B, M, K, Fe]
        # prepare indexing to gather efficiently
        batch_idx = torch.arange(B, device=device)[:, None, None]  # [B,1,1]
        # topk_idx is indexes into N
        x_gather = x[batch_idx, topk_idx, :]  # [B, M, K, Fe]
        # compute angular distances for these neighbors
        cos_neighbors = topk_vals  # [B, M, K], these are cos values
        d_neighbors = torch.acos(cos_neighbors.clamp(-1.0, 1.0))  # [B, M, K]

        # 5) compute attention logits from (feature, distance)
        # build feature vector per neighbor: concat feature and distance
        d_feat = d_neighbors.unsqueeze(-1)  # [B, M, K, 1]
        feat = torch.cat([x_gather, d_feat], dim=-1)  # [B, M, K, Fe+1]
        # run MLP: flatten first three dims to run MLP efficiently
        BmK = B * M * K
        feat_flat = feat.view(BmK, Fe + 1)
        logits = self.att_mlp(feat_flat).view(B, M, K)  # [B, M, K]

        # 6) softmax over K (LOS) per pixel, numerically stable
        logits_max, _ = logits.max(dim=2, keepdim=True)
        logits_stable = logits - logits_max
        weights = torch.softmax(logits_stable, dim=2)  # [B, M, K]

        # 7) weighted aggregation
        weights_exp = weights.unsqueeze(-1)  # [B, M, K, 1]
        pooled = torch.sum(weights_exp * x_gather, dim=2)  # [B, M, Fe]

        return pooled, topk_idx  # topk indices may be useful for diagnostics

class main_net(nn.Module):
    def __init__(self,
                 nside_out,
                 pix_theta_phi,      # tensor [M,2] radians (passed at init or forward)
                 in_channels=1,
                 sampler_hidden=32,
                 sampler_K=64,
                 gnn_hidden=64,
                 gnn_layers=3,
                 graph_k=8,
                 num_coeffs=200):
        """
        nside_out: healpix nside for output grid (M = 12*nside_out^2)
        pix_theta_phi: [M,2] tensor of HEALPix pixel centers (radians)
        in_channels: LOS feature dim (e.g., 1)
        sampler_*: sampler hyperparams
        gnn_*: GNN hyperparams
        graph_k: k for knn_graph on pooled nodes
        num_coeffs: final number of spherical harmonic coefficients
        """
        super().__init__()
        self.idd = idd
        self.nside_out = nside_out
        self.pix_theta_phi = pix_theta_phi.clone()  # CPU or device moved later
        self.M = pix_theta_phi.shape[0]

        # sampler
        self.sampler = SparseHealpixSampler(in_channels, sampler_hidden, sampler_K)

        # small input MLP on pooled features
        self.input_proj = nn.Linear(in_channels, gnn_hidden)

        # graph conv layers
        self.convs = nn.ModuleList([GraphConv(gnn_hidden, gnn_hidden) for _ in range(gnn_layers)])

        # output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, num_coeffs)
        )

        self.graph_k = graph_k
        self.l1 = nn.L1Loss()

    @staticmethod
    def sph_to_cart(theta, phi):
        st = torch.sin(theta)
        x = st * torch.cos(phi)
        y = st * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    def forward(self, xxx):
        """
        Args:
            x: [B, N, F] LOS features
            los_theta_phi: [B, N, 2] positions in radians
        Returns:
            out: [B, num_coeffs]
        """
        los_theta_phi = xxx[:,:2,:].transpose(1,2)
        x = xxx[:,2,:].unsqueeze(-1)
        device = x.device
        B, N, Fe = x.shape
        M = self.M
        # ensure pixel centers on same device
        pix_theta_phi = self.pix_theta_phi.to(device)  # [M,2]

        # 1) pooled HEALPix map with learnable sampler
        pooled, neighbor_idx = self.sampler(x, los_theta_phi, pix_theta_phi)  # [B, M, Fe]

        # 2) node embedding projection
        h = F.relu(self.input_proj(pooled))  # [B, M, gnn_hidden]

        # 3) build graph across M nodes per graph, for all B graphs flattened
        # flatten nodes for PyG-style processing: [B*M, gnn_hidden]
        h_flat = h.view(B * M, -1)

        # node positions for knn_graph: repeat pix centers for each batch
        pix_theta = pix_theta_phi[:, 0]  # [M]
        pix_phi = pix_theta_phi[:, 1]    # [M]
        pix_xyz = self.sph_to_cart(pix_theta, pix_phi)  # [M,3]
        # repeat for each batch
        pix_xyz_rep = pix_xyz.unsqueeze(0).repeat(B, 1, 1).view(B * M, 3).to(device)  # [B*M,3]

        # batch vector telling which node belongs to which graph
        batch_vec = torch.repeat_interleave(torch.arange(B, device=device), repeats=M)  # [B*M]

        # construct knn graph among nodes, respecting batch (so only edges within same sample)
        # knn_graph returns edge_index in shape [2, E]
        edge_index = knn_graph(pix_xyz_rep, k=self.graph_k, batch=batch_vec, loop=False)

        # 4) GNN layers
        h_conv = h_flat
        for conv in self.convs:
            h_conv = F.relu(conv(h_conv, edge_index))

        # 5) global pooling per batch graph
        graph_feat = global_mean_pool(h_conv, batch_vec)  # [B, gnn_hidden]

        # 6) output MLP -> spherical harmonic coefficients
        out = self.output_mlp(graph_feat)  # [B, num_coeffs]
        return out
    def criterion(self,guess,target):
        return self.l1(guess,target)


