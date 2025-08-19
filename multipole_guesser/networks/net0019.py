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
import torch.optim as optim
import pdb
idd = 19
what = "18, but with a big MLP"

def thisnet(Nell):
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
    #Nell = 5
    num_coeffs = Nell**2+2*Nell

    model = main_net(Nside=nside_out, hidden=512, layers=6, normalize_rm=False, Nell=Nell)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs  = 300
    lr = 1e-3
    batch_size=4 #net 8
    trainer(model,data,parameters,validatedata,validateparams,epochs=epochs,lr=lr,batch_size=batch_size, weight_decay=0)

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
    def __init__(self, data, targets, rm_mean=None, rm_std=None, fit_stats=False, nside=8):
        assert data.ndim == 3 and data.size(1) == 3
        self.data = data.clone()
        self.targets = targets.clone()

        rm    = self.data[:, 2, :]

        # rm standardize
        if rm_mean is None or rm_std is None or fit_stats:
            rm_mean = rm.mean()
            rm_std  = rm.std().clamp_min(1e-6)
        rm_scaled = (rm - rm_mean) / rm_std

        self.data[:, 2, :] = rm

        theta = self.data[:,0,:]
        phi = self.data[:,1,:]
        rm  = self.data[:,2,:]

        self.rm_mean = rm_mean
        self.rm_std  = rm_std
        sampler = sampler1(nside)
        self.theta_pix, self.phi_pix, self.rm_pix = sampler(theta, phi, rm)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        #return self.data[idx], self.targets[idx]
        return self.rm_pix[idx], self.targets[idx]

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed=8675309):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    torch.set_num_threads(16)
    #model = torch.nn.DataParallel(model).to('cpu')
    #model.nside = model.module.nside
    #model.criterion = model.module.criterion



    # Fit normalization on train set only
    print('object and pooling')
    ds_train = SphericalDataset(train_data, train_targets, fit_stats=True, nside=model.nside)
    ds_val   = SphericalDataset(val_data,   val_targets,
                                rm_mean=ds_train.rm_mean, rm_std=ds_train.rm_std, fit_stats=False, nside=model.nside)

    print('on we go')
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=max(64, batch_size), shuffle=False, drop_last=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    print("Total Steps", total_steps)
    warmup_steps = int(warmup_frac * total_steps)
    #scheduler = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100,300,600],  # change after N and N+M steps
        gamma=0.1             # multiply by gamma each time
    )


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

            running += loss.item() * xb.size(0)

        scheduler.step()
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
        lr = optimizer.param_groups[0]['lr']

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
    plt.clf()
    plt.plot(train_curve, label="train")
    plt.plot(val_curve,   label="val")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/plots/errtime_net%04d"%(os.environ['HOME'], idd))

    return model, {"train": train_curve, "val": val_curve, "best_val": best_val}

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# PyG / scatter imports
from torch_geometric.nn import GraphConv, global_mean_pool, knn_graph
from torch_scatter import scatter_add

class sampler1(torch.nn.Module):
    def __init__(self, nside: int, fill_value: float = 0.0, device=None):
        """
        nside: HEALPix nside
        fill_value: value to put in pixels with no contributing rays
        """
        super().__init__()
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.fill_value = fill_value

        # Precompute HEALPix pixel centers (same for all batches)
        pix = torch.arange(self.npix, dtype=torch.long)
        th, ph = hp.pix2ang(nside, pix.cpu().numpy())  # radians
        theta_hp = torch.tensor(th, dtype=torch.float32)
        phi_hp   = torch.tensor(ph, dtype=torch.float32)

        if device is not None:
            theta_hp = theta_hp.to(device)
            phi_hp   = phi_hp.to(device)

        # Buffers move with .to(device) and are saved with the module
        self.register_buffer("theta_healpy", theta_hp)   # [M]
        self.register_buffer("phi_healpy",   phi_hp)     # [M]

    @torch.no_grad()
    def forward(self, theta: torch.Tensor, phi: torch.Tensor, rm: torch.Tensor):
        """
        theta, phi, rm: [B, N]  (radians; θ=colatitude, φ=longitude)
        Returns:
          theta_healpy: [M]
          phi_healpy:   [M]
          rm_mean:      [B, M]
        """
        assert theta.shape == phi.shape == rm.shape, "theta, phi, rm must all be [B, N]"
        B, N = theta.shape
        M = self.npix
        device = rm.device

        # Compute pixel indices for each ray in each batch on CPU (healpy requirement), then back to device
        # (If θ/φ are already on CPU, .cpu() is a no-op.)
        pix_np = hp.ang2pix(self.nside, theta.detach().cpu().numpy(), phi.detach().cpu().numpy())  # [B, N]
        pix = torch.as_tensor(pix_np, dtype=torch.long, device=device)  # [B, N]

        # Flatten batch with non-overlapping index ranges per batch: [0..M-1], [M..2M-1], ...
        batch_offsets = (torch.arange(B, device=device).unsqueeze(1) * M)  # [B,1]
        flat_idx = (pix + batch_offsets).reshape(-1)                       # [B*N]

        # Values to pool
        vals = rm.reshape(-1)                                              # [B*N]

        # Sum and counts per (batch,pixel)
        sum_flat   = scatter_add(vals,                flat_idx, dim=0, dim_size=B * M)  # [B*M]
        count_flat = scatter_add(torch.ones_like(vals), flat_idx, dim=0, dim_size=B * M)  # [B*M]

        # Mean with safe divide; fill empty pixels
        mean_flat = sum_flat / count_flat.clamp(min=1)
        mean = mean_flat.view(B, M)                                        # [B, M]
        if self.fill_value != 0.0:
            mean = mean.masked_fill(count_flat.view(B, M) == 0, self.fill_value)

        # Return pixel centers (shared) and per-batch pooled map
        return self.theta_healpy, self.phi_healpy, mean

# Requires: torch, torch_geometric, (optionally) cuda
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, knn_graph

class ComplexHeadConv(nn.Module):
    def __init__(self, hidden_dim: int, num_coeff: int):
        super().__init__()
        self.num_coeff = num_coeff

        # Project features into sequence of length num_coeff
        self.linear_expand = nn.Linear(hidden_dim, num_coeff)

        # Conv1d to produce 2 channels (real + imag) for each coefficient
        self.conv_out = nn.Conv1d(
            in_channels=1, out_channels=2, kernel_size=1
        )

    def forward(self, h: torch.Tensor):
        """
        h: [B, H] or [B, N, H] (after pooling -> [B,H])
        """
        B, H = h.shape

        # Expand to [B, num_coeff]
        coeffs = self.linear_expand(h)        # [B, num_coeff]

        # Add channel dim for conv: [B, 1, num_coeff]
        coeffs = coeffs.unsqueeze(1)

        # Conv1d: [B, 2, num_coeff]
        out = self.conv_out(coeffs)

        # Final output: [B, num_coeff, 2]
        #out = out.permute(0, 2, 1)

        return out

def error_real_imag(guess,target):
    g_real = guess[:,0,:]
    g_imag = guess[:,1,:]
    t_real = target.real
    t_imag = target.imag
    L1  = F.l1_loss(g_real, t_real)
    L1 += F.l1_loss(g_imag, t_imag)
    return L1

def sph_to_cart(theta, phi):
    # theta: colatitude [0,pi], phi: longitude [0,2pi]
    st = torch.sin(theta)
    return torch.stack([st*torch.cos(phi), st*torch.sin(phi), torch.cos(theta)], dim=-1)  # [N,3]

class main_net(nn.Module):
    """
    Input:
      - theta: [N], phi: [N] (fixed across dataset; radians)
      - forward(rm): rm is [B, N]
    Output:
      - [B, num_coeffs]
    """
    def __init__(self,
                 Nside: int=8,
                 Nell: int=3,
                 k: int = 16,
                 hidden: int = 128,
                 layers: int = 4,
                 add_pos_enc: bool = True,
                 dropout: float = 0.0,
                 normalize_rm: bool = True,
                 device: str = None):
        super().__init__()

        self.nside=Nside
        #num_coeffs = Nell**2+2*Nell #no monopole
        num_coeffs = (np.arange(Nell)+1).sum()
        self.num_coeffs = num_coeffs
        self.Nell=Nell
        npix = hp.nside2npix(Nside)
        theta, phi = hp.pix2ang(Nside, np.arange(npix), nest=False)  # RING order
        theta = torch.tensor(theta,dtype=torch.float32)
        phi   = torch.tensor(phi,dtype=torch.float32)
        N = theta.shape[0]

        assert theta.ndim == 1 and phi.ndim == 1 and theta.shape[0] == phi.shape[0]
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Static geometry (buffers move with .to(device)) ---
        theta = theta.to(device)
        phi   = phi.to(device)
        self.register_buffer("theta", theta)
        self.register_buffer("phi",   phi)
        xyz = sph_to_cart(theta, phi)                       # [N,3]
        self.register_buffer("xyz", xyz)

        # KNN graph built once on static positions
        with torch.no_grad():
            edge_index = knn_graph(xyz, k=k, loop=False)    # [2,E]
        self.register_buffer("edge_index", edge_index)

        # Optional positional encodings per node (broadcast to batches)
        pos_dim = 0
        if add_pos_enc:
            pe = torch.stack(
                [torch.sin(theta), torch.cos(theta),
                 torch.sin(phi),   torch.cos(phi)], dim=-1)  # [N,4]
            self.register_buffer("pos_enc", pe)
            pos_dim = pe.shape[-1]
        else:
            self.register_buffer("pos_enc", torch.empty(N,0, device=device))

        self.normalize_rm = normalize_rm

        in_dim = 1 + pos_dim  # RM + (optional) pos features
        widths = [hidden, 2*hidden, 4*hidden, 2*hidden, hidden]
        self.input_proj =nn.Sequential(nn.Linear(in_dim, 2*hidden), nn.ReLU(), nn.Linear(2*hidden, widths[0]))

        # Stacked GraphConv with residuals + LayerNorm
        #self.convs = nn.ModuleList([GraphConv(hidden, hidden) for _ in range(layers)])
        #self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.convs = nn.ModuleList([
                GraphConv(widths[i], widths[i+1]) for i in range(len(widths)-1)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(w) for w in widths[1:]])
        # projection layers for residual when dims differ
        self.res_projs = nn.ModuleList([
                (nn.Linear(widths[i], widths[i+1]) if widths[i] != widths[i+1] else nn.Identity())
                for i in range(len(widths)-1)
        ])


        # Readout → SH coefficients
        #self.head = nn.Sequential(
        #    nn.Linear(hidden, hidden),
        #    nn.ReLU(),
        #    nn.Linear(hidden, num_coeffs)
        #)
        self.head = ComplexHeadConv( hidden_dim=hidden, num_coeff=num_coeffs)

        # Small learnable output scale to keep logits sane early on
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, rm: torch.Tensor):
        """
        rm: [B, N]
        returns: [B, num_coeffs]
        """
        assert rm.ndim == 2 and rm.shape[1] == self.theta.shape[0]
        B, N = rm.shape
        device = rm.device
        edge_index = self.edge_index  # [2,E] on correct device via buffer

        # Optional per-sample normalization (helps stability for wide RM ranges)
        if self.normalize_rm:
            rm_mean = rm.mean(dim=1, keepdim=True)
            rm_std  = rm.std(dim=1, keepdim=True).clamp_min(1e-6)
            rm_norm = (rm - rm_mean) / rm_std
        else:
            rm_norm = rm

        # Build node features: [B,N,1+pos_dim]
        x = rm_norm.unsqueeze(-1)  # [B,N,1]
        if self.pos_enc.numel() > 0:
            pe = self.pos_enc.unsqueeze(0).expand(B, -1, -1)  # [B,N,pos_dim]
            x = torch.cat([x, pe], dim=-1)

        # Flatten to PyG format
        x = self.input_proj(x)              # [B,N,H]
        x = F.relu(x)
        x = x.reshape(B*N, -1)              # [B*N,H]

        # Batch vector maps nodes to graphs
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)  # [B*N]

        # Message passing
        h = x
        h = x   # [B*N, widths[0]]
        for conv, ln, proj in zip(self.convs, self.norms, self.res_projs):
            h_new = conv(h, edge_index)             # [B*N, out_dim]
            # residual: project previous h to out_dim then add
            h = ln(F.relu(h_new) + proj(h))         # residual + norm + activation
            h = self.dropout(h)


        # Global mean pool per graph (sample)
        g = global_mean_pool(h, batch_vec)  # [B,H]

        out = self.head(g) * self.output_scale  # [B,num_coeffs]
        return out

    def criterion(self,guess, target, kind="l1"):
        #err = error_spherical(guess,target,self.Nell)
        #err = error_mag_phase(guess,target)
        err = error_real_imag(guess,target)
        return err




