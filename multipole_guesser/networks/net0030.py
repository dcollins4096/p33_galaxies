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
import loader

idd = 30
what = "3d conv.  L=8.   Learn"

fname = "clm_take16_L=8.h5"
#ntrain = 600
#ntrain = 20
ntrain = 100
#nvalid=3
nvalid=4
def load_data():

    sky, clm, Nell = loader.loader(fname,ntrain=ntrain, nvalid=1)
    return sky, clm, Nell

def thisnet(Nell):

    model = main_net(Nell, hidden_channels=1024, num_encodings=16, fc_layers=[2048,1024,512,256])

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.ntheta = 32
    model.nphi   = 32

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs  = 250
    lr = 1e-4
    batch_size=10 
    lr_schedule=[900]
    trainer(model,data,parameters,validatedata,validateparams,epochs=epochs,lr=lr,batch_size=batch_size, weight_decay=0, lr_schedule=lr_schedule)

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
    def __init__(self, data, targets, rm_mean=None, rm_std=None, fit_stats=False, ntheta=32,nphi=32):
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
        sampler = sampler2d(ntheta,nphi, n_smooth=1)
        self.data = sampler(theta, phi, rm)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        #return self.data[idx], self.targets[idx]
        return self.data[idx], self.targets[idx]

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
    lr_schedule=[900],
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
    ds_train = SphericalDataset(train_data, train_targets, fit_stats=True, ntheta=model.ntheta, nphi=model.nphi)
    ds_val   = SphericalDataset(val_data,   val_targets,
                                rm_mean=ds_train.rm_mean, rm_std=ds_train.rm_std, fit_stats=False,ntheta=model.ntheta, nphi=model.nphi) 

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
        milestones=lr_schedule, #[100,300,600],  # change after N and N+M steps
        gamma=0.1             # multiply by gamma each time
    )


    #scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

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
            #scaler.scale(loss).backward()
            loss.backward()
            #if grad_clip is not None:
                #scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if verbose:
                print("  steps")
            #scaler.step(optimizer)
            optimizer.step()
            #scaler.update()

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

        print(f"[{epoch:3d}/{epochs}] net{idd:d}  train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr:.2e} | bad {bad_epochs:02d} | ETA {eta}")
        if nowdate.day - etad.day != 0:
            print('tomorrow')

        if bad_epochs >= patience and False:
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

import torch
from torch_scatter import scatter_add


from scipy.ndimage import gaussian_filter

class sampler2d(torch.nn.Module):
    def __init__(self, n_theta: int, n_phi: int, fill_value: float = 0.0, device=None, n_smooth=0):
        """
        n_theta: number of bins along theta (colatitude)
        n_phi:   number of bins along phi   (longitude)
        fill_value: value to put in pixels with no contributing rays
        """
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.n_smooth=n_smooth
        self.fill_value = fill_value
        self.M = n_theta * n_phi

        # Precompute grid centers
        theta_edges = torch.linspace(0, torch.pi, n_theta + 1)
        phi_edges   = torch.linspace(0, 2 * torch.pi, n_phi + 1)

        # Bin centers
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        phi_centers   = 0.5 * (phi_edges[:-1] + phi_edges[1:])

        # Meshgrid → [M]
        theta_grid, phi_grid = torch.meshgrid(theta_centers, phi_centers, indexing="ij")
        theta_grid = theta_grid.reshape(-1)
        phi_grid   = phi_grid.reshape(-1)

        if device is not None:
            theta_grid = theta_grid.to(device)
            phi_grid   = phi_grid.to(device)

        self.register_buffer("theta_grid", theta_grid)  # [M]
        self.register_buffer("phi_grid",   phi_grid)    # [M]

    @torch.no_grad()
    def forward(self, theta: torch.Tensor, phi: torch.Tensor, rm: torch.Tensor):
        """
        theta, phi, rm: [B, N]  (radians; θ in [0,π], φ in [0,2π])
        Returns:
          theta_grid: [M]
          phi_grid:   [M]
          rm_mean:    [B, M]
        """
        assert theta.shape == phi.shape == rm.shape, "theta, phi, rm must all be [B, N]"
        B, N = theta.shape
        M = self.M
        device = rm.device

        # Compute integer bin indices
        th_bin = torch.clamp((theta / torch.pi * self.n_theta).long(), 0, self.n_theta - 1)
        ph_bin = torch.clamp((phi / (2*torch.pi) * self.n_phi).long(), 0, self.n_phi - 1)
        pix = th_bin * self.n_phi + ph_bin   # [B, N]

        # Flatten batch with offsets
        batch_offsets = (torch.arange(B, device=device).unsqueeze(1) * M)  # [B,1]
        flat_idx = (pix + batch_offsets).reshape(-1)                       # [B*N]

        # Values
        vals = rm.reshape(-1)

        # Sum and counts
        sum_flat   = scatter_add(vals, flat_idx, dim=0, dim_size=B * M)     # [B*M]
        count_flat = scatter_add(torch.ones_like(vals), flat_idx, dim=0, dim_size=B * M)

        # Mean with safe divide
        mean_flat = sum_flat / count_flat.clamp(min=1)
        mean = mean_flat.view(B, M)
        if self.fill_value != 0.0:
            mean = mean.masked_fill(count_flat.view(B, M) == 0, self.fill_value)

        theta_grid = self.theta_grid.view(self.n_theta, self.n_phi)
        phi_grid = self.phi_grid.view(self.n_theta, self.n_phi)
        mean = mean.view(B,self.n_theta,self.n_phi)
        mean = gaussian_filter(mean, sigma = (0, self.n_smooth,self.n_smooth))
        mean = torch.tensor(mean, dtype=torch.float32)

        #stack.
        theta_exp = theta_grid.unsqueeze(0).expand(B, -1, -1)
        phi_exp   = phi_grid.unsqueeze(0).expand(B, -1, -1)

        out = torch.stack([theta_exp, phi_exp, mean], dim=1)

        return out

def error_real_imag(guess,target):

    L1  = F.l1_loss(guess.real, target.real)
    L1 += F.l1_loss(guess.imag, target.imag)
    return L1


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, num_encoding_functions=4):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions

    def forward(self, x):
        """
        x: [B,3,H,W]  (theta, phi, value)
        Returns: [B,3+2*num_encoding_functions*2,H,W]
        """
        B, C, H, W = x.shape
        device = x.device

        # Make normalized coordinates in [-1,1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        coords = torch.stack([xx, yy], dim=0)  # [2,H,W]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B,2,H,W]

        # Positional encodings (Fourier features)
        encodings = []
        for i in range(self.num_encoding_functions):
            freq = 2.0 ** i * math.pi
            encodings.append(torch.sin(freq * coords))
            encodings.append(torch.cos(freq * coords))
        encodings = torch.cat(encodings, dim=1)  # [B,2*2*num_enc,H,W]

        # Concatenate input with encodings
        return torch.cat([x, encodings], dim=1)


class main_net(nn.Module):
    def __init__(self, Nell, hidden_channels=64, num_encodings=4, fc_layers=[512,1024,512]):
        super().__init__()
    
        ell_list = np.arange(Nell)+1
        num_coeffs = (ell_list+1).sum()
        self.n_coeff = num_coeffs
        self.Nell = Nell
        self.posenc = PositionalEncoding2D(num_encodings)
        
        in_channels = 3 + 2 * 2 * num_encodings  # (θ,φ,value) + PE
        in_channels = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )   
        pool_h, pool_w = 3, 3   # or whatever you choose
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))

# Build FC stack dynamically
        in_dim = hidden_channels * pool_h * pool_w
        dims = [in_dim] + fc_layers + [2 * num_coeffs]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.fc2 = nn.Sequential(*layers)
            
        # Global pooling (so features are [B,hidden_channels])
            
    def forward(self, x):              
        """ 
        x: [B,3,H,W]
        returns: [B,2,n_coeff] as complex
        """
        #x = self.posenc(x)
        feats = self.conv1(x)
        feats = self.pool(feats).flatten(1)  # [B, hidden_channels]

        out = self.fc2(feats)                # [B, 2*n_coeff]
        out = out.view(-1, 2, self.n_coeff)
        out = out[:,0,:] + 1j * out[:,1,:]   # make complex

        return out



    def criterion(self,guess, target, kind="l1"):
        #err = error_spherical(guess,target,self.Nell)
        #err = error_mag_phase(guess,target)
        err = error_real_imag(guess,target)
        return err




