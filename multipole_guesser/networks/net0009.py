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
import random
import os
import matplotlib.pyplot as plt
idd = 9
what = "Net 7 but rewritten for awesome by gpt"

def thisnet():

    d_model = 128
    n_heads = 8
    n_layers = 6
    mpl_ratio = 4
    max_ell = 2
    model = main_net(d_model, n_heads, n_layers, sh_max_l = max_ell)
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs  = 300
    lr = 3e-4
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

        # angle scaling to [-1,1]
        theta_scaled = (theta / math.pi) * 2.0 - 1.0
        phi_scaled   = (phi / math.pi) - 1.0

        # rm standardize
        if rm_mean is None or rm_std is None or fit_stats:
            rm_mean = rm.mean()
            rm_std  = rm.std().clamp_min(1e-6)
        rm_scaled = (rm - rm_mean) / rm_std

        self.data[:, 0, :] = theta_scaled
        self.data[:, 1, :] = phi_scaled
        self.data[:, 2, :] = rm_scaled

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
    warmup_steps = int(warmup_frac * total_steps)
    scheduler = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_val = float("inf")
    best_state = None
    patience = 25
    bad_epochs = 0

    train_curve, val_curve = [], []
    t0 = time.time()
    verbose=True

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
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                preds = model(xb)
                print("  crit")
                loss  = model.criterion(preds, yb)

            print("  scale backward")
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            print("  steps")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * xb.size(0)

        train_loss = running / len(ds_train)
        train_curve.append(train_loss)

        # validate
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
        eta = datetime.datetime.fromtimestamp(now + secs_left).strftime("%H:%M:%S")
        print(f"[{epoch:3d}/{epochs}] train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | bad {bad_epochs:02d} | ETA {eta}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. Best val {best_val:.4f}.")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

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

# ---------------------------
# Utilities
# ---------------------------
def sph_to_cart(theta, phi):
    st = torch.sin(theta)
    x = st * torch.cos(phi)
    y = st * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

class SphereTokenEmbed(nn.Module):
    def __init__(self, d_model=128, pos_emb_dim=32, use_learned_pos=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.use_learned_pos = use_learned_pos

        self.token_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        if use_learned_pos:
            self.pos_mlp = nn.Sequential(
                nn.Linear(3, pos_emb_dim),
                nn.GELU(),
                nn.Linear(pos_emb_dim, d_model),
            )
        else:
            self.pos_mlp = None

        self.ln = nn.LayerNorm(d_model)

    def forward(self, theta, phi, rm):
        xyz = sph_to_cart(theta, phi)    # (B, N, 3)
        rm = rm.unsqueeze(-1)            # (B, N, 1)
        base = torch.cat([rm, xyz], dim=-1)  # (B, N, 4)
        tok = self.token_mlp(base)           # (B, N, d_model)
        if self.pos_mlp is not None:
            tok = tok + self.pos_mlp(xyz)
        return self.ln(tok), xyz

class GeoBiasAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, bias_mlp_hidden=32, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.bias_mlp = nn.Sequential(
            nn.Linear(3, bias_mlp_hidden),
            nn.ReLU(),
            nn.Linear(bias_mlp_hidden, n_heads)
        )

    def forward(self, x, xyz, attn_mask=None):
        B, N, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,N,N)

        # Relative position bias via small MLP on (x_i - x_j)
        rel_pos = xyz[:, :, None, :] - xyz[:, None, :, :]   # (B,N,N,3)
        bias_out = self.bias_mlp(rel_pos.view(B * N * N, 3)).view(B, N, N, H).permute(0, 3, 1, 2)
        scores = scores + bias_out

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1).bool(), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out), attn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GeoBiasAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, xyz, attn_mask=None):
        x = x + self.attn(self.ln1(x), xyz, attn_mask)[0]
        x = x + self.ff(self.ln2(x))
        return x

class main_net(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=6, mlp_ratio=4.0,
                 sh_max_l=3, use_cls_token=True, dropout=0.1, out_activation=None):
        super().__init__()
        self.idd = idd

        self.d_model = d_model
        self.token_embed = SphereTokenEmbed(d_model=d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # (l+1)^2 coefficients up to sh_max_l
        self.sh_max_l = sh_max_l
        out_dim = sh_max_l**2+2*sh_max_l

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )
        self.out_activation = out_activation  # e.g., nn.Tanh() if targets in [-1,1]
        self.loss_fn = nn.SmoothL1Loss(beta=0.01)  # Huber

    def forward(self, data, attn_mask=None):
        # data: (B, 3, N_tokens) with rows [theta, phi, rm]
        theta, phi, rm = data[:, 0, :], data[:, 1, :], data[:, 2, :]
        x, xyz = self.token_embed(theta, phi, rm)  # (B,N,d), (B,N,3)

        if self.use_cls_token:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            dummy = xyz.new_zeros(B, 1, 3)
            dummy[..., 2] = 1.0
            xyz = torch.cat([dummy, xyz], dim=1)

        for layer in self.layers:
            x = layer(x, xyz, attn_mask)

        pooled = x[:, 0, :] if self.use_cls_token else x.mean(dim=1)
        out = self.regressor(pooled)
        if self.out_activation is not None:
            out = self.out_activation(out)
        return out

    def criterion(self, pred, target):
        return self.loss_fn(pred, target)

