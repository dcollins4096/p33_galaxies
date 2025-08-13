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

idd = 7
what = "Net 2 with more details"

def init_weights_constant(m):
    if isinstance(m, nn.Linear):
        #nn.init.constant_(m.weight, 0.5)
        nn.init.constant_(m.bias, 0.1)

def thisnet():

    d_model = 512
    n_heads = 16
    n_layers = 6
    mpl_ratio = 4
    max_ell = 3
    model = main_net(d_model, n_heads, n_layers)
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs  = 300
    lr = 3e-4
    batch_size=3
    trainer(model,data,parameters,validatedata,validateparams,epochs=epochs,lr=lr,batch_size=batch_size)

def trainer(model, data,parameters, validatedata,validateparams,epochs=1, lr=1e-3, batch_size=10, test_num=0, weight_decay=None):
    optimizer = optim.AdamW( model.parameters(), lr=lr)
    from torch.optim.lr_scheduler import CyclicLR
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #        optimizer, max_lr=1e-3, total_steps=epochs
    #)
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
        output1 =model(data_n)
        loss = model.criterion(output1, param_n)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        tnow = time.time()
        tel = tnow-t0
        if epoch > 0: #(epoch>0 and epoch%100==0) or epoch==10:
            model.eval()
            with torch.no_grad():
                mod1 = model(validatedata)
                this_loss = model.criterion(mod1, validateparams)
                losslist.append(this_loss.item())
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utilities
# ---------------------------
def sph_to_cart(theta, phi):
    """theta: polar angle [0, pi], phi: azimuth [0, 2pi).
    Both tensors of shape (...). Returns (..., 3)."""
    st = torch.sin(theta)
    x = st * torch.cos(phi)
    y = st * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


# ---------------------------
# Token embedding
# ---------------------------
class SphereTokenEmbed(nn.Module):
    """
    Embed each sightline (theta, phi, rm) into d_model features.
    We use Cartesian xyz + rm, then an MLP, plus a learned positional embedding
    (optionally simple).
    """
    def __init__(self, d_model=128, pos_emb_dim=32, use_learned_pos=True):
        super().__init__()
        self.d_model = d_model
        self.use_learned_pos = use_learned_pos

        # Base token embed from [rm, x,y,z]
        self.token_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Optional learned positional encoding of xyz
        if use_learned_pos:
            self.pos_mlp = nn.Sequential(
                nn.Linear(3, pos_emb_dim),
                nn.GELU(),
                nn.Linear(pos_emb_dim, d_model),
            )
        else:
            # sinusoidal style positional encoding on theta/phi could be added
            self.pos_mlp = None

        # Small layer norm
        self.ln = nn.LayerNorm(d_model)

    def forward(self, theta, phi, rm):
        # theta,phi,rm: (batch, n_tokens)
        # convert to cartesian
        xyz = sph_to_cart(theta, phi)   # (batch, n_tokens, 3)
        rm = rm.unsqueeze(-1)          # (batch, n_tokens, 1)
        base = torch.cat([rm, xyz], dim=-1)  # (batch, n_tokens, 4)
        tok = self.token_mlp(base)          # (batch, n_tokens, d_model)

        if self.pos_mlp is not None:
            pos = self.pos_mlp(xyz)   # (batch, n_tokens, d_model)
            tok = tok + pos

        tok = self.ln(tok)
        return tok, xyz  # return xyz for geodesic computations


# ---------------------------
# Attention with geodesic bias
# ---------------------------
class GeoBiasAttention(nn.Module):
    """
    Multi-head attention where we add a learned bias depending on angular distance.
    - q,k,v are projected from inputs.
    - pairwise bias b(angle) is computed by a small MLP applied to cos(angle)=dot(x_i, x_j).
    - bias is added to attention logits before softmax.
    """
    def __init__(self, d_model, n_heads=8, bias_mlp_hidden=32, dropout=0.0):
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

        # small MLP to map cos(angle) -> scalar bias per head (we output per-head biases)
        self.bias_mlp = nn.Sequential(
            nn.Linear(3, bias_mlp_hidden),
            nn.ReLU(),
            nn.Linear(bias_mlp_hidden, n_heads)
        )

    def forward(self, x, xyz, attn_mask=None):
        # x: (batch, n_tokens, d_model)
        # xyz: (batch, n_tokens, 3) unit vectors
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, d_head)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, N, N)
        scores = scores / math.sqrt(self.d_head)

        # Compute pairwise cos(angle) = dot(x_i, x_j) for geodesic information
        # xyz: (B,N,3). Compute (B, N, N) cosines
        # To save memory, compute per-batch
        # cos_theta = xyz @ xyz.transpose(-2,-1)  # (B,N,N)
        cos_theta = torch.einsum('b i d, b j d -> b i j', xyz, xyz)  # (B,N,N)
        cos_theta_clamped = cos_theta.unsqueeze(1)  # (B,1,N,N)

        # Suppose xyz is computed from theta, phi inside forward()
        # xyz: [B, N, 3]

        # Compute bias from relative positions
        rel_pos = xyz[:, :, None, :] - xyz[:, None, :, :]   # [B, N, N, 3]
        bias_in = rel_pos.reshape(B * N * N, 3)             # [B*N*N, 3]
        bias_out = self.bias_mlp(bias_in)                   # [B*N*N, H]

        # Reshape to [B, N, N, H]
        bias_out = bias_out.view(B, N, N, self.n_heads)

        # Move heads dimension before N,N: [B, H, N, N]
        bias_out = bias_out.permute(0, 3, 1, 2)


        # Add bias to scores
        scores = scores + bias_out

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1).bool(), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)   # (B, H, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)  # (B, N, d_model)
        out = self.out_proj(out)
        return out, attn  # return attention for possible diagnostics


# ---------------------------
# Transformer block (attention + FFN + skip + LN)
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = GeoBiasAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, xyz, attn_mask=None):
        x_norm = self.ln1(x)
        attn_out, attn_map = self.attn(x_norm, xyz, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_map


# ---------------------------
# SphereFormer Model
# ---------------------------
class main_net(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=6, mlp_ratio=4.0,
                 sh_max_l=3, use_cls_token=True, dropout=0.0):
        """
        sh_max_l: maximum spherical harmonic degree to predict. Output size =
                  (l+1)^2 for l in [0..sh_max_l], i.e., (sh_max_l+1)**2 coefficients.
        """
        super().__init__()
        self.idd = idd
        self.d_model = d_model
        self.token_embed = SphereTokenEmbed(d_model=d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.use_cls_token = use_cls_token
        if use_cls_token:
            # learnable cls token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # final pooling: we will either use cls or mean pooling
        out_dim = sh_max_l**2 + 2*sh_max_l
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim)
        )
        self.L1 = nn.L1Loss()

    def forward(self, data, attn_mask=None):
        """
        theta, phi, rm: tensors of shape (batch, n_tokens)
        returns: sh_coeffs (batch, out_dim)
        """
        theta = data[:,0,:]
        phi   = data[:,1,:]
        rm    = data[:,2,:]

        x, xyz = self.token_embed(theta, phi, rm)  # (B,N,d), (B,N,3)
        B, N, _ = x.shape

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,d)
            x = torch.cat([cls, x], dim=1)          # (B, N+1, d)
            # pad xyz with dummy for cls token (unit z)
            dummy = xyz.new_zeros(B, 1, 3)
            dummy[..., 2] = 1.0
            xyz = torch.cat([dummy, xyz], dim=1)
        # optional attn_mask full logic should account for cls token shape if used

        attn_maps = []
        for layer in self.layers:
            x, attn_map = layer(x, xyz, attn_mask=attn_mask)
            attn_maps.append(attn_map)

        # pooling
        if self.use_cls_token:
            pooled = x[:, 0, :]  # CLS
        else:
            pooled = x.mean(dim=1)

        out = self.regressor(pooled)
        #return out, attn_maps
        return out
    def criterion(self, guess, target):
        return self.L1(guess,target)


