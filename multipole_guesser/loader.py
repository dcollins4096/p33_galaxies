
import torch
import h5py
import pdb


def loader(fname, nvalid=1, ntrain=30):
    fptr = h5py.File(fname,'r')
    clm =  torch.tensor(fptr['Clm'][()], dtype=torch.float)
    rm  =  torch.tensor(fptr['Rm'][()], dtype=torch.float)
    theta =torch.tensor(fptr['theta'][()], dtype=torch.float)
    phi   =torch.tensor(fptr['phi'][()], dtype=torch.float)

    sky_all = torch.stack([theta,phi,rm], axis=1)
    clm_all = clm
    sky={}
    clm={}
    sky['train']=sky_all[:ntrain]
    sky['valid']=sky_all[ntrain:ntrain+nvalid]
    sky['test'] =sky_all[ntrain+nvalid:]
    clm['train']=clm_all[:ntrain]
    clm['valid']=clm_all[ntrain:ntrain+nvalid]
    clm['test'] =clm_all[ntrain+nvalid:]
    return sky, clm



