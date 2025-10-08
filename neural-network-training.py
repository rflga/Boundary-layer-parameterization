import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.gridspec as gridspec
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import matplotlib.colors as clrs
import matplotlib.cm as cm
import os

# Atmos and Ocean datasets
Zatm = np.arange(4, 2052, 8)
Zoce = np.arange(-255, 1, 2)
nSample_atm = 5920
nSample_oce = 4004
nSample = nSample_atm + nSample_oce

# Inputs
ucoarse = xr.open_dataset('ucoarse.nc')
vcoarse = xr.open_dataset('vcoarse.nc')
tke     = xr.open_dataset('tke.nc')

# Outputs
subgrid_wu = xr.open_dataset('subgrid_wu.nc')
subgrid_wv = xr.open_dataset('subgrid_wv.nc')

# Boundary layer height / depth
blheight = xr.open_dataset('blheight.nc')



# --------- 1. Normalize by boundary layer height / depth ---------
normalized_Z  = np.arange(0.05, 1.20, 0.02)
new_nz = len(normalized_Z)

ucoarse_znorm  = np.nan*np.zeros((new_nz, nSample))
vcoarse_znorm  = np.nan*np.zeros((new_nz, nSample))
tke_znorm      = np.nan*np.zeros((new_nz, nSample))
subgrid_wu_znorm = np.nan*np.zeros((new_nz, nSample))
subgrid_wv_znorm = np.nan*np.zeros((new_nz, nSample))

for sample in range(nSample):
  if sample < nSample_atmos:
    znorm_sample  = Zatm / blheight[sample].values
  else:
    znorm_sample  = Zoce / blheight[sample].values

  tkesample = tke[:, sample]
  usample, vsample, = ucoarse[:, sample], vcoarse[:, sample]
  subwu_sample, subwv_sample = subgrid_wu[:, sample], subgrid_wv[:, sample]
  
  VARS     = [usample, vsample, subwu_sample, subwv_sample, tkesample]
  NORM_DS  = [ucoarse_znorm, vcoarse_znorm, subgrid_wu_znorm, subgrid_wv_znorm, tke_znorm]

  for v, ds in zip(VARS, NORM_DS):
      v = v.assign_coords({'z_index':znorm_sample})
      v_norm = v.interp(z_index=normalized_Z)

      ds[:, sample] = v_norm

z  = normalized_Z

ucoarse_znorm = xr.DataArray(ucoarse_znorm, dims=['normalized_z', 'sample'], coords=[normalized_Z, ucoarse.sample])
vcoarse_znorm = xr.DataArray(vcoarse_znorm, dims=['normalized_z', 'sample'], coords=[normalized_Z, ucoarse.sample])
tke_znorm     = xr.DataArray(tke_znorm,     dims=['normalized_z', 'sample'], coords=[normalized_Z, ucoarse.sample])
subgrid_wu_znorm = xr.DataArray(subgrid_wu_znorm, dims=['normalized_z', 'sample'], coords=[normalized_Z, ucoarse.sample])
subgrid_wv_znorm = xr.DataArray(subgrid_wv_znorm, dims=['normalized_z', 'sample'], coords=[normalized_Z, ucoarse.sample])
# -------------------------------------------------------------


# --------------- 2. Normalize inputs / outputs ---------------







