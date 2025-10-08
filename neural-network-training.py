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
Normalized_U = (ucoarse_znorm - ucoarse_znorm.mean(dim='normalized_z')) / ucoarse_znorm.std(dim='normalized_z')
Normalized_V = (vcoarse_znorm - vcoarse_znorm.mean(dim='normalized_z')) / vcoarse_znorm.std(dim='normalized_z')
Normalized_tke = tke_znorm / tke_znorm[0, :]

Normalized_subWU = subgrid_wu_znorm / subgrid_wu_znorm[0, :]
Normalized_subWV = subgrid_wv_znorm / subgrid_wv_znorm[0, :]
Normalized_subConcat = xr.concat((Normalized_subWU, Normalized_subWV), dim='normalized_z')

INPUTS = [Normalized_U, Normalized_V,  Normalized_tke]
nVariables = len(INPUTS)
nFeatures  = nVariables * new_nz
Normalized_inputs = np.nan*np.zeros([nFeatures, nSample])
for i in range(nVariables):
    Normalized_inputs[i*new_nz: (i+1)*new_nz] = INPUTS[i]

Normalized_inputs_DA = xr.DataArray(Normalized_inputs, dims=['input_feature', 'sample'], coords=[np.arange(nFeatures), Normalized_U.sample])
# ------------------------------------------------------------


# --------------- 3. Train / Test split and ANN training ---------------
inputs_tensor = torch.tensor(Normalized_inputs_DA.values, dtype=torch.float32).T
output_tensor = torch.tensor(Normalized_subConcat.values, dtype=torch.float32).T

indices = np.arange(nSample)

rs1 = 43
rs2 = 42
X_train, X_test_val, y_train, y_test_val, train_indices, test_val_indices = train_test_split(inputs_tensor, 
                                                                                 output_tensor, 
                                                                                 indices,
                                                                                 test_size=0.2, 
                                                                                 random_state=rs1)

X_test, X_val, y_test, y_val, test_indices, val_indices = train_test_split(X_test_val, 
                                                                           y_test_val, 
                                                                           test_val_indices,
                                                                           test_size=0.5, 
                                                                           random_state=rs2)

nTrain = len(X_train)
nVal   = len(X_val)
nTest  = len(X_test)


# Hyperparameters
nfeat = Normalized_inputs_DA.shape[0]
nout  = Normalized_subConcat.shape[0]

nNeurons      = 128
n_epochs      = 200
batch_size    = 200
learning_rate = 0.0001
drop          = 0.2
weight_decay  = 1e-4

model = nn.Sequential(
    nn.Linear(nfeat, nNeurons),
    nn.ReLU(),
    nn.Dropout(p=drop),
    nn.Linear(nNeurons, nNeurons),
    nn.ReLU(),
    nn.Dropout(p=drop),
    nn.Linear(nNeurons, nNeurons),
    nn.ReLU(),
    nn.Dropout(p=drop),
    nn.Linear(nNeurons, nout))

Loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        Xbatch = X_train[i:i+batch_size]
        ybatch = y_train[i:i+batch_size]

        optimizer.zero_grad()
        y_pred = model(Xbatch)

        loss = Loss(y_pred, ybatch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * Xbatch.size(0)

    train_loss /= X_train.shape[0]
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss  = Loss(y_val_pred, y_val)
        val_losses.append(val_loss.item())










