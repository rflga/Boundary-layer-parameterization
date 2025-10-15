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
nSample_atm = 5920
nSample_oce = 4004
nSample = nSample_atm + nSample_oce

# Inputs - Normalized by zi
Inputs_znorm  = xr.open_dataset('Inputs_znorm.nc')
ucoarse_znorm = Inputs_znorm.ucoarse_znorm
vcoarse_znorm = Inputs_znorm.vcoarse_znorm
tke_znorm     = Inputs_znorm.tke_znorm

# Outputs - Normalized by zi
Outputs_znorm    = xr.open_dataset('Outputs_znorm.nc')
subgrid_wu_znorm = Outputs_znorm.subgrid_wu_znorm
subgrid_wv_znorm = Outputs_znorm.subgrid_wv_znorm



# --------------- 1. Normalize inputs / outputs ---------------
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



# --------------- 2. Plot the outputs - with normalization (Figure 1) ---------------
# Take log of stability parameter values for visualization
StabParams_atm = np.array([-0.6 , -0.51,  0.18,  0.53,  0.64,  1.03,  1.5 ,  1.65,  1.84, 1.86,  1.99,  2.48,  2.64,  2.99,  3.45])
StabParams_oce = np.array([-1.05,  0.41,  0.83,  1.03,  1.22,  1.82,  2.08,  2.71,  2.93, 3.08,  3.88,  4.73])
StabParams = np.append(StabParams_atm, StabParams_oce)

# List of LES simulations
SIMUS_list_atm = np.array(['Ug25Q002', 'Ug16Q001', 'Ug12Q001','Ug16Q003', 'Ug20Q005', 'Ug16Q006','Ug16Q010', 'Ug12Q005', 'Ug8Q003','Ug5Q001', 'Ug10Q005', 'Ug8Q006', 'Ug12Q020', 'Ug10Q020', 'Ug8Q020'])
SIMUS_list_oce = np.array(['U15HF25', 'U15HF75', 'U8HF25', 'U10HF50', 'U15HF150', 'U10HF100', 'U8HF75', 'U10HF200', 'U8HF150', 'U5HF50', 'U5HF100', 'U5HF200'])
SIMUS_list = np.append(SIMUS_list_atm, SIMUS_list_oce)

# Simulation labels
SP_list_atm = ['$ζ_{0.55}$', '$ζ_{0.60}$', '$ζ_{1.2}$', '$ζ_{1.7}$', '$ζ_{1.9}$', '$ζ_{2.8}$', '$ζ_{4.5}$', '$ζ_{5.2}$', '$ζ_{6.3}$', '$ζ_{6.4}$', '$ζ_{7.3}$', '$ζ_{11.9}$', '$ζ_{14}$', '$ζ_{19.9}$', '$ζ_{31.4}$']
SP_list_oce = ['$ζ_{0.35}$', '$ζ_{1.5}$', '$ζ_{2.3}$', '$ζ_{2.8}$', '$ζ_{3.4}$', '$ζ_{6.2}$', '$ζ_{8.0}$', '$ζ_{15.0}$', '$ζ_{48.4}$', '$ζ_{112.8}$']
SP_list = SP_list_atm + SP_list_oce

norm_atm = clrs.Normalize(vmin=-0.5, vmax=3.5)
norm_oce = clrs.Normalize(vmin=0.5,  vmax=3.3)
cmap = cm.rainbow

# figure
fig = plt.figure(figsize=(12,7))
VARS = [subgrid_wu_znorm, subgrid_wv_znorm]

GS_out = gridspec.GridSpec(1,2, wspace=0.5)
GS_in1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=GS_out[0], wspace=0.03, hspace=0.5)
GS_in2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=GS_out[1], wspace=0.03, hspace=0.5)

z = subgrid_wu_znorm.normalized_z

for fld_lab, GS in zip(['HF', 'Ug'], [GS_in1, GS_in2]):
  for c1, (var, Norm_var, tit) in enumerate(zip(VARS, [Normalized_subWU, Normalized_subWV], ['WU', 'WV'])):
    ax1, ax2 = fig.add_subplot(GS[0, c1]), fig.add_subplot(GS[1, c1])

    for c, (simu, sp, st) in enumerate(zip(SIMUS_list, SP_list, StabParams)):
        var_simu      = var.where(var.sample.str.contains(simu), drop=True)
        Norm_var_simu = Norm_var.where(Norm_var.sample.str.contains(simu), drop=True)
  
        if 'Ug' in simu: norm = norm_atm
        if 'HF' in simu: norm = norm_oce
            
        if fld_lab in simu:
            continue
  
        var_simu_mean = var_simu.mean(dim='sample')
        var_simu_std  = var_simu.std(dim='sample')
        Norm_var_simu_mean = Norm_var_simu.mean(dim='sample')
        Norm_var_simu_std  = Norm_var_simu.std(dim='sample')
        
        label = sp
  
        ax1.plot(var_simu_mean, z, label=label, color=cmap(norm(st)), marker='+')
        ax1.fill_betweenx( z, (var_simu_mean-var_simu_std), (var_simu_mean+var_simu_std), color=cmap(norm(st)), alpha=0.2)
  
        ax2.plot(Norm_var_simu_mean, z, label=label, color=cmap(norm(st)), marker='+')
        ax2.fill_betweenx( z, (Norm_var_simu_mean-Norm_var_simu_std), (Norm_var_simu_mean+Norm_var_simu_std), color=cmap(norm(st)), alpha=0.2)
  
    if c1==0:
        if fld_lab=='HF': 
            tit_lab='Atmospheric'
            ax1.set_ylabel('$z/z_i$', fontname=fontname, fontsize=14)
            ax1.legend(fontsize=9, bbox_to_anchor=(2.52, 0.5))
  
        else:
            tit_lab='Oceanic'
            ax1.set_ylabel('$z/z_i$', labelpad=-2, fontname=fontname, fontsize=14)
            ax1.legend(fontsize=9, bbox_to_anchor=(2.55, 0.3))
            ax1.set_xticks([-0.0001, -0.0003])

        ax2.set_ylabel('$z/z_i$', fontsize=14, fontname=fontname)
        flux='\overline{u\'w\'}'
  
    if c1==1:
        ax1.set_yticks([])
        ax2.set_yticks([])
        flux='\overline{v\'w\'}'
  
        if fld_lab=='Ug': 
            ax1.set_xticks(ax1.get_xticks()[1::2])
  
    ax1.tick_params(axis='both', labelsize=11)
    ax2.tick_params(axis='both', labelsize=11)
  
    ax1.set_title(tit_lab+' $'+flux+'$', fontname=fontname)
    ax2.set_title(tit_lab+' $'+flux+' / '+flux+'_{surf}$', fontsize=11, fontname=fontname)
    ax1.set_xlabel('Momentum flux ($m^2/s^{-2}$)', fontname=fontname)
    ax2.set_xlabel('Normalized flux', fontname=fontname)

fig.savefig("Fig1.png", dpi=500)
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
# ------------------------------------------------------------









