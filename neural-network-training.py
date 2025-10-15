import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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
import random

# Atmos and Ocean datasets
nSample_atm = 5920
nSample_oce = 4004
nSample = nSample_atm + nSample_oce
index_atmos = np.arange(nSample_atm)
index_ocean = np.arange(nSample_atm, nSample)

# Inputs - Normalized by zi
Inputs_znorm  = xr.open_dataset('Inputs_znorm.nc')
ucoarse_znorm = Inputs_znorm.ucoarse_znorm
vcoarse_znorm = Inputs_znorm.vcoarse_znorm
tke_znorm     = Inputs_znorm.tke_znorm

# Outputs - Normalized by zi
Outputs_znorm    = xr.open_dataset('Outputs_znorm.nc')
subgrid_wu_znorm = Outputs_znorm.subgrid_wu_znorm
subgrid_wv_znorm = Outputs_znorm.subgrid_wv_znorm

z = subgrid_wu_znorm.normalized_z
nz = len(z)




# --------------- 1. Normalize inputs / outputs ---------------
Normalized_U = (ucoarse_znorm - ucoarse_znorm.mean(dim='normalized_z')) / ucoarse_znorm.std(dim='normalized_z')
Normalized_V = (vcoarse_znorm - vcoarse_znorm.mean(dim='normalized_z')) / vcoarse_znorm.std(dim='normalized_z')
Normalized_tke = tke_znorm / tke_znorm[0, :]

Normalized_subWU = subgrid_wu_znorm / subgrid_wu_znorm[0, :]
Normalized_subWV = subgrid_wv_znorm / subgrid_wv_znorm[0, :]
Normalized_subConcat = xr.concat((Normalized_subWU, Normalized_subWV), dim='normalized_z')

INPUTS = [Normalized_U, Normalized_V,  Normalized_tke]
nVariables = len(INPUTS)
nFeatures  = nVariables * nz
Normalized_inputs = np.nan*np.zeros([nFeatures, nSample])
for i in range(nVariables):
    Normalized_inputs[i*nz: (i+1)*nz] = INPUTS[i]

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










# --------------- 4. Plot predicted vs truth - With density plots (Figure 2) ---------------
def plot_density(pred, true, ax=None, cmap='viridis', gridsize=200):
    minval = min(pred.min(), true.min())
    maxval = max(pred.max(), true.max())
    if ax is not None:
        hb = ax.hexbin(pred, true, gridsize=gridsize,
               cmap=cmap, mincnt=1, norm=LogNorm(vmin=1, vmax=500))
        
        ax.plot([minval, maxval], [minval, maxval], 'k--', lw=1)
        ax.set_xlabel('Predicted Flux ($m^2/s^2$)', fontsize=12)
        ax.set_ylabel('True Flux ($m^2/s^2$)', fontsize=12)
        ax.set_title(f'Predicted vs True Fluxes', fontsize=14)
        cbar = fig.colorbar(hb, ax=ax, shrink=0.7) 
        cbar.set_label('Counts', rotation=270, labelpad=10)


model.eval()
# True fluxes
subgrid_wu_test = subgrid_wu_znorm[:, test_indices]
subgrid_wv_test = subgrid_wv_znorm[:, test_indices]

subgrid_wu_test_atmos = subgrid_wu_test.where(subgrid_wu_test.sample.str.contains('ATMOS'), drop=True)
subgrid_wv_test_atmos = subgrid_wv_test.where(subgrid_wv_test.sample.str.contains('ATMOS'), drop=True)
subgrid_wu_test_ocean = subgrid_wu_test.where(subgrid_wu_test.sample.str.contains('OCEAN'), drop=True)
subgrid_wv_test_ocean = subgrid_wv_test.where(subgrid_wv_test.sample.str.contains('OCEAN'), drop=True)

# ANN-Predicted fluxes
predicted_fluxes = model(X_test)
predicted_wu = predicted_fluxes[:, :nz].detach().numpy()
predicted_wv = predicted_fluxes[:, nz:].detach().numpy()

predicted_wu = xr.DataArray(predicted_wu, dims=['sample', 'normalized_z'], coords=[subgrid_wu[:, test_indices].sample, z])
predicted_wv = xr.DataArray(predicted_wv, dims=['sample', 'normalized_z'], coords=[subgrid_wu[:, test_indices].sample, z])

predicted_wu_atmos = predicted_wu.where(predicted_wu.sample.str.contains('ATMOS'), drop=True)
predicted_wv_atmos = predicted_wv.where(predicted_wv.sample.str.contains('ATMOS'), drop=True)
predicted_wu_ocean = predicted_wu.where(predicted_wu.sample.str.contains('OCEAN'), drop=True)
predicted_wv_ocean = predicted_wv.where(predicted_wv.sample.str.contains('OCEAN'), drop=True)

# Un-normalize the prediction
unnorm_predicted_wu_atmos = predicted_wu_atmos * subgrid_wu_znorm[:, test_indices].where(subgrid_wu_znorm.sample.str.contains('ATMOS'), drop=True)[0]
unnorm_predicted_wv_atmos = predicted_wv_atmos * subgrid_wv_znorm[:, test_indices].where(subgrid_wv_znorm.sample.str.contains('ATMOS'), drop=True)[0]
unnorm_predicted_wu_ocean = predicted_wu_ocean * subgrid_wu_znorm[:, test_indices].where(subgrid_wu_znorm.sample.str.contains('OCEAN'), drop=True)[0]
unnorm_predicted_wv_ocean = predicted_wv_ocean * subgrid_wv_znorm[:, test_indices].where(subgrid_wv_znorm.sample.str.contains('OCEAN'), drop=True)[0]

# 1D predicted and truth for density plot and R2
unorm_predicted_atmos_1d = np.append(unnorm_predicted_wu_atmos.T.values.flatten(), unnorm_predicted_wv_atmos.T.values.flatten())
unorm_predicted_ocean_1d = np.append(unnorm_predicted_wu_ocean.T.values.flatten(), unnorm_predicted_wv_ocean.T.values.flatten())

unnorm_predicted_wu_1d = np.append(unnorm_predicted_wu_atmos.T.values.flatten(), unnorm_predicted_wu_ocean.T.values.flatten())
unnorm_predicted_wv_1d = np.append(unnorm_predicted_wv_atmos.T.values.flatten(), unnorm_predicted_wv_ocean.T.values.flatten())

subgrid_test_atmos_1d = np.append(subgrid_wu_test_atmos.values.flatten(), subgrid_wv_test_atmos.values.flatten())
subgrid_test_ocean_1d = np.append(subgrid_wu_test_ocean.values.flatten(), subgrid_wv_test_ocean.values.flatten())

r2_atm = r2_score(subgrid_test_atmos_1d, unorm_predicted_atmos_1d)
r2_oce = r2_score(subgrid_test_ocean_1d, unorm_predicted_ocean_1d)

print('R2 in Atmos: ', r2_atm)
print('R2 in Ocean: ', r2_oce)

# Plot the figure
GS_out = gridspec.GridSpec(1,2, wspace=0.5)
GS_in1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=GS_out[0], wspace=0.03, hspace=0.4)
GS_in2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=GS_out[1], wspace=0.03, hspace=0.4)

fig1 = plt.figure(figsize=(10,6))
ax1, ax2, ax3, ax4 = fig1.add_subplot(GS_in1[0,0]), fig1.add_subplot(GS_in1[0,1]), fig1.add_subplot(GS_in2[0,0]), fig1.add_subplot(GS_in2[0,1])
ax5, ax6 = fig1.add_subplot(GS_in1[1,:]), fig1.add_subplot(GS_in2[1,:])

plot_density(unorm_predicted_atmos_1d, subgrid_test_atmos_1d, ax=ax5)
plot_density(unorm_predicted_ocean_1d, subgrid_test_ocean_1d, ax=ax6)

# Atmos w'u'
ax1.plot(unnorm_predicted_wu_atmos.mean(dim='sample'), z, label='Prediction', color='blue')
ax1.plot(subgrid_wu_test_atmos.mean(dim='sample'), z, label='Truth', color='orange')
ax1.fill_betweenx(z, unnorm_predicted_wu_atmos.mean(dim='sample')-unnorm_predicted_wu_atmos.std(dim='sample'), unnorm_predicted_wu_atmos.mean(dim='sample')+unnorm_predicted_wu_atmos.std(dim='sample'), color='blue', alpha=0.1)
ax1.fill_betweenx(z, subgrid_wu_test_atmos.mean(dim='sample') - subgrid_wu_test_atmos.std(dim='sample'), subgrid_wu_test_atmos.mean(dim='sample') + subgrid_wu_test_atmos.std(dim='sample'), color='orange', alpha=0.1)

# Atmos v'w'
ax2.plot(unnorm_predicted_wv_atmos.mean(dim='sample'), z, label='Prediction', color='blue')
ax2.plot(subgrid_wv_test_atmos.mean(dim='sample'), z, label='Truth', color='orange')
ax2.fill_betweenx(z, unnorm_predicted_wv_atmos.mean(dim='sample')-unnorm_predicted_wv_atmos.std(dim='sample'), unnorm_predicted_wv_atmos.mean(dim='sample')+unnorm_predicted_wv_atmos.std(dim='sample'), color='blue', alpha=0.1)
ax2.fill_betweenx(z, subgrid_wv_test_atmos.mean(dim='sample') - subgrid_wv_test_atmos.std(dim='sample'), subgrid_wv_test_atmos.mean(dim='sample') + subgrid_wv_test_atmos.std(dim='sample'), color='orange', alpha=0.1)

# Ocean w'u'
ax3.plot(unnorm_predicted_wu_ocean.mean(dim='sample'), z, label='Prediction', color='blue')
ax3.plot(subgrid_wu_test_ocean.mean(dim='sample'), z, label='Truth', color='orange')
ax3.fill_betweenx(z, unnorm_predicted_wu_ocean.mean(dim='sample')-unnorm_predicted_wu_ocean.std(dim='sample'), unnorm_predicted_wu_ocean.mean(dim='sample')+unnorm_predicted_wu_ocean.std(dim='sample'), color='blue', alpha=0.1)
ax3.fill_betweenx(z, subgrid_wu_test_ocean.mean(dim='sample') - subgrid_wu_test_ocean.std(dim='sample'), subgrid_wu_test_ocean.mean(dim='sample') + subgrid_wu_test_ocean.std(dim='sample'), color='orange', alpha=0.1)

# Ocean v'w'
ax4.plot(unnorm_predicted_wv_ocean.mean(dim='sample'), z, label='Prediction', color='blue')
ax4.plot(subgrid_wv_test_ocean.mean(dim='sample'), z, label='Truth', color='orange')
ax4.fill_betweenx(z, unnorm_predicted_wv_ocean.mean(dim='sample')-unnorm_predicted_wv_ocean.std(dim='sample'), unnorm_predicted_wv_ocean.mean(dim='sample')+unnorm_predicted_wv_ocean.std(dim='sample'), color='blue', alpha=0.1)
ax4.fill_betweenx(z, subgrid_wv_test_ocean.mean(dim='sample') - subgrid_wv_test_ocean.std(dim='sample'), subgrid_wv_test_ocean.mean(dim='sample') + subgrid_wv_test_ocean.std(dim='sample'), color='orange', alpha=0.1)

# Legends
ax1.legend()
ax2.legend()
ax3.legend(loc='upper left')
ax4.legend(loc='upper left')

# Ticks
ax2.set_yticks([])
ax3.set_xticks(ax3.get_xticks()[::2])
ax4.set_xticks(ax4.get_xticks()[2::2])
ax4.set_yticks

# Titles and labels
ax1.set_title('Atmospheric $\overline{u\'w\'}$')
ax2.set_title('Atmospheric $\overline{v\'w\'}$')
ax3.set_title('Oceanic $\overline{u\'w\'}$')
ax4.set_title('Oceanic $\overline{v\'w\'}$')
ax1.set_xlabel('Momentum flux ($m^2/s^{-2}$)', fontsize=9)
ax2.set_xlabel('Momentum flux ($m^2/s^{-2}$)', fontsize=9)
ax3.set_xlabel('Momentum flux ($m^2/s^{-2}$)', fontsize=9)
ax4.set_xlabel('Momentum flux ($m^2/s^{-2}$)', fontsize=9)
ax1.set_ylabel('Altitude (m)')
ax3.set_ylabel('Depth (m)')

fig1.savefig('Prez_fig2.png', dpi=500)
# ------------------------------------------------------------










# --------------- 5. Plot upgradient fluxes (Figure 3) ---------------
ucoarse_test  = ucoarse_znorm[:, test_indices]
vcoarse_test  = vcoarse_znorm[:, test_indices]

ucoarse_test_atm  = ucoarse_test.where(ucoarse_test.sample.str.contains('ATMOS'), drop=True)
ucoarse_test_oce  = ucoarse_test.where(ucoarse_test.sample.str.contains('OCEAN'), drop=True)
vcoarse_test_atm  = vcoarse_test.where(vcoarse_test.sample.str.contains('ATMOS'), drop=True)
vcoarse_test_oce  = vcoarse_test.where(vcoarse_test.sample.str.contains('OCEAN'), drop=True)

GS = gridspec.GridSpec(1,2, wspace=0.3)
gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=GS[0], wspace=0)
gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=GS[1], wspace=0)

fig = plt.figure(figsize=(13,4))
ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
ax3, ax4 = fig.add_subplot(gs2[0]), fig.add_subplot(gs2[1])

for i, axes in zip([0,1], [[ax1,ax2], [ax3,ax4]]):
    ax_u, ax_suw = axes[0], axes[1]
    
    if i==0:
        # Atmospheric example
        sample = 3348
        test_sample = np.where(unnorm_predicted_wu_atmos.sample.str.contains(f'i_sample={str(sample)}'))[0][0]

        suw_s      = unnorm_predicted_wv_atmos[test_sample].values
        suw_s_true = subgrid_wv_test_atmos[:, test_sample].values

        us = vcoarse_test_atm[:, test_sample].values
    
    if i==1:
        # Oceanic example
        sample = 6380
        test_sample = np.where(unnorm_predicted_wu_ocean.sample.str.contains(f'i_sample={str(sample)}'))[0][0]

        suw_s      = unnorm_predicted_wv_ocean[test_sample].values
        suw_s_true = subgrid_wv_test_ocean[:, test_sample].values
        
        us = vcoarse_znorm[:, sample].values

    ax_suw.axvline(0, linestyle='--', color='black', linewidth=0.5)
    ax_suw.plot(suw_s_true, z, linewidth=2.5, color='orange',    label='True flux')
    ax_suw.plot(suw_s,      z, linewidth=2.5, color='royalblue', label='Predicted flux')
    
    ax_u.plot(us, z, linewidth=2.5, color='brown', label='u profile')

    # Compute RMSE
    rmse = np.sqrt(np.mean((suw_s_true - suw_s) ** 2))
    print('RMSE:', rmse)

alt1, alt2 = 0.43, 0.70
ax1.axhspan(alt1, alt2, color='gray', alpha=0.3, label='Upgradient flux zone') 
ax2.axhspan(alt1, alt2, color='gray', alpha=0.3)

alt3, alt4 = -0.34, -0.73
ax3.axhspan(alt3, alt4, color='gray', alpha=0.3, label='Upgradient flux zone') 
ax4.axhspan(alt3, alt4, color='gray', alpha=0.3)

# Legend
ax1.legend(loc='upper left', fontsize=10)
ax2.legend(loc='upper left', fontsize=10)
ax3.legend(loc='lower left', fontsize=10)
ax4.legend(loc='lower left', fontsize=10)


# Ticks
ax2.set_yticks([])
ax4.set_yticks([])
ax1.tick_params(axis='both', labelsize=11)
ax2.tick_params(axis='both', labelsize=11)
ax3.tick_params(axis='both', labelsize=11)
ax4.tick_params(axis='both', labelsize=11)
ax1.set_xticks([8.0, 8.4])
ax2.set_xticks(ax2.get_xticks()[1::2])
ax3.set_xticks([0.000, -0.006])

# Titles and Labels
ax1.set_ylabel('$z/z_i$', fontsize=15)
ax1.set_xlabel('Wind speed $(m/s)$', fontsize=10)
ax1.set_title('Zonal Wind $u$', fontsize=12)

ax3.set_ylabel('$z/z_i$', fontsize=15)
ax3.set_xlabel('Current $(m/s)$', fontsize=10)
ax3.set_title('Zonal Current $u$', fontsize=12)

offset_text = ax4.xaxis.get_offset_text()
offset_text.set_x(1.1)     
offset_text.set_y(0.5) 

fig.savefig('Fig3.png', dpi=500)
# ------------------------------------------------------------








# --------------- 6. Ocean data-limited regime augmented with atmospheric data (Figure 4) ---------------
np.random.seed(42)

nTrain = 100
nTest  = 100
nVal   = 100

shuffled_index_atmos = np.random.permutation(index_atmos)

nRS = 12    # Twelve ocean simulations
nInit = 10  # 10 different initialization of weights for ANN

rmse_oce_only = np.nan*np.zeros([nRS, nInit])
rmse_oce_atmo = np.nan*np.zeros([nRS, nInit])

# Train on one Ocean simu, test on the rest of ocean simus (+ atmos sometimes)
for c_rs, simu in enumerate(Simu_list_oce):
    # Training data
    masked_data_simu  = Normalized_inputs_DA.sample.str.contains(simu)

    # Create a mask that contains only ocean samples, but doesn't contain the simulation we are training on
    masked_data_rest  = ~Normalized_inputs_DA.sample.str.contains(simu)
    masked_data_ocean =  Normalized_inputs_DA.sample.str.contains('HF')
    mask_testval      =  masked_data_rest & masked_data_ocean

    indices_simu    = np.array([i for i, val in enumerate(masked_data_simu.values) if val])
    indices_testval = np.array([i for i, val in enumerate(mask_testval.values) if val])

    np.random.seed(0)
    shuffled_index_ocean   = np.random.permutation(indices_simu)
    shuffled_index_testval = np.random.permutation(indices_testval)

    for train_settings in ['OCEAN_ONLY', 'OCEAN_ATMOS']:  #, 'OCEAN_MORE']:
        if train_settings == 'OCEAN_ONLY':
            # Only take 100 ocean samples from the simulation
            train_indices = shuffled_index_ocean[:nTrain]
            
        if train_settings == 'OCEAN_ATMOS':
            # Second we take the 100 ocean samples + 1000 random atmos
            train_indices = np.append(shuffled_index_atmos[:1000], shuffled_index_ocean[:nTrain])

        test_indices = shuffled_index_testval[:nTest]
        val_indices  = shuffled_index_testval[nTest:nTest+nVal]

        X_train = torch.tensor(Normalized_inputs_DA.T[train_indices].values,  dtype=torch.float32)
        X_test  = torch.tensor( Normalized_inputs_DA.T[test_indices].values,  dtype=torch.float32)
        X_val   = torch.tensor(  Normalized_inputs_DA.T[val_indices].values,  dtype=torch.float32)

        y_train = torch.tensor(Normalized_subConcat.T[train_indices].values,  dtype=torch.float32)
        y_test  = torch.tensor( Normalized_subConcat.T[test_indices].values,  dtype=torch.float32)
        y_val   = torch.tensor(  Normalized_subConcat.T[val_indices].values,  dtype=torch.float32)

        nNeurons = 128
        batch_size = 200
        n_epochs = 200
        learning_rate = 0.0001
        drop = 0.2
        weight_decay = 1e-4

        nfeat = Normalized_inputs_DA.shape[0]
        nout  = Normalized_subConcat.shape[0]

        for init in range(nInit):
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

                # Average losses
                train_loss /= X_train.shape[0]
                train_losses.append(train_loss)
                
                if X_val is not None:
                    # Validation step
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        y_val_pred = model(X_val)
                        val_loss = Loss(y_val_pred, y_val)
                        val_losses.append(val_loss.item())

            model.eval()
            y_pred = model(X_test)
            rmse = root_mean_squared_error(y_test.detach().numpy().flatten(), y_pred.detach().numpy().flatten())

            if train_settings == 'OCEAN_ONLY':
                rmse_oce_only[c_rs, init] = rmse
            if train_settings == 'OCEAN_ATMOS':
                rmse_oce_atmo[c_rs, init] = rmse

rmse_oce_only_da = xr.DataArray(rmse_oce_only, dims=['random_seeds', 'NN_init'])
rmse_oce_atmo_da = xr.DataArray(rmse_oce_atmo, dims=['random_seeds', 'NN_init'])

# Plot the figure
mean_rmse_oce_only = rmse_oce_only_da.mean(dim='NN_init')
mean_rmse_oce_atmo = rmse_oce_atmo_da.mean(dim='NN_init')
std_rmse_oce_only  = rmse_oce_only_da.std(dim='NN_init')
std_rmse_oce_atmo  = rmse_oce_atmo_da.std(dim='NN_init')

fig = plt.figure(figsize=(12, 5))

X  = np.arange(nRS)
dx = 0.03

for x in X:
    plt.axvline(x, linestyle='--', color='black', linewidth=0.1)

plt.errorbar(X-dx, mean_rmse_oce_only, xerr=0, yerr=std_rmse_oce_only, fmt='none', capsize=5, color='black', zorder=1)
plt.errorbar(X+dx, mean_rmse_oce_atmo, xerr=0, yerr=std_rmse_oce_atmo, fmt='none', capsize=5, color='black', zorder=1)

plt.scatter(X-dx, mean_rmse_oce_only, marker='d', s=70, edgecolor='black', color='cyan',  label='Train on 100 Ocean samples')
plt.scatter(X+dx, mean_rmse_oce_atmo, marker='d', s=70, edgecolor='black', color='green', label='Train on 100 Ocean + 1000 Atmos. samples')

offset_yp = np.array([0.003, 0.004, 0.005, 0.004, 0.004, 0.004, 0.003, 0.004, 0.005, 0.004, 0.004, 0.005])
offset_ym = np.array([0.005, 0.005, 0.005, 0.006, 0.006, 0.006, 0.004, 0.005, 0.006, 0.005, 0.005, 0.009])

for xi, y, offp in zip(X, mean_rmse_oce_only, offset_yp):
    plt.text(xi-dx, y+offp, f"{y:.3f}", ha='center', va='bottom', fontsize=10)
for xi, y, offm in zip(X, mean_rmse_oce_atmo, offset_ym):  
    plt.text(xi+dx, y-offm, f"{y:.3f}", ha='center', va='top', fontsize=10)

plt.legend(loc='upper right')

plt.yticks([0.15, 0.17, 0.19, 0.21, 0.23])
plt.xticks(ticks=X, labels=Labels_oce, fontsize=14)
plt.tick_params(axis='y', labelsize=12)

plt.title('Effect of including atmospheric data on ocean prediction', fontsize=14)
plt.ylabel('Normalized RMSE', fontsize=14)
plt.xlabel('Ocean simulation sampled for training', fontsize=12)
plt.ylim(0.145, 0.235)

fig.savefig('Fig4.png', dpi=500)
# ------------------------------------------------------------

