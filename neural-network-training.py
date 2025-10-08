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


# Inputs
ucoarse = xr.open_dataset('ucoarse.nc')
vcoarse = xr.open_dataset('vcoarse.nc')
tke     = xr.open_dataset('tke.nc')


# Outputs
subgrid_wu = xr.open_dataset('subgrid_wu.nc')
subgrid_wv = xr.open_dataset('subgrid_wv.nc')
