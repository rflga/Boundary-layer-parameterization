import numpy as np
import netCDF4 as nc

float_type = "f8"
# float_type = "f4"

# Get number of vertical levels and size from .ini file
with open('moistcblles.ini') as f:
    for line in f:
        if(line.split('=')[0]=='ktot'):
            kmax = int(line.split('=')[1])
        if(line.split('=')[0]=='zsize'):
            zsize = float(line.split('=')[1])

dz = zsize / kmax

z   = np.linspace(0.5 * dz, zsize - 0.5 * dz, kmax)
th  = np.empty(z.shape)
thl = np.empty(z.shape)
qt  = np.empty(z.shape)
u   = np.empty(z.shape)
ug  = np.empty(z.shape)

dthetadz = 0.005
dthetadz = 0.008 # for q17

dqtdz    = -5e-6

# linearly stratified profile
z0  = 500     # Initial BLH
th0 = 300.    # Ref temp, K
qt0 = 0.019   # Ref qt,   kg/kg

for k in range(kmax):
    if z[k] <= z0:
        th[k]  = th0
        thl[k] = th0
        qt[k]  = qt0
    else:
        th[k]  = th0 + dthetadz * (z[k] - z0)
        thl[k] = th0 + dthetadz * (z[k] - z0)
        qt[k]  = qt0 +    dqtdz * (z[k] - z0)

wind  = 10.
u[:]  = wind
ug[:] = wind

nc_file = nc.Dataset("moistcblles_input.nc", mode="w", datamodel="NETCDF4", clobber=False)

nc_file.createDimension("z", kmax)
nc_z  = nc_file.createVariable("z" , float_type, ("z"))

nc_group_init = nc_file.createGroup("init");
nc_u   = nc_group_init.createVariable("u" , float_type, ("z"))
nc_th  = nc_group_init.createVariable("th", float_type, ("z"))
nc_thl = nc_group_init.createVariable("thl", float_type, ("z"))
nc_qt  = nc_group_init.createVariable("qt", float_type, ("z"))
nc_ug  = nc_group_init.createVariable("u_geo", float_type, ("z"))

nc_z [:]  = z [:]
nc_u [:]  = u [:]
nc_th[:]  = th[:]
nc_thl[:] = thl[:]
nc_qt[:]  = qt[:]
nc_ug[:]  = ug[:]

nc_file.close()
