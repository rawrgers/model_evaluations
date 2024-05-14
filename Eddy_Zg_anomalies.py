import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.util as util
import matplotlib.pyplot as plt
import matplotlib
from shapely import geometry
from collections import namedtuple
import os
from dask.distributed import Client,LocalCluster
import dask
import glob as glob
from datetime import datetime, timedelta
from matplotlib import colors as mplc
import regionmask
from tqdm import tqdm
import xesmf as xe
import zarr
import gcsfs
from shapely.geometry.polygon import LinearRing
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sompy
import matplotlib.patches as mpatches
cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)

dask.config.set({"array.slicing.split_large_chunks": True,"allow_rechunk":True})
################################################################################
def regrid_bi(ds):

# Create new grid
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90,91,1)),
                         'lon': (['lon'], np.arange(0,360,1))
                        }
                       )
# Build Regridder
    regridder = xe.Regridder(ds,ds_out,'bilinear')
# Apply Regridder
    ds_out = regridder(ds)
    return ds_out
################################################################################
def regrid_con(ds):

# Create new grid
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(0,91.0,1)),
                         'lon': (['lon'], np.arange(200,301,1))
                        }
                       )
# Build Regridder
    # Assuming your latitude and longitude variables are named 'latitude' and 'longitude'
    ds = ds.to_dataset().sel(latitude=slice(None, None, -1))
    # Set latitude and longitude as spatial dimensions
    ds = ds.set_coords(['latitude', 'longitude'])
    # ds = ds.swap_dims({'latitude': 'y', 'longitude': 'x'})
    regridder = xe.Regridder(ds,ds_out,'conservative')
# Apply Regridder
    ds_out = regridder(ds)
    return ds_out
################################################################################
def load_ERA5(zg):
    zg = xr.open_mfdataset(zg).sel(time=slice('1981','2010')).Z500/9.81
    zg = regrid_con(zg)
    zg = zg.sel(time=zg['time.season']=='DJF')

    # zg = zg.groupby('time.month')-zg.groupby('time.month').mean('time')
    zg = zg.groupby('lat') - zg.groupby('lat').mean('lon')
    zg = zg.mean('time')

    return zg
################################################################################
def load_CMIP(tmin,zg):
    """
    Load data for the given source
    """
    dst={};dsp={};
    keys = ['BCC-CSM2-MR', 'BCC-ESM1', 'SAM0-UNICON', 'CanESM5', 'MRI-ESM2-0', 'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'NorESM2-LM', 'MIROC6', 'ACCESS-CM2', 'NorESM2-MM', 'ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'TaiESM1', 'NorCPM1', 'CMCC-ESM2']

    for source_id in tqdm(keys):
        source_id = source_id

        vad = zg[(zg.source_id==source_id)].zstore.values[0]      # zg
        gcs = gcsfs.GCSFileSystem(token='anon')
        dsp[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True)

        dsp[source_id] = regrid_bi(dsp[source_id])
        dsp[source_id] = dsp[source_id].sel(time=dsp[source_id]['time.season']=='DJF').zg
        dsp[source_id] = dsp[source_id].sel(plev = 50000,method='nearest')

        # zg = zg.groupby('time.month')-zg.groupby('time.month').mean('time')
        dsp[source_id] = dsp[source_id].groupby('lat') - dsp[source_id].groupby('lat').mean('lon')
        dsp[source_id] = dsp[source_id].mean('time')

    return dsp
################################################################################
def COMP(ERA5,CMIP6):
    lvls = levels=np.arange(-250, 251, 25)
    units = 'm'
    mod = np.empty((1,181*360))
    for model in CMIP6:
        ny,nx = CMIP6[model].shape
        mod = np.vstack((mod,np.reshape(CMIP6[model].values, [ny*nx], order='F')))

    mod = mod[1:,:]
    mod = mod.mean(axis=0).reshape(ny,nx, order='F')
    rean = ERA5
    # x,y = np.meshgrid(da[model].lon, da[model].lat)
    # ERA5, lons = util.add_cyclic_point(ERA5, coord=ERA.lon)
    x,y = np.meshgrid(rean.lon, rean.lat)

    proj = ccrs.PlateCarree(central_longitude=180)
    cmap  = matplotlib.cm.bwr.copy()
    cmap.set_bad('black',1.)
    fig, ax = plt.subplots(figsize=(12,8), subplot_kw=dict(projection=proj))
    cs = ax.contourf(x, y, rean,
                        levels=lvls,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap)

    cb=fig.colorbar(cs, ax=ax, shrink=0.8, aspect=20)
    cb.set_label('[{}]'.format(units),labelpad=1,fontsize='xx-large')
    cb.ax.tick_params(labelsize='x-large')

    # ax.add_patch(mpatches.Rectangle(xy=[190, 48], width=40, height=17,
    #                                 edgecolor='black',
    #                                 linewidth=3,
    #                                 fill=False,
    #                                 transform=ccrs.PlateCarree()))

    ax.set_title('(a) ERA5 500mb DJF Eddy Geopotential Height Anomaly',fontsize=22,loc='left')
    ax.coastlines()
    ax.set_global()
    # ax.set_extent((150,270,30,80), crs = ccrs.PlateCarree())
    plt.tight_layout()

    plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/ERA5_500_eddy_anomaly.png',dpi=400)
    return fig,rean,mod
################################################################################
def bootstrap(ds,da):
    lat,lon = (da.lat,da.lon)
    da = da.values
    pval = np.ones((lat.size*lon.size))*np.nan
    niter = 5000
    mc = np.ones((niter,da.size))*np.nan
    ds = ds.values

    for n in range(niter):
        t = np.random.choice(range(ds.shape[0]))
        mc[n,:] = t.reshape(:,lat.size*lon.size)

    for i in range(Q.size):
        pval[i] = stats.percentileofscore(mc[:,i],Q[i])/100.

    pval = pval.reshape((lat.size,lon.size))
    return sig
################################################################################

def main():

    ERA5 = load_ERA5('/home/disk/rocinante/DATA/ERA5/monthly/ERA5_monthly_Z500.nc').compute(scheduler="single-threaded")

    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')

    CMIP6 = load_CMIP(df.query("activity_id=='CMIP' & experiment_id=='historical' & member_id=='r1i1p1f1' & table_id=='day' & variable_id=='tasmin' & grid_label=='gn'"),
                        df.query("activity_id=='CMIP' & experiment_id=='historical' & member_id=='r1i1p1f1' & table_id=='Amon' & variable_id=='zg' & grid_label=='gn'"))#.compute(scheduler="single-threaded")

    CMIP6 = {model:CMIP6[model].assign_coords(model=model) for model in CMIP6}

    fig,rean,mod = COMP(ERA5,CMIP6)
    plt.show(block=False)
################
    nt,ny,nx = ERA5.shape
    x,y = np.meshgrid(ERA5.lon, ERA5.lat)
    proj = ccrs.Orthographic(210,55)
    fig, axes = plt.subplots(7,4, figsize=(12,16), subplot_kw=dict(projection=proj))

    lvls = levels=np.arange(-60, 61, 5)
    units = 'hPa'

    for i in range(ERA5.time.size):
        cs = axes.flat[i].contourf(x, y, ERA5.values[i,:,:]/100,
                                   levels=lvls,
                                   transform=ccrs.PlateCarree(),
                                   cmap='RdBu_r')

        cb=fig.colorbar(cs, ax=axes.flat[i], shrink=0.8, aspect=20)
        cb.set_label('[unit: {}]'.format(units),labelpad=1,fontsize='xx-small')
        cb.ax.tick_params(labelsize='xx-small')
        axes.flat[i].coastlines()
        axes.flat[i].set_global()
        axes.flat[i].set_extent((150,270,30,80), crs = ccrs.PlateCarree())

    plt.tight_layout()
        # plt.suptitle('{} Self Organized Maps (n=28)'.format(model),fontsize=22,y=0.9)
    plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/SOM/{}_full_maps_q=0.01.png'.format(model),dpi=400)

    plt.show(block=False)
