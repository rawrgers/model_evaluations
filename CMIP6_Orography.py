# Need to demonstrate differences in orography spatially. Maybe look do something
# similar to the model bias and agreement plots?

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.pylab import *
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

# cluster = LocalCluster(n_workers=4, threads_per_worker=1)
# client = Client(cluster)
#
# dask.config.set({"array.slicing.split_large_chunks": True,"allow_rechunk":True})
################################################################################
def regrid(ds,ERA5):

# Create new grid
    ds_out = ERA5
    # ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90.0,91.0,1)),
    #                      'lon': (['lon'], np.arange(0.0,360.0,1))
    #                     }
    #                    )
# Build Regridder
    regridder = xe.Regridder(ds,ds_out,'bilinear')
# Apply Regridder
    ds_out = regridder(ds)
    return ds_out
################################################################################
def load_CMIP(df,ERA5):
    """
    Load data for the given source
    """
    ds = {}
    # keys = [source_id for source_id in df['source_id']]
    keys = ['GISS-E2-1-G','BCC-CSM2-MR','AWI-CM-1-1-MR','BCC-ESM1','SAM0-UNICON',
 'CanESM5','MRI-ESM2-0','HadGEM3-GC31-LL','MPI-ESM-1-2-HAM','UKESM1-0-LL',
 'MPI-ESM1-2-LR','MPI-ESM1-2-HR','NESM3','NorESM2-LM','FGOALS-g3',
 'NorCPM1','MIROC6','ACCESS-CM2','NorESM2-MM','ACCESS-ESM1-5','MIROC-ES2L',
 'HadGEM3-GC31-MM','AWI-ESM-1-1-LR','TaiESM1','CMCC-ESM2']
    for source_id in tqdm(keys):
        try:
            vad = df[(df.source_id==source_id)].zstore.values[0]
            gcs = gcsfs.GCSFileSystem(token='anon')
            ds[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True,use_cftime=True)
            ds[source_id] = regrid(ds[source_id],ERA5)
        except:
            print('No Elevation Data for {}'.format(source_id))
    return ds
################################################################################
def load_ERA(file):
    da = xr.open_mfdataset(file,chunks='auto').z/9.81 #change geopotential to height
    da = da.sel(expver=1)
    da = da.sel(time=slice('1981','2010'))
    da = da.sel(time=da['time.season']=='DJF').mean('time')
    return da
################################################################################
def main():
    ERA5 = load_ERA(glob.glob('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/fixed/geopotential.nc'))
    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
    orog = load_CMIP(df.query("activity_id=='CMIP'& experiment_id=='historical'& table_id=='fx' & variable_id=='orog' & grid_label=='gn'").drop_duplicates(subset=['source_id']),ERA5)


    bias = {}
    # bias = {model:orog[model].orog-ERA5 for model in orog}

    combined = {model:orog[model].orog.assign_coords(model=model) for model in orog}
    combined = xr.concat([combined[model] for model in combined],dim='model')
    # MMQ = combined.chunk(dict(model=-1)).quantile(q=0.5,dim='model')
    MMM = combined.chunk(dict(model=-1)).mean(dim='model')
    # fig, ax = plt.subplots(figsize=(8,8),subplot_kw={'projection': ccrs.PlateCarree()})
    ERA5_summary = ERA5.sel(latitude=slice(51.5,46.5),longitude=slice(235,244)).mean(['latitude','longitude']).values # 969m
    MMM_summary = MMM.sel(latitude=slice(51.5,46.5),longitude=slice(235,244)).mean(['latitude','longitude']).values # 962m


    map_proj = ccrs.PlateCarree()
    font = matplotlib.font_manager.FontProperties(family='times new roman', size=14)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8),subplot_kw={'projection': map_proj})
    # cbk = {'label':'Mean Bias (m)','orientation':'vertical','location':'right'}
    lons = [-125, -125, -116, -116]
    lats = [46.5, 51.5, 51.5, 46.5]
    ring = LinearRing(list(zip(lons, lats)))

    extent = [225, 255, 35, 60]
    ax1.set_extent(extent)
    ax1.coastlines(resolution='50m')
    img = ERA5.drop('expver').plot(ax=ax1,vmin=0,vmax=2500,transform=ccrs.PlateCarree(),add_colorbar=False)
    cbar = plt.colorbar(img,ax=ax1,extend='max',spacing='proportional',
                orientation='horizontal',pad=0.05, format="%.0f",label='Elevation (m)')
    text = cbar.ax.xaxis.label
    ax1.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=2)
    text.set_font_properties(font)
    ax1.set_title('(a)',fontsize=16,loc='left')
    ax1.set_title('ERA5',fontsize=16,loc='center')
    ax2.set_extent(extent)

    # ax.gridlines()
    ax2.coastlines(resolution='50m')
    img = MMM.plot(ax=ax2,vmin=0,vmax=2500,transform=ccrs.PlateCarree(),add_colorbar=False)
    cbar = plt.colorbar(img,ax=ax2,extend='max',spacing='proportional',
                orientation='horizontal',pad=0.05, format="%.0f",label='Elevation (m)')
    text = cbar.ax.xaxis.label
    ax2.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=2)
    font = matplotlib.font_manager.FontProperties(family='times new roman', size=14)
    text.set_font_properties(font)
    ax2.set_title('(b)',fontsize=16,loc='left')
    ax2.set_title('CMIP6',fontsize=16,loc='center')
    plt.tight_layout()
    plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/Orography.png',dpi=400)
    plt.show(block=False)
