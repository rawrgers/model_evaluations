import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
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
dask.config.set({"array.slicing.split_large_chunks": True,"allow_rechunk":True})
################################################################################
def regrid_1x1_bi(ds):

# Create new grid

    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90.0,91.0,1)),
                         'lon': (['lon'], np.arange(0.0,360,1))
                        }
                       )
# Build Regridder
    regridder = xe.Regridder(ds,ds_out,'bilinear')
# Apply Regridder
    ds_out = regridder(ds)
    return ds_out
################################################################################
def regrid_1x1_con(ds):

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
def load_CMIP(df):
    """
    Load data for the given source
    """
    keys = df['source_id'].values
    ds = {}
    skew = {}
    for source_id in tqdm(keys):
        vad = df[(df.source_id==source_id)].zstore.values[0]
        gcs = gcsfs.GCSFileSystem(token='anon')
        ds[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True,use_cftime=True)
        ds[source_id] = regrid_1x1_bi(ds[source_id])
        ds[source_id] = ds[source_id].sel(time=slice('1981','2010'))
        ds[source_id] = ds[source_id].sel(time=ds[source_id]['time.season']=='DJF')
        skew[source_id] = ds[source_id].reduce(func=scipy.stats.skew,dim='time')
        ds[source_id] = ds[source_id].groupby('time.month') - ds[source_id].groupby('time.month').mean('time')
        ds[source_id] = ds[source_id].tasmin.chunk(dict(time=-1)).quantile(q=0.01,dim='time')
    return ds,skew
################################################################################
def load_ERA(files):
    da = xr.open_mfdataset(files,chunks='auto')
    # da = da.resample(time='1D').min()
    # da.to_netcdf(glob.glob('/home/disk/rocinante/DATA/ERA5/daily/T02MIN/*.nc'))
    # da.close()
    # da = da.t2m.chunk(dict(time=10)).to_netcdf('/home/disk/rocinante/DATA/ERA5/daily/ERA5_daily_T2MIN.nc')
    da = da.t2m.chunk(dict(time=-1)).sel(latitude=slice(90,0),longitude=slice(200,300)).compute()
    da = da.sel(time=da['time.season']=='DJF').compute()
    da = da.groupby('time.month') - da.groupby('time.month').mean('time')
    skew = da.reduce(func=scipy.stats.skew,dim='time')
    da = regrid_1x1_con(da)
    skew = regrid_1x1_con(skew)
    ERA5 = da.quantile(q=0.01,dim='time')
    return ERA5,skew
################################################################################
def main():
    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        tasmin,model_skew = load_CMIP(df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='tasmin' & grid_label=='gn'").drop_duplicates(subset=['source_id']))
    ERA5,era5_skew = load_ERA(sorted(glob.glob('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/t2m/t2m.day.min.*.nc')))
    ERA5 = ERA5['t2m'].compute()#.rename({'latitude':'lat','longitude':'lon'}).drop('quantile')
    era5_skew = era5_skew['t2m'].compute()

    bias = {}
    keys = ['MPI-ESM-1-2-HAM',
             'NorESM2-MM',
             'ACCESS-CM2',
             'TaiESM1',
             'UKESM1-0-LL',
             'ACCESS-ESM1-5',
             'MIROC6',
             'CMCC-ESM2',
             'HadGEM3-GC31-MM',
             'MPI-ESM1-2-HR',
             'MIROC-ES2L',
             'AWI-ESM-1-1-LR',
             'NorCPM1',
             'CanESM5',
             'BCC-CSM2-MR',
             'MRI-ESM2-0',
             'HadGEM3-GC31-LL',
             'MPI-ESM1-2-LR',
             'NorESM2-LM',
             'BCC-ESM1',
             'SAM0-UNICON']
    bias = {model:tasmin[model]-ERA5 for model in keys}
    skew_bias = {model:model_skew[model]['tasmin']-era5_skew for model in keys}

    for model in tqdm(bias):
        bias[model].to_netcdf('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/1st_pctl_bias/{}.nc'.format(model))
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
        extent = [220, 270, 20, 70]
        ax.set_extent(extent)
        ax.gridlines()
        ax.coastlines(resolution='50m')
        ax.set_title(model,loc='left',fontsize=20)
        biasplot = bias[model].plot(vmin=-8,transform=ccrs.PlateCarree(),add_colorbar=False)
        ax.set_title('',loc='center')
        cb = {'label': 'Bias (°C)', 'orientation': 'horizontal', 'shrink': 0.9}
        cbar = plt.colorbar(biasplot, ax=ax, **cb)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(cb['label'], fontsize=20)
        plt.tight_layout()
        plt.savefig('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/1st_pctl_bias/{}.png'.format(model),dpi=400)

    for model in tqdm(skew_bias):
        skew_bias[model].to_netcdf('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/1st_pctl_bias/{}_skew.nc'.format(model))
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
        extent = [220, 270, 20, 70]
        ax.set_extent(extent)
        ax.gridlines()
        ax.coastlines(resolution='50m')
        ax.set_title(model,loc='left',fontsize=20)
        biasplot = skew_bias[model].plot(vmin=-1,transform=ccrs.PlateCarree(),add_colorbar=False)
        ax.set_title('',loc='center')
        cb = {'label': 'Bias in Skewness', 'orientation': 'horizontal', 'shrink': 0.9}
        cbar = plt.colorbar(biasplot, ax=ax, **cb)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(cb['label'], fontsize=20)
        plt.tight_layout()
        plt.savefig('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/1st_pctl_bias/{}_skew.png'.format(model),dpi=400)


# Absolute Bias
    # MMM = sum(bias[model] for model in bias)/len(bias)
    combined = {model:bias[model].assign_coords(model=model) for model in bias}
    combined = xr.concat([combined[model] for model in combined],dim='model')
    # MMQ = combined.chunk(dict(model=-1)).quantile(q=0.5,dim='model')
    # MMM = combined.chunk(dict(model=-1)).mean(dim='model').compute()
    MMM = combined.chunk(dict(model=-1)).quantile(q=0.5,dim='model').compute()
    fontsize=22
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
    cbk = {'label':'Median Bias (°C)','orientation':'horizontal','location':'bottom'}
    ax.gridlines()
    ax.coastlines(resolution='50m')
    # ax.set_xlim([])
    ax.set_ylim([20,70])


    MMM.drop('quantile').plot(vmin=-8,transform=ccrs.PlateCarree(),cmap=plt.cm.bwr,cbar_kwargs=cbk)
    lons = [-125, -125, -120.5, -120.5]
    lats = [47, 50, 50, 47]
    ring = LinearRing(list(zip(lons, lats)))
    ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='white')
    # plt.title('CMIP6 Multi-model Median Bias in 1st Percentile DJF Minimum Temperature (1981-2010)',fontsize=8)
    plt.tight_layout()
    plt.savefig('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/1st_pctl_bias/Median_Bias_PNW.png',dpi=400)

#######
    agreement = {}
    for model in tqdm(bias):
        agreement[model] = bias[model].copy()
        agreement[model] = agreement[model].where((bias[model]<0),1)
        agreement[model] = agreement[model].where((agreement[model]>0),0)

    counts = sum(agreement[model] for model in agreement).compute()
    map_proj = ccrs.PlateCarree()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16, 12),subplot_kw={'projection': map_proj})
    extent1 = [205, 295, 30, 70]
    extent2 = [220, 260, 40, 60]
    # Colorbar label font size
    fontsize = 36

    # Modify the cbar_kwargs dictionaries to include the fontsize parameter
    cbk1 = {'label': 'Mean Bias (°C)', 'orientation': 'horizontal', 'shrink': 0.9}
    cbk2 = {'label': 'Model Count', 'orientation': 'horizontal', 'shrink': 0.9}
    mean1 = MMM.drop('quantile').plot(ax=ax1,vmin=-8,cmap=plt.cm.bwr,add_colorbar=False)

    ax1.set_extent(extent1)
    ax1.coastlines(resolution='50m')
    ax1.set_title('(a)',loc='center',fontsize=fontsize)

    count1 = counts.drop('quantile').plot(ax=ax2,vmin=0,vmax=21,levels=22,cmap=plt.cm.bwr,add_colorbar=False)

    ax2.set_extent(extent1)
    ax2.coastlines(resolution='50m')
    ax2.set_title('(b)',loc='center',fontsize=fontsize)

    lons = [-125, -125, -116, -116]
    lats = [46.5, 51.5, 51.5, 46.5]
    ring = LinearRing(list(zip(lons, lats)))
    ax1.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=3)
    ax2.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=3)
    # plt.tight_layout()
    # plt.show(block=False)
    # plt.title('CMIP6 Model Agreement in + 1st Percentile DJF Minimum Temperature Bias (1981-2010)',fontsize=8)
#----------

    mean2 = MMM.drop('quantile').plot(ax=ax3,vmin=-8,cmap=plt.cm.bwr,add_colorbar=False)
    # After creating the colorbars, adjust the font size of colorbar labels
    mean_cbar = plt.colorbar(mean2, ax=ax3, **cbk1)
    mean_cbar.ax.tick_params(labelsize=fontsize)
    mean_cbar.set_label(cbk1['label'], fontsize=fontsize)
    ax3.set_extent(extent2)
    ax3.coastlines(resolution='50m')
    # ax3.set_title('(a)',loc='left',fontsize=18)

    count2 = counts.drop('quantile').plot(ax=ax4,vmin=0,vmax=21,levels=22,cmap=plt.cm.bwr,add_colorbar=False)
    count_cbar = plt.colorbar(count2, ax=ax4, **cbk2)
    count_cbar.ax.tick_params(labelsize=fontsize)
    count_cbar.set_label(cbk2['label'], fontsize=fontsize)
    ax4.set_extent(extent2)
    ax4.coastlines(resolution='50m')
    # ax4.set_title('(b)',loc='left',fontsize=18)

    lons = [-125, -125, -116, -116]
    lats = [46.5, 51.5, 51.5, 46.5]
    ring = LinearRing(list(zip(lons, lats)))
    ax3.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=3)
    ax4.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=3)
    # plt.tight_layout()
    # plt.show(block=False)
    # plt.title('CMIP6 Model Agreement in + 1st Percentile DJF Minimum Temperature Bias (1981-2010)',fontsize=8)
    for ax in (ax1, ax2,ax3, ax4):
        ax.set_aspect(1, adjustable="box")

    plt.tight_layout()

    plt.savefig('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/1st_pctl_bias/mean_count_NA_multi.png',dpi=400)

# Difference in Skewness



    # MMM = sum(bias[model] for model in bias)/len(bias)
    combined = {model:skew_bias[model].assign_coords(model=model) for model in bias}
    combined = xr.concat([combined[model] for model in combined],dim='model')
    # MMQ = combined.chunk(dict(model=-1)).quantile(q=0.5,dim='model')
    # MMM = combined.chunk(dict(model=-1)).mean(dim='model').compute()
    MMM = combined.chunk(dict(model=-1)).quantile(q=0.5,dim='model').compute()
    fontsize=22
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))
    cbk = {'label':'Difference in Skewness','orientation':'horizontal','location':'bottom'}
    ax.gridlines()
    ax.coastlines(resolution='50m')
    # ax.set_xlim([])
    ax.set_ylim([20,70])


    MMM.drop('quantile').plot(vmin=-2,transform=ccrs.PlateCarree(),cmap=plt.cm.bwr,cbar_kwargs=cbk)
    lons = [-125, -125, -120.5, -120.5]
    lats = [47, 50, 50, 47]
    ring = LinearRing(list(zip(lons, lats)))
    ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='white')
    # plt.title('CMIP6 Multi-model Median Bias in 1st Percentile DJF Minimum Temperature (1981-2010)',fontsize=8)
    plt.tight_layout()
    plt.savefig('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/1st_pctl_bias/Median_Skewness_PNW.png',dpi=400)

#######
    agreement = {}
    for model in tqdm(skew_bias):
        agreement[model] = skew_bias[model].copy()
        agreement[model] = agreement[model].where((skew_bias[model]<0),1)
        agreement[model] = agreement[model].where((skew_agreement[model]>0),0)

    counts = sum(agreement[model] for model in agreement).compute()
    map_proj = ccrs.PlateCarree()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16, 12),subplot_kw={'projection': map_proj})
    extent1 = [205, 295, 30, 70]
    extent2 = [220, 260, 40, 60]
    # Colorbar label font size
    fontsize = 36

    # Modify the cbar_kwargs dictionaries to include the fontsize parameter
    cbk1 = {'label': 'Difference in Skewness', 'orientation': 'horizontal', 'shrink': 0.9}
    cbk2 = {'label': 'Model Count', 'orientation': 'horizontal', 'shrink': 0.9}
    mean1 = MMM.drop('quantile').plot(ax=ax1,vmin=-2,cmap=plt.cm.bwr,add_colorbar=False)

    ax1.set_extent(extent1)
    ax1.coastlines(resolution='50m')
    ax1.set_title('(a)',loc='center',fontsize=fontsize)

    count1 = counts.drop('quantile').plot(ax=ax2,vmin=0,vmax=21,levels=22,cmap=plt.cm.bwr,add_colorbar=False)

    ax2.set_extent(extent1)
    ax2.coastlines(resolution='50m')
    ax2.set_title('(b)',loc='center',fontsize=fontsize)

    lons = [-125, -125, -116, -116]
    lats = [46.5, 51.5, 51.5, 46.5]
    ring = LinearRing(list(zip(lons, lats)))
    ax1.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=3)
    ax2.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=3)
    # plt.tight_layout()
    # plt.show(block=False)
    # plt.title('CMIP6 Model Agreement in + 1st Percentile DJF Minimum Temperature Bias (1981-2010)',fontsize=8)
#----------

    mean2 = MMM.drop('quantile').plot(ax=ax3,vmin=-2,cmap=plt.cm.bwr,add_colorbar=False)
    # After creating the colorbars, adjust the font size of colorbar labels
    mean_cbar = plt.colorbar(mean2, ax=ax3, **cbk1)
    mean_cbar.ax.tick_params(labelsize=fontsize)
    mean_cbar.set_label(cbk1['label'], fontsize=fontsize)
    ax3.set_extent(extent2)
    ax3.coastlines(resolution='50m')
    # ax3.set_title('(a)',loc='left',fontsize=18)

    count2 = counts.drop('quantile').plot(ax=ax4,vmin=0,vmax=21,levels=22,cmap=plt.cm.bwr,add_colorbar=False)
    count_cbar = plt.colorbar(count2, ax=ax4, **cbk2)
    count_cbar.ax.tick_params(labelsize=fontsize)
    count_cbar.set_label(cbk2['label'], fontsize=fontsize)
    ax4.set_extent(extent2)
    ax4.coastlines(resolution='50m')
    # ax4.set_title('(b)',loc='left',fontsize=18)

    lons = [-125, -125, -116, -116]
    lats = [46.5, 51.5, 51.5, 46.5]
    ring = LinearRing(list(zip(lons, lats)))
    ax3.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=3)
    ax4.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=3)
    # plt.tight_layout()
    # plt.show(block=False)
    # plt.title('CMIP6 Model Agreement in + 1st Percentile DJF Minimum Temperature Bias (1981-2010)',fontsize=8)
    for ax in (ax1, ax2,ax3, ax4):
        ax.set_aspect(1, adjustable="box")

    plt.tight_layout()

    plt.savefig('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/1st_pctl_bias/median_skew_count_NA_multi.png',dpi=400)
