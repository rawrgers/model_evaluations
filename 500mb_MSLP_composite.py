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
from xbootstrap import block_bootstrap
# import sompy
import matplotlib.patches as mpatches
from joblib import Parallel, delayed
from scipy import stats
# cluster = LocalCluster(n_workers=4, threads_per_worker=1)
# client = Client(cluster)

dask.config.set({"array.slicing.split_large_chunks": True,"allow_rechunk":True})
################################################################################
def regrid_bi(ds):

# Create new grid
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90,91,1)),
                         'lon': (['lon'], np.arange(0,360,1))
                        }
                       )
# Build Regridder
    regridder = xe.Regridder(ds.chunk({'lat':-1,'lon':-1}),ds_out,'bilinear')
# Apply Regridder
    ds_out = regridder(ds)
    return ds_out
################################################################################
def regrid_con(ds):

# Create new grid
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90,91,1)),
                         'lon': (['lon'], np.arange(0,360,1))
                        }
                       )
# Build Regridder
    regridder = xe.Regridder(ds.chunk({'lat':-1,'lon':-1}),ds_out,'conservative')
# Apply Regridder
    ds_out = regridder(ds)
    return ds_out
################################################################################
def ERA5_psl(tmin,psl,q,norm):
    psl = xr.open_mfdataset(psl,chunks='auto').sel(time=slice('1981','2010'))
    # tmin = xr.open_mfdataset(tmin).sel(time=slice('1981','2010'))

    psl = psl.rename({'latitude':'lat','longitude':'lon'})
    # tmin = tmin.rename({'latitude':'lat','longitude':'lon'})
    #
    # tmin = tmin.sel(lat=slice(51.5,46.5),lon=slice(235,244)).mean(['lat','lon']).t2m
    # tmin = tmin.sel(time=tmin['time.season']=='DJF')
    #
    # tmin = tmin.chunk(dict(time=-1))
    # min_times = tmin.where(tmin<tmin.quantile(q=q),drop=True).time
    psl = psl.sel(time=psl['time.season']=='DJF')
    psl = regrid_con(psl.chunk(dict(lat=-1,lon=-1)))
    psl = psl.sel(lat=slice(30,80),lon=slice(150,270))


    if norm ==True:
        psl = psl.apply(normalize).squeeze()
    else:
        psl = psl.groupby('time.month')-psl.groupby('time.month').mean('time')
    # psl = psl.sel(time=min_times)
    return psl
################################################################################
def normalize(da):
    mn = da.groupby('time.month').mean('time')
    st = da.groupby('time.month').std('time')
    return ((da.groupby('time.month')-mn).groupby('month')/st).drop('month')
################################################################################
def load_CMIP(tmin,psl,q,norm):
    """
    Load data for the given source
    """
    dst={};dsp={};
    for source_id in tqdm(pd.merge(tmin['source_id'],psl['source_id'],how='inner').values):
            source_id = source_id[0]
            # vad = tmin[(tmin.source_id==source_id)].zstore.values[0]    # tmin
            # gcs = gcsfs.GCSFileSystem(token='anon')
            # dst[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True,chunks='auto',use_cftime=True)

            vad = psl[(psl.source_id==source_id)].zstore.values[0]      # psl
            gcs = gcsfs.GCSFileSystem(token='anon')
            dsp[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True,chunks='auto',use_cftime=True)

            # dst[source_id] = dst[source_id].sel(time=slice('1981','2010'))
            # dst[source_id] = dst[source_id].sel(time=dst[source_id]['time.season']=='DJF')
            # dst[source_id] = regrid(dst[source_id])

            # dst[source_id] = dst[source_id].sel(lat=slice(46.5,51.5),lon=slice(235,244)).mean(['lat','lon']).tasmin
            # dst[source_id] = dst[source_id].chunk(dict(time=-1))
            # min_times = dst[source_id].where(dst[source_id]<=dst[source_id].quantile(q=q),drop=True).time

            dsp[source_id] = dsp[source_id].sel(time=slice('1981','2010'))
            dsp[source_id] = dsp[source_id].sel(time=dsp[source_id]['time.season']=='DJF')
            # dsp[source_id] = regrid(dsp[source_id])
            dsp[source_id] = dsp[source_id].sel(lat=slice(30,80),lon=slice(150,270))
            dsp[source_id] = dsp[source_id].chunk(dict(time=-1))

            if norm == True:
                dsp[source_id] = dsp[source_id].apply(normalize).squeeze()
            else:
                dsp[source_id] = dsp[source_id].groupby('time.month')-dsp[source_id].groupby('time.month').mean('time').psl
            # dsp[source_id] = dsp[source_id].sel(time=min_times)

    return dsp
################################################################################
# def SOM(da,name,q,globe,norm):
#     if globe == False:
#         try:
#             da = {model:da[model].sel(lat=slice(30,80),lon=slice(150,270)) for model in da}
#         except:
#             pass
#         dmn = 'small'
#     else:
#         dmn = 'globe'
#
#     if norm == True:
#         lvls = np.arange(-3, 3.1, 0.5)
#         units = 'std dev'
#         try:
#             data = np.array()
#             for model in da:
#                 data.append(da[model],psl.values,dim=0)
#         except:
#             data=da.values
#
#     else:
#         lvls = levels=np.arange(-50, 51, 5)
#         units = 'hPa'
#         try:
#             data = np.empty((27,6171))
#             for model in da:
#                 nt,ny,nx = da[model].psl.shape
#                 data = np.vstack((data,np.reshape(da[model].psl.values/100, [nt, ny*nx], order='F')))
#         except:
#             data=da.values/100
#
#     nt,ny,nx = data.shape
#     data = np.reshape(data, [nt, ny*nx], order='F')
#     # data = data[27:,:]
#
#     sm = sompy.SOMFactory().build(data, mapsize=(2,5), normalization=None, initialization='random')
#     sm.train(n_job=1, verbose=False)
#
#     codebook =  sm.codebook.matrix
#     x,y = np.meshgrid(da.lon, da.lat)
#     proj = ccrs.Orthographic(210,55)
#     fig, axes = plt.subplots(2,5, figsize=(16,8), subplot_kw=dict(projection=proj))
#
#     for i in range(sm.codebook.nnodes):
#         onecen = codebook[i,:].reshape(ny,nx, order='F')
#         cs = axes.flat[i].contourf(x, y, onecen,
#                                    levels=lvls,
#                                    transform=ccrs.PlateCarree(),
#                                    cmap='RdBu_r')
#
#         cb=fig.colorbar(cs, ax=axes.flat[i], shrink=0.8, aspect=20)
#         cb.set_label('[unit: {}]'.format(units),labelpad=1)
#         axes.flat[i].coastlines()
#         axes.flat[i].set_global()
#         axes.flat[i].set_extent((150,270,30,80), crs = ccrs.PlateCarree())
#
#
#     plt.suptitle('{} mean SLP SOMs'.format(name),fontsize=22,y=0.9)
#     # plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/SOM/{}_q={}_{}.png'.format(name,str(q),dmn),dpi=400)
#
#     from sompy.visualization.bmuhits import BmuHitsView
#
#     vhts  = BmuHitsView(3, 4, "Amount of each regime: {}".format(model),text_size=16)
#
# # do the K-means clustering on the SOM grid, sweep across k = 2 to 20
#     from sompy.visualization.hitmap import HitMapView
#     K = 20 # stop at this k for SSE sweep
#     [labels, km, data, SSE] = sm.cluster(K,opt=0)
#     # print(labels)
#     # print(km)
#     # hits = HitMapView(4,7,title="Clustering",text_size=12)
#     # a=hits.show(sm)
#
#     # vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="RdBu_r", logaritmic=False)
#     # plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/SOM/{}_counts.png'.format(model),dpi=400)
#
#     # plt.show(block=False)
#
#     return sm
################################################################################
def COMP(da,name,q,globe,norm,sig_points):
    if globe == False:
        da = da.sel(lat=slice(30,80),lon=slice(150,270))
        dmn = 'small'
    else:
        dmn = 'globe'

    if norm == True:
        lvls = np.arange(-3, 3.1, 0.5)
        units = 'std dev'
        try:
            data = np.empty((1,181,360))
            for model in da:
                data.append(da[model].psl.mean('time').values,dim=0)
        except:
            data=da.mean('time').values

    else:
        lvls = levels=np.arange(-40, 41, 5)
        units = 'hPa'
        data=da.values/100

    # data = data[1:,:]
    # data = data.mean(axis=0).reshape(ny,nx, order='F')

    # x,y = np.meshgrid(da[model].lon, da[model].lat)
    x,y = np.meshgrid(da.lon, da.lat)

    proj = ccrs.PlateCarree(central_longitude=210)
    fig, ax = plt.subplots(figsize=(12,8), subplot_kw=dict(projection=proj))
    cs = ax.contourf(x, y, data,
                        levels=lvls,
                        transform=ccrs.PlateCarree(),
                        cmap='RdBu_r')
    ax.scatter(x[sig_points], y[sig_points], color='black', transform=ccrs.PlateCarree(), s=1)
    cb=fig.colorbar(cs, ax=ax, shrink=0.7, aspect=20)
    cb.set_label('[{}]'.format(units),labelpad=1,fontsize='xx-large')
    cb.ax.tick_params(labelsize='xx-large')

    ax.add_patch(mpatches.Rectangle(xy=[190, 48], width=40, height=17,
                                    edgecolor='black',
                                    linewidth=3,
                                    fill=False,
                                    transform=ccrs.PlateCarree()))
    ax.set_title('(a)',fontsize=22,loc='left')
    ax.coastlines()
    ax.set_global()
    ax.set_extent((150,270,30,80), crs = ccrs.PlateCarree())
    plt.tight_layout()

    plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/{}_q={}_{}_composite.png'.format(name,str(q),dmn),dpi=400)
    return fig,ax
################################################################################
def bootstrap(ds,da):
    lat,lon = (da.lat,da.lon)
    da = da.values
    pval = np.ones((lat.size*lon.size))*np.nan
    niter = 5000
    mc = np.ones((niter,da.size))*np.nan
    ds = ds.values

    for n in tqdm(range(niter)):
        t = np.random.choice(range(ds.shape[0]), size=28, replace=False)
        mc[n,:] = ds[t,:,:].reshape(-1,lat.size*lon.size).mean(axis=0).squeeze()

    for i in range(da.size):
        pval[i] = stats.percentileofscore(mc[:,i],da.reshape(lat.size*lon.size)[i])/100.

    pval = pval.reshape((lat.size,lon.size))
    significant_points = (pval > 0.975) | (pval < 0.025)
    return significant_points
################################################################################
def save_bootstrap_cmip(da,da_mean,itr):
    lat,lon = (da_mean.lat,da_mean.lon)
    # pval = np.ones((lat.size*lon.size))*np.nan
    # mc = np.ones((niter,da.size))*np.nan

    my_list = []
    for model in da:
        dd = da[model].psl.expand_dims('model')
        # Get the number of time steps
        n_times = len(dd['time'])

        # Choose 27 random indices
        random_time_indices = np.random.choice(n_times, size=int(n_times*0.01), replace=False)

        # Select the corresponding times from the DataArray
        random_times = dd.isel(time=random_time_indices).mean('time')
        my_list.append(random_times)
    combined = xr.combine_by_coords(my_list).mean('model').compute()
    combined['itr'] = itr
    combined = combined.assign_coords(itr=itr)
    combined = combined.expand_dims('itr')
    combined.to_netcdf(f'/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/bootstrap_mslp/{itr}.nc')
    return combined
    # results.append(xr.combine_by_coords(my_list).mean('model').stack(space=('lat', 'lon')).psl)
    # combined_data = xr.concat(results, dim='iterations')
    # da_mean = da_mean.stack(space=('lat', 'lon')).psl
    # Calculate the percentile of da_mean relative to combined_data

    # # Initialize an empty array to store percentile scores
    # percentile_scores = np.empty_like(da_mean)
    #
    # # Iterate over each point in da_mean
    # for index, value in tqdm(np.ndenumerate(da_mean)):
    #     # Get the corresponding value in combined_data
    #     combined_data_value = combined_data.sel(space = combined_data['space'][index]).values
    #
    #     # Calculate the percentile score for the current point in da_mean relative to combined_data
    #     percentile_scores[index] = stats.percentileofscore(combined_data_value, value)
    # return sig
################################################################################
def test(ds,da):
    lat,lon = (da.lat,da.lon)
    da = da.values.reshape(lat.size*lon.size)
    pval = np.ones((lat.size*lon.size))*np.nan
    niter = len(ds.itr)
    mc = np.ones((niter,da.size))*np.nan

    mc[:,:] = ds.values.reshape(niter,lat.size*lon.size)

    for i in range(da.size):
        pval[i] = stats.percentileofscore(mc[:,i],da[i])/100.

    pval = pval.reshape((lat.size,lon.size))
    significant_points = (pval > 0.975) | (pval < 0.025)
    nonsig = (pval < 0.975) & (pval > 0.025)

    # Assign values of 1 to the points that pass the test
    pval[significant_points] = 1
    pval[nonsig] = 0
    return significant_points
################################################################################

def main():

    q=0.01
    norm=False
    ERA5 = ERA5_psl(sorted(glob.glob('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/t2m/t2m.day.min.*.nc')),
                    '/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/msl/msl.day.mean.nc',
                    q=q,norm=norm)


    ERA5_mins = xr.open_dataset('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/ERA5_mintemp_msl_values.nc')

    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
    # tmin = df.query("activity_id=='CMIP' & experiment_id=='historical' & member_id=='r1i1p1f1' & table_id=='day' & variable_id=='tasmin' & grid_label=='gn'")
    # psl = df.query("activity_id=='CMIP' & experiment_id=='historical' & member_id=='r1i1p1f1' & table_id=='day' & variable_id=='psl' & grid_label=='gn'")

    psl_data = load_CMIP(df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='tasmin' & grid_label=='gn'").drop_duplicates(subset=['source_id']),
                            df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='psl' & grid_label=='gn'").drop_duplicates(subset=['source_id']),
                            q=q,norm=norm)#.compute(scheduler="single-threaded")

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

    da = {model:psl_data[model].assign_coords(model=model).drop(['month','time_bnds']).chunk(dict(lat=-1,lon=-1)).compute() for model in keys}

    da_times = {model:xr.open_dataset(f'/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/{model}_mintemp_msl_values.nc',use_cftime=True).expand_dims('model') for model in da}
    # np.sum([len(xr.open_dataset(f'/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/{model}_mintemp_msl_values.nc',use_cftime=True).expand_dims('model').time) for model in da])
    da_mean = xr.combine_by_coords([da_times[model].mean('time') for model in da_times]).mean('model')
    da_mean = da_mean.sel(lat=slice(30,80),lon=slice(150,270))



    boots = xr.open_mfdataset(glob.glob('/home/disk/rocinante/rawrgers/Data/CMIP/Cold_Events_Project/bootstrap_mslp/*.nc')).chunk(dict(itr=-1,lat=-1,lon=-1)).load()
    sig_points = test(boots.psl,da_mean.psl)
    # itrs = range(5000)
    # Parallel(n_jobs=10)(delayed(save_bootstrap_cmip)(da, da_mean, itr) for itr in itrs)
    COMP(da_mean.psl,'CMIP6',q=q,globe=False,norm=norm,sig_points=sig_points)

    sig_points = bootstrap(ERA5.msl.drop('month'),ERA5_mins.msl.sel(lat=slice(30,80),lon=slice(150,270)).mean('time'))
    COMP(ERA5_mins.msl.mean('time'),'ERA5',q=q,globe=False,norm=norm,sig_points=sig_points)
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
