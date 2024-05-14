import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import proplot as plt
import os
from dask.distributed import Client,LocalCluster
import dask
import glob as glob
import zarr
import gcsfs
import seaborn as sns
import matplotlib.lines as mlines

#===============================================================================

def set_vars():
    lats = {
    'wa':[46.5,51.5],
    'ak':[48,67]
    }
    lons = {
    'wa':[235,244],
    'ak':[190,230]
    }
    return lats,lons
#===============================================================================

def load_ERA(file,lats,lons):
    da = xr.open_mfdataset(file,chunks='auto')
    if 't2m' in da.variables:
        da = da.sel(latitude=slice(lats['wa'][1],lats['wa'][0]),longitude=slice(lons['wa'][0],lons['wa'][1])) # WA WCASC is to 45.5 and 120.5
        da = da.groupby('time.month') - da.groupby('time.month').mean('time')
        da = da.mean(['latitude','longitude']).t2m
    else:
        da = da.sel(latitude=slice(lats['ak'][1],lats['ak'][0]),longitude=slice(lons['ak'][0],lons['ak'][1]))
        da = da.groupby('time.month') - da.groupby('time.month').mean('time')
        da = da.max(['latitude','longitude']).msl
    da = da.sel(time=slice('1981','2010'))
    da = da.sel(time=da['time.season']=='DJF')

    return da
#===============================================================================
def load_CMIP(df,lats,lons):
    """
    Load data for the given source
    """
    ds = {}
    for source_id in df['source_id']:
        try:
            vad = df[(df.source_id==source_id)].zstore.values[0]

            gcs = gcsfs.GCSFileSystem(token='anon')
            ds[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True,use_cftime=True,chunks='auto')
            ds[source_id] = ds[source_id].sel(time=slice('1981','2010'))
            ds[source_id] = ds[source_id].sel(time=ds[source_id]['time.season']=='DJF')

            if any(df.variable_id == 'tasmin'):
                ds[source_id] = ds[source_id].sel(lat=slice(lats['wa'][0],lats['wa'][1]),lon=slice(lons['wa'][0],lons['wa'][1]))
                ds[source_id] = ds[source_id].tasmin
            else:
                ds[source_id] = ds[source_id].sel(lat=slice(lats['ak'][0],lats['ak'][1]),lon=slice(lons['ak'][0],lons['ak'][1]))
                ds[source_id] = ds[source_id].psl
        except:
            pass
    return ds
#===============================================================================

def main():
    cluster = LocalCluster(n_workers=10, threads_per_worker=1)
    client = Client(cluster)

    dask.config.set({"array.slicing.split_large_chunks": True})

    lats,lons = set_vars()

    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')

    tasmin = client.submit(load_CMIP,df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='tasmin' & grid_label=='gn'").drop_duplicates(subset=['source_id']),lats,lons).result()
    psl = client.submit(load_CMIP,df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='psl' & grid_label=='gn'").drop_duplicates(subset=['source_id']),lats,lons).result()

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

    tasmin = {model:tasmin[model].groupby('time.month')-tasmin[model].groupby('time.month').mean(['time']) for model in keys}
    psl = {model:psl[model].groupby('time.month')-psl[model].groupby('time.month').mean(['time']) for model in keys}

    tasmin = {model:tasmin[model].mean(['lat','lon']) for model in tasmin}
    psl = {model:psl[model].max(['lat','lon']) for model in psl}

    psl_keys_set = set(psl.keys())
    tasmin_keys_set = set(tasmin.keys())

    # Find the overlap between the keys
    overlap_keys = psl_keys_set.intersection(tasmin_keys_set)

    # Convert the result to a list if needed
    overlap = list(overlap_keys)

    # mdls = ['SAM0-UNICON', 'CanESM5', 'MRI-ESM2-0', 'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'MIROC6', 'ACCESS-CM2', 'NorESM2-MM','ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'TaiESM1', 'NorCPM1', 'CMCC-ESM2']
    tasmin = {k: v for k, v in tasmin.items() if k in overlap}
    psl = {k: v for k, v in psl.items() if k in overlap}

    tmin = np.empty(1)
    mslp = np.empty(1)
    for model in tasmin:
        try:
            mslp = np.concatenate((mslp,psl[model].values))
            tmin = np.concatenate((tmin,tasmin[model].values))
        except:
            pass
    mslp = mslp[1:]
    tmin = tmin[1:]

    era_mslp = load_ERA('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/msl/msl.day.mean.nc',lats,lons)
    era_tmin = load_ERA(glob.glob('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/t2m/t2m.day.min.nc'),lats,lons)

    era = pd.DataFrame(columns=['Average Daily Minimum Temperature (°C)','Maximum Daily Pressure Anomaly (Pa)'])

    era['Average Daily Minimum Temperature (°C)'] = era_tmin.values
    era['Maximum Daily Pressure Anomaly (Pa)'] = era_mslp.values
    era['Dataset'] = 'ERA5'



    data = pd.DataFrame(columns=['Average Daily Minimum Temperature (°C)','Maximum Daily Pressure Anomaly (Pa)'])
    data['Average Daily Minimum Temperature (°C)'] = tmin
    data['Maximum Daily Pressure Anomaly (Pa)'] = mslp
    data['Dataset'] = 'CMIP6 Models'

    # data = data.append(era,ignore_index=True)

    graph = sns.jointplot(x=data['Average Daily Minimum Temperature (°C)'],y=data['Maximum Daily Pressure Anomaly (Pa)'],
                 color='b', kind="kde",height=8)      ## CHANGE HERE

    graph.x = era['Average Daily Minimum Temperature (°C)']
    graph.y = era['Maximum Daily Pressure Anomaly (Pa)']
    graph.plot_joint(sns.kdeplot, color='r', label='ERA5')   ## CHANGE HERE

    mods = mlines.Line2D([], [], color='blue', marker='s', ls='', label='CMIP6 Models')
    rean = mlines.Line2D([], [], color='red', marker='s', ls='', label='ERA5')
    # etc etc
    plt.legend(handles=[mods, rean],loc='lower left',fontsize='large')
    plt.xlabel('Daily Minimum Temperature (°C)',fontsize=18)
    plt.ylabel('Maximum Daily Pressure Anomaly (Pa)', fontsize=18)
    plt.tight_layout()

    graph.plot_marginals(sns.kdeplot, color='r', shade=False, legend=False)

    # graph = sns.jointplot(x=data['Average Daily Minimum Temperature (°C)'],y=data['Maximum Daily Pressure Anomaly (Pa)'], kind="kde",data=data,height=8,label='CMIP6 Models')
    #
    # graph.x = data['Average Daily Minimum Temperature (°C)']
    # graph.y = data['Maximum Daily Pressure Anomaly (Pa)']
    # # graph.plot_joint(sns.kdeplot, shade=True, cmap='Blues',label='CMIP6 Models',legend=True)
    # graph.plot_joint(sns.kdeplot, color='b',label='CMIP6 Models')
    #
    # graph.x = era['Average Minimum Temperature (°C)']
    # graph.y = era['Maximum Pressure (Pa)']
    # # graph.plot_joint(sns.kdeplot, shade=True, cmap='Reds',label='ERA5',legend=True)
    # graph.plot_joint(sns.kdeplot, color='r',label='ERA5')
    #
    # sns.distplot(era['Average Minimum Temperature (°C)'], kde=True,hist=False, color="r", ax=graph.ax_marg_x)
    # sns.distplot(data['Average Daily Minimum Temperature (°C)'], kde=True,hist=False, color="b", ax=graph.ax_marg_x)
    # sns.distplot(era['Maximum Pressure (Pa)'], kde=True,hist=False, color="r",vertical=True, ax=graph.ax_marg_y)
    # sns.distplot(data['Maximum Daily Pressure Anomaly (Pa)'], kde=True,hist=False, color="b",vertical=True, ax=graph.ax_marg_y)


    plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/density.png',dpi=400)
    plt.show(block=False)
