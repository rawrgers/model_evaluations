from dask.distributed import Client,LocalCluster,as_completed
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import seaborn as sns
import re
import dask
import zarr
import gcsfs
import statsmodels.api as sm
import numbers
from glob import glob
from scipy.stats import linregress

cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)

dask.config.set({"array.slicing.split_large_chunks": True})
################################################################################
def set_vars(): # Seattle (47.6062, -122.3321), Spokane (47.6588, -117.4260), Billings (45.7833,-108.5007)
    place = pd.DataFrame()
    place['lat'] = [47.6062]
    place['lon'] = [-122.3321]
    # place['name'] = ['Seattle']
    place['name'] = ['WCASC']
    return place
################################################################################
def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_med=None, rug_kwargs=None, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)
        rug_medx_params = dict(ymin=0, ymax=rug_length, c='red', alpha=1.0)
        rug_medy_params = dict(xmin=0, xmax=rug_length, c='red', alpha=1.0)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

        if rug_med is not None:
            ax.axvline(np.quantile(x,q=0.5),**rug_medx_params)
            ax.axhline(np.quantile(y,q=0.5),**rug_medy_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)
################################################################################
def load_ERA(file,place):
    df = xr.open_mfdataset(file,chunks='auto')
    df = df.sel(time=slice('1981','2010'))
    df = df.drop(['time_bnds'])
    # df = df.interp(latitude=place['lat'],longitude=place['lon'])
    df = df.sel(latitude=slice(51.5,46.5),longitude=slice(235,244)).mean(['latitude','longitude'])
    df = df.sel(time=df['time.season']=='DJF')
    df['t2m'] = (df['t2m']-273.15)
    df = df.to_dataframe()

    return df
################################################################################
def load_CMIP(df):
    """
    Load data for the given source
    """
    ds = {}
    for source_id in tqdm(df['source_id']):
        vad = df[(df.source_id==source_id)].zstore.values[0]

        gcs = gcsfs.GCSFileSystem(token='anon')
        if any(df.variable_id=='tasmin'):
            ds[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True,chunks='auto').tasmin
        elif any(df.variable_id=='tas'):
            ds[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True,chunks='auto').tas
        else:
            ds[source_id] = xr.open_zarr(gcs.get_mapper(vad),consolidated=True,chunks='auto').zg
    return ds
################################################################################
def load_obs(file):
    df = pd.read_csv(file,header=None,names=['Year','Month','Day','tmin'])
    df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day']]
        .astype(str).agg('-'.join, axis=1),infer_datetime_format=True)
    df = df.set_index(df['time'])
    df = df['1981':'2010']
    da = df.to_xarray()
    da = da.sel(time=da['time.season']=='DJF')
    df = da.to_dataframe()
    return df
################################################################################
def get_temps(df,place):
    """
    finding the time of the lowest minimum temps
    """
    temps = {}
    for model in tqdm(df.keys()):
        # print(model)
        # temps[model] = df[model].interp(lat=place['lat'],lon=(360+place['lon']))
        temps[model] = df[model].sel(lat=slice(46.5,51.5),lon=slice(235,244)).mean(['lat','lon'])
        try:
            temps[model] = temps[model].sel(time=slice('1981-01','2010-12'))
        except:
            dates = xr.cftime_range(start='1981-01-01 12:00:00',end='2010-01-01 12:00:00',freq='D',calendar='noleap')
            temps[model] = temps[model].sel(time=slice(dates[0],dates[-1]))
        try:
            temps[model] = temps[model].sel(time=temps[model]['time.season']=='DJF').drop('height')
        except:
            temps[model] = temps[model].sel(time=temps[model]['time.season']=='DJF')
        temps[model] = temps[model]-273.15
        temps[model] = temps[model].to_dataframe()
    return temps
################################################################################
def get_annual_min(df):

    temps={}
    for model in tqdm(df.keys()):
        # print(model)
        # temps[model] = df[model].interp(lat=place['lat'],lon=(360+place['lon']))
        temps[model] = df[model].sel(lat=slice(46.5,51.5),lon=slice(235,244)).mean(['lat','lon'])
        try:
            temps[model] = temps[model].sel(time=slice('1981-01','2010-12'))
        except:
            dates = xr.cftime_range(start='1981-01-01 12:00:00',end='2010-12-31 12:00:00',freq='D',calendar='noleap')
            temps[model] = temps[model].sel(time=slice(dates[0],dates[-1]))
        temps[model] = temps[model].resample(time='1Y').min().drop('height')
        temps[model] = temps[model]-273.15
        temps[model] = temps[model].to_dataframe()
    return temps
################################################################################
def get_annual_mean(df):

    temps = {}
    for model in tqdm(df.keys()):
        # print(model)
        # temps[model] = df[model].interp(lat=place['lat'],lon=(360+place['lon']))
        temps[model] = df[model].sel(lat=slice(46.5,51.5),lon=slice(235,244)).mean(['lat','lon'])
        try:
            temps[model] = temps[model].sel(time=slice('1981-01','2010-12'))
        except:
            dates = xr.cftime_range(start='1981-01-01 12:00:00',end='2010-12-31 12:00:00',freq='D',calendar='noleap')
            temps[model] = temps[model].sel(time=slice(dates[0],dates[-1]))
        temps[model] = temps[model].resample(time='1Y').mean().drop('height')
        temps[model] = temps[model]-273.15
        temps[model] = temps[model].to_dataframe()
    return temps
################################################################################
def main():
    # Load Data
    place = set_vars()
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

    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')

    if var == 'tasmin':
        mod_tasmin = client.submit(load_CMIP,df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='tasmin' & grid_label=='gn'").drop_duplicates(subset=['source_id'])).result()
        era_tasmin = client.submit(load_ERA,glob('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/t2m/t2m.day.min.nc'),(place)).result()
    else:
        # This is still only grabbing DJF for ERA5.
        era_tasmin = client.submit(load_ERA,glob('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/t2m/t2m.day.mean.nc'),(place)).result()
        era_tasmin = era_tasmin.reset_index()[['time','t2m']].set_index('time')
        mod_tasmin = client.submit(load_CMIP,df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='tas' & grid_label=='gn'").drop_duplicates(subset=['source_id'])).result()
    resolution = {model:len(mod_tasmin[model].lon) for model in mod_tasmin}

    mod_tasmin = {model: mod_tasmin[model] for model in keys}

# GHCN DATA
    obs = load_obs('/home/disk/picea/mauger/GHCN/DATA/TMIN_USW00024233.csv')
    # Data: /home/disk/picea/mauger/GHCN/DATA
    # TMIN_USW00024233.csv SeaTac Airport
    # Scripts/Refs: /home/disk/margaret/mauger/GHCND


    # obs_tasmin = obs_tas.rename({'t2m':'tasmin'}).tas
    # obs_tas['time'] = mod_tasmin['GISS-E2-1-G'].time

    # Grab seasonal mean first, then comput bias
    # obs_tas = obs_tas.groupby(obs_tas.time.dt.season).mean()
    # obs_tas = obs_tas.sel(season='DJF')

    mod_temps = get_temps(mod_tasmin,place)

    # for model in mod_temps:
    #     plt.figure()
    #     ax = mod_temps[model]['tasmin'].plot.kde(label=model)
    #     ax = obs_tasmin['t2m'].plot.kde(label='ERA5')
    #     ax.legend()
    #     ax.set_title('PDF of minimum daily temperatures (1981-2010)')
    #     plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/PDF/'+model+'_tasmin.png',dpi=400)
    fontsize=14
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    for model in mod_temps:
        try:
            mod_temps[model]['tasmin'].plot.kde(ax=ax1,color='grey',alpha=0.5,label='CMIP6 Models')
        except:
            mod_temps[model].plot.kde(ax=ax1,color='grey',alpha=0.5,label='CMIP6 Models')
    ax1.set_title('(a)',loc='left',fontsize=16)
    ax1.tick_params(axis='both',labelsize=fontsize)

    era_tasmin['t2m'].plot.kde(ax=ax1,label='ERA5',color='red')
    handles, labels = ax1.get_legend_handles_labels()
    display = (0,21)
    ax1.legend([handle for i,handle in enumerate(handles) if i in display],
      [label for i,label in enumerate(labels) if i in display], loc = 'upper left',
      fontsize='x-large')
    ax1.set_xlabel('Temperature (°C)',fontsize=fontsize)
    ax1.set_xlim([-60,20])

    # Now to plot a histogram
    bias = pd.DataFrame(columns=['50th Percentile','1st Percentile'])
    for model in sorted(mod_temps,reverse=True):
        try:
            dict = pd.DataFrame.from_dict({model:[mod_temps[model].tasmin.quantile(q=0.5)-era_tasmin['t2m'].quantile(q=0.5),
                    mod_temps[model].tasmin.quantile(q=0.01)-era_tasmin['t2m'].quantile(q=0.01)]},orient='index',
                    columns=['50th Percentile','1st Percentile'])
        except:
            dict = pd.DataFrame.from_dict({model:[mod_temps[model].tas.quantile(q=0.5)-era_tasmin['t2m'].quantile(q=0.5),
                    mod_temps[model].tas.quantile(q=0.01)-era_tasmin['t2m'].quantile(q=0.01)]},orient='index',
                    columns=['50th Percentile','1st Percentile'])
        bias = bias.append(dict)

    resolution = pd.DataFrame(resolution.items(), columns=['model', 'longitude_points'])
    resolution = resolution.set_index('model')
    resolution['bias'] = bias['1st Percentile']
    resolution = resolution.dropna(subset=['bias'])
    resolution['longitude_points'] = resolution['longitude_points']

    # Linear Regression
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=resolution, x='longitude_points', y='bias')

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(resolution['longitude_points'], resolution['bias'])
    x_values = np.array([min(resolution['longitude_points']), max(resolution['longitude_points'])])
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values, color='red', linestyle='--')

    s_slope, s_intercept, s_r_value, s_p_value, s_std_err = linregress(resolution.drop(['HadGEM3-GC31-MM','MIROC-ES2L'])['longitude_points'], resolution.drop(['HadGEM3-GC31-MM','MIROC-ES2L'])['bias'])
    sx_values = np.array([min(resolution.drop(['HadGEM3-GC31-MM','MIROC-ES2L'])['longitude_points']), max(resolution.drop(['HadGEM3-GC31-MM','MIROC-ES2L'])['longitude_points'])])
    sy_values = s_slope * sx_values + s_intercept
    plt.plot(sx_values, sy_values, color='blue', linestyle='--')

    # label 'HadGEM3-GC31-MM' and 'MIROC-ES2L'
    plt.text(resolution.loc['HadGEM3-GC31-MM', 'longitude_points'], resolution.loc['HadGEM3-GC31-MM', 'bias'], 'HadGEM3-GC31-MM', fontsize=10, ha='center',va='top')
    plt.text(resolution.loc['MIROC-ES2L', 'longitude_points'], resolution.loc['MIROC-ES2L', 'bias'], 'MIROC-ES2L', fontsize=10, ha='center',va='top')
    # Add regression equation to the plot
    plt.text(370, -12, f'y = {slope:.2f}x + {intercept:.2f}\nR-squared = {r_value**2:.2f}', fontsize=12,color='red')
    plt.text(370, -3, f'y = {s_slope:.2f}x + {s_intercept:.2f}\nR-squared = {s_r_value**2:.2f}', fontsize=12,color='blue')
    # Add labels and title
    plt.xlabel('Number of Longitude Points')
    plt.ylabel('Magnitude of Bias (°C)')
    plt.title('Scatter Plot and Linear Regression between Bias and Horizontal Resolution')

    # Show plot
    # plt.grid(True)
    # plt.tight_layout()
    plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/PDF/resolution_linreg.png',dpi=400)
    plt.show()



    # Plots
    bias.plot.barh(ax=ax2,color={"50th Percentile": "green", "1st Percentile": "purple"})
    ax2.set_xlabel('Temperature (°C)',fontsize=fontsize)
    ax2.axvline(0,color='k',ls='--')
    ax2.legend(loc='lower left',fontsize='small')
    ax2.set_title('(b)',loc='left',fontsize=fontsize)
    ax2.set_xlim([-20,5])
    ax2.tick_params(axis='both',labelsize=fontsize)


    plt.tight_layout()
    if var == 'tasmin':
        plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/PDF/ensemble_DJF_tasmin_1981-2010.png',dpi=400)
    else:
        plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/PDF/ensemble_DJF_tas_1981-2010.png',dpi=400)


    # for model in mod_temps:
    #     plt.figure(figsize = (8,8))
    #     # pp_mod = sm.ProbPlot(mod_temps[model]['tasmin'].values, fit=True)
    #     # ax1 = pp_mod.qqplot(other=era_tasmin['t2m'].values, line='45')
    #     y = mod_temps[model]['tasmin'].values
    #     x = era_tasmin['t2m'].values
    #     qqplot(x,y,alpha=0.1, edgecolor='k', rug=True,rug_med=True)
    #     plt.title('QQ plot of DJF minimum daily temperatures (1981-2010) for '+place['name'].values[0])
    #     plt.ylabel(model+' (°C)')
    #     plt.xlabel('ERA5 (°C)')
    #     plt.xlim([np.floor(y.min()),np.ceil(x.max())])
    #     plt.ylim([np.floor(y.min()),np.ceil(x.max())])
    #     ref = np.arange(np.floor(y.min()),np.ceil(x.max())+1,1)
    #     plt.plot(ref,ref,color='k')
    #     plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/PDF/QQ/'+model+'_DJF_QQ_plot_'+place['name'].values[0]+'_1981-2010.png',dpi=400)

    # plt.figure(figsize = (8,8))
    # y = era_tasmin['t2m'].values
    # x = obs['tmin'].values
    # qqplot(x,y,alpha=0.1, edgecolor='k', rug=True,rug_med=True)
    # plt.title('QQ plot of DJF minimum daily temperatures (1981-2010) for '+place['name'].values[0])
    # plt.ylabel('ERA5 (°C)')
    # plt.xlabel('OBS (°C)')
    # plt.xlim([np.floor(x.min()),np.ceil(y.max())])
    # plt.ylim([np.floor(x.min()),np.ceil(y.max())])
    # ref = np.arange(np.floor(x.min()),np.ceil(y.max())+1,1)
    # plt.plot(ref,ref,color='k')
    # plt.savefig('/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/PDF/QQ/OBS_ERA5_DJF_QQ_plot_'+place['name'].values[0]+'_1981-2010.png',dpi=400)


    # Scatterplot of coldest day in each year compared to annual average
    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
    mod_tas = client.submit(load_CMIP,df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='tas' & grid_label=='gn'").drop_duplicates(subset=['source_id'])).result()
    mod_tasmin = client.submit(load_CMIP,df.query("activity_id=='CMIP' & experiment_id=='historical' & table_id=='day' & variable_id=='tasmin' & grid_label=='gn'").drop_duplicates(subset=['source_id'])).result()
    mod_tas = {model: mod_tas[model] for model in keys}
    mod_tasmin = {model: mod_tasmin[model] for model in keys}

    ann = get_annual_mean(mod_tas)
    min = get_annual_min(mod_tasmin)
    fs=16
    for model in ann:
        plt.figure(figsize=(8,8))
        plt.scatter(ann[model]['tas'].values,min[model]['tasmin'].values)
        plt.xlabel('Mean Annual Temperature',fontsize=fs)
        plt.ylabel('Minimum Annual Temperature',fontsize=fs)
        slope, intercept, r_value, p_value, std_err = linregress(ann[model]['tas'].values,min[model]['tasmin'].values)
        x_values = np.array([ann[model]['tas'].values.min(), ann[model]['tas'].values.max()])
        y_values = slope * x_values + intercept
        plt.plot(x_values, y_values, color='red', linestyle='--')
        plt.text(ann[model]['tas'].values.min()+0.5, min[model]['tasmin'].values.min()+0.5, f'y = {slope:.2f}x + {intercept:.2f}\nR-squared = {r_value**2:.2f}', fontsize=12,color='red')
        plt.title(f'{model}',loc='left',fontsize=fs)
        plt.title('1981-2010',loc='right',fontsize=fs)
        plt.savefig(f'/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/PDF/{model}_ann_min_linreg_1981-2010.png',dpi=400)
    plt.show(block=False)


    # do the same for ERA5 now
    era_tasmin = glob('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/t2m/t2m.day.min.nc')
    era_tasmin = xr.open_mfdataset(era_tasmin,chunks='auto')
    era_tasmin = era_tasmin.sel(time=slice('1979','2020'))
    # df = df.interp(latitude=place['lat'],longitude=place['lon'])
    era_tasmin = era_tasmin.sel(latitude=slice(51.5,46.5),longitude=slice(235,244)).mean(['latitude','longitude'])
    era_tasmin['t2m'] = (era_tasmin['t2m']-273.15)
    era_tasmin = era_tasmin.to_dataframe()
    era_tasmin = era_tasmin.reset_index()[['time','t2m']].set_index('time').drop_duplicates()

    era_tas = glob('/home/disk/becassine/DATA/Reanalysis/ERA5/raw/daily/t2m/t2m.day.mean.nc')
    era_tas = xr.open_mfdataset(era_tas,chunks='auto')
    era_tas = era_tas.sel(time=slice('1979','2020'))
    # df = df.interp(latitude=place['lat'],longitude=place['lon'])
    era_tas = era_tas.sel(latitude=slice(51.5,46.5),longitude=slice(235,244)).mean(['latitude','longitude'])
    era_tas['t2m'] = (era_tas['t2m']-273.15)
    era_tas = era_tas.to_dataframe()
    era_tas = era_tas.reset_index()[['time','t2m']].set_index('time').drop_duplicates()

    era_ann = era_tas.resample('1Y').mean()
    era_min = era_tasmin.resample('1Y').min()

    plt.figure(figsize=(8,8))
    plt.scatter(era_ann['t2m'].values,era_min['t2m'].values)
    plt.xlabel('Mean Annual Temperature',fontsize=fs)
    plt.ylabel('Minimum Annual Temperature',fontsize=fs)
    slope, intercept, r_value, p_value, std_err = linregress(era_ann['t2m'].values,era_min['t2m'].values)
    x_values = np.array([era_ann['t2m'].values.min(), era_ann['t2m'].values.max()])
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values, color='red', linestyle='--')
    plt.text(era_ann['t2m'].values.max()-1, era_min['t2m'].values.min()+0.5, f'y = {slope:.2f}x + {intercept:.2f}\nR-squared = {r_value**2:.2f}', fontsize=12,color='red')
    plt.title(f'ERA5',loc='left',fontsize=fs)
    plt.title('1979-2020',loc='right',fontsize=fs)
    plt.savefig(f'/home/disk/rocinante/rawrgers/Figures/CMIP6_DataScience/PDF/ERA5_ann_min_linreg_1979-2020.png',dpi=400)
if __name__ == "__main__":
    main()
