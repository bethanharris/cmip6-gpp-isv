import iris
from iris.util import equalise_attributes
import os
import numpy as np
import xarray as xr
import cf_units
import dask.array as da
from tqdm import tqdm
from scipy.stats import linregress


def detrend_missing_values(data):
    x = np.arange(data.size)
    valid_idx = ~np.logical_or(data>998, data<-998)
    if valid_idx.sum() > 0:
        m, b, r_val, p_val, std_err = linregress(x[valid_idx], data[valid_idx])
        detrended_data = data - (m*x + b)
    else:
        detrended_data = data
    return detrended_data


def detrend_cube(cube, dimension='time'):
    """
    Adapted from esmvalcore to work with missing values.
    Detrend data along a given dimension.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    dimension: str
        Dimension to detrend
    method: str
        Method to detrend. Available: linear, constant. See documentation of
        'scipy.signal.detrend' for details

    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    coord = cube.coord(dimension)
    axis = cube.coord_dims(coord)[0]
    detrended = da.apply_along_axis(
        detrend_missing_values,
        axis=axis,
        arr=cube.lazy_data().rechunk([45, 500, 500]),
        shape=(cube.shape[axis],)
    )
    return cube.copy(detrended)


def monthly_anomalies(cube, detrend=False):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    climatology = dxr.groupby("time.month").mean("time")
    anomalies_xr = (dxr.groupby("time.month") - climatology)
    coord_names = anomalies_xr.coords._names
    if 'lat' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lon': 'longitude'})
    anomalies = anomalies_xr.to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def daily_anomalies(cube, detrend=False):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    climatology = dxr.groupby(month_day_str).mean("time")
    anomalies = (dxr.groupby(month_day_str) - climatology).to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def monthly_anomalies_standardised(cube, detrend=False):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    climatology_mean = dxr.groupby("time.month").mean("time")
    climatology_std = dxr.groupby("time.month").std("time")
    anomalies_xr = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        dxr.groupby("time.month"),
        climatology_mean,
        climatology_std, dask='allowed'
    )
    coord_names = anomalies_xr.coords._names
    if 'lat' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lon': 'longitude'})
    anomalies = anomalies_xr.to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def daily_anomalies_standardised(cube, detrend=False):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    climatology_mean = dxr.groupby(month_day_str).mean("time")
    climatology_std = dxr.groupby(month_day_str).std("time")
    anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        dxr.groupby(month_day_str),
        climatology_mean,
        climatology_std, dask='allowed'
    ).to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def daily_anomalies_standardised_rolling_std(cube, detrend=False, window=7):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    climatology_mean = dxr.groupby(month_day_str).mean("time")
    rolling_dxr = dxr.rolling(min_periods=1, center=True, time=window).construct("window")
    grouped_rolling = rolling_dxr.groupby(month_day_str)
    climatology_std = grouped_rolling.std(["time", "window"])
    anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        dxr.groupby(month_day_str),
        climatology_mean,
        climatology_std, dask='allowed'
    ).to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        anomalies.coord(coord_key).bounds = None
        anomalies.coord(coord_key).guess_bounds()
        anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
        anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def daily_anomalies_standardised_rolling_mean_std(cube, detrend=False, window=7):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    rolling_dxr = dxr.rolling(min_periods=1, center=True, time=window).construct("window")
    grouped_rolling = rolling_dxr.groupby(month_day_str)
    climatology_mean = grouped_rolling.mean(["time", "window"])
    climatology_std = grouped_rolling.std(["time", "window"])
    anomalies_xr = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        dxr.groupby(month_day_str),
        climatology_mean,
        climatology_std, dask='allowed'
    )
    coord_names = anomalies_xr.coords._names
    if 'lat' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lon': 'longitude'})
    anomalies = anomalies_xr.to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        if anomalies.coord(coord_key).points.size > 1:
            anomalies.coord(coord_key).bounds = None
            anomalies.coord(coord_key).guess_bounds()
            anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
            anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies


def rolling_mean_std_for_standardisation(cube, detrend=False, window=7):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    rolling_dxr = dxr.rolling(min_periods=1, center=True, time=window).construct("window")
    grouped_rolling = rolling_dxr.groupby(month_day_str)
    climatology_mean = grouped_rolling.mean(["time", "window"])
    climatology_std = grouped_rolling.std(["time", "window"])
    mean_xr = xr.apply_ufunc(
        lambda x, m: (x/x) * m,
        dxr.groupby(month_day_str),
        climatology_mean, dask='allowed'
    )
    std_xr = xr.apply_ufunc(
        lambda x, s: (x/x) * s,
        dxr.groupby(month_day_str),
        climatology_std, dask='allowed'
    )
    coord_names = std_xr.coords._names
    if 'lat' in coord_names:
        mean_xr = mean_xr.rename({'lat': 'latitude'})
        std_xr = std_xr.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        mean_xr = mean_xr.rename({'lon': 'longitude'})
        std_xr = std_xr.rename({'lon': 'longitude'})
    means = mean_xr.to_iris()
    stdevs = std_xr.to_iris()
    means.standard_name = cube.standard_name
    stdevs.standard_name = cube.standard_name
    means.long_name = cube.long_name
    stdevs.long_name = cube.long_name
    means.units = cube.units
    stdevs.units = cube.units
    calendar = stdevs.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    means.coord('time').convert_units(common_time_unit)
    stdevs.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        if stdevs.coord(coord_key).points.size > 1:
            stdevs.coord(coord_key).bounds = None
            means.coord(coord_key).bounds = None
            stdevs.coord(coord_key).guess_bounds()
            means.coord(coord_key).guess_bounds()
            stdevs.coord(coord_key).bounds = np.round(stdevs.coord(coord_key).bounds, 3)
            means.coord(coord_key).bounds = np.round(means.coord(coord_key).bounds, 3)
            stdevs.coord(coord_key).points = np.round(stdevs.coord(coord_key).points, 3)
            means.coord(coord_key).points = np.round(means.coord(coord_key).points, 3)
    return means, stdevs



def daily_anomalies_rolling_mean(cube, detrend=False, window=7):
    if detrend:
        cube = detrend_cube(cube)
    dxr = xr.DataArray.from_iris(cube)
    month_day_str = xr.DataArray(dxr.indexes['time'].strftime('%m-%d'), coords=dxr.coords['time'].coords, name='month_day_str')
    rolling_dxr = dxr.rolling(min_periods=1, center=True, time=window).construct("window")
    grouped_rolling = rolling_dxr.groupby(month_day_str)
    climatology_mean = grouped_rolling.mean(["time", "window"])
    anomalies_xr = xr.apply_ufunc(
        lambda x, m: (x - m),
        dxr.groupby(month_day_str),
        climatology_mean, dask='allowed'
    )
    coord_names = anomalies_xr.coords._names
    if 'lat' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lat': 'latitude'})
    if 'lon' in coord_names:
        anomalies_xr = anomalies_xr.rename({'lon': 'longitude'})
    anomalies = anomalies_xr.to_iris()
    anomalies.standard_name = cube.standard_name
    anomalies.long_name = cube.long_name
    anomalies.units = cube.units
    calendar = anomalies.coord('time').units.calendar
    common_time_unit = cf_units.Unit('days since 1970-01-01', calendar=calendar)
    anomalies.coord('time').convert_units(common_time_unit)
    for coord_key in ['time', 'latitude', 'longitude']:
        if anomalies.coord(coord_key).points.size > 1:
            anomalies.coord(coord_key).bounds = None
            anomalies.coord(coord_key).guess_bounds()
            anomalies.coord(coord_key).bounds = np.round(anomalies.coord(coord_key).bounds, 3)
            anomalies.coord(coord_key).points = np.round(anomalies.coord(coord_key).points, 3)
    return anomalies
