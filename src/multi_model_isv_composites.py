import os
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cftime
import iris
import iris.coord_categorisation
from bandpass_filters import lanczos_lowpass_filter_missing_data
from detrend_anomalies import daily_anomalies_rolling_mean, daily_anomalies_standardised_rolling_mean_std


### EDIT THIS SECTION TO CHANGE RESOLUTION/REGION/SEASON/ANOMALY STANDARDISATION/SM MASKING ###

regrid_label = 'regrid_1_by_1_deg'
standardise_anomalies = True
use_obs_sm_mask = True # only analyse pixels with valid ESA CCI soil moisture data
                       # Note this doesn't work for native model resolution data.
region_name = '60S80N'
region_lon_west = -180.
region_lon_east = 180.
region_lat_south = -60.
region_lat_north = 80.
season_months = np.arange(1, 13).astype(int)

### END OF CONFIGURATION SECTION ###

standardise_label = 'standardised' if standardise_anomalies else 'unstandardised'
composite_name = f'{region_name}_{regrid_label}_{standardise_label}'
if use_obs_sm_mask:
    if regrid_label == 'native':
        raise ValueError('ESA CCI soil moisture data has not been regridded to model resolutions.')
    composite_name += '_CCI-SM-mask'
composite_name += '_maskfrozen'
os.system(f'mkdir -p ../data/composites/multimodel/{composite_name}')

if use_obs_sm_mask:
    obs_sm = iris.load_cube("/prj/nceo/bethar/esm-isv/OBS/regrid_1_by_1_deg/OBS_mrsos-ESACCI.nc")
    obs_sm_data = ma.filled(obs_sm.data, np.nan)
    obs_sm_mask = np.all(np.isnan(obs_sm_data), axis=0)

if region_lat_south < -60 or region_lat_north > 80:
    raise ValueError("No obs data outside 60S-80N")

cmip6_models = ['NorESM2-LM', 'CNRM-ESM2-1', 'BCC-CSM2-MR', 'ACCESS-ESM1-5', 'UKESM1-0-LL']


def model_colours(model):
    colours = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7'] # Wong, 2011, doi:10.1038/NMETH.1618
    cmip_colours =  {'ACCESS-ESM1-5': colours[5],
                     'BCC-CSM2-MR': colours[2],
                     'CNRM-ESM2-1': colours[3], 
                     'NorESM2-LM': colours[4],
                     'UKESM1-0-LL': colours[0]}
    if model.startswith('OBS'):
        model_colour = 'k'
    elif model in cmip_colours.keys():
        model_colour = cmip_colours[model]
    else:
        raise KeyError(f'Unknown model type {model}')
    return model_colour


def obs_type_linestyles(obs_type):
    obs_linestyles = {'OBS_pr-IMERG': '-',
                      'OBS_mrsos-ESACCI': '-',
                      'OBS_mrsos-GLEAM': '--',
                      'OBS_tasmax-ERA5': '-',
                      'OBS_tasmax-MERRA2': '--',
                      'OBS_gpp-VPM': '-',
                      'OBS_gpp-FLUXCOM-ERA5': '-',
                      'OBS_gpp-FLUXCOM-CRUJRAv1': '--',
                      'OBS_gpp-MODIS-TERRA': '--',
                      'OBS_gpp-SIF-GOME2-JJ': ':',
                      'OBS_gpp-SIF-GOME2-PK': 'dashdot',
                      'OBS_gpp-VOD': 'dashdot',
                      'OBS_gpp-VODCA2GPP': 'dashdot',
                      'OBS_gpp-NDVI': '--',
                      'OBS_hfls-GLEAM': '-',
                      'OBS_hfls-Lu2021': '--',
                      'OBS_rsds-CERES': '-',
                      'OBS_vpd-ERA5': '-'
                      }
    if obs_type in obs_linestyles.keys():
        linestyle = obs_linestyles[obs_type]
    else:
        linestyle = '-'
    return linestyle


def obs_only_linestyles(obs_type):
    obs_linestyles = {'OBS_pr-IMERG': '-',
                      'OBS_mrsos-ESACCI': '-',
                      'OBS_mrsos-GLEAM': '--',
                      'OBS_tasmax-ERA5': '-',
                      'OBS_tasmax-MERRA2': '--',
                      'OBS_gpp-VPM': '--',
                      'OBS_gpp-FLUXCOM-ERA5': '-',
                      'OBS_gpp-FLUXCOM-CRUJRAv1': '--',
                      'OBS_gpp-MODIS-TERRA': '-',
                      'OBS_gpp-SIF-GOME2-JJ': '-',
                      'OBS_gpp-SIF-GOME2-PK': '--',
                      'OBS_gpp-VOD': '--',
                      'OBS_gpp-VODCA2GPP': '-',
                      'OBS_gpp-NDVI': '-',
                      'OBS_hfls-GLEAM': '-',
                      'OBS_hfls-Lu2021': '--',
                      'OBS_rsds-CERES': '-',
                      'OBS_vpd-ERA5': '-'
                      }
    if obs_type in obs_linestyles.keys():
        linestyle = obs_linestyles[obs_type]
    else:
        linestyle = '-'
    return linestyle


def obs_only_colours(obs_type):
    obs_colours = {'OBS_pr-IMERG': 'k',
                      'OBS_mrsos-ESACCI': 'k',
                      'OBS_mrsos-GLEAM': 'r',
                      'OBS_tasmax-ERA5': 'k',
                      'OBS_tasmax-MERRA2': 'r',
                      'OBS_gpp-VPM': '#66a61e',
                      'OBS_gpp-FLUXCOM-ERA5': '#e6ab02',
                      'OBS_gpp-FLUXCOM-CRUJRAv1': '#e6ab02',
                      'OBS_gpp-MODIS-TERRA': '#66a61e',
                      'OBS_gpp-SIF-GOME2-JJ': '#e7298a',
                      'OBS_gpp-SIF-GOME2-PK': '#e7298a',
                      'OBS_gpp-VOD': '#7570b3',
                      'OBS_gpp-VODCA2GPP': '#7570b3',
                      'OBS_gpp-NDVI': '#d95f02',
                      'OBS_hfls-GLEAM': 'k',
                      'OBS_hfls-Lu2021': 'r',
                      'OBS_rsds-CERES': 'k',
                      'OBS_vpd-ERA5': 'k'
                      }
    if obs_type in obs_colours.keys():
        colour = obs_colours[obs_type]
    else:
        colour = '-'
    return colour


variable_units = {'pr': 'kg m^-2 s^-1',
                  'mrsos': 'kg m^-2',
                  'mrso': 'kg m^-2',
                  'mrsol_1.0': 'kg m^-2',
                  'mrsol_3.0': 'kg m^-2',
                  'tasmax': 'K',
                  'hfls': 'W m^-2',
                  'lai': 'unitless',
                  'gpp': 'kg m^-2 s^-1',
                  'rsds': 'W m^-2',
                  'vpd': 'hPa'
                  }


def standardised_anomaly(cube):
    print(f'computing standardised anomalies of {cube.standard_name}')
    if cube.name() in ['GPP-VPM', 'GPP-MODIS-TERRA', 'NDVI', 'gpp-VODCA2GPP'] or cube.name().startswith('SIF'):  #8-daily data
        anomaly = daily_anomalies_standardised_rolling_mean_std(cube, detrend=True, window=31)
    else:
        anomaly = daily_anomalies_standardised_rolling_mean_std(cube, detrend=True)
    return anomaly


def unstandardised_anomaly(cube):
    print(f'computing unstandardised anomalies of {cube.standard_name}')
    if cube.name() in ['GPP-VPM', 'GPP-MODIS-TERRA', 'GPP' 'NDVI', 'gpp-VODCA2GPP'] or cube.name().startswith('SIF'): #8-daily data
        anomaly = daily_anomalies_rolling_mean(cube, detrend=True, window=31)
    else:
        anomaly = daily_anomalies_rolling_mean(cube, detrend=True)
    return anomaly


def get_dates_for_box(precip_lowfreq, lat_idx, lon_idx):
    filtered_precip_px = precip_lowfreq[:, lat_idx, lon_idx]
    candidate_maxima = argrelextrema(filtered_precip_px, np.greater)[0]
    m = np.nanmean(filtered_precip_px)
    s = np.nanstd(filtered_precip_px)
    sig_season_maxima = filtered_precip_px[candidate_maxima] > m + s
    sig_event_idx = candidate_maxima[sig_season_maxima]
    return sig_event_idx


def grid_coords_from_id(precip_lowfreq, id):
    ids = np.arange(precip_lowfreq[0].size).reshape(precip_lowfreq.shape[1], precip_lowfreq.shape[2])
    coords = np.where(ids==id)
    return coords[0][0], coords[1][0]


def filter_precip(precip_cube):
    lats = precip_cube.coord('latitude').points
    precip_anom = ma.filled(precip_cube.data, np.nan)
    precip_lowfreq = np.empty_like(precip_anom, dtype=np.float32)
    for i in tqdm(range(precip_anom.shape[1]), desc='filtering precip to ISV'):
        if lats[i] > -60:
            for j in range(precip_anom.shape[2]):
                precip_lowfreq[:, i, j] = lanczos_lowpass_filter_missing_data(precip_anom[:, i, j], 1./25., 
                                                                              window=121, min_slice_size=365)
    return precip_lowfreq


def get_isv_event_dates(precip_lowfreq):
    events = []
    box_ids = np.arange(precip_lowfreq[0].size)
    for box_id in tqdm(box_ids, desc='finding dates of ISV maxima'):
        lat_idx, lon_idx = grid_coords_from_id(precip_lowfreq, box_id)
        sig_event_idx = get_dates_for_box(precip_lowfreq, lat_idx, lon_idx)
        for event in range(len(sig_event_idx)):
            events.append(((lat_idx, lon_idx), sig_event_idx[event]))
    return events


def time_series_around_date(data_grid, lat_idx, lon_idx, date_idx, days_range=60):
    box_whole_time_series = data_grid[:, lat_idx, lon_idx]
    end_buffer = np.ones(days_range)*np.nan
    data_pad = np.hstack((end_buffer, box_whole_time_series, end_buffer))
    time_series = data_pad[date_idx+days_range-days_range:date_idx+days_range+(days_range+1)]
    return time_series


def composite_events(events, data_grid, days_range=60, existing_composite=None, existing_n=None):
    days_around = np.arange(-days_range, days_range+1)
    if existing_composite is not None:
        composite = existing_composite
        n = existing_n
        start_idx = 0
    else:
        event = events[0]
        composite = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        n = (~np.isnan(composite)).astype(float)
        start_idx = 1
    for event in tqdm(events[start_idx:], desc='creating composite'):
        event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
        first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
        valid_days = np.logical_or(additional_valid_day, first_valid_day)
        n[valid_days] += 1
        composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
        composite[first_valid_day] = event_series[first_valid_day]
    return days_around, composite, n


def save_composites(model):
    if regrid_label == 'native':
        data_dir = f"/prj/nceo/bethar/esm-isv/{model}/"
    else:
        data_dir = f"/prj/nceo/bethar/esm-isv/{model}/{regrid_label}"
    if model == 'OBS':
        precip = iris.load_cube(f"{data_dir}/{model}_pr-IMERG.nc")
    else:
        precip = iris.load_cube(f"{data_dir}/{model}_pr.nc")
    precip_anom = unstandardised_anomaly(precip)
    precip_lowfreq = filter_precip(precip_anom)
    events = get_isv_event_dates(precip_lowfreq)

    if model == 'OBS':
        tasmax = iris.load_cube(f"{data_dir}/{model}_tasmax-ERA5.nc")
    else:
        tasmax = iris.load_cube(f"{data_dir}/{model}_tasmax.nc")

    iris.coord_categorisation.add_month_number(tasmax, 'time', name='month_number')
    tasmax_monthly_median = tasmax.aggregated_by('month_number', iris.analysis.MEDIAN)
    frozen_month = tasmax_monthly_median.data < 273.15

    lats = precip.coord('latitude').points
    lons = np.copy(precip.coord('longitude').points)
    time = precip.coord('time')
    months = [cftime.num2date(d, time.units.origin, calendar=time.units.calendar).month for d in time.points]
    if max(lons) > 200.:
        lons[lons>180.] -= 360.

    region_events = []
    for event in events:
        lat, lon = event[0]
        lat_in_region = (lats[lat] <= region_lat_north and lats[lat] >= region_lat_south)
        lon_in_region = (lons[lon] <= region_lon_east and lons[lon] >= region_lon_west)
        day = event[1]
        month = months[day]
        if (lat_in_region and lon_in_region) and (month in season_months):
            if not frozen_month[month-1, lat, lon]:
                region_events.append(event)

    if use_obs_sm_mask:
        sm_mask = obs_sm_mask
    else:
        if model == 'OBS':
            sm = iris.load_cube(f"{data_dir}/{model}_mrsos-GLEAM.nc")
        else:
            sm = iris.load_cube(f"{data_dir}/{model}_mrsos.nc")
        sm_anom = standardised_anomaly(sm)
        sm_data = ma.filled(sm_anom.data, np.nan)
        sm_mask = np.all(np.isnan(sm_data), axis=0)

    variables_for_model = [x.split(f'{model}_')[-1].split('.nc')[0] for x in os.listdir(data_dir) if x.endswith('.nc')]
    for v in variables_for_model:
        composite_save_dir = f'../data/composites/multimodel/{composite_name}'
        os.system(f'mkdir -p {composite_save_dir}')
        composite_save_name = f'{composite_save_dir}/{model}_{v}_composite_{composite_name}.csv'
        file_already_saved  = os.path.isfile(composite_save_name)
        if file_already_saved:
            print(f'{composite_save_name} already exists, skipping')
        else:
            cube = iris.load_cube(f"{data_dir}/{model}_{v}.nc")
            cube_has_time_coord = len(cube.coords(standard_name='time')) > 0
            if cube_has_time_coord: # don't composite e.g. land area fraction
                number_time_steps = cube.coord('time').points.size
                if number_time_steps > 1:
                    if standardise_anomalies:
                        anom = standardised_anomaly(cube)
                    else:
                        anom = unstandardised_anomaly(cube)
                    data = ma.filled(anom.data, np.nan)
                    if model == 'OBS' and not standardise_anomalies:
                        if v == 'pr-IMERG':
                            data /= 86400.
                        if v in ['mrsos-ESACCI', 'mrsos-GLEAM']:
                            data *= 100.
                    for t in range(data.shape[0]):
                        data[t, sm_mask] = np.nan
                    days_around, composite, n = composite_events(region_events, data)
                    np.savetxt(composite_save_name, composite, delimiter=',')
                    np.savetxt(f'{composite_save_dir}/{model}_{v}_n_{composite_name}.csv', n, delimiter=',', fmt='%d')


def compute_scaling_factors(scale_variable, obs=[]):
    obs_composite = np.genfromtxt(f'../data/composites/multimodel/{composite_name}/OBS_{scale_variable}_composite_{composite_name}.csv', delimiter=',')
    max_obs_anom = np.nanmax(obs_composite)
    scaling_factor = {}
    cmip6_variable_name = scale_variable.split('-')[0]
    for model in cmip6_models:
        model_composite = np.genfromtxt(f'../data/composites/multimodel/{composite_name}/{model}_{cmip6_variable_name}_composite_{composite_name}.csv', delimiter=',')
        max_model_anom = np.nanmax(model_composite)
        scaling_factor[model] = max_obs_anom/max_model_anom
    for plot_obs in obs:
        scaling_factor[f'OBS_{plot_obs}'] = 1.
    return scaling_factor


def obs_legend_entries(handles, labels, cmip6_variable_name):
    label_prefix = f'OBS_{cmip6_variable_name}-'
    obs_handles_labels = [(h, l.split(label_prefix)[-1]) for h, l in zip(handles, labels) if l.startswith(label_prefix)]
    obs_handles = [hl[0] for hl in obs_handles_labels]
    obs_labels = [hl[1] for hl in obs_handles_labels]
    return obs_handles, obs_labels


def single_obs_legend_entry(handles, labels):
    first_obs_handle = True
    for i, label in enumerate(labels):
        if label.startswith('OBS'):
            if not first_obs_handle:
                _ = handles.pop(i)
                _ = labels.pop(i)
            first_obs_handle = False
    model_labels = [l.split('_')[0] for l in labels]
    return handles, model_labels


def plot_variable_comparison(cmip6_variable_name, scale_by_max=None,
                             ax=None, plot_models_legend=True, plot_obs_legend=True, title=False, 
                             save=False, show=True, std_error=False, obs=[]):
    if isinstance(obs, str):
        obs = [obs]
    obs_for_variable = [o for o in obs if o.startswith(f'{cmip6_variable_name}-')]
    if scale_by_max is not None:
        scaling_factor = compute_scaling_factors(scale_by_max, obs=obs)
    composite_directory = f'../data/composites/multimodel/{composite_name}'
    region_filenames = [f for f in os.listdir(composite_directory) if f.endswith('.csv')]
    filenames_for_variable = [f for f in region_filenames if f'_{cmip6_variable_name}_composite_{composite_name}' in f]
    filenames_for_variable.sort()
    model_composites = {f.split('_')[0]: np.genfromtxt(f'{composite_directory}/{f}', delimiter=',') for f in filenames_for_variable}
    for plot_obs in obs:
        if plot_obs.startswith(f'{cmip6_variable_name}-'):
            model_composites[f'OBS_{plot_obs}'] = np.genfromtxt(f'{composite_directory}/OBS_{plot_obs}_composite_{composite_name}.csv', delimiter=',')
    if ax is None:
        fig = plt.figure(figsize=(6, 3.85))#plt.figure(figsize=(5, 3.2))
        ax = plt.gca()
    days_around = np.arange(-60, 61)
    for model in model_composites.keys():
        composite = model_composites[model]
        if scale_by_max is not None:
            composite *= scaling_factor[model]
        ax.plot(days_around, composite, label=model, color=model_colours(model), linestyle=obs_type_linestyles(model))
    handles, labels = ax.get_legend_handles_labels()
    if plot_obs_legend and len(obs_for_variable) > 0:
        obs_handles, obs_labels = obs_legend_entries(handles, labels, cmip6_variable_name)
        obs_legend = ax.legend(obs_handles, obs_labels, loc='lower right', fontsize=10)
    if plot_models_legend:
        model_handles, model_labels = single_obs_legend_entry(handles, labels)        
        ax.legend(model_handles, model_labels, loc='upper left', fontsize=10)
        if plot_obs_legend and len(obs_for_variable) > 0: # need to add the obs legend back on if it exists or won't get both legends showing
            ax.add_artist(obs_legend)
    if title:
        ax.set_title(f'{cmip6_variable_name}', fontsize=14)
    ax.set_xlabel('days since precipitation ISV maximum', fontsize=14)
    ax.set_xlim([-60, 60])
    if standardise_anomalies:
        ax.set_ylabel('standardised anomaly', fontsize=14)
    else:
        ax.set_ylabel(f'anomaly ({variable_units[cmip6_variable_name]})', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.axhline(0, color='gray', alpha=0.5)
    ax.axvline(0, color='gray', alpha=0.5)
    if save:
        save_dir = f'../figures/multimodel/{composite_name}'
        os.system(f'mkdir -p {save_dir}')
        save_filename = f'{save_dir}/{cmip6_variable_name}_global_multimodel_comparison'
        if scale_by_max is not None:
            save_filename += f'_scale_by_max_{scale_by_max}'
        if len(obs) > 0:
            save_filename += f"_OBS_{'_'.join(obs_for_variable)}"
        plt.tight_layout()
        plt.savefig(f'{save_filename}.png', dpi=300)
        print(save_filename)
    if show:
        plt.show()


def plot_variable_comparison_obs_only(cmip6_variable_name, scale_by_max=None,
                                      ax=None, plot_models_legend=False, plot_obs_legend=True, title=False, 
                                      save=False, show=True, std_error=False, obs=[], split_legend=False):
    if isinstance(obs, str):
        obs = [obs]
    obs_for_variable = [o for o in obs if o.startswith(f'{cmip6_variable_name}-')]
    if scale_by_max is not None:
        scaling_factor = compute_scaling_factors(scale_by_max, obs=obs)
    composite_directory = f'../data/composites/multimodel/{composite_name}'
    region_filenames = [f for f in os.listdir(composite_directory) if f.endswith('.csv')]
    filenames_for_variable = [f for f in region_filenames if f'_{cmip6_variable_name}_composite_{composite_name}' in f]
    filenames_for_variable.sort()
    model_composites = {f.split('_')[0]: np.genfromtxt(f'{composite_directory}/{f}', delimiter=',') for f in filenames_for_variable}
    for plot_obs in obs:
        if plot_obs.startswith(f'{cmip6_variable_name}-'):
            model_composites[f'OBS_{plot_obs}'] = np.genfromtxt(f'{composite_directory}/OBS_{plot_obs}_composite_{composite_name}.csv', delimiter=',')
    if ax is None:
        fig = plt.figure(figsize=(6, 3.85))
        ax = plt.gca()
    days_around = np.arange(-60, 61)
    for model in model_composites.keys():
        if model.startswith('OBS'):
            composite = model_composites[model]
            if scale_by_max is not None:
                composite *= scaling_factor[model]
            ax.plot(days_around, composite, label=model, color=obs_only_colours(model), linestyle=obs_only_linestyles(model))
    handles, labels = ax.get_legend_handles_labels()
    if plot_obs_legend and len(obs_for_variable) > 0:
        obs_handles, obs_labels = obs_legend_entries(handles, labels, cmip6_variable_name)
        if split_legend:
            number_entries = len(obs_labels)
            split_idx = number_entries//2 + 1
            obs_legend = ax.legend(obs_handles[0:split_idx], obs_labels[0:split_idx], loc='upper left', fontsize=10)
            obs_legend2 = ax.legend(obs_handles[split_idx:], obs_labels[split_idx:], loc='lower left', fontsize=10)
            ax.add_artist(obs_legend)
        else:
            obs_legend = ax.legend(obs_handles, obs_labels, loc='upper left', fontsize=10)
    if plot_models_legend:
        model_handles, model_labels = single_obs_legend_entry(handles, labels)        
        ax.legend(model_handles, model_labels, loc='lower right', fontsize=10)
        if plot_obs_legend and len(obs_for_variable) > 0: # need to add the obs legend back on if it exists or won't get both legends showing
            ax.add_artist(obs_legend)
            if split_legend:
                ax.add_artist(obs_legend2)
    if title:
        ax.set_title(f'{cmip6_variable_name}', fontsize=14)
    ax.set_xlabel('days since precipitation ISV maximum', fontsize=14)
    ax.set_xlim([-60, 60])
    if standardise_anomalies:
        ax.set_ylabel('standardised anomaly', fontsize=14)
    else:
        ax.set_ylabel(f'anomaly ({variable_units[cmip6_variable_name]})', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.axhline(0, color='gray', alpha=0.5)
    ax.axvline(0, color='gray', alpha=0.5)
    if save:
        save_dir = f'../figures/multimodel/{composite_name}'
        os.system(f'mkdir -p {save_dir}')
        save_filename = f'{save_dir}/{cmip6_variable_name}_obs_comparison'
        if scale_by_max is not None:
            save_filename += f'_scale_by_max_{scale_by_max}'
        plt.tight_layout()
        plt.savefig(f'{save_filename}.png', dpi=300)
        print(save_filename)
    if show:
        plt.show()


def combine_aqua_terra_ndvi():
    aqua_composite = np.genfromtxt(f'../data/composites/multimodel/{composite_name}/OBS_gpp-NDVI-AQUA_composite_{composite_name}.csv', delimiter=",")
    terra_composite = np.genfromtxt(f'../data/composites/multimodel/{composite_name}/OBS_gpp-NDVI-TERRA_composite_{composite_name}.csv', delimiter=",")
    aqua_n = np.genfromtxt(f'../data/composites/multimodel/{composite_name}/OBS_gpp-NDVI-AQUA_n_{composite_name}.csv', delimiter=",")
    terra_n = np.genfromtxt(f'../data/composites/multimodel/{composite_name}/OBS_gpp-NDVI-TERRA_n_{composite_name}.csv', delimiter=",")
    combined_composite = (aqua_composite*aqua_n + terra_composite*terra_n)/(aqua_n + terra_n)
    combined_n = aqua_n + terra_n
    np.savetxt(f'../data/composites/multimodel/{composite_name}/OBS_gpp-NDVI_composite_{composite_name}.csv', combined_composite, delimiter=",", fmt='%.3e')
    np.savetxt(f'../data/composites/multimodel/{composite_name}/OBS_gpp-NDVI_n_{composite_name}.csv', combined_n, delimiter=",", fmt='%d')


def composite_subplots():
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.65), sharex=True)
    ((pr_ax, mrsos_ax), (gpp_ax, gpp_obs_ax)) = axs
    plot_variable_comparison('pr', obs=['pr-IMERG'], ax=pr_ax, save=False, show=False)
    plot_variable_comparison('mrsos', obs=['mrsos-ESACCI', 'mrsos-GLEAM'], ax=mrsos_ax, 
                             scale_by_max='pr-IMERG', save=False, show=False)
    plot_variable_comparison('gpp', obs=['gpp-FLUXCOM-ERA5','gpp-MODIS-TERRA'], ax=gpp_ax, 
                             scale_by_max='pr-IMERG', save=False, show=False)
    gpp_obs = ['gpp-FLUXCOM-ERA5', 'gpp-FLUXCOM-CRUJRAv1',
               'gpp-MODIS-TERRA','gpp-VPM', 
               'gpp-SIF-GOME2-JJ', 'gpp-SIF-GOME2-PK',
               'gpp-VODCA2GPP']
    plot_variable_comparison_obs_only('gpp', scale_by_max='pr-IMERG', obs=gpp_obs, ax=gpp_obs_ax, 
                                      save=False, show=False, split_legend=True)
    pr_ax.set_title("$\\bf{(a)}$ " + 'precipitation', fontsize=14)
    mrsos_ax.set_title("$\\bf{(b)}$ " + 'surface soil moisture', fontsize=14)
    gpp_ax.set_title("$\\bf{(c)}$ " + 'GPP', fontsize=14)
    gpp_obs_ax.set_title("$\\bf{(d)}$ " + 'GPP observational products', fontsize=14)
    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()
    save_dir = f'../figures/multimodel/{composite_name}'
    os.system(f'mkdir -p {save_dir}')
    plt.savefig(f'{save_dir}/composite_subplots.png', dpi=400, bbox_inches='tight')
    plt.savefig(f'{save_dir}/composite_subplots.pdf', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # for model in cmip6_models + ['OBS']:
        # save_composites(model)
    # combine_aqua_terra_ndvi()
    # composite_subplots()
    # plot_variable_comparison('pr', obs=['pr-IMERG'],save=True)
    # plot_variable_comparison('mrsos', obs=['mrsos-ESACCI', 'mrsos-GLEAM'], save=True)
    plot_variable_comparison('mrsos', obs=['mrsos-ESACCI', 'mrsos-GLEAM'], scale_by_max='pr-IMERG', save=True)
    # plot_variable_comparison('mrsol_1.0', obs=['mrsol_1.0-GLEAM'], scale_by_max='pr-IMERG', save=True)
    # plot_variable_comparison('mrsol_1.0', obs=['mrsol_1.0-GLEAM'], scale_by_max=None, save=True)
    # plot_variable_comparison('mrso', scale_by_max='pr-IMERG', save=True)
    # plot_variable_comparison('mrso', scale_by_max=None, save=True)
    # plot_variable_comparison('rsds', obs=['rsds-CERES'], scale_by_max=None, save=True)
    # plot_variable_comparison('rsds', obs=['rsds-CERES'], scale_by_max='pr-IMERG', save=True)
    # plot_variable_comparison('vpd', obs=['vpd-ERA5'], scale_by_max=None, save=True)
    # plot_variable_comparison('vpd', obs=['vpd-ERA5'], scale_by_max='pr-IMERG', save=True)
    # plot_variable_comparison('gpp',scale_by_max='mrsos-GLEAM', obs=['gpp-FLUXCOM-ERA5','gpp-MODIS-TERRA'],save=True)
    plot_variable_comparison('gpp',scale_by_max='pr-IMERG', obs=['gpp-FLUXCOM-ERA5','gpp-MODIS-TERRA'],save=True)
    # plot_variable_comparison('gpp',obs=['gpp-FLUXCOM-ERA5','gpp-MODIS-TERRA'], save=True)    
    plot_variable_comparison_obs_only('gpp',scale_by_max='pr-IMERG', obs=['gpp-FLUXCOM-ERA5',
                             'gpp-FLUXCOM-CRUJRAv1','gpp-MODIS-TERRA','gpp-VPM','gpp-SIF-GOME2-JJ',
                             'gpp-SIF-GOME2-PK', 'gpp-VODCA2GPP'], save=True, split_legend=True)
