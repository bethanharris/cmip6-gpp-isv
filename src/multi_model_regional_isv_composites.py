import os
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import cftime
import iris
import iris.coord_categorisation
from bandpass_filters import lanczos_lowpass_filter_missing_data
from detrend_anomalies import daily_anomalies_rolling_mean, daily_anomalies_standardised_rolling_mean_std
import regionmask
import calendar


### EDIT THIS SECTION TO CHANGE RESOLUTION/REGION/SEASON/NORMALISATION/SM MASKING' ###

regrid_label = 'regrid_1_by_1_deg'
standardise_anomalies = True
use_obs_sm_mask = True # only include model data at grid boxes where there are ESA CCI SM observations
analysis_version_name = 'rolling_7d_mean_stdev_maskfrozen_60S60N'
season_months = np.arange(1, 13).astype(int) # include intraseasonal maxima only if peaking in these months
regions_to_include = np.arange(46) # all land regions

cmip6_models = ['NorESM2-LM', 'CNRM-ESM2-1', 'BCC-CSM2-MR', 'ACCESS-ESM1-5', 'UKESM1-0-LL']

### NO NEED TO EDIT BELOW HERE ###

def season_label(season_months):
    number_months = np.array(season_months).size
    if number_months == 12:
        label = ''
    else:
        month_initials = [m[0] for m in calendar.month_abbr[1:]]
        abbr = ''.join([month_initials[m-1] for m in season_months])
        label = f'_{abbr}'
    return label


ar6_land = regionmask.defined_regions.ar6.land
region_names = ar6_land.names
region_abbreviations = ar6_land.abbrevs

standardise_label = 'standardised' if standardise_anomalies else 'unstandardised'
composite_name = f'{analysis_version_name}_{regrid_label}_{standardise_label}'
if use_obs_sm_mask:
    if regrid_label == 'native':
        raise ValueError('ESA CCI soil moisture data has not been regridded to model resolutions.')
    composite_name += '_CCI-SM-mask'
composite_name += season_label(season_months)
composite_save_dir = f'../data/composites/multimodel/regional/{composite_name}'
os.system(f'mkdir -p {composite_save_dir}')

if use_obs_sm_mask:
    obs_sm = iris.load_cube("/prj/nceo/bethar/esm-isv/OBS/regrid_1_by_1_deg/OBS_mrsos-ESACCI.nc")
    obs_sm_data = ma.filled(obs_sm.data, np.nan)
    obs_sm_mask = np.all(np.isnan(obs_sm_data), axis=0)


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
                  'huss': 'unitless',
                  'vpd': 'hPa'}


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
    number_lats = precip_cube.coord('latitude').points.size
    number_lons = precip_cube.coord('longitude').points.size
    if len(precip_anom.shape) == 1 and number_lats == 1 and number_lons == 1:
        precip_anom = np.expand_dims(precip_anom, axis=1)
        precip_anom = np.expand_dims(precip_anom, axis=2)
    precip_lowfreq = np.ones_like(precip_anom, dtype=np.float32)*np.nan
    for i in tqdm(range(number_lats), desc='filtering precip to ISV'):
        if lats[i] > -60 and lats[i]<60:
            for j in range(number_lons):
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
    if len(data_grid.shape) == 1:
        data_grid = np.expand_dims(data_grid, axis=1)
        data_grid = np.expand_dims(data_grid, axis=2)
    box_whole_time_series = data_grid[:, lat_idx, lon_idx]
    end_buffer = np.ones(days_range)*np.nan
    data_pad = np.hstack((end_buffer, box_whole_time_series, end_buffer))
    time_series = data_pad[date_idx+days_range-days_range:date_idx+days_range+(days_range+1)]
    return time_series


def composite_events(events, data_grid, days_range=60):
    days_around = np.arange(-days_range, days_range+1)
    if len(events) == 0:
        composite = np.ones_like(days_around) * np.nan
        composite_std = np.ones_like(days_around) * np.nan
        n = np.zeros_like(days_around)
    else:
        event = events[0]
        composite = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
        n = (~np.isnan(composite)).astype(float)
        squaresum = np.ones_like(n, dtype=np.float64) * np.nan
        squaresum[~np.isnan(composite)] = 0.
        start_idx = 1
        for event in events[start_idx:]:
            event_series = time_series_around_date(data_grid, event[0][0], event[0][1], event[1], days_range=days_range)
            additional_valid_day = np.logical_and(~np.isnan(event_series), ~np.isnan(composite))
            first_valid_day = np.logical_and(~np.isnan(event_series), np.isnan(composite))
            valid_days = np.logical_or(additional_valid_day, first_valid_day)
            n[valid_days] += 1
            old_mean = np.copy(composite)
            composite[additional_valid_day] = composite[additional_valid_day] + (event_series[additional_valid_day] - composite[additional_valid_day])/n[additional_valid_day]
            composite[first_valid_day] = event_series[first_valid_day]
            squaresum[first_valid_day] = 0.
            squaresum[additional_valid_day] = squaresum[additional_valid_day] + ((event_series[additional_valid_day] - old_mean[additional_valid_day]) * (event_series[additional_valid_day] - composite[additional_valid_day]))
        composite_std = np.sqrt(squaresum/n)
    return days_around, composite, composite_std, n


def save_composites(model, regions_to_include):
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

    ar6_mask = ar6_land.mask(precip.coord('longitude').points, precip.coord('latitude').points).data
    if regrid_label == 'native':
        land_mask = ma.filled(iris.load_cube(f"{data_dir}/{model}_sftlf.nc").data, -999)
        ar6_mask[land_mask<=0.] = np.nan

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
    month_bins = np.arange(13) + 0.5
    if max(lons) > 200.:
        lons[lons>180.] -= 360.

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
    for v in tqdm(variables_for_model, desc=f'{model}: compositing variables for all regions'):
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
                if model=='OBS' and not standardise_anomalies:
                    if v=='pr-IMERG':
                        data /= 86400.
                    if v in ['mrsos-ESACCI', 'mrsos-GLEAM']:
                        data *= 100.
                for t in range(data.shape[0]):
                    data[t, sm_mask] = np.nan
                for region in regions_to_include:
                    region_events = []
                    region_event_months = []
                    for event in events:
                        lat, lon = event[0]
                        gridbox_in_region = (ar6_mask[lat, lon] == region)
                        day = event[1]
                        month = months[day]
                        if gridbox_in_region and (month in season_months):
                            if not frozen_month[month-1, lat, lon]:
                                region_events.append(event)
                                region_event_months.append(month)
                    if v in ['pr', 'pr-IMERG']:
                        month_counts, _ = np.histogram(region_event_months, bins=month_bins)
                        np.savetxt(f'{composite_save_dir}/{model}_event_months_{composite_name}_region{int(region)}.csv', month_counts, delimiter=",", fmt='%d')
                    days_around, composite, composite_std, n = composite_events(region_events, data)
                    np.savetxt(f'{composite_save_dir}/{model}_{v}_composite_{composite_name}_region{int(region)}.csv', composite, delimiter=",", fmt='%.3e')
                    np.savetxt(f'{composite_save_dir}/{model}_{v}_composite_std_{composite_name}_region{int(region)}.csv', composite_std, delimiter=",", fmt='%.3e')
                    np.savetxt(f'{composite_save_dir}/{model}_{v}_n_{composite_name}_region{int(region)}.csv', n, delimiter=",", fmt='%d')


def save_event_months(model, regions_to_include):
    if regrid_label == 'native':
        data_dir = f"/prj/nceo/bethar/esm-isv/{model}/"
    else:
        data_dir = f"/prj/nceo/bethar/esm-isv/{model}/{regrid_label}"
    precip = iris.load_cube(f"{data_dir}/{model}_pr.nc")
    precip_anom = unstandardised_anomaly(precip)
    precip_lowfreq = filter_precip(precip_anom)
    events = get_isv_event_dates(precip_lowfreq)

    ar6_mask = ar6_land.mask(precip.coord('longitude').points, precip.coord('latitude').points).data
    if regrid_label == 'native':
        land_mask = ma.filled(iris.load_cube(f"{data_dir}/{model}_sftlf.nc").data, -999)
        ar6_mask[land_mask<=0.] = np.nan

    lats = precip.coord('latitude').points
    lons = np.copy(precip.coord('longitude').points)
    time = precip.coord('time')
    months = [cftime.num2date(d, time.units.origin, calendar=time.units.calendar).month for d in time.points]
    month_bins = np.arange(13) + 0.5
    if max(lons) > 200.:
        lons[lons>180.] -= 360.

    for region in regions_to_include:
        region_events = []
        region_event_months = []
        for event in events:
            lat, lon = event[0]
            gridbox_in_region = (ar6_mask[lat, lon] == region)
            day = event[1]
            month = months[day]
            if gridbox_in_region and (month in season_months):
                region_events.append(event)
                region_event_months.append(month)
        month_counts, _ = np.histogram(region_event_months, bins=month_bins)
        np.savetxt(f'{composite_save_dir}/{model}_event_months_{composite_name}_region{int(region)}.csv', month_counts, delimiter=",", fmt='%d')


def combine_aqua_terra_ndvi(region_number):
    aqua_composite = np.genfromtxt(f'{composite_save_dir}/OBS_gpp-NDVI-AQUA_composite_{composite_name}_region{int(region_number)}.csv', delimiter=",")
    terra_composite = np.genfromtxt(f'{composite_save_dir}/OBS_gpp-NDVI-TERRA_composite_{composite_name}_region{int(region_number)}.csv', delimiter=",")
    aqua_n = np.genfromtxt(f'{composite_save_dir}/OBS_gpp-NDVI-AQUA_n_{composite_name}_region{int(region_number)}.csv', delimiter=",")
    terra_n = np.genfromtxt(f'{composite_save_dir}/OBS_gpp-NDVI-TERRA_n_{composite_name}_region{int(region_number)}.csv', delimiter=",")
    combined_composite = (aqua_composite*aqua_n + terra_composite*terra_n)/(aqua_n + terra_n)
    combined_n = aqua_n + terra_n
    np.savetxt(f'{composite_save_dir}/OBS_gpp-NDVI_composite_{composite_name}_region{int(region_number)}.csv', combined_composite, delimiter=",", fmt='%.3e')
    np.savetxt(f'{composite_save_dir}/OBS_gpp-NDVI_n_{composite_name}_region{int(region_number)}.csv', combined_n, delimiter=",", fmt='%d')


def compute_scaling_factors(scale_variable, region_number, obs=[]):
    obs_composite = np.genfromtxt(f'{composite_save_dir}/OBS_{scale_variable}_composite_{composite_name}_region{region_number}.csv', delimiter=',')
    max_obs_anom = np.nanmax(obs_composite)
    scaling_factor = {}
    cf_standard_name = scale_variable.split('-')[0]
    for model in ['NorESM2-LM', 'CNRM-ESM2-1', 'BCC-CSM2-MR', 'ACCESS-ESM1-5', 'UKESM1-0-LL']:
        model_composite = np.genfromtxt(f'{composite_save_dir}/{model}_{cf_standard_name}_composite_{composite_name}_region{region_number}.csv', delimiter=',')
        max_model_anom = np.nanmax(model_composite)
        scaling_factor[model] = max_obs_anom/max_model_anom
    for plot_obs in obs:
        scaling_factor[f'OBS_{plot_obs}'] = 1.
    return scaling_factor


def obs_legend_entries(handles, labels, cf_standard_name):
    label_prefix = f'OBS_{cf_standard_name}-'
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


def plot_variable_comparison(cf_standard_name, region_number, scale_by_max=None,
                             ax=None, plot_models_legend=True, plot_obs_legend=True, title=False, 
                             save=False, show=True, std_error=False, obs=[]):
    if isinstance(obs, str):
        obs = [obs]
    obs_for_variable = [o for o in obs if o.startswith(f'{cf_standard_name}-')]
    if scale_by_max is not None:
        scaling_factor = compute_scaling_factors(scale_by_max, region_number, obs=obs)
    region_filenames = [f for f in os.listdir(composite_save_dir) if f.endswith(f'_region{region_number}.csv')]
    filenames_for_variable = [f for f in region_filenames if f'_{cf_standard_name}_composite_{composite_name}' in f]
    filenames_for_variable.sort()
    model_composites = {f.split('_')[0]: np.genfromtxt(f'{composite_save_dir}/{f}', delimiter=',') for f in filenames_for_variable}
    if std_error:
        filenames_for_std = [f for f in region_filenames if f'_{cf_standard_name}_composite_std' in f]
        filenames_for_std.sort()
        model_composites_std = {f.split('_')[0]: np.genfromtxt(f'{composite_save_dir}/{f}', delimiter=',') for f in filenames_for_std}            
        filenames_for_n = [f for f in region_filenames if f'_{cf_standard_name}_n_' in f]
        filenames_for_n.sort()
        model_composites_n = {f.split('_')[0]: np.genfromtxt(f'{composite_save_dir}/{f}', delimiter=',') for f in filenames_for_n}
    for plot_obs in obs:
        if plot_obs.startswith(f'{cf_standard_name}-'):
            model_composites[f'OBS_{plot_obs}'] = np.genfromtxt(f'{composite_save_dir}/OBS_{plot_obs}_composite_{composite_name}_region{region_number}.csv', delimiter=',')
            if std_error:
                model_composites_std[f'OBS_{plot_obs}'] = np.genfromtxt(f'{composite_save_dir}/OBS_{plot_obs}_composite_std_{composite_name}_region{region_number}.csv', delimiter=',')
                model_composites_n[f'OBS_{plot_obs}'] = np.genfromtxt(f'{composite_save_dir}/OBS_{plot_obs}_n_{composite_name}_region{region_number}.csv', delimiter=',')
    if ax is None:
        fig = plt.figure(figsize=(6, 3.85))
        ax = plt.gca()
    days_around = np.arange(-60, 61)
    for model in model_composites.keys():
        composite = model_composites[model]
        if std_error:
            err = model_composites_std[model]/np.sqrt(model_composites_n[model])
            upper_95ci = composite + 1.96 * err
            lower_95ci = composite - 1.96 * err
        if scale_by_max is not None:
            composite *= scaling_factor[model]
            if std_error:
                upper_95ci *= scaling_factor[model]
                lower_95ci *= scaling_factor[model]
        ax.plot(days_around, composite, label=model, color=model_colours(model), linestyle=obs_type_linestyles(model))
        if std_error:
            ax.fill_between(days_around, lower_95ci, y2=upper_95ci, edgecolor=model_colours(model), facecolor=model_colours(model), alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if plot_obs_legend and len(obs_for_variable) > 0:
        obs_handles, obs_labels = obs_legend_entries(handles, labels, cf_standard_name)
        obs_legend = ax.legend(obs_handles, obs_labels, loc='lower right', fontsize=10)
    if plot_models_legend:
        model_handles, model_labels = single_obs_legend_entry(handles, labels)        
        ax.legend(model_handles, model_labels, loc='upper left', fontsize=10)
        if plot_obs_legend and len(obs_for_variable) > 0: # need to add the obs legend back on if it exists or won't get both legends showing
            ax.add_artist(obs_legend)
    if title:
        ax.set_title(f'{cf_standard_name}, {region_names[region_number]}', fontsize=14)
    ax.set_xlabel('days since precipitation ISV maximum', fontsize=14)
    ax.set_xlim([-60, 60])
    if standardise_anomalies:
        ax.set_ylabel('standardised anomaly', fontsize=14)
    else:
        ax.set_ylabel(f'anomaly ({variable_units[cf_standard_name]})', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.axhline(0, color='gray', alpha=0.5)
    ax.axvline(0, color='gray', alpha=0.5)
    if save:
        save_dir = f'../figures/multimodel/regional/{composite_name}'
        os.system(f'mkdir -p {save_dir}')
        save_filename = f'{save_dir}/{cf_standard_name}_{region_abbreviations[region_number]}_multimodel_comparison'
        if scale_by_max is not None:
            save_filename += f'_scale_by_max_{scale_by_max}'
        if len(obs) > 0:
            save_filename += f"_OBS_{'_'.join(obs_for_variable)}"
        plt.tight_layout()
        plt.savefig(f'{save_filename}.png', dpi=300)
        print(save_filename)
    if show:
        plt.show()


def subplot_regional_response(region_number, scale_by_max=None, std_error=False, 
                              obs=[], show=False):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=standardise_anomalies, figsize=(9, 6.75))
    axs_list = axs.flatten()
    plot_vars = ['pr', 'mrsos', 'vpd', 'gpp']
    for i, v in enumerate(plot_vars):
        ax = axs_list[i]
        plot_models_legend = (i==0)
        plot_variable_comparison(v, region_number, scale_by_max=scale_by_max, 
                                 obs=obs, ax=ax, plot_models_legend=plot_models_legend, 
                                 plot_obs_legend=True, title=False, 
                                 save=False, show=False, std_error=std_error)
        ax.set_title(v)
        if standardise_anomalies:
            ax.label_outer()
    abbrev = region_abbreviations[region_number]
    plt.suptitle(f'{region_names[region_number]} ({abbrev})', fontsize=14)
    plt.tight_layout()
    save_dir = f'../figures/multimodel/regional/{composite_name}/subplots/multi_obs'
    os.system(f"mkdir -p {save_dir}")
    save_filename = f'subplots_region_{abbrev}'
    if scale_by_max is not None:
        save_filename += f'_scale_by_max_{scale_by_max}'
    if std_error:
        save_filename += '_stderror'
    if len(obs) > 0:
        save_filename += f"_OBS_{'_'.join(obs)}"
    plt.savefig(f'{save_dir}/{save_filename}.png', dpi=600, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_event_months(regions_to_include):
    month_range = np.arange(12) + 1
    month_labels = [m[0] for m in list(calendar.month_abbr)[1:]]
    save_dir = f'{composite_save_dir}/month_counts/'
    os.system(f"mkdir -p {save_dir}")
    for region in regions_to_include:
        fig, ax = plt.subplots(figsize=(6, 4))
        for model in ['NorESM2-LM', 'CNRM-ESM2-1', 'BCC-CSM2-MR', 'ACCESS-ESM1-5', 'UKESM1-0-LL', 'OBS']:
            month_counts = np.genfromtxt(f'{composite_save_dir}/{model}_event_months_{composite_name}_region{int(region)}.csv', delimiter=',')
            plt.plot(month_range, month_counts, '-o', ms=2, label=model, color=model_colours(model))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.tick_params(labelsize=14)
        ax.set_xticks(month_range)
        ax.set_xticklabels(month_labels, fontsize=16)
        ax.set_ylabel('number of events', fontsize=16)
        ax.set_title(f'{region_names[region]} ({region_abbreviations[region]})', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/event_month_counts_{region_abbreviations[region]}.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    for model in cmip6_models + ['OBS']:
        save_composites(model, regions_to_include)
    plot_event_months(regions_to_include)
    plot_obs = ['pr-IMERG', 'mrsos-ESACCI', 'mrsos-GLEAM', 'tasmax-ERA5', 'tasmax-MERRA2', 'gpp-FLUXCOM-ERA5', 'gpp-MODIS-TERRA', 'gpp-SIF-GOME2-JJ', 'hfls-GLEAM', 'hfls-Lu2021', 'rsds-CERES']
    for region_number in regions_to_include:
        subplot_regional_response(region_number, scale_by_max='pr-IMERG', obs=[])
        plot_variable_comparison('gpp', region_number, scale_by_max='pr-IMERG',
                                 ax=None, plot_models_legend=True, plot_obs_legend=False, title=True, 
                                 save=True, show=False, std_error=False, obs=[])