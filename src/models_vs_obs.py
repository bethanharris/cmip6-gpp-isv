import numpy as np
import matplotlib.pyplot as plt
import iris
import xarray as xr
import scipy.stats
import numpy.ma as ma
import os
import regionmask
import pandas as pd


analysis_version_name = 'rolling_7d_mean_stdev_maskfrozen'
regrid_label = 'regrid_1_by_1_deg'
standardise_anomalies = True
use_obs_sm_mask = True

cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'NorESM2-LM', 'UKESM1-0-LL']

standardise_label = 'standardised' if standardise_anomalies else 'unstandardised'
composite_name = f'{analysis_version_name}_{regrid_label}_{standardise_label}'
if use_obs_sm_mask:
    composite_name += '_CCI-SM-mask'

composite_save_dir = f'../data/composites/multimodel/regional/{composite_name}'

ar6_land = regionmask.defined_regions.ar6.land
region_names = ar6_land.names
region_abbreviations = ar6_land.abbrevs

opposite_season = {'JJA': 'DJF',
                   'MAM': 'SON',
                   'DJF': 'JJA',
                   'SON': 'MAM',
                   'all': 'all'}

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


def gpp_obs_only_colours(obs_type):
    obs_colours = {
                   'OBS-VPM': '#66a61e',
                   'OBS-FLUXCOM-ERA5': '#e6ab02',
                   'OBS-FLUXCOM-CRUJRAv1': '#FFDF83',
                   'OBS-MODIS-TERRA': '#43710F',
                   'OBS-SIF-GOME2-JJ': '#e7298a',
                   'OBS-SIF-GOME2-PK': '#EF92C1',
                   'OBS-VOD': '#7570b3',
                   'OBS-VODCA2GPP': '#7570b3',
                   'OBS-NDVI': '#d95f02',
                    }
    if obs_type in obs_colours.keys():
        colour = obs_colours[obs_type]
    else:
        colour = '-'
    return colour


def grid_box_weights(model):
    if regrid_label != 'regrid_1_by_1_deg':
        raise KeyError('Have not set up grid box land area weighting for non-1deg grids')
    deg_spacing = 1.
    radius_earth = 6371000.
    global_lon = (np.arange(0, 360, deg_spacing) + 0.5*deg_spacing)
    global_lat = (np.arange(-90, 90, deg_spacing) + 0.5*deg_spacing)
    _, lat_grid = np.meshgrid(global_lon, global_lat)
    grid_area = (np.pi / 180.) * (radius_earth ** 2) * deg_spacing * np.abs(
    np.sin(np.deg2rad(lat_grid+0.5*deg_spacing)) - np.sin(np.deg2rad(lat_grid-0.5*deg_spacing)))
    if model == 'OBS':
        land_frac = iris.load_cube("/prj/nceo/bethar/modis_land_percentage_1deg.nc").data
        land_frac = np.roll(land_frac, 180)
    else:
        land_frac = iris.load_cube(f"/prj/nceo/bethar/esm-isv/{model}/regrid_1_by_1_deg/{model}_sftlf.nc").data
    weights = grid_area * land_frac
    return weights


def regions_with_enough_data(models, variable):
    all_regions = np.arange(46)
    valid_regions = []
    for region_number in all_regions:
        n_all_models = []
        for model in models:
            n_filename = f'{model}_{variable}_n_{composite_name}_region{int(region_number)}.csv'
            n = np.loadtxt(f'{composite_save_dir}/{n_filename}', delimiter=',')
            n_all_models.append(min(n))
        if min(n_all_models) >= 20:
            valid_regions.append(region_number)
    return np.array(valid_regions)


def hemisphere_regions():
    test_data = iris.load_cube("/prj/nceo/bethar/esm-isv/UKESM1-0-LL/regrid_1_by_1_deg/UKESM1-0-LL_pr.nc")
    data_xr = xr.DataArray.from_iris(test_data)
    coord_names = data_xr.coords._names
    if 'latitude' in coord_names:
        data_xr = data_xr.rename({'latitude': 'lat'})
    if 'longitude' in coord_names:
        data_xr = data_xr.rename({'longitude': 'lon'})
    mask_3D = ar6_land.mask_3D(data_xr)
    data_lats = data_xr.lat
    weights = grid_box_weights('OBS')
    regional_means = data_lats.weighted(mask_3D*ma.filled(weights,0)).mean(dim=("lat", "lon"))
    regions = regional_means.region.data
    regional_mean_values = regional_means.data
    nhem_regions = regions[regional_mean_values >= 0.]
    shem_regions = regions[regional_mean_values < 0.]
    return nhem_regions, shem_regions


def get_composite(model, variable_name, obs_product, region_number, season):
    season_label = '' if season == 'all' else f'_{season}'
    composite_directory = f'{composite_save_dir}{season_label}'
    variable_label = obs_product if model == 'OBS' else variable_name
    filename = f"{composite_directory}/{model}_{variable_label}_composite_{composite_name}{season_label}_region{int(region_number)}.csv"
    return np.genfromtxt(filename, delimiter=',')


def get_n_events(model, variable_name, obs_product, region_number, season):
    season_label = '' if season == 'all' else f'_{season}'
    composite_directory = f'{composite_save_dir}{season_label}'
    variable_label = obs_product if model == 'OBS' else variable_name
    filename = f"{composite_directory}/{model}_{variable_label}_n_{composite_name}{season_label}_region{int(region_number)}.csv"
    return np.genfromtxt(filename, delimiter=',')


def compute_scaling_factor(model, scale_variable, region_number, season, scale_final=False):
    season_label = '' if season == 'all' else f'_{season}'
    obs_composite = np.genfromtxt(f'{composite_save_dir}{season_label}/OBS_{scale_variable}_composite_{composite_name}{season_label}_region{region_number}.csv', delimiter=',')
    max_obs_anom = np.nanmax(obs_composite)
    final_obs_anom = np.nanmean(obs_composite[-20:])
    variable_name = scale_variable.split('-')[0]
    model_composite = np.genfromtxt(f'{composite_save_dir}{season_label}/{model}_{variable_name}_composite_{composite_name}{season_label}_region{region_number}.csv', delimiter=',')
    max_model_anom = np.nanmax(model_composite)
    final_model_anom = np.nanmean(model_composite[-20:])
    if scale_final:
        scaling_factor = final_obs_anom/final_model_anom
    else:
        scaling_factor = max_obs_anom/max_model_anom
    return scaling_factor


def amplitude_and_lag(model, variable, obs_product, region_number, season, 
                      force_positive=False, scale_by=None):
    composite = get_composite(model, variable, obs_product, region_number, season)
    n = get_n_events(model, variable, obs_product, region_number, season)
    if np.logical_or(np.all(np.isnan(composite)), np.nanmax(n)<50):
        amplitude = np.nan
        lag = np.nan
        final_amplitude = np.nan
    else:
        if model != 'OBS' and scale_by is not None:
            peak_scaling_factor = compute_scaling_factor(model, scale_by, region_number, season)
            final_scaling_factor = compute_scaling_factor(model, scale_by, region_number, season,
                                                          scale_final=True)
        else:
            peak_scaling_factor = 1.
            final_scaling_factor = 1.
        days_either_side = (composite.size - 1) // 2
        days_around = np.arange(-days_either_side, days_either_side+1)
        if force_positive and variable == 'pr':
            print('Not forcing pr lag to be positive')
            force_positive = False
        if force_positive:
            lag = np.nanargmax(composite[days_around>=0.])
            amplitude = composite[lag+days_either_side]
        else:
            amplitude = composite[np.nanargmax(np.abs(composite))]
            lag = days_around[np.nanargmax(np.abs(composite))]
        if np.isnan(amplitude):
            lag = np.nan
        amplitude *= peak_scaling_factor
        final_amplitude = np.nanmean(composite[-20:]) * final_scaling_factor
    return amplitude, lag, final_amplitude


def all_region_lags_amplitudes(model, composite_variable, obs_product, nhem_regions, shem_regions, season, 
                               force_positive=False, scale_by=None):
    all_lags = {}
    all_amplitudes = {}
    all_final_amplitudes = {}
    for region in nhem_regions:
        amplitude, lag, final_amplitude = amplitude_and_lag(model, composite_variable, obs_product, region, 
                                                            season, force_positive=force_positive, scale_by=scale_by)
        all_lags[str(region)] = lag
        all_amplitudes[str(region)] = amplitude
        all_final_amplitudes[str(region)] = final_amplitude
    for region in shem_regions:
        amplitude, lag, final_amplitude = amplitude_and_lag(model, composite_variable, obs_product, region, 
                                                            opposite_season[season], force_positive=force_positive, scale_by=scale_by)
        all_lags[str(region)] = lag
        all_amplitudes[str(region)] = amplitude
        all_final_amplitudes[str(region)] = final_amplitude
    return all_amplitudes, all_lags, all_final_amplitudes


def remove_nans(tuple_of_lists):
    stack_lists = np.vstack(tuple_of_lists)
    any_nans = np.any(np.isnan(stack_lists), axis=0)
    nan_idcs = np.where(any_nans)[0]
    removed_nans = np.delete(stack_lists, nan_idcs, axis=1)
    split_lists = np.vsplit(removed_nans, len(tuple_of_lists))
    return (l.flatten().tolist() for l in split_lists)


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def save_amplitudes_lags(cmip_variable_name, obs_products=[], season='all', scale_by=None):

    force_positive = cmip_variable_name.startswith('mrso') or cmip_variable_name in ['gpp', 'hfls']

    regions_to_include = regions_with_enough_data(cmip6_models, 'gpp')
    regions_to_include = regions_to_include[~np.isin(regions_to_include, [20, 36])]
    regions_in_nhem, regions_in_shem = hemisphere_regions()
    nhem_regions = np.intersect1d(regions_to_include, regions_in_nhem)
    shem_regions = np.intersect1d(regions_to_include, regions_in_shem)

    save_amplitudes = {}
    save_lags = {}
    save_final_amplitudes = {}

    region_numbers = [int(r) for r in nhem_regions] + [int(r) for r in shem_regions]
    save_amplitudes['region'] = region_numbers
    save_lags['region'] = region_numbers
    save_final_amplitudes['region'] = region_numbers

    for obs_product in obs_products:
        obs_amp, obs_lag, obs_final_amp = all_region_lags_amplitudes('OBS', cmip_variable_name, f'{cmip_variable_name}-{obs_product}', 
                                                                     nhem_regions, shem_regions, season,
                                                                     force_positive=force_positive)
        region_amps_obs = [obs_amp[str(r)] for r in nhem_regions] + [obs_amp[str(r)] for r in shem_regions]
        region_lags_obs = [obs_lag[str(r)] for r in nhem_regions] + [obs_lag[str(r)] for r in shem_regions]
        region_final_amps_obs = [obs_final_amp[str(r)] for r in nhem_regions] + [obs_final_amp[str(r)] for r in shem_regions]
        save_amplitudes[obs_product] = region_amps_obs
        save_lags[obs_product] = region_lags_obs
        save_final_amplitudes[obs_product] = region_final_amps_obs
    for model in cmip6_models:
        if not (model == 'BCC-CSM2-MR' and cmip_variable_name in ['vpd', 'mrsol_1.0']):
            amplitudes, lags, final_amplitudes = all_region_lags_amplitudes(model, cmip_variable_name, '', nhem_regions, shem_regions, season,
                                                                            force_positive=force_positive, scale_by=scale_by)
            region_amps = [amplitudes[str(r)] for r in nhem_regions] + [amplitudes[str(r)] for r in shem_regions]
            region_lags = [lags[str(r)] for r in nhem_regions] + [lags[str(r)] for r in shem_regions]
            region_final_amps = [final_amplitudes[str(r)] for r in nhem_regions] + [final_amplitudes[str(r)] for r in shem_regions]
            save_amplitudes[model] = region_amps
            save_lags[model] = region_lags
            save_final_amplitudes[model] = region_final_amps
    save_filename = f'{cmip_variable_name}_obs_vs_models' if season == 'all' else f'{cmip_variable_name}_obs_vs_models_{season}'
    if scale_by is not None:
        save_filename += f'_scale_by_{scale_by}'
    save_dir = f'../data/amplitudes_lags/{composite_name}'
    os.system(f'mkdir -p {save_dir}')
    pd.DataFrame(save_amplitudes).sort_values('region').to_csv(f'{save_dir}/amplitude_{save_filename}.csv', index=False)
    pd.DataFrame(save_lags).sort_values('region').to_csv(f'{save_dir}/lag_{save_filename}.csv', index=False)
    pd.DataFrame(save_final_amplitudes).sort_values('region').to_csv(f'{save_dir}/final_amplitude_{save_filename}.csv', index=False)


def plot_linear_regression(ax, x, y, model, legend_r_only=False, use_obs_colours=False):
    regression = scipy.stats.linregress(x, y)
    if legend_r_only:
        # model_initial = model[0]
        # model_label = f'{model_initial}: r={regression.rvalue:.2f}'
        model_label = f'r={regression.rvalue:.2f}'
    else:
        model_name = model[4:] if model.startswith('OBS') else model
        model_label = f'{model_name}, r={regression.rvalue:.2f}'
    if regression.pvalue < 0.05:
        model_label += '*'
    colour = gpp_obs_only_colours(model) if use_obs_colours else model_colours(model)
    ax.scatter(x, y, s=18, c=colour, label=model_label)
    x_range = np.linspace(min(x), max(x), 10)
    fit_line = regression.slope * x_range + regression.intercept
    ax.plot(x_range, fit_line, c=colour, linestyle='--', linewidth=0.95)


def scatter_models_vs_obs(models, cmip_variable_name, obs_product, season='all', save=False, scale_by=None,
                          amp_ax=None, lag_ax=None, final_amp_ax=None, legends_in_frame=False, subplot_labels=False):
    if amp_ax is None:
        amp_fig, amp_ax = plt.subplots(figsize=(6, 4.5))
    if lag_ax is None:
        lag_fig, lag_ax = plt.subplots(figsize=(6, 4.5))
    if final_amp_ax is None:
        final_amp_fig, final_amp_ax = plt.subplots(figsize=(6, 4.5))

    season_label = '' if season == 'all' else f'_{season}'
    scale_by_label = '' if scale_by is None else f'_scale_by_{scale_by}'
    amplitude_lag_save_dir = f'../data/amplitudes_lags/{composite_name}'
    amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_{cmip_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')
    lag_data = pd.read_csv(f'{amplitude_lag_save_dir}/lag_{cmip_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')
    final_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_{cmip_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')

    amplitude_data.dropna(inplace=True)
    lag_data.dropna(inplace=True)
    final_amplitude_data.dropna(inplace=True)

    region_amps_obs = amplitude_data[obs_product]
    region_lags_obs = lag_data[obs_product]
    region_final_amps_obs = final_amplitude_data[obs_product]

    for model in models:
        region_amps = amplitude_data[model]
        region_lags = lag_data[model]
        region_final_amps = final_amplitude_data[model]

        plot_linear_regression(amp_ax, region_amps_obs, region_amps, model, legend_r_only=False)
        plot_linear_regression(lag_ax, region_lags_obs, region_lags, model, legend_r_only=legends_in_frame)
        plot_linear_regression(final_amp_ax, region_final_amps_obs, region_final_amps, model, legend_r_only=legends_in_frame)

    season_title = '' if season=='all' else f' {season}'
    amp_title = f'{cmip_variable_name} peak amplitude' + season_title
    lag_title = f'{cmip_variable_name} lag' + season_title + ' (days)'
    final_amp_title = f'{cmip_variable_name} post-event amplitude' + season_title
    if subplot_labels:
        amp_title = "$\\bf{(b)}$ " + amp_title
        lag_title = "$\\bf{(d)}$ " + lag_title
        final_amp_title = "$\\bf{(c)}$ " + final_amp_title
    amp_ax.set_title(amp_title, fontsize=14)
    lag_ax.set_title(lag_title, fontsize=14)
    final_amp_ax.set_title(final_amp_title, fontsize=14)
    for ax in [amp_ax, lag_ax, final_amp_ax]:
        ax.set_xlabel('obs', fontsize=14)
        ax.set_ylabel('model', fontsize=14)
        ax.tick_params(labelsize=14)
        if legends_in_frame:
            ax.legend(loc='best', handletextpad=0.1)
        else:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        add_identity(ax, color='gray', alpha=0.5, zorder=0)
    if save:
        season_label = '' if season == 'all' else f'_{season}'
        save_dir = f'../figures/multimodel/regional/model_vs_obs_scatters/{composite_name}'
        save_filename = f'{cmip_variable_name}_{obs_product}_{season}'
        if scale_by is not None:
            save_filename += f'_scale_by_{scale_by}'
        os.system(f'mkdir -p {save_dir}')
        plt.figure(amp_fig)
        plt.savefig(f'{save_dir}/amplitude_{save_filename}.png', dpi=400, bbox_inches='tight')
        plt.figure(lag_fig)
        plt.savefig(f'{save_dir}/lag_{save_filename}.png', dpi=400, bbox_inches='tight')
        plt.figure(final_amp_fig)
        plt.savefig(f'{save_dir}/final_amplitude_{save_filename}.png', dpi=400, bbox_inches='tight')
        plt.close('all')


def illustrate_amplitudes_lags(ax=None, subplot_label=False):
    if ax is None:
        fig = plt.figure(figsize=(6, 4.5))
        ax = plt.gca()
    example_composite = np.genfromtxt("/users/global/bethar/python/cmip6-gpp-isv/data/composites/multimodel/regional/rolling_7d_mean_stdev_maskfrozen_regrid_1_by_1_deg_standardised_CCI-SM-mask/UKESM1-0-LL_gpp_composite_rolling_7d_mean_stdev_maskfrozen_regrid_1_by_1_deg_standardised_CCI-SM-mask_region39.csv")
    days_around = 60
    x = np.arange(-days_around, days_around+1)
    lag = np.argmax(example_composite[x>=0.])
    amplitude = example_composite[lag+days_around]
    final_amplitude = np.nanmean(example_composite[-20:])
    ax.plot(x, example_composite, 'k', linewidth=1.5)
    peak_amplitude_colour = '#EE0000'
    lag_colour = '#EE0000'
    final_amplitude_colour = '#EE0000'
    # for 3-colour version suggest #DC267F for lag, #FFB000 for post-event amplitude, #648FFF for peak amplitude
    top_ylim = ax.get_ylim()[-1]
    ax.set_ylim(top=top_ylim*1.07) # make some extra room at top of plot for the lag label
    final_x = np.arange(40, 61)
    final_amplitude_line = np.ones_like(final_x) * final_amplitude
    ax.plot(final_x, final_amplitude_line, color=final_amplitude_colour)
    ax.axvline(0, color='#BBBBBB', linestyle='--')
    ax.axhline(0, color='#BBBBBB', linestyle='--')
    arrow_style = {'arrowstyle': '<|-|>', 'shrinkA': 0, 'shrinkB': 0, 'linewidth': 1.5}
    ax.annotate('', xy=(0, amplitude), xytext=(lag, amplitude), ha='center',
                arrowprops={'fc': lag_colour, 'ec': lag_colour, **arrow_style})
    ax.annotate('', xy=(lag, 0), xytext=(lag, amplitude), va='center',
                arrowprops={'fc': peak_amplitude_colour, 'ec': peak_amplitude_colour, **arrow_style})
    ax.annotate('', xy=(50, 0), xytext=(50, final_amplitude), va='center',
                arrowprops={'fc': final_amplitude_colour, 'ec': final_amplitude_colour, **arrow_style})
    ax.set_xlim([-days_around, days_around])
    bbox = {'boxstyle': 'square', 'ec': 'w', 'fc': 'w', 'alpha': 1, 'pad': 0.05}
    ax.text(48, 0.5*final_amplitude, 'post-event\namplitude', ha='right', 
            color=final_amplitude_colour, fontsize=12, bbox=bbox)
    ax.text(lag-2, 0.5*amplitude, 'peak\namplitude', ha='right', va='center',
            color=peak_amplitude_colour, fontsize=12, bbox=bbox)
    ax.text(0.4*lag, amplitude*1.025, ' lag', va='bottom', ha='center',
            color=lag_colour, fontsize=12, bbox=bbox)
    ax.tick_params(labelsize=14)
    if subplot_label:
        ax.set_title("$\\bf{(a)}$", fontsize=14)
    ax.set_xlabel('days since intraseasonal precipitation maximum', fontsize=14)
    ax.set_ylabel('standardised anomaly', fontsize=14)


def model_vs_obs_subplots(models, cmip_variable_name, obs_product, season='all', scale_by=None, save=False):
    fig, ((schematic_ax, amp_ax), (final_amp_ax, lag_ax)) = plt.subplots(2, 2, figsize=(12, 9))

    illustrate_amplitudes_lags(ax=schematic_ax, subplot_label=True)
    scatter_models_vs_obs(models, cmip_variable_name, obs_product, season=season, save=False, scale_by=scale_by,
                          amp_ax=amp_ax, lag_ax=lag_ax, final_amp_ax=final_amp_ax, legends_in_frame=True,
                          subplot_labels=True)
    plt.subplots_adjust(hspace=0.3)
    if save:
        season_label = '' if season == 'all' else f'_{season}'
        save_dir = f'../figures/multimodel/regional/model_vs_obs_scatters/{composite_name}'
        save_filename = f'subplots_{cmip_variable_name}_{obs_product}_{season}'
        if scale_by is not None:
            save_filename += f'_scale_by_{scale_by}'
        os.system(f'mkdir -p {save_dir}')
        plt.savefig(f'{save_dir}/{save_filename}.png', dpi=400, bbox_inches='tight')
        plt.savefig(f'{save_dir}/{save_filename}.pdf', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    gpp_obs = ['FLUXCOM-ERA5', 'FLUXCOM-CRUJRAv1', 'VODCA2GPP', 'MODIS-TERRA', 'SIF-GOME2-JJ', 'SIF-GOME2-PK', 'VPM']
    save_amplitudes_lags('gpp', obs_products=gpp_obs, scale_by='mrsos-ESACCI')
    save_amplitudes_lags('mrsos', obs_products=['ESACCI', 'GLEAM'], scale_by=None)
    save_amplitudes_lags('vpd', obs_products=['ERA5'], scale_by=None)
    model_vs_obs_subplots(cmip6_models, 'gpp', 'FLUXCOM-CRUJRAv1', scale_by='mrsos-ESACCI', save=True)