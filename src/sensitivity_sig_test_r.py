import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
import os


emmeans = importr('emmeans')
pandas2ri.activate()
R = ro.r

analysis_version_name = 'rolling_7d_mean_stdev_maskfrozen_regrid_1_by_1_deg_standardised_CCI-SM-mask'
os.system(f'mkdir -p ../data/amplitudes_lags/{analysis_version_name}')


def regions_with_enough_data(models, variable, obs_product):
    all_regions = np.arange(46)
    valid_regions = []
    for region_number in all_regions:
        n_all_models = []
        composite_save_dir = f'../data/composites/multimodel/regional/{analysis_version_name}'
        for model in models:
            variable_label = f'{variable}-{obs_product}' if model == 'OBS' else variable
            n_filename = f'{model}_{variable_label}_n_{analysis_version_name}_region{int(region_number)}.csv'
            n = np.loadtxt(f'{composite_save_dir}/{n_filename}', delimiter=',')
            n_all_models.append(min(n))
        if min(n_all_models) >= 20:
            valid_regions.append(region_number)
    return np.array(valid_regions)


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


def compute_scaling_factor(model, variable, scale_variable, region_number):
    if model.startswith('OBS'):
        scaling_factor = 1.
    else:
        obs_composite = np.genfromtxt(f'../data/composites/multimodel/regional/{analysis_version_name}/OBS_{scale_variable}_composite_{analysis_version_name}_region{region_number}.csv', delimiter=',')
        max_obs_anom = np.nanmax(obs_composite)
        cf_standard_name = scale_variable.split('-')[0]
        model_composite = np.genfromtxt(f'../data/composites/multimodel/regional/{analysis_version_name}/{model}_{cf_standard_name}_composite_{analysis_version_name}_region{region_number}.csv', delimiter=',')
        max_model_anom = np.nanmax(model_composite)
        scaling_factor= max_obs_anom/max_model_anom
    return scaling_factor


def get_composite(model, variable_name, region_number, obs_product, scale_by_max=None):
    composite_directory = f'../data/composites/multimodel/regional/{analysis_version_name}'
    variable_label = f'{variable_name}-{obs_product}' if model == 'OBS' else variable_name
    filename = f"{composite_directory}/{model}_{variable_label}_composite_{analysis_version_name}_region{int(region_number)}.csv"
    composite = np.genfromtxt(filename, delimiter=',')
    if scale_by_max is not None:
        scaling_factor = compute_scaling_factor(model, variable_name, scale_by_max, region_number)
        composite *= scaling_factor
    return composite


def get_n_events(model, variable_name, region_number, obs_product):
    composite_directory = f'../data/composites/multimodel/regional/{analysis_version_name}'
    variable_label = f'{variable_name}-{obs_product}' if model == 'OBS' else variable_name
    filename = f"{composite_directory}/{model}_{variable_label}_n_{analysis_version_name}_region{int(region_number)}.csv"
    n = np.genfromtxt(filename, delimiter=',')
    return n


def amplitudes_and_lag(model, variable_name, region_number, obs_product, force_positive=False, scale_by_max=None):
    composite = get_composite(model, variable_name, region_number, obs_product, scale_by_max=scale_by_max)
    n = get_n_events(model, variable_name, region_number, obs_product)
    if np.nanmax(n) < 50:
        amplitude = np.nan
        lag = np.nan
        final_amplitude = np.nan
    else:
        days_either_side = (composite.size - 1) // 2
        days_around = np.arange(-days_either_side, days_either_side+1)
        if force_positive and variable_name == 'pr':
            print('Not forcing pr lag to be positive')
            force_positive = False
        if force_positive:
            lag = np.argmax(composite[days_around>=0.])
            amplitude = composite[lag+days_either_side]
        else:
            amplitude = composite[np.argmax(np.abs(composite))]
            lag = days_around[np.argmax(np.abs(composite))]
        if np.isnan(amplitude):
            lag = np.nan
        final_amplitude = np.nanmean(composite[-20:])
    return amplitude, lag, final_amplitude


def all_region_amps_lags(model, variable_name, obs_product, force_positive=False, scale_by_max=None):
    amplitudes = []
    lags = []
    final_amplitudes = []
    regions_to_include = np.arange(46)
    regions_to_include = np.delete(regions_to_include, [0, 20, 36])
    for region_number in regions_to_include:
        amplitude, lag, final_amplitude, = amplitudes_and_lag(model, variable_name, region_number,
                                                              obs_product,
                                                              force_positive=force_positive,
                                                              scale_by_max=scale_by_max)
        amplitudes.append(amplitude)
        lags.append(lag)
        final_amplitudes.append(final_amplitude)
    return amplitudes, lags, final_amplitudes


def remove_nans(tuple_of_lists):
    stack_lists = np.vstack(tuple_of_lists)
    any_nans = np.any(np.isnan(stack_lists), axis=0)
    nan_idcs = np.where(any_nans)[0]
    removed_nans = np.delete(stack_lists, nan_idcs, axis=1)
    split_lists = np.vsplit(removed_nans, len(tuple_of_lists))
    return (l.flatten().tolist() for l in split_lists)


def compare_trends_mrsos():
    models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'NorESM2-LM', 'UKESM1-0-LL']
    gpp_products = ['FLUXCOM-ERA5', 'FLUXCOM-CRUJRAv1', 'VODCA2GPP', 'MODIS-TERRA', 'SIF-GOME2-JJ', 'SIF-GOME2-PK', 'VPM']
    mrsos_obs = 'ESACCI'
    model_types = []
    mrsos_all_amps = []
    gpp_all_amps = []
    gpp_all_lags = []
    mrsos_all_final_amps = []
    gpp_all_final_amps = []
    for model in models:
        gpp_amps, gpp_lags, gpp_final_amps = all_region_amps_lags(model, 'gpp', None, force_positive=True)
        mrsos_amps, _, mrsos_final_amps = all_region_amps_lags(model, 'mrsos', None, force_positive=True)
        mrsos_all_amps += mrsos_amps
        mrsos_all_final_amps += mrsos_final_amps
        gpp_all_amps += gpp_amps
        gpp_all_lags += gpp_lags
        gpp_all_final_amps += gpp_final_amps
        model_types += [model]*len(gpp_amps)
    for gpp_obs in gpp_products:
        gpp_amps, gpp_lags, gpp_final_amps = all_region_amps_lags('OBS', 'gpp', gpp_obs, force_positive=True)
        mrsos_amps, _, mrsos_final_amps = all_region_amps_lags('OBS', 'mrsos', mrsos_obs, force_positive=True)
        mrsos_all_amps += mrsos_amps
        mrsos_all_final_amps += mrsos_final_amps
        gpp_all_amps += gpp_amps
        gpp_all_lags += gpp_lags
        gpp_all_final_amps += gpp_final_amps
        model_types += [gpp_obs]*len(gpp_amps)
    df = pd.DataFrame({'mrsos_amp': mrsos_all_amps,
                       'gpp_amp': gpp_all_amps,
                       'model': model_types
    })
    m = R.lm('mrsos_amp ~ model*gpp_amp', df)
    p = R.pairs(emmeans.emtrends(m, 'model', var="gpp_amp"))
    print(p)
    R.summary(p).to_csvfile(f'../data/amplitudes_lags/{analysis_version_name}/emtrends_mrsos_amplitude.csv')

    df = pd.DataFrame({'mrsos_amp': mrsos_all_amps,
                       'gpp_lag': gpp_all_lags,
                       'model': model_types
    })
    m = R.lm('mrsos_amp ~ model*gpp_lag', df)
    p = R.pairs(emmeans.emtrends(m, 'model', var="gpp_lag"))
    print(p)
    R.summary(p).to_csvfile(f'../data/amplitudes_lags/{analysis_version_name}/emtrends_mrsos_lag.csv')

    df = pd.DataFrame({'mrsos_final_amp': mrsos_all_final_amps,
                       'gpp_final_amp': gpp_all_final_amps,
                       'model': model_types
    })
    m = R.lm('mrsos_final_amp ~ model*gpp_final_amp', df)
    p = R.pairs(emmeans.emtrends(m, 'model', var="gpp_final_amp"))
    print(p)
    R.summary(p).to_csvfile(f'../data/amplitudes_lags/{analysis_version_name}/emtrends_mrsos_final_amp.csv')


def compare_trends_vpd():
    models = ['ACCESS-ESM1-5', 'CNRM-ESM2-1', 'NorESM2-LM', 'UKESM1-0-LL']
    gpp_products = ['FLUXCOM-ERA5', 'FLUXCOM-CRUJRAv1', 'VODCA2GPP', 'MODIS-TERRA', 'SIF-GOME2-JJ', 'SIF-GOME2-PK', 'VPM']
    vpd_obs = 'ERA5'
    model_types = []
    vpd_all_amps = []
    vpd_all_final_amps = []
    gpp_all_amps = []
    gpp_all_lags = []
    gpp_all_final_amps = []
    for model in models:
        gpp_amps, gpp_lags, gpp_final_amps = all_region_amps_lags(model, 'gpp', None, force_positive=True)
        vpd_amps, _, vpd_final_amps = all_region_amps_lags(model, 'vpd', None, force_positive=False)
        vpd_all_amps += vpd_amps
        gpp_all_amps += gpp_amps
        vpd_all_final_amps += vpd_final_amps
        gpp_all_final_amps += gpp_final_amps
        gpp_all_lags += gpp_lags
        model_types += [model]*len(gpp_amps)
    for gpp_obs in gpp_products:
        gpp_amps, gpp_lags, gpp_final_amps = all_region_amps_lags('OBS', 'gpp', gpp_obs, force_positive=True)
        vpd_amps, _, vpd_final_amps = all_region_amps_lags('OBS', 'vpd', vpd_obs, force_positive=False)
        vpd_all_amps += vpd_amps
        vpd_all_final_amps += vpd_final_amps
        gpp_all_amps += gpp_amps
        gpp_all_lags += gpp_lags
        gpp_all_final_amps += gpp_final_amps
        model_types += [gpp_obs]*len(gpp_amps)
    df = pd.DataFrame({'vpd_amp': vpd_all_amps,
                       'gpp_amp': gpp_all_amps,
                       'model': model_types
    })
    m = R.lm('vpd_amp ~ model*gpp_amp', df)
    p = R.pairs(emmeans.emtrends(m, 'model', var="gpp_amp"))
    print(p)
    R.summary(p).to_csvfile(f'../data/amplitudes_lags/{analysis_version_name}/emtrends_vpd_amplitude.csv')

    df = pd.DataFrame({'vpd_amp': vpd_all_amps,
                       'gpp_lag': gpp_all_lags,
                       'model': model_types
    })
    m = R.lm('vpd_amp ~ model*gpp_lag', df)
    p = R.pairs(emmeans.emtrends(m, 'model', var="gpp_lag"))
    print(p)
    R.summary(p).to_csvfile(f'../data/amplitudes_lags/{analysis_version_name}/emtrends_vpd_lag.csv')

    df = pd.DataFrame({'vpd_final_amp': vpd_all_final_amps,
                       'gpp_final_amp': gpp_all_final_amps,
                       'model': model_types
    })
    m = R.lm('vpd_final_amp ~ model*gpp_final_amp', df)
    p = R.pairs(emmeans.emtrends(m, 'model', var="gpp_final_amp"))
    print(p)
    R.summary(p).to_csvfile(f'../data/amplitudes_lags/{analysis_version_name}/emtrends_vpd_final_amplitude.csv')


if __name__ == '__main__':
    compare_trends_mrsos()
    compare_trends_vpd()