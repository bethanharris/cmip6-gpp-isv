import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import regionmask
import types


analysis_version_name = 'rolling_7d_mean_detrend_filternonorm_norm_globalsmmask'

ar6_land = regionmask.defined_regions.ar6.land
region_names = ar6_land.names
region_abbreviations = ar6_land.abbrevs

model_colours = {'ACCESS-ESM1-5': 'C0',
                 'BCC-CSM2-MR': 'C1',
                 'CNRM-ESM2-1': 'C2', 
                 'NorESM2-LM': 'C3',
                 'UKESM1-0-LL': 'C4',
                 'OBS': 'k'}

variable_units = {'pr': 'kg m^-2 s^-1',
                  'mrsos': 'kg m^-2',
                  'mrso': 'kg m^-2',
                  'tasmax': 'K',
                  'hfls': 'W m^-2',
                  'lai': 'unitless',
                  'gpp': 'kg m^-2 s^-1',
                  'rsds': 'W m^-2',
                  'huss': 'unitless',
                  'vpd': 'hPa'}


def model_colours(model):
    cmip_colours =  {'ACCESS-ESM1-5': 'C0',
                     'BCC-CSM2-MR': 'C1',
                     'CNRM-ESM2-1': 'C2', 
                     'NorESM2-LM': 'C3',
                     'UKESM1-0-LL': 'C4'}
    if model.startswith('OBS'):
        model_colour = 'k'
    elif model in cmip_colours.keys():
        model_colour = cmip_colours[model]
    else:
        raise KeyError(f'Unknown model type {model}')
    return model_colour   


def regions_with_enough_data(models, variable):
    all_regions = np.arange(46)
    valid_regions = []
    for region_number in all_regions:
        n_all_models = []
        composite_save_dir = f'./composites/multimodel/regional/{analysis_version_name}'
        for model in models:
            n_filename = f'{model}_{variable}_n_{analysis_version_name}_region{int(region_number)}.csv'
            n = np.loadtxt(f'{composite_save_dir}/{n_filename}', delimiter=',')
            n_all_models.append(min(n))
        if min(n_all_models) >= 20:
            valid_regions.append(region_number)
    return np.array(valid_regions)


def sort_regions_by_max_amplitude(models, variable, valid_regions, force_positive=False):
    regional_maxima = []
    for region_number in valid_regions:
        amp_all_models = []
        for model in models:
            amp, _ = amplitude_and_lag(model, variable, region_number, force_positive=force_positive)
            amp_all_models.append(amp)
        regional_maxima.append(max(amp_all_models))
    sorted_valid_regions = valid_regions[np.argsort(np.array(regional_maxima))]
    return sorted_valid_regions[::-1]


def load_composite(model, variable, region_number):
    composite_save_dir = f'./composites/multimodel/regional/{analysis_version_name}'
    if model.startswith('OBS'):
        composite_filename = f'{model}_composite_{analysis_version_name}_region{int(region_number)}.csv'
    else:
        composite_filename = f'{model}_{variable}_composite_{analysis_version_name}_region{int(region_number)}.csv'
    return np.loadtxt(f'{composite_save_dir}/{composite_filename}', delimiter=',')


def amplitude_and_lag(model, variable, region_number, force_positive=False):
    composite = load_composite(model, variable, region_number)
    days_either_side = (composite.size - 1) // 2
    days_around = np.arange(-days_either_side, days_either_side+1)
    if force_positive and variable == 'pr':
        print('Not forcing pr lag to be positive')
        force_positive = False
    if force_positive:
        lag = np.argmax(composite[days_around>=0.])
        amplitude = composite[lag+days_either_side]
    else:
        amplitude = composite[np.argmax(np.abs(composite))]
        lag = days_around[np.argmax(np.abs(composite))]
    return amplitude, lag


def regional_final_anomaly(model, variable, region_number, force_positive=False):
    composite = load_composite(model, variable, region_number)
    return np.nanmean(composite[-10:])


def all_models_in_region(models, variable, region_number, obs=[], full_composites=False, force_positive=False):
    if full_composites:
        model_composites = {}
        for model in models + [f'OBS_{obs_type}' for obs_type in obs]:
            composite = load_composite(model, variable, region_number)
            model_composites[model] = composite
        return model_composites
    else:
        model_amplitudes = {}
        model_lags = {}
        for model in models + [f'OBS_{obs_type}' for obs_type in obs]:
            amplitude, lag = amplitude_and_lag(model, variable, region_number, force_positive=force_positive)
            model_amplitudes[model] = (amplitude)
            model_lags[model] = lag
        return model_amplitudes, model_lags


def compute_scaling_factors(models, scale_variable, region_number, obs=[]):
    obs_composite = np.genfromtxt(f'composites/multimodel/regional/{analysis_version_name}/OBS_{scale_variable}_composite_{analysis_version_name}_region{region_number}.csv', delimiter=',')
    max_obs_anom = np.nanmax(obs_composite)
    scaling_factor = {}
    cf_standard_name = scale_variable.split('-')[0]
    for model in models:
        model_composite = np.genfromtxt(f'composites/multimodel/regional/{analysis_version_name}/{model}_{cf_standard_name}_composite_{analysis_version_name}_region{region_number}.csv', delimiter=',')
        max_model_anom = np.nanmax(model_composite)
        scaling_factor[model] = max_obs_anom/max_model_anom
    for plot_obs in obs:
        scaling_factor[f'OBS_{plot_obs}'] = 1.
    return scaling_factor


def plot_region(models, variable, region_number, full_composites=False, obs=[], scale_by_max=None, force_positive=False, ax=None):
    create_ax = (ax is None)
    if create_ax:
        plt.figure(figsize=(6, 4.5))
        ax = plt.gca()
    if scale_by_max is not None:
        model_factors = compute_scaling_factors(models, scale_by_max, region_number, obs=obs)
    model_results = all_models_in_region(models, variable, region_number, obs=obs, full_composites=full_composites, force_positive=force_positive)
    for model in model_results.keys():
        if full_composites:
            model_composites = model_results
            composite = model_composites[model]
            linestyle = '-'
            if scale_by_max is not None:
                scale_factor = model_factors[model]
                composite *= scale_factor
                if np.logical_or(scale_factor<0.5, scale_factor>2.):
                    linestyle = '--'
            ax.plot(np.arange(-60, 61), composite, model_colours(model),
                    linestyle=linestyle, label=model)
        else:
            model_amplitudes, model_lags = model_results
            amplitude = model_amplitudes[model]
            facecolors = model_colours[model]
            if scale_by_max is not None:
                scale_factor = model_factors[model]
                amplitude *= scale_factor
                if np.logical_or(scale_factor<0.5, scale_factor>2.):
                    facecolors = 'None'
            ax.scatter(model_lags[model], amplitude, edgecolors=model_colours(model),
                        facecolors=facecolors, label=model)
    if create_ax:
        plt.title(region_abbreviations[region_number], fontsize=14)
        plt.xlabel(f'{variable} lag (days)', fontsize=16)
        plt.ylabel(f'{variable} amplitude ({variable_units[variable]})', fontsize=16)
        plt.gca().tick_params(labelsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.show()
    

def subplot_all_regions(models, variable, obs=[], full_composites=False, scale_by_max=None, force_positive=False):
    valid_regions = regions_with_enough_data(models, variable)
    sorted_regions = sort_regions_by_max_amplitude(models, variable, valid_regions, force_positive=force_positive)
    fig, axs = plt.subplots(nrows=7, ncols=6, figsize=(10, 10), sharex=True, sharey='row')
    axlist = axs.ravel()
    for i, ax in enumerate(axlist):
        if i < sorted_regions.size:
            plot_region(models, variable, sorted_regions[i], obs=obs, force_positive=force_positive,
                        full_composites=full_composites, scale_by_max=scale_by_max, ax=ax)
            ax.tick_params(labelsize=12)
            ax.text(0.05, 0.85, region_abbreviations[sorted_regions[i]], 
                    ha='left', va='center', transform=ax.transAxes, fontsize=12)            
            ax.axhline(0, color='gray', linestyle='-', alpha=0.2)
            ax.axvline(0, color='gray', linestyle='-', alpha=0.2)
            if variable == 'gpp':
                ax.set_xticks([0, 30, 60])
            if full_composites:
                ax.set_xticks([-60, 0, 60])
            fmt = mpl.ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.offsetText.set_fontsize(9)
            ax.yaxis.offsetText.set_position((-0.1, 1))
        else:
            ax.remove()
    if full_composites:
        fig.text(0.5, 0.04, 'days since intraseasonal precipitation maximum', ha='center')
        fig.text(0.04, 0.5, f'{variable} anomaly ({variable_units[variable]})', va='center', rotation='vertical')
        # fig.text(0.04, 0.5, f'standardised {variable} anomaly', va='center', rotation='vertical')
    else:
        fig.text(0.5, 0.04, f'{variable} lag (days)', ha='center')
        fig.text(0.04, 0.5, f'{variable} amplitude ({variable_units[variable]})', va='center', rotation='vertical')
        # fig.text(0.04, 0.5, f'{variable} amplitude', va='center', rotation='vertical')

    save_filename = f'figures/multimodel/regional/lag_amplitude_{variable}_{analysis_version_name}'
    if full_composites:
        save_filename += '_full_composites'
    if scale_by_max is not None:
        save_filename += f'_scale_by_max_{scale_by_max}'
    if force_positive:
        save_filename += '_force_positive'
    if len(obs)>0:
        obs_for_variable = [o for o in obs if o.startswith(f'{variable}-')]
        save_filename += f"_OBS_{'_'.join(obs_for_variable)}"
    plt.savefig(f'{save_filename}.pdf', dpi=800, bbox_inches='tight')
    plt.savefig(f'{save_filename}.png', dpi=800, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    models = ['NorESM2-LM', 'CNRM-ESM2-1', 'BCC-CSM2-MR', 'ACCESS-ESM1-5', 'UKESM1-0-LL']
    # subplot_all_regions(models, 'pr', obs=['pr-IMERG'], scale_by_max='pr-IMERG', full_composites=True)
    subplot_all_regions(models, 'gpp', obs=['gpp-SIF-GOME2-JJ'], scale_by_max='mrsos-ESACCI', full_composites=True)
    # subplot_all_regions(models, 'gpp', obs=['gpp-MODIS-TERRA'], scale_by_max='pr-IMERG', full_composites=True)
