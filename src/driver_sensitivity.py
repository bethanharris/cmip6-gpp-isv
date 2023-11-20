import numpy as np
import os
import string
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from models_vs_obs import plot_linear_regression, gpp_obs_only_colours


cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'NorESM2-LM', 'UKESM1-0-LL']
gpp_obs_products = ['FLUXCOM-ERA5', 'FLUXCOM-CRUJRAv1',  'MODIS-TERRA', 'VPM', 'SIF-GOME2-JJ', 'SIF-GOME2-PK', 'VODCA2GPP']

analysis_version_name = 'rolling_7d_mean_stdev_maskfrozen_regrid_1_by_1_deg_standardised_CCI-SM-mask'
amplitude_lag_save_dir = f'../data/amplitudes_lags/{analysis_version_name}'


def regions_with_enough_data(models, variable, obs_product):
    all_regions = np.arange(46)
    valid_regions = []
    for region_number in all_regions:
        n_all_models = []
        composite_save_dir = f'./composites/multimodel/regional/{analysis_version_name}'
        for model in models:
            variable_label = f'{variable}-{obs_product}' if model == 'OBS' else variable
            n_filename = f'{model}_{variable_label}_n_{analysis_version_name}_region{int(region_number)}.csv'
            n = np.loadtxt(f'{composite_save_dir}/{n_filename}', delimiter=',')
            n_all_models.append(min(n))
        if min(n_all_models) >= 20:
            valid_regions.append(region_number)
    return np.array(valid_regions)


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


def scatter_mrsos_vs_vpd(mrsos_obs_product, vpd_obs_product, season='all', 
                         amp_ax=None, lag_ax=None, final_amp_ax=None, legend_r_only=False,
                         legend_outside_plot=False, save=False):
    if amp_ax is None:
        amp_fig = plt.figure(figsize=(7, 3.5))
        amp_ax = plt.gca()
    if lag_ax is None:
        lag_fig = plt.figure(figsize=(7, 3.5))
        lag_ax = plt.gca()
    if final_amp_ax is None:
        final_amp_fig = plt.figure(figsize=(7, 3.5))
        final_amp_ax = plt.gca()

    season_label = '' if season == 'all' else f'_{season}'
    scale_by_label = ''
    amplitude_lag_save_dir = f'../data/amplitudes_lags/{analysis_version_name}'
    mrsos_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_mrsos_obs_vs_models{season_label}{scale_by_label}.csv')
    mrsos_lag_data = pd.read_csv(f'{amplitude_lag_save_dir}/lag_mrsos_obs_vs_models{season_label}{scale_by_label}.csv')
    mrsos_final_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_mrsos_obs_vs_models{season_label}{scale_by_label}.csv')
    vpd_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_vpd_obs_vs_models{season_label}{scale_by_label}.csv')
    vpd_lag_data = pd.read_csv(f'{amplitude_lag_save_dir}/lag_vpd_obs_vs_models{season_label}{scale_by_label}.csv')
    vpd_final_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_vpd_obs_vs_models{season_label}{scale_by_label}.csv')

    mrsos_amplitude_data.dropna(inplace=True)
    mrsos_lag_data.dropna(inplace=True)
    mrsos_final_amplitude_data.dropna(inplace=True)

    vpd_amplitude_data.dropna(inplace=True)
    vpd_lag_data.dropna(inplace=True)
    vpd_final_amplitude_data.dropna(inplace=True)

    mrsos_regions = mrsos_amplitude_data['region']
    vpd_amplitude_data = vpd_amplitude_data[np.isin(vpd_amplitude_data['region'], mrsos_regions)]
    vpd_lag_data = vpd_lag_data[np.isin(vpd_lag_data['region'], mrsos_regions)]
    vpd_final_amplitude_data = vpd_final_amplitude_data[np.isin(vpd_final_amplitude_data['region'], mrsos_regions)]

    if mrsos_obs_product is not None and vpd_obs_product is not None:
        mrsos_amps_obs = mrsos_amplitude_data[mrsos_obs_product]
        mrsos_lags_obs = mrsos_lag_data[mrsos_obs_product]
        mrsos_final_amps_obs = mrsos_final_amplitude_data[mrsos_obs_product]

        vpd_amps_obs = vpd_amplitude_data[vpd_obs_product]
        vpd_lags_obs = vpd_lag_data[vpd_obs_product]
        vpd_final_amps_obs = vpd_final_amplitude_data[vpd_obs_product]

        plot_linear_regression(amp_ax, vpd_amps_obs, mrsos_amps_obs, 'OBS-OBS', legend_r_only=legend_r_only)
        plot_linear_regression(lag_ax, vpd_amps_obs, mrsos_lags_obs, 'OBS-OBS', legend_r_only=legend_r_only)
        plot_linear_regression(final_amp_ax, vpd_final_amps_obs, mrsos_final_amps_obs, 'OBS-OBS', legend_r_only=legend_r_only)

    models = cmip6_models.copy()
    models.remove('BCC-CSM2-MR')

    for model in models:
        model_mrsos_amps = mrsos_amplitude_data[model]
        model_mrsos_lags = mrsos_lag_data[model]
        model_mrsos_final_amps = mrsos_final_amplitude_data[model]

        model_vpd_amps = vpd_amplitude_data[model]
        model_vpd_lags = vpd_lag_data[model]
        model_vpd_final_amps = vpd_final_amplitude_data[model]

        plot_linear_regression(amp_ax, model_vpd_amps, model_mrsos_amps, model, legend_r_only=legend_r_only)
        plot_linear_regression(lag_ax, model_vpd_amps, model_mrsos_lags, model, legend_r_only=legend_r_only)
        plot_linear_regression(final_amp_ax, model_vpd_final_amps, model_mrsos_final_amps, model, legend_r_only=legend_r_only)

    amp_ax.set_ylabel('mrsos peak amplitude', fontsize=14)
    amp_ax.set_xlabel('vpd peak amplitude', fontsize=14)
    lag_ax.set_ylabel('mrsos lag (days)', fontsize=14)
    lag_ax.set_xlabel('vpd peak amplitude', fontsize=14)
    final_amp_ax.set_ylabel('mrsos post-event amplitude', fontsize=14)
    final_amp_ax.set_xlabel('vpd post-event amplitude', fontsize=14)
    for ax in [amp_ax, lag_ax, final_amp_ax]:
        if legend_outside_plot:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        else:
            ax.legend(loc='best', fontsize=10, handletextpad=0.1)
    if save:
        scale_label = ''
        plt.figure(amp_fig)
        plt.tight_layout()
        fig_save_dir = f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}'
        os.system(f'mkdir -p {fig_save_dir}')
        plt.savefig(f'{fig_save_dir}/mrsos_vs_vpd_amplitude{scale_label}.png', dpi=400, bbox_inches='tight')
        plt.figure(lag_fig)
        plt.tight_layout()
        plt.savefig(f'{fig_save_dir}/mrsos_lag_vs_vpd_amplitude{scale_label}.png', dpi=400, bbox_inches='tight')
        plt.figure(final_amp_fig)
        plt.tight_layout()
        plt.savefig(f'{fig_save_dir}/mrsos_vs_vpd_final_amplitudes{scale_label}.png', dpi=400, bbox_inches='tight')
        plt.close('all')



def scatter_driver_vs_gpp(driver_variable_name, gpp_obs_product, driver_obs_product, season='all', 
                          amp_ax=None, lag_ax=None, final_amp_ax=None, legend_r_only=False,
                          legend_outside_plot=False, save=False):
    if amp_ax is None:
        amp_fig = plt.figure(figsize=(7, 3.5))
        amp_ax = plt.gca()
    if lag_ax is None:
        lag_fig = plt.figure(figsize=(7, 3.5))
        lag_ax = plt.gca()
    if final_amp_ax is None:
        final_amp_fig = plt.figure(figsize=(7, 3.5))
        final_amp_ax = plt.gca()

    season_label = '' if season == 'all' else f'_{season}'
    scale_by_label = ''
    amplitude_lag_save_dir = f'../data/amplitudes_lags/{analysis_version_name}'
    gpp_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_gpp_obs_vs_models{season_label}{scale_by_label}.csv')
    gpp_lag_data = pd.read_csv(f'{amplitude_lag_save_dir}/lag_gpp_obs_vs_models{season_label}{scale_by_label}.csv')
    gpp_final_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_gpp_obs_vs_models{season_label}{scale_by_label}.csv')
    driver_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_{driver_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')
    driver_lag_data = pd.read_csv(f'{amplitude_lag_save_dir}/lag_{driver_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')
    driver_final_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_{driver_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')

    gpp_amplitude_data.dropna(inplace=True)
    gpp_lag_data.dropna(inplace=True)
    gpp_final_amplitude_data.dropna(inplace=True)

    driver_amplitude_data.dropna(inplace=True)
    driver_lag_data.dropna(inplace=True)
    driver_final_amplitude_data.dropna(inplace=True)

    gpp_regions = gpp_amplitude_data['region']
    driver_amplitude_data = driver_amplitude_data[np.isin(driver_amplitude_data['region'], gpp_regions)]
    driver_lag_data = driver_lag_data[np.isin(driver_lag_data['region'], gpp_regions)]
    driver_final_amplitude_data = driver_final_amplitude_data[np.isin(driver_final_amplitude_data['region'], gpp_regions)]

    if gpp_obs_product is not None and driver_obs_product is not None:
        gpp_amps_obs = gpp_amplitude_data[gpp_obs_product]
        gpp_lags_obs = gpp_lag_data[gpp_obs_product]
        gpp_final_amps_obs = gpp_final_amplitude_data[gpp_obs_product]

        driver_amps_obs = driver_amplitude_data[driver_obs_product]
        driver_lags_obs = driver_lag_data[driver_obs_product]
        driver_final_amps_obs = driver_final_amplitude_data[driver_obs_product]

        plot_linear_regression(amp_ax, driver_amps_obs, gpp_amps_obs, 'OBS', legend_r_only=legend_r_only)
        plot_linear_regression(lag_ax, driver_amps_obs, gpp_lags_obs, 'OBS', legend_r_only=legend_r_only)
        plot_linear_regression(final_amp_ax, driver_final_amps_obs, gpp_final_amps_obs, 'OBS', legend_r_only=legend_r_only)

    models = cmip6_models.copy()
    if driver_variable_name == 'vpd':
        models.remove('BCC-CSM2-MR')

    for model in models:
        model_gpp_amps = gpp_amplitude_data[model]
        model_gpp_lags = gpp_lag_data[model]
        model_gpp_final_amps = gpp_final_amplitude_data[model]

        model_driver_amps = driver_amplitude_data[model]
        model_driver_lags = driver_lag_data[model]
        model_driver_final_amps = driver_final_amplitude_data[model]

        plot_linear_regression(amp_ax, model_driver_amps, model_gpp_amps, model, legend_r_only=legend_r_only)
        plot_linear_regression(lag_ax, model_driver_amps, model_gpp_lags, model, legend_r_only=legend_r_only)
        plot_linear_regression(final_amp_ax, model_driver_final_amps, model_gpp_final_amps, model, legend_r_only=legend_r_only)

    amp_ax.set_ylabel('gpp peak amplitude', fontsize=14)
    amp_ax.set_xlabel(f'{driver_variable_name} peak amplitude', fontsize=14)
    lag_ax.set_ylabel('gpp lag (days)', fontsize=14)
    lag_ax.set_xlabel(f'{driver_variable_name} peak amplitude', fontsize=14)
    final_amp_ax.set_ylabel('gpp post-event amplitude', fontsize=14)
    final_amp_ax.set_xlabel(f'{driver_variable_name} post-event amplitude', fontsize=14)
    for ax in [amp_ax, lag_ax, final_amp_ax]:
        if legend_outside_plot:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        else:
            ax.legend(loc='best', fontsize=10, handletextpad=0.1)
    if save:
        scale_label = ''
        plt.figure(amp_fig)
        plt.tight_layout()
        fig_save_dir = f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}'
        os.system(f'mkdir -p {fig_save_dir}')
        plt.savefig(f'{fig_save_dir}/gpp_vs_{driver_variable_name}_amplitude{scale_label}.png', dpi=400, bbox_inches='tight')
        plt.figure(lag_fig)
        plt.tight_layout()
        plt.savefig(f'{fig_save_dir}/gpp_lag_vs_{driver_variable_name}_amplitude{scale_label}.png', dpi=400, bbox_inches='tight')
        plt.figure(final_amp_fig)
        plt.tight_layout()
        plt.savefig(f'{fig_save_dir}/gpp_vs_{driver_variable_name}_final_amplitudes{scale_label}.png', dpi=400, bbox_inches='tight')
        plt.close('all')


def driver_subplots():
    fig, axs = plt.subplots(3, 2, figsize=(9, 10))
    ((mrsos_amp_ax, vpd_amp_ax), (mrsos_final_amp_ax, vpd_final_amp_ax), (mrsos_lag_ax, vpd_lag_ax)) = axs
    scatter_driver_vs_gpp('mrsos', 'FLUXCOM-CRUJRAv1', 'ESACCI', amp_ax=mrsos_amp_ax, lag_ax=mrsos_lag_ax, 
                          final_amp_ax=mrsos_final_amp_ax, legend_r_only=True, save=False)
    scatter_driver_vs_gpp('vpd', 'FLUXCOM-CRUJRAv1', 'ERA5', amp_ax=vpd_amp_ax, lag_ax=vpd_lag_ax, 
                          final_amp_ax=vpd_final_amp_ax, legend_r_only=True, save=False)
    alphabet = string.ascii_lowercase[0:6]
    for i, ax in enumerate(axs.ravel()):
         ax.set_title(f'$\\bf{{({alphabet[i]})}}$', fontsize=14)
    plt.tight_layout()

    model_lines = [Line2D([0], [0], marker='o', color=model_colours(model), label=model,
                   markerfacecolor=model_colours(model), lw=0, markersize=7,
                   markeredgecolor='None') for model in ['OBS'] + cmip6_models]

    fig.legend(model_lines, ['OBS'] + cmip6_models, ncol=len(cmip6_models)+1, 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=13,
               handletextpad=-0.2, columnspacing=0.5)

    plt.savefig(f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}/driver_sensitivity_subplots_CRUJRAv1.png', 
                dpi=400, bbox_inches='tight')
    plt.savefig(f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}/driver_sensitivity_subplots_CRUJRAv1.pdf', 
                dpi=400, bbox_inches='tight')
    plt.show()


def scatter_driver_vs_gpp_obs_only(driver_variable_name, driver_obs_product, season='all', 
                                   amp_ax=None, lag_ax=None, final_amp_ax=None, legend_r_only=False,
                                   legend_outside_plot=False, save=False):
    if amp_ax is None:
        amp_fig = plt.figure(figsize=(7, 3.5))
        amp_ax = plt.gca()
    if lag_ax is None:
        lag_fig = plt.figure(figsize=(7, 3.5))
        lag_ax = plt.gca()
    if final_amp_ax is None:
        final_amp_fig = plt.figure(figsize=(7, 3.5))
        final_amp_ax = plt.gca()

    season_label = '' if season == 'all' else f'_{season}'
    scale_by_label = ''
    amplitude_lag_save_dir = f'../data/amplitudes_lags/{analysis_version_name}'
    gpp_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_gpp_obs_vs_models{season_label}{scale_by_label}.csv')
    gpp_lag_data = pd.read_csv(f'{amplitude_lag_save_dir}/lag_gpp_obs_vs_models{season_label}{scale_by_label}.csv')
    gpp_final_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_gpp_obs_vs_models{season_label}{scale_by_label}.csv')
    driver_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_{driver_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')
    driver_lag_data = pd.read_csv(f'{amplitude_lag_save_dir}/lag_{driver_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')
    driver_final_amplitude_data = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_{driver_variable_name}_obs_vs_models{season_label}{scale_by_label}.csv')

    gpp_amplitude_data.dropna(inplace=True)
    gpp_lag_data.dropna(inplace=True)
    gpp_final_amplitude_data.dropna(inplace=True)

    driver_amplitude_data.dropna(inplace=True)
    driver_lag_data.dropna(inplace=True)
    driver_final_amplitude_data.dropna(inplace=True)

    gpp_regions = gpp_amplitude_data['region']
    driver_amplitude_data = driver_amplitude_data[np.isin(driver_amplitude_data['region'], gpp_regions)]
    driver_lag_data = driver_lag_data[np.isin(driver_lag_data['region'], gpp_regions)]
    driver_final_amplitude_data = driver_final_amplitude_data[np.isin(driver_final_amplitude_data['region'], gpp_regions)]

    for gpp_obs in gpp_obs_products:
        plot_linear_regression(amp_ax, driver_amplitude_data[driver_obs_product], gpp_amplitude_data[gpp_obs], f'OBS-{gpp_obs}', legend_r_only=legend_r_only, use_obs_colours=True)
        plot_linear_regression(lag_ax, driver_amplitude_data[driver_obs_product], gpp_lag_data[gpp_obs], f'OBS-{gpp_obs}', legend_r_only=legend_r_only, use_obs_colours=True)
        plot_linear_regression(final_amp_ax, driver_final_amplitude_data[driver_obs_product], gpp_final_amplitude_data[gpp_obs], f'OBS-{gpp_obs}', legend_r_only=legend_r_only, use_obs_colours=True)

    amp_ax.set_ylabel('gpp peak amplitude', fontsize=14)
    amp_ax.set_xlabel(f'{driver_variable_name} peak amplitude', fontsize=14)
    lag_ax.set_ylabel('gpp lag (days)', fontsize=14)
    lag_ax.set_xlabel(f'{driver_variable_name} peak amplitude', fontsize=14)
    final_amp_ax.set_ylabel('gpp post-event amplitude', fontsize=14)
    final_amp_ax.set_xlabel(f'{driver_variable_name} post-event amplitude', fontsize=14)
    for ax in [amp_ax, lag_ax, final_amp_ax]:
        if legend_outside_plot:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        else:
            ax.legend(loc='best', fontsize=10, handletextpad=0.1)
        ax.tick_params(labelsize=12)
    if save:
        scale_label = ''
        plt.figure(amp_fig)
        plt.tight_layout()
        fig_save_dir = f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}'
        os.system(f'mkdir -p {fig_save_dir}')
        plt.savefig(f'{fig_save_dir}/gpp_vs_{driver_variable_name}_amplitude{scale_label}_all_gpp_obs.png', dpi=400, bbox_inches='tight')
        plt.figure(lag_fig)
        plt.tight_layout()
        plt.savefig(f'{fig_save_dir}/gpp_lag_vs_{driver_variable_name}_amplitude{scale_label}_all_gpp_obs.png', dpi=400, bbox_inches='tight')
        plt.figure(final_amp_fig)
        plt.tight_layout()
        plt.savefig(f'{fig_save_dir}/gpp_vs_{driver_variable_name}_final_amplitudes{scale_label}_all_gpp_obs.png', dpi=400, bbox_inches='tight')
        plt.close('all')


def driver_subplots_gpp_obs_only():
    fig, axs = plt.subplots(3, 2, figsize=(9, 10))
    ((mrsos_amp_ax, vpd_amp_ax), (mrsos_final_amp_ax, vpd_final_amp_ax), (mrsos_lag_ax, vpd_lag_ax)) = axs
    scatter_driver_vs_gpp_obs_only('mrsos', 'ESACCI', amp_ax=mrsos_amp_ax, lag_ax=mrsos_lag_ax, 
                                   final_amp_ax=mrsos_final_amp_ax, legend_r_only=True, save=False)
    scatter_driver_vs_gpp_obs_only('vpd', 'ERA5', amp_ax=vpd_amp_ax, lag_ax=vpd_lag_ax, 
                                   final_amp_ax=vpd_final_amp_ax, legend_r_only=True, save=False)
    alphabet = string.ascii_lowercase[0:6]
    for i, ax in enumerate(axs.ravel()):
         ax.set_title(f'$\\bf{{({alphabet[i]})}}$', fontsize=14)
    plt.tight_layout()

    obs_lines = [Line2D([0], [0], marker='o', color=gpp_obs_only_colours(f'OBS-{obs}'), label=obs,
                 markerfacecolor=gpp_obs_only_colours(f'OBS-{obs}'), lw=0, markersize=7,
                 markeredgecolor='None') for obs in gpp_obs_products]

    fig.legend(obs_lines, gpp_obs_products, ncol=len(gpp_obs_products), 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=13,
               handletextpad=-0.2, columnspacing=0.5)

    plt.savefig(f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}/driver_sensitivity_subplots_gpp_obs_only.png', 
                dpi=400, bbox_inches='tight')
    plt.savefig(f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}/driver_sensitivity_subplots_gpp_obs_only.pdf', 
                dpi=400, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    driver_subplots()
    driver_subplots_gpp_obs_only()
