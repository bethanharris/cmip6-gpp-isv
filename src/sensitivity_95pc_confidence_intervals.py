import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt


analysis_version_name = 'rolling_7d_mean_stdev_maskfrozen_60S60N_regrid_1_by_1_deg_standardised_CCI-SM-mask'


def plot_slope_cis(driver_variable, composite_property, ax=None, save=False, show=True, generic_y_label=False):
    emtrends = pd.read_csv(f'/users/global/bethar/python/cmip6-gpp-isv/data/amplitudes_lags/{analysis_version_name}/emtrends_{driver_variable}_{composite_property}.csv')
    cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'NorESM2-LM', 'UKESM1-0-LL']
    obs_products = ['FLUXCOM-CRUJRAv1', 'FLUXCOM-ERA5', 'MODIS-TERRA', 'VPM', 'SIF-GOME2-JJ', 'SIF-GOME2-PK', 'VODCA2GPP',]
    sorted_models = cmip6_models+obs_products
    emtrends.model = emtrends.model.astype("category")
    emtrends.model = emtrends.model.cat.set_categories(sorted_models)
    emtrends.sort_values(['model'], inplace=True)
    trend_variable_key = 'amp' if composite_property=='amplitude' or composite_property=='lag' else 'final_amp'
    trend_key = f'{driver_variable}_{trend_variable_key}.trend'
    errorbars = np.vstack([emtrends[trend_key] - emtrends['lower.CL'], emtrends['upper.CL'] - emtrends[trend_key]])
    if ax is None:
        plt.figure()
        ax = plt.gca()
    x = np.arange(len(emtrends['model']))
    ax.errorbar(emtrends['model'], emtrends[trend_key], yerr=errorbars, fmt='-o', linestyle="None", color='k')
    ax.set_xticks(ticks=x, labels=emtrends['model'], rotation=45, ha='right', va='top', position=(1.15, 0.01))
    driver_label = {'mrsos': 'SSM', 'vpd': 'VPD'}[driver_variable]
    property_label = {'amplitude': 'peak amplitude', 'final_amplitude': 'post-event amplitude', 'lag': 'lag'}[composite_property]
    if generic_y_label:
        ax.set_ylabel('sensitivity', fontsize=12)
    else:
        ax.set_ylabel(f'{property_label} sensitivity\nof GPP to {driver_label}', fontsize=12)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=10)
    if save:
        plt.savefig(f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}/sensitivity_95CI_{driver_variable}_{composite_property}.png', dpi=400, bbox_inches='tight')
    if show:
        plt.show()


def sensitivity_subplots():
    fig, axs = plt.subplots(3, 2, figsize=(9, 10), gridspec_kw={'wspace': 0.4, 'hspace': 1.1})
    ((mrsos_amp_ax, vpd_amp_ax), (mrsos_final_amp_ax, vpd_final_amp_ax), (mrsos_lag_ax, vpd_lag_ax)) = axs
    plot_slope_cis('mrsos', 'amplitude', ax=mrsos_amp_ax, show=False, generic_y_label=True)
    plot_slope_cis('vpd', 'amplitude', ax=vpd_amp_ax, show=False, generic_y_label=True)
    plot_slope_cis('mrsos', 'lag', ax=mrsos_lag_ax, show=False, generic_y_label=True)
    plot_slope_cis('vpd', 'lag', ax=vpd_lag_ax, show=False, generic_y_label=True)
    plot_slope_cis('mrsos', 'final_amplitude', ax=mrsos_final_amp_ax, show=False, generic_y_label=True)
    plot_slope_cis('vpd', 'final_amplitude', ax=vpd_final_amp_ax, show=False, generic_y_label=True)
    alphabet = string.ascii_lowercase[0:6]
    titles = ['GPP peak amplitude\nvs SSM peak amplitude',
              'GPP peak amplitude\nvs VPD peak amplitude',
              'GPP post-event amplitude\nvs SSM post-event amplitude',
              'GPP post-event amplitude\nvs VPD post-event amplitude',
              'GPP lag\nvs SSM peak amplitude',        
              'GPP lag\nvs VPD peak amplitude']
    for i, ax in enumerate(axs.ravel()):
         ax.set_title(f'$\\bf{{({alphabet[i]})}}$ {titles[i]}', fontsize=12, ma='center')
    plt.savefig(f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}/sensitivity_95CI_subplots.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'../figures/multimodel/regional/driver_sensitivity/{analysis_version_name}/sensitivity_95CI_subplots.pdf', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    sensitivity_subplots()

