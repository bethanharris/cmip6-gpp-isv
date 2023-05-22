from taylor_diagrams import TaylorDiagram
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects


analysis_version = 'rolling_7d_mean_stdev_maskfrozen_regrid_1_by_1_deg_standardised_CCI-SM-mask'
amplitude_lag_save_dir = f'../data/amplitudes_lags/{analysis_version}'


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


def obs_shapes(obs_product):
    shapes = {'FLUXCOM-ERA5': '*',
              'FLUXCOM-CRUJRAv1': 'P',
              'VODCA2GPP': 'd',
              'MODIS-TERRA': 's',
              'SIF-GOME2-JJ': '^',
              'SIF-GOME2-PK': 'v',
              'VPM': 'p'
              }
    return shapes[obs_product]


models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'NorESM2-LM', 'UKESM1-0-LL']
obs = ['FLUXCOM-ERA5', 'FLUXCOM-CRUJRAv1', 'MODIS-TERRA', 'VPM', 'SIF-GOME2-JJ', 'SIF-GOME2-PK', 'VODCA2GPP']


def amplitude_taylor():
    season_label = ''
    scale_label = '_scale_by_mrsos-ESACCI'
    amp = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_gpp_obs_vs_models{season_label}{scale_label}.csv')
    lag = pd.read_csv(f'{amplitude_lag_save_dir}/lag_gpp_obs_vs_models{season_label}{scale_label}.csv')
    final_amp = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_gpp_obs_vs_models{season_label}{scale_label}.csv')
    amp.dropna(inplace=True)
    lag.dropna(inplace=True)
    final_amp.dropna(inplace=True)

    fig = plt.figure()

    stdref = 1.#np.std(amp['FLUXCOM-ERA5'])

    dia = TaylorDiagram(stdref, fig=fig, label='FLUXCOM-ERA5', normalised_stdev=True)#, extend=True)

    for model in models:
        for obs_product in obs:
            model_data = amp[model]
            obs_data = amp[obs_product]
            r = np.corrcoef(model_data, obs_data)[0, 1]
            # model_stdev = np.std(model_data)
            stdev_ratio = np.std(model_data)/np.std(obs_data)
            dia.add_sample(stdev_ratio, r,
                           marker=obs_shapes(obs_product), ms=10, ls='',
                           mfc=model_colours(model), mec='k')

    dia.add_refstd(1., '', '')    
    # for obs_product in obs:
    #     obs_data = amp[obs_product]
    #     stddev = np.std(obs_data)
    #     marker = obs_shapes(obs_product)
    #     dia.add_refstd(stddev, marker, obs_product)


    dia.add_grid()
    contours = dia.add_contours(colors='0.5')
    clbls = plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f', use_clabeltext=True)
    plt.setp(clbls, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
    box = dia.ax.get_position()
    dia.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # obs_legend = dia.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 0.5))
    model_lines = [Line2D([0], [0], marker='o', color=model_colours(model), label=model,
                          markerfacecolor=model_colours(model), lw=0, markersize=7,
                          markeredgecolor='k') for model in models]

    obs_markers = [Line2D([0], [0], marker=obs_shapes(obs_product), label=obs_product,
                   mfc='k', mec='0.5', ls='', ms=10) for obs_product in obs]

    obs_legend = dia.ax.legend(obs_markers, obs, loc='upper left', bbox_to_anchor=(1.05, 0.5))

    dia.ax.legend(model_lines, models, loc='lower left', bbox_to_anchor=(1.05, 0.5))
    dia.ax.add_artist(obs_legend)
    fig_save_dir = f'../figures/multimodel/regional/taylor_diagrams/{analysis_version}'
    os.system(f'mkdir -p {fig_save_dir}')
    plt.savefig(f'{fig_save_dir}/taylor_diagram_gpp_amplitude_normalised_std.png',
                dpi=400, bbox_extra_artists=plt.gcf().get_children(), bbox_inches='tight', pad_inches=0.3)
    plt.show()


def subplots_taylor():
    season_label = ''
    scale_label = '_scale_by_mrsos-ESACCI'
    amp = pd.read_csv(f'{amplitude_lag_save_dir}/amplitude_gpp_obs_vs_models{season_label}{scale_label}.csv')
    lag = pd.read_csv(f'{amplitude_lag_save_dir}/lag_gpp_obs_vs_models{season_label}{scale_label}.csv')
    final_amp = pd.read_csv(f'{amplitude_lag_save_dir}/final_amplitude_gpp_obs_vs_models{season_label}{scale_label}.csv')
    amp.dropna(inplace=True)
    lag.dropna(inplace=True)
    final_amp.dropna(inplace=True)

    max_amp_std_ratio = 0
    max_lag_std_ratio = 0
    max_final_amp_std_ratio = 0
    for model in models:
        for obs_product in obs:
            model_data = amp[model]
            obs_data = amp[obs_product]
            amp_std_ratio = np.std(amp[model])/np.std(amp[obs_product])
            lag_std_ratio = np.std(lag[model])/np.std(lag[obs_product])
            final_amp_std_ratio = np.std(final_amp[model])/np.std(final_amp[obs_product])
            max_amp_std_ratio = max(max_amp_std_ratio, amp_std_ratio)
            max_lag_std_ratio = max(max_lag_std_ratio, lag_std_ratio)
            max_final_amp_std_ratio = max(max_final_amp_std_ratio, final_amp_std_ratio)

    fig = plt.figure(figsize=(12, 9))
    amp_dia = TaylorDiagram(1., fig=fig, rect=221, rotate_stdev_labels=True, srange=(0, 1.05*max_amp_std_ratio), normalised_stdev=True)
    lag_dia = TaylorDiagram(1., fig=fig, rect=223, extend=True, srange=(0, 1.05*max_lag_std_ratio), normalised_stdev=True)
    final_amp_dia = TaylorDiagram(1., fig=fig, rect=222, srange=(0, 1.05*max_final_amp_std_ratio), rotate_stdev_labels=True, normalised_stdev=True)

    for model in models:
        for obs_product in obs:
            model_data = amp[model]
            obs_data = amp[obs_product]
            r = np.corrcoef(model_data, obs_data)[0, 1]
            stdev_ratio = np.std(model_data)/np.std(obs_data)
            amp_dia.add_sample(stdev_ratio, r,
                               marker=obs_shapes(obs_product), ms=10, ls='',
                               mfc=model_colours(model), mec='k')
            model_data = lag[model]
            obs_data = lag[obs_product]
            r = np.corrcoef(model_data, obs_data)[0, 1]
            stdev_ratio = np.std(model_data)/np.std(obs_data)
            lag_dia.add_sample(stdev_ratio, r,
                               marker=obs_shapes(obs_product), ms=10, ls='',
                               mfc=model_colours(model), mec='k')
            model_data = final_amp[model]
            obs_data = final_amp[obs_product]
            r = np.corrcoef(model_data, obs_data)[0, 1]
            stdev_ratio = np.std(model_data)/np.std(obs_data)
            final_amp_dia.add_sample(stdev_ratio, r,
                                     marker=obs_shapes(obs_product), ms=10, ls='',
                                     mfc=model_colours(model), mec='k')

    for dia in amp_dia, lag_dia, final_amp_dia:
        dia.add_grid()
        contours = dia.add_contours(colors='0.5')
        clbls = plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f', use_clabeltext=True)
        plt.setp(clbls, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")], clip_on=False)
        box = dia.ax.get_position()
        dia.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        obs_markers = [Line2D([0], [0], marker=obs_shapes(obs_product), label=obs_product,
                   mfc='k', mec='0.5', ls='', ms=10) for obs_product in obs]
        model_lines = [Line2D([0], [0], marker='o', color=model_colours(model), label=model,
                              markerfacecolor=model_colours(model), lw=0, markersize=7,
                              markeredgecolor='k') for model in models]

    legend_ax = fig.add_subplot(224)
    model_legend = legend_ax.legend(model_lines, models, loc='lower left', bbox_to_anchor=(0.25, 0.6), fontsize=14)
    legend_ax.legend(obs_markers, obs, loc='upper left', bbox_to_anchor=(0.25, 0.6), fontsize=14)
    legend_ax.add_artist(model_legend)
    legend_ax.axis('off')

    amp_dia._ax.set_title("$\\bf{(a)}$ " + 'gpp peak amplitude', pad=15, fontsize=16)
    final_amp_dia._ax.set_title("$\\bf{(b)}$ " + 'gpp post-event amplitude', pad=15, fontsize=16)
    lag_dia._ax.set_title("$\\bf{c)}$ " + 'gpp lag (days)', pad=35, fontsize=16)
    fig_save_dir = f'../figures/multimodel/regional/taylor_diagrams/{analysis_version}'
    os.system(f'mkdir -p {fig_save_dir}')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'{fig_save_dir}/taylor_diagram_gpp_subplots_normalised_std.png', dpi=400)
    plt.savefig(f'{fig_save_dir}/taylor_diagram_gpp_subplots_normalised_std.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    subplots_taylor()