import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris
import iris.quickplot as qplt
import iris.coord_categorisation
from iris.util import equalise_attributes, unify_time_units
from pyesgf.search import SearchConnection
from tqdm import tqdm
import pickle
import regionmask


models = ['ACCESS-ESM1-5', 'CNRM-ESM2-1', 'UKESM1-0-LL']

variants = {'ACCESS-ESM1-5': 'r1i1p1f1',
           'BCC-CSM2-MR': 'r1i1p1f1',
           'CNRM-ESM2-1': 'r1i1p1f3',
           'NorESM2-LM': 'r1i1p1f1',
           'UKESM1-0-LL': 'r2i1p1f2'}


lc_colours = {'treeFrac': '#117733',
              'baresoilFrac': '#ffee88',
              'cropFrac': '#44aa88',
              'grassFrac': '#99bb55',
              'shrubFrac': '#ddcc66'}

ar6_land = regionmask.defined_regions.ar6.land
region_abbrevs = ar6_land.abbrevs


def open_cube(model, cmip_variable_name, frequency, experiment_id='esm-hist', variant=None, start_year=2000, end_year=2014):
    if variant is None:
        variant = variants[model]

    conn = SearchConnection('http://esgf-data.dkrz.de/esg-search', distrib=True)
    ctx = conn.new_context(
        project="CMIP6",
        source_id=model,
        experiment_id=experiment_id,
        frequency=frequency,
        variant_label=variant,
        latest=True,
        variable=cmip_variable_name,
        facets="data_node"
        )
    ctx.hit_count

    result = ctx.search()[0]
    result.dataset_id
    files = result.file_context().search()
    
    opendap_urls = []
    for file in files:
        print(file.download_url)
        opendap_urls.append(file.opendap_url)

    if frequency == 'fx':
        all_cubes = iris.load(opendap_urls)
    else:
        date_range = iris.Constraint(time=lambda cell: start_year <= cell.point.year <= end_year)
        all_cubes = iris.load(opendap_urls, date_range)
    if len(all_cubes) > 1:
        equalise_attributes(all_cubes)
        unify_time_units(all_cubes)
        variable_cube = all_cubes.concatenate_cube()
    else:
        variable_cube = all_cubes[0]
    return variable_cube


def read_mean_land_cover_fracs(model):
    cell_area_experiment_id = 'piControl' if model=='UKESM1-0-LL' else 'esm-hist'
    cell_area_variant = 'r1i1p1f2' if model=='UKESM1-0-LL' else None
    land_cover_frac_types = ['grassFrac', 'treeFrac', 'shrubFrac', 'baresoilFrac', 'cropFrac']
    if model == 'CNRM-ESM2-1':
        land_cover_frac_types.remove('shrubFrac')
    model_data = {}
    model_data['area'] = open_cube(model, 'areacella', 'fx', 
                                   experiment_id=cell_area_experiment_id, variant=cell_area_variant)
    for lc_frac in tqdm(land_cover_frac_types):
        model_data[lc_frac] = open_cube(model, lc_frac, 'mon').collapsed('time', iris.analysis.MEAN)
    return model_data


def lc_regional_fractions_of_total_area(model):
    model_data = read_mean_land_cover_fracs(model)
    lc_fracs = {}
    lc_types = list(model_data.keys())
    lc_types.remove('area')
    for lc_type in tqdm(lc_types, desc=f'Computing {model} regional land cover fracs'):
        lc = model_data[lc_type]
        area = model_data['area']
        lc_region_fracs = np.ones(46)*np.nan
        for region in np.arange(46):
            region_grid_area = np.ones(area.shape)*np.nan
            region_lc_area = np.ones(area.shape)*np.nan
            ar6_mask = ma.filled(ar6_land.mask(area.coord('longitude').points,area.coord('latitude').points).data, np.nan)
            in_region = (ar6_mask==region)
            region_grid_area[in_region] = ma.filled(area.data, np.nan)[in_region]
            region_lc_area[in_region] = ma.filled(lc.data, 0.)[in_region]
            total_lc_area = np.nansum(region_grid_area * region_lc_area)
            total_area = np.nansum(region_grid_area)
            lc_fraction_of_land = total_lc_area/total_area
            lc_region_fracs[region] = lc_fraction_of_land
        lc_fracs[lc_type] = lc_region_fracs
    return lc_fracs


def all_region_fractions_total_area():
    if os.path.isfile('../data/land_cover/all_model_region_fracs_total_area.p'):
        lc_fracs = pickle.load(open('../data/land_cover/all_model_region_fracs_total_area.p', 'rb'))
    else:
        lc_fracs = {model: lc_regional_fractions_of_total_area(model) for model in models}
        pickle.dump(lc_fracs, open('../data/land_cover/all_model_region_fracs_total_area.p', 'wb'))
    return lc_fracs


def get_region_frac(fracs, model, region_number, lc_type):
    if lc_type in fracs[model].keys():
        region_frac = fracs[model][lc_type][region_number]
    else:
        region_frac = 0.
    return region_frac


def plot_region_fracs(region_number, ax=None, title=True, save=False, show=False, compress_width=False):
    fracs = all_region_fractions_total_area()
    all_lcs = [list(fracs[model].keys()) for model in models]
    lc_types = list(set([lc for i in all_lcs for lc in i]))
    lc_types.sort()
    if 'shrubFrac' in lc_types:
        lc_types.append(lc_types.pop(lc_types.index('shrubFrac')))
    total_lc = 0
    new_figure = (ax is None)
    if new_figure:
        fig, ax = plt.subplots(figsize=(6, 4))
    for lc in lc_types:
        frac_all_models = np.array([get_region_frac(fracs, model, region_number, lc) for model in models])
        b = ax.bar(models, frac_all_models, bottom=total_lc, label=lc[:-4], color=lc_colours[lc])
        total_lc += frac_all_models
    residual = 100. - total_lc
    ax.bar(models, residual, bottom=total_lc, label='other', color='w', hatch='//')
    box = ax.get_position()
    width_factor = 0.4 if compress_width else 0.8
    legend_position = 0.4 if compress_width else 0.5
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_ylim([0, 100])
    ax.set_ylabel('region area (%)', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5))
    if title:
        ax.set_title(region_abbrevs[region_number])
    ax.tick_params(labelsize=14)
    label_rotation = 15 if compress_width else 10
    ax.tick_params(axis='x', labelsize=12, rotation=label_rotation)
    if new_figure:
        if save:
            save_dir = '../data/figures/regional_land_cover/'
            os.system(f'mkdir -p {save_dir}')
            plt.savefig(f'{save_dir}/landCover_area_percents_{region_abbrevs[region_number]}.png', bbox_inches='tight', dpi=300)
            plt.close()
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':
    for region_number in np.arange(46):
        plot_region_fracs(region_number, save=True)
