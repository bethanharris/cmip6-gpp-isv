import os
import iris
from iris.util import equalise_attributes, unify_time_units
from pyesgf.search import SearchConnection
from download_and_process import regrid_cube


model_data_directory = '/prj/nceo/bethar/esm-isv'

models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'NorESM2-LM', 'UKESM1-0-LL']

variants = {'ACCESS-ESM1-5': 'r1i1p1f1',
           'BCC-CSM2-MR': 'r1i1p1f1',
           'CNRM-ESM2-1': 'r1i1p1f3',
           'NorESM2-LM': 'r1i1p1f1',
           'UKESM1-0-LL': 'r2i1p1f2'}

data_nodes = {'ACCESS-ESM1-5': 'esgf-data1.llnl.gov',
              'BCC-CSM2-MR': 'esgf.ceda.ac.uk',
              'CNRM-ESM2-1': 'esgf-data1.llnl.gov',
              'NorESM2-LM': 'noresg.nird.sigma2.no',
              'UKESM1-0-LL': 'esgf-data1.llnl.gov'}#'esgf.ceda.ac.uk'}


def daily_data(model, variable_name):
    conn = SearchConnection('http://esgf-data.dkrz.de/esg-search', distrib=True)
    ctx = conn.new_context(
        project="CMIP6",
        source_id=model,
        experiment_id='esm-hist',
        frequency='day',
        variant_label=variants[model],
        latest=True,
        variable=variable_name,
        data_node=data_nodes[model],
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

    date_range = iris.Constraint(time=lambda cell: 2000 <= cell.point.year <= 2014)
    all_cubes = iris.load(opendap_urls, date_range)
    if len(all_cubes) > 1:
        equalise_attributes(all_cubes)
        unify_time_units(all_cubes)
        cube = all_cubes.concatenate_cube()
    else:
        cube = all_cubes[0]
    return cube


def daily_vpd(model):
    rh = daily_data(model, 'hurs')
    tas = daily_data(model, 'tas')
    tas_celsius = tas - 273.15
    e_sat = 6.112 * iris.analysis.maths.exp((17.67 * tas_celsius)/(tas_celsius + 243.5))
    vpd = e_sat * (1. - rh/100.)
    return vpd


def save_vpd_cube(model):
    os.chdir(f'{model_data_directory}/{model}')
    save_filename_regridded = f'regrid_1_by_1_deg/{model}_vpd.nc'
    regridded_file_already_saved = os.path.isfile(save_filename_regridded)
    if regridded_file_already_saved:
        print(f'{save_filename_regridded} already exists, skipping')
    else:
        save_filename_native = f'{model}_vpd.nc'
        native_file_already_saved = os.path.isfile(save_filename_native)
        if native_file_already_saved:
            vpd = iris.load_cube(save_filename_native)
        else:
            vpd = daily_vpd(model)
            iris.save(vpd, save_filename_native)
        target_cube = iris.load_cube(f'{model_data_directory}/regrid_target_cube_1_by_1_deg.nc')
        regridded_cube = regrid_cube(model, 'vpd', vpd, target_cube)
        iris.save(regridded_cube, save_filename_regridded)


def save_vpd_all_models():
    for model in models:
        if model != 'BCC-CSM2-MR':
            save_vpd_cube(model)


if __name__ == '__main__':
    save_vpd_all_models()