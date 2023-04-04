import iris
import iris.coord_categorisation
from iris.util import equalise_attributes, unify_time_units
import xarray as xr
import os
import sys
import numpy as np
import numpy.ma as ma


model_data_directory = '/prj/nceo/bethar/esm-isv'


def download_cmip6_data(model):
    cmip6_files_list = f'{model_data_directory}/file_list_{model}.txt'
    with open(cmip6_files_list) as file:
        filenames = [line.rstrip() for line in file]
    for cmip6_datafile in filenames:
        cmip6_saved_filename = cmip6_datafile.split('/')[-1]
        cmip6_data_downloaded = os.path.isfile(cmip6_saved_filename)
        cmip_variable_name = cmip6_saved_filename.split('_')[0]
        processed_cube_filenames = os.listdir('.')
        processed_cubes_matching_name = [f for f in processed_cube_filenames if f'_{cmip_variable_name}.' in f or f'_{cmip_variable_name}_' in f]
        cube_already_processed = True if len(processed_cubes_matching_name) > 0 else False
        if cmip6_data_downloaded:
            print(f'{cmip6_saved_filename} already exists, skipping')
        elif cube_already_processed:
            print(f'{cmip6_saved_filename} already processed to {processed_cubes_matching_name[0]}, skipping')
        else:
            os.system(f'wget {cmip6_datafile}')


def consolidate_variable(model, cmip_variable_name):
    save_filename = f'{model}_{cmip_variable_name}.nc'
    file_already_saved  = os.path.isfile(save_filename.split('/')[-1])
    if file_already_saved:
        print(f'{save_filename} already exists, skipping')
    else:
        date_range = iris.Constraint(time=lambda cell: 2000 <= cell.point.year <= 2014)
        all_cubes = iris.load(f'{cmip_variable_name}_*', date_range)
        equalise_attributes(all_cubes)
        unify_time_units(all_cubes)
        variable_cube = all_cubes.concatenate_cube()
        iris.save(variable_cube, save_filename)


def daily_mean_from_3hr(model, cmip_variable_name):
    save_filename = f'{model}_{cmip_variable_name}.nc'
    file_already_saved  = os.path.isfile(save_filename.split('/')[-1])
    if file_already_saved:
        print(f'{save_filename} already exists, skipping')
    else:
        date_range = iris.Constraint(time=lambda cell: 2000 <= cell.point.year <= 2014)
        all_cubes = iris.load(f'{cmip_variable_name}*', date_range)
        equalise_attributes(all_cubes)
        unify_time_units(all_cubes)
        variable_cube = all_cubes.concatenate_cube()
        iris.coord_categorisation.add_month_number(variable_cube, 'time', name='month')
        iris.coord_categorisation.add_day_of_year(variable_cube, 'time', name='day')
        iris.coord_categorisation.add_year(variable_cube, 'time', name='year')
        daily_mean = variable_cube.aggregated_by(['day', 'month', 'year'], iris.analysis.MEAN)
        iris.save(daily_mean, save_filename)


def save_fixed_variable(model, cmip_variable_name):
    save_filename = f'{model}_{cmip_variable_name}.nc'
    file_already_saved  = os.path.isfile(save_filename.split('/')[-1])
    if file_already_saved:
        print(f'{save_filename} already exists, skipping')
    else:
        variable_cube = iris.load(f'{cmip_variable_name}*').concatenate_cube()
        iris.save(variable_cube, save_filename)


def soil_moisture_to_depths(model):
    depth_coord_name = 'sdepth' if model=='CNRM-ESM2-1' else 'depth'
    depth_thresholds = [1., 3.]
    for depth_limit in depth_thresholds:
        save_filename = f'{model}_{cmip_variable_name}_{depth_limit}.nc'
        file_already_saved  = os.path.isfile(save_filename.split('/')[-1])
        if file_already_saved:
            print(f'{save_filename} already exists, skipping')
        else:
            date_range = iris.Constraint(time=lambda cell: 2000 <= cell.point.year <= 2014)
            all_cubes = iris.load('mrsol*', date_range)
            equalise_attributes(all_cubes)
            unify_time_units(all_cubes)
            mrsol = all_cubes.concatenate_cube()
            layer_depths = mrsol.coord(depth_coord_name).bounds
            last_full_layer = np.where(layer_depths[:,-1] < depth_limit)[0][-1]
            next_layer_bounds = layer_depths[last_full_layer+1]
            fraction_of_next_layer = (depth_limit - next_layer_bounds[0])/(next_layer_bounds[1] - next_layer_bounds[0])
            soil_moisture_to_depth = mrsol[:, 0:last_full_layer+1, :, :].collapsed(depth_coord_name, iris.analysis.SUM)
            soil_moisture_to_depth += mrsol[:, last_full_layer+1, :, :] * fraction_of_next_layer
            iris.save(soil_moisture_to_depth, save_filename)


def process_downloads(model, cmip_variable_name):
    if cmip_variable_name == 'gpp':
        daily_mean_from_3hr(model, cmip_variable_name)
    elif cmip_variable_name == 'sftlf':
        save_fixed_variable(model, cmip_variable_name)
    elif cmip_variable_name == 'mrsol':
        soil_moisture_to_depths(model)
    else:
        consolidate_variable(model, cmip_variable_name)


def clean_up_downloads(model):
    os.system('rm *esm-hist*')
    os.system('rm *piControl*')
    os.system('rm *1pctCO2*')


def mask_cube_by_other_cube(to_mask, take_mask_from):
    to_mask_xr = xr.DataArray.from_iris(to_mask)
    take_mask_from_xr = xr.DataArray.from_iris(take_mask_from)
    masked_xr = xr.where(take_mask_from_xr.isnull(), np.nan, to_mask_xr)
    masked_xr = masked_xr.rename({'lat': 'latitude'})
    masked_xr = masked_xr.rename({'lon': 'longitude'})
    masked_cube = masked_xr.to_iris()
    masked_cube.remove_coord("latitude")
    masked_cube.add_dim_coord(to_mask.coord('latitude'), 1)
    masked_cube.remove_coord("longitude")
    masked_cube.add_dim_coord(to_mask.coord('longitude'), 2)
    masked_cube.standard_name = to_mask.standard_name
    masked_cube.long_name = to_mask.long_name
    masked_cube.units = to_mask.units
    return masked_cube


def ensure_cube_lat_lon_bounds(cube):
    if cube.coord('latitude').bounds is None:
        cube.coord('latitude').guess_bounds()
    if cube.coord('longitude').bounds is None:
        cube.coord('longitude').guess_bounds()


def area_weighted_regrid(model, cube, target_cube):
    ensure_cube_lat_lon_bounds(cube)
    ensure_cube_lat_lon_bounds(target_cube)
    scheme = iris.analysis.AreaWeighted(mdtol=1.)
    regridded_cube = cube.regrid(target_cube, scheme)
    return regridded_cube


def land_area_weighted_regrid(model, cube, target_cube):
    ensure_cube_lat_lon_bounds(cube)
    ensure_cube_lat_lon_bounds(target_cube)
    land_percent_filename = f'{model}_sftlf.nc'
    if os.path.isfile(land_percent_filename):
        land_frac = iris.load_cube(land_percent_filename)/100.
    else:
        raise OSError(f'No land mask downloaded for {model}')
    land_frac_grid = iris.util.broadcast_to_shape(land_frac.data, cube.shape, (1, 2))
    land_frac_cube = cube.copy(data=ma.masked_values(land_frac_grid.data, 0))
    land_frac_cube = mask_cube_by_other_cube(land_frac_cube, cube)
    land_weighted_cube = cube * land_frac_cube
    scheme = iris.analysis.AreaWeighted(mdtol=1.)
    regridded_cube = land_weighted_cube.regrid(target_cube, scheme)
    regridded_land_frac = land_frac_cube.regrid(target_cube, scheme)
    final_regridded_cube = regridded_cube / regridded_land_frac
    return final_regridded_cube
    

def regrid_cube(model, cmip_variable_name, cube, target_cube):
    if cmip_variable_name == 'sftlf':
        regridded_cube = area_weighted_regrid(model, cube, target_cube)
    else:
        regridded_cube = land_area_weighted_regrid(model, cube, target_cube)
    return regridded_cube


def regrid_all_cubes(model, regrid_to):
    os.system(f'mkdir -p regrid_{regrid_to}')
    target_cube = iris.load_cube(f'{model_data_directory}/regrid_target_cube_{regrid_to}.nc')
    all_cubes = [cube for cube in os.listdir('.') if cube.endswith('.nc')]
    for cube_filename in all_cubes:
        filename_no_ext = cube_filename.split('.nc')[0]
        cmip_variable_name = filename_no_ext.split('_')[-1]
        save_regridded_filename = f'regrid_{regrid_to}/{filename_no_ext}.nc'
        file_already_saved  = os.path.isfile(save_regridded_filename)
        if file_already_saved:
            print(f'{save_regridded_filename} already exists, skipping')
        else:
            cube = iris.load_cube(cube_filename)
            regridded_cube = regrid_cube(model, cmip_variable_name, cube, target_cube)
            iris.save(regridded_cube, save_regridded_filename)


if __name__ == '__main__':
    model = sys.argv[1]
    os.system(f'mkdir -p {model_data_directory}/{model}')
    os.chdir(f'{model_data_directory}/{model}')
    if model == 'BCC-CSM2-MR':
        if not os.path.isfile("BCC-CSM2-MR_sftlf.nc"):
            raise IOError('''Need to download land mask from CMIP5 archives and place in directory manually. Obtain file from https://dap.ceda.ac.uk/badc/cmip5/data/cmip5/output1/BCC/bcc-csm1-1-m/piControl/fx/atmos/fx/r0i0p0/v20120705/sftlf/sftlf_fx_bcc-csm1-1-m_piControl_r0i0p0.nc and place in BCC-CSM2-MR directory with filename BCC-CSM2-MR_sftlf.nc. Then rerun this script.''')
    download_cmip6_data(model)
    for cmip_variable_name in ['sftlf', 'mrsos', 'mrso', 'mrsol', 'pr', 'gpp', 'rsds', 'tasmax', 'hfls', 'lai']:
        try:
            process_downloads(model, cmip_variable_name)
        except:
            print(f'No {cmip_variable_name} for {model}')
    clean_up_downloads(model)
    regrid_all_cubes(model, '1_by_1_deg')
