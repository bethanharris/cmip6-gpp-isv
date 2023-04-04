import iris
import iris.coords
import iris.cube
import numpy as np
import numpy.ma as ma


model_data_directory = '/prj/nceo/bethar/esm-isv'

def rectangular_grid(resolution_degrees_lat, resolution_degrees_lon): 
    lat_centres = np.arange(-90, 90, resolution_degrees_lat) + 0.5*resolution_degrees_lat
    lon_centres = np.arange(0, 360, resolution_degrees_lon) + 0.5*resolution_degrees_lon
    latitude = iris.coords.DimCoord(lat_centres, standard_name='latitude', units='degrees')
    longitude = iris.coords.DimCoord(lon_centres, standard_name='longitude', units='degrees')
    dummy_data = ma.masked_all((lat_centres.size, lon_centres.size))
    cube = iris.cube.Cube(dummy_data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
    for coord_key in ['latitude', 'longitude']:
        cube.coord(coord_key).bounds = None
        cube.coord(coord_key).guess_bounds()
        cube.coord(coord_key).bounds = np.round(cube.coord(coord_key).bounds, 3)
        cube.coord(coord_key).points = np.round(cube.coord(coord_key).points, 3)
    lat_label = str(resolution_degrees_lat).replace('.','pt')
    lon_label = str(resolution_degrees_lon).replace('.','pt')
    iris.save(cube, f'{model_data_directory}/regrid_target_cube_{lat_label}_by_{lon_label}_deg.nc')


if __name__ == '__main__':
    rectangular_grid(1, 1)