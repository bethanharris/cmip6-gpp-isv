# cmip6-gpp-isv
Code for Harris et al. "Contrasting responses of vegetation productivity to intraseasonal rainfall in Earth System Models"

This repository contains code to reproduce the analysis and figures of Harris et al. (2023) *Contrasting responses of vegetation productivity to intraseasonal rainfall in Earth System Models*, submitted to Earth System Dynamics. The code includes automatic downloading and processing of CMIP6 model data, but does not include the processing of observational datasets, which must be done separately. Details of the Python enviroments used to run the analysis are included in [cmip6-gpp-isv.yml](cmip6-gpp-isv.yml) and [r_env.yml](r_env.yml). All analysis was run in the environment detailed in [cmip6-gpp-isv.yml](cmip6-gpp-isv.yml), other than [sensitivity_sig_tests.py](src/sensitivity_sig_tests.py), which requires the environment from [r_env.yml](r_env.yml) due to the use of rpy2 to run R packages.

# Running the analysis

- Running [download_model_data.sh](src/download_model_data.sh) will download and regrid data from CMIP6 models. The directory to which data is downloaded is specified at the top of [download_and_process.py](src/download_and_process.py). This includes VPD by default. Any other variables need to be manually included by creating a txt file at /\<download_directory\>/\<model-name.txt\> and adding URLs corresponding to CMIP6 data files. Files for the 5 models tested in the paper are included in [data](data). These need to be moved to the chosen download directory before running [download_model_data.sh](src/download_model_data.sh). URLs were obtained from the Earth System Grid Federation CMIP6 data portal at [https://esgf-index1.ceda.ac.uk/search/cmip6-ceda/](https://esgf-index1.ceda.ac.uk/search/cmip6-ceda/).
- Default regridding is to 1x1 degree horizontal resolution, but this can be customised by creating a new grid template using [create_regrid_target_cube.py](src/create_regrid_target_cube.py) and changing the *regrid_to* argument fed to regrid_all_cubes() in [download_model_data.sh](src/download_model_data.sh).
- The regridded model data is saved using the naming format "/\<download_directory\>/\<model-name\>/\<regrid_code\>/\<model-name\>\_\<cmip-variable-name\>.nc". To add observational data products to the analysis, follow this format by creating a directory "/\<download_directory\>/OBS/\<regrid_code\>/" and placing regridded datasets in it with names "OBS\_\<cmip-variable-name\>-\<product-name\>.nc", e.g. "OBS_gpp-VODCA2GPP". These observations will then be usable in the following scripts.
- Run [multi_model_isv_composites.py](src/multi_model_isv_composites.py) to save global composites for all downloaded CMIP6 models/variables and observations. Regional composites (based on a lat-lon rectangle region) can also be saved. The configuration section at the top of the script allows you to set this region, limit rainfall events to those occurring in particular months, choose whether to standardise anomalies, and whether composites should be restricted to grid boxes in which there are valid ESA CCI soil moisture observations. This is the script used to produce Figure 1.
- [multi_model_regional_isv_composites.py](src/multi_model_regional_isv_composites.py) performs the same function as [multi_model_isv_composites.py](src/multi_model_isv_composites.py), with similar configuration choices available, but composites are produced and saved for each of the IPCC AR6 regions. Also contains functions to plot these composites regionally.
- Figure 2 was produced by running [models_vs_obs.py](src/models_vs_obs.py). This script reads in the composites produced by [multi_model_regional_isv_composites.py](src/multi_model_regional_isv_composites.py), then computes and saves the peak amplitude, post-event amplitude and lag values discussed in the paper. It then compares the modelled regional values to observed values for a single choice of observational product.
- A modified version of code for producing Taylor diagrams from [Yannick Copin](https://doi.org/10.5281/zenodo.5548061) is included as [taylor_diagrams.py](src/taylor_diagrams.py). This is used by [models_vs_obs_taylor.py](src/models_vs_obs_taylor.py) to compare modelled regional GPP lags and amplitudes to multiple observational products, as in Figure 3.
- Plots of the sensitivity of GPP responses to surface soil moisture and atmospheric vapour pressure deficit drivers (Figure 4) are produced using [driver_sensitivity.py](src/driver_sensitivity.py). The statistical testing of whether these sensitivities differ between models/observations is done by [sensitivity_sig_tests.py](src/sensitivity_sig_tests.py).
- To compare land covers between the models, [compare_regional_land_cover.py](src/compare_regional_land_cover.py) downloads, processes and saves the land cover fractions in each IPCC AR6 region, for each model included in Figure A1. Figure A1 itself is then produced by [region_examples_with land_cover.py](src/region_examples_with_land_cover.py).