Code for the BACDM model is from https://zenodo.org/records/15788378, with the corresponding paper "Faster, better, and more accurate mapping of burned areas using Sentinel-2 multispectral images" by Liu et al. (https://doi.org/10.1016/j.rse.2025.115137).

To test locally the results of BADCM over a collection of 16-band chips, and for the burned area product from ICNF, one can perform the following set of steps:

1. Update `AAA_Configs.py`, namely
  ```
  # Input directory containing all available 16-band GeoTIFF files
chip_source_folder = r'H:\new_parquets_2017_2025\tabular\T29TNF\processed_outputs\chips'
# Temp directory to store the selected  16-band TIF files
#Input_dir = r".\chips_test\TQG_burn_area" # Dominic tests
Input_dir = r".\chips_test\TNF_BA_20241155265"
# where before and after 6-channel geo-referenced tifs are stored
Test_im_pathA = r".\test_data\before_TNF_BA_20241155265"
Test_im_pathB = r".\test_data\after_TNF_BA_20241155265"
# where predicted change maps will be saved (both as png and geotiff)
Test_det_path = r".\test_data\predictions_TNF_BA_20241155265"
# ICNF burned areas or another shape file reference file for tests:
shp_path = r'H:\ref_datasets\BDR_ICNF\ardida_2024\ardida_2024.shp'
# CCD results (rasters or vectors) for comparison with our predictions
CCD_raster_results_path = r"H:\new_parquets_2017_2025\tabular\T29TNF\processed_outputs\rasters" # bimonthly, for 2023 and 2024
CCD_vector_results_path = r"H:\new_parquets_2017_2025\tabular\T29TNF\processed_outputs\vectors" # bimonthly, for 2023 and 2024
  ```
2. Execute in QGIS `qgis_read_area_ardida_ICNF.py`. Set `CLEAR=True` to clear the current QGIS project. The legend for the burned area indicates the month and the labels indicate the day.
3. In QGIS, select manualy a burned area polygon for the local analysis
4. With that polygon selected, execute in QGIS `qgis_read_tif_files_intersect_selected_BA.py`. This will select the available chips in `chip_source_folder` that intersect the selected burned area, and will create copies of those chips in `Input_dir`
5. In VSCode, execute `split_tifs.py` to create the `before` and `after` chips in folders `Test_im_pathA` and `Test_im_pathB`;
6. In VSCode, execute `test.py` to apply BADCM and obtain the prediction;
7. In QGIS, execute `qgis_load_before_after_prediction.py` to clear the current project and create a new one with layers `before`, `after`, and `predict`;
8. In QGIS, execute again  `qgis_read_area_ardida_ICNF.py`. Set `CLEAR=False` to add the ICNF burned area layer to the existing layers `before`, `after`, and `predict`.
9. In QGIS, add to the map the raster and vector CCD raw results for the same tile, and for all time periods. The legend should match the legend of the reference data set.
