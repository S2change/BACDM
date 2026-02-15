Code for the BACDM model is from https://zenodo.org/records/15788378, with the corresponding paper "Faster, better, and more accurate mapping of burned areas using Sentinel-2 multispectral images" by Liu et al. (https://doi.org/10.1016/j.rse.2025.115137).

To test locally the results of BADCM over a collection of 16-band chips, and for some reference data set (e.g. burned area product from ICNF), one can perform the following set of steps. It also allows to compare with the raw CCD vectorized product.

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
2. Execute in QGIS `qgis_read_reference_data_set_and_CCD_vector.py`. Set `CLEAR=True` to clear the current QGIS project. The legend for the burned area indicates the month and the labels indicate the day. If available, add to the map the vectorized CCD raw results results for the same ROI and time period of the reference data. Use same legend for both reference and vectorized CCD datasets (but `width` is different to distinguish both data sets).
3. In QGIS, execute `qgis_add_chips_boundaries_to_project.py` to create a layer that shows the locations of the available chips available in `chip_source_folder`; 
4. In QGIS, select manually a reference polygon (e.g. some burned area) for the local analysis at a location where chips are available; 
5. With that polygon selected, execute in QGIS `qgis_read_chip_tif_files_intersect_selected_reference.py`. This will select the available chips in `chip_source_folder` that intersect the selected feature, and will create copies of those chips in `Input_dir`
6. In VSCode, execute `split_tifs.py` to create the `before` and `after` chips in folders `Test_im_pathA` and `Test_im_pathB`;
7. In VSCode, execute `test.py` to apply BADCM and obtain the prediction;
8. In QGIS, execute `qgis_load_before_after_prediction.py` to clear the current project and create a new one with layers `before`, `after`, and `predict`;
9. In QGIS, execute `qgis_read_reference_data_set_and_CCD_vector.py`. Set `ADD_CCD_VECTOR_LAYER=False` if you wish and set `CLEAR=False` to add the reference layer  to the existing layers `before`, `after`, and `predict`.

