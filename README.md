Code for the BACDM model is from https://zenodo.org/records/15788378, with the corresponding paper "Faster, better, and more accurate mapping of burned areas using Sentinel-2 multispectral images" by Liu et al. (https://doi.org/10.1016/j.rse.2025.115137).

To test locally the results of BADCM over a collection of 16-band chips, and for some reference data set (e.g. burned area product from ICNF), one can perform the following set of steps. It also allows to compare with the raw CCD vectorized product.

# A. Create chips from hdf5 file around a given reference feature 

1. Open in QGIS a layer with reference data (e.g. BDR_expanded_v0.gpkg), which has fields `Data0` and `Data1`.
2. Select manually a feature where `Data0` and `Data1` are close and within the range of dates and within the tile of the input `hdf5` input file.
3. In QGIS console, execute `scripts\chips\qgis_convert_ref_vector_to_input_tif.py`. The ouput is a 10 m resolution `tif` file e.g. `harmonized_to_tifs\output_20210322_604495_607495_4444065_4447065.tif` in `EPSG:32629`. The output tif file has 3 bands, but just band 1 and 3 are relevant for the next step. Band1 has the mean date of the vector feature in format YYYYMMDD for the pixels within the feature and 65535 (NoData) otherwise. Band3 is 1 for the pixels within the feature and 0 otherwise.
5. In VSCode, execute `python -m chips.chips_S2_dates_hdf5` with `SPATIAL_BOUNDS=None`, `MAX_DATE = None`  and `MIN_DATE = None`, with inputs `output_20210322_604495_607495_4444065_4447065.tif` and the `hdf5` file. The output is a set of chips that cover the extent of `output_20210322_604495_607495_4444065_4447065.tif` for the mean date between `Data0` and `Data1`.
   
# B. Access chips and apply DL model

1. Update `AAA_Configs.py`, namely
  ```
working_dir = r"C:\Users\mlc\Downloads\temp\test_tif_to_hdf5"
suffix_test_files = "TNE_buf_322" # around buffer_id=...(BDR_expanded_v0)
# where 16 bands chips are stored
Input_dir =  os.path.join(working_dir, "chips", "all", suffix_test_files)
# where before and after 6-channel geo-referenced tifs are stored
Test_im_pathA = os.path.join(working_dir, "chips", "before",suffix_test_files)
Test_im_pathB = os.path.join(working_dir, "chips", "after",suffix_test_files)
# where predicted change maps will be saved (both as png and geotiff)
Test_det_path = os.path.join(working_dir, "chips", "predictions",suffix_test_files) 
# ICNF burned areas or another vector georeferenced file for tests:
shp_path, DATA0 =  os.path.join(working_dir, "harmonized", "BDR_expanded_v0.gpkg"), "Data0" 
temp_raster_reference= os.path.join(working_dir, "harmonized_to_tifs") # temporary raster version of the reference vector file; 
  ```
2. (optional if A) Execute in QGIS `qgis_read_reference_data_set_and_CCD_vector.py`. Set `CLEAR=True` to clear the current QGIS project. The legend for the burned area indicates the month and the labels indicate the day. If available, add to the map the vectorized CCD raw results results for the same ROI and time period of the reference data. Use same legend for both reference and vectorized CCD datasets (but `width` is different to distinguish both data sets).
3. In QGIS, execute `qgis_add_chips_boundaries_to_project.py` to create a layer that shows the locations of the available chips available in `chip_source_folder`; 
4. In QGIS, select manually a reference polygon (e.g. some burned area) for the local analysis at a location where chips are available; 
5. With that polygon selected, execute in QGIS `qgis_read_chip_tif_files_intersect_selected_reference.py`. This will select the available chips in `chip_source_folder` that intersect the selected feature, and will create copies of those chips in `Input_dir`
6. In VSCode, set the working directory as `repos\mrs_bacd_2025\BACDM_9feb_2026`. Execute `python split_tifs.py` to create the `before` and `after` chips in folders `Test_im_pathA` and `Test_im_pathB`;
7. In VSCode, execute `test.py` to apply BADCM and obtain the prediction;
8. In QGIS, execute `qgis_load_before_after_prediction.py` to clear the current project and create a new one with layers `before`, `after`, and `predict`;
9. In QGIS, execute `qgis_read_reference_data_set_and_CCD_vector.py`. Set `ADD_CCD_VECTOR_LAYER=False` if you wish and set `CLEAR=False` to add the reference layer  to the existing layers `before`, `after`, and `predict`.

