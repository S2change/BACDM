# test_data
Directory for testing chips created from CCD results

### chips_to_test
Directory of chips created from CCD results to process through BACDM

- 01_T29TQG_2018010_20211231 - First chips used to test BACDM
- TQG_burn_area - A set of specific tiles where a large fire happened in 2024

## Processed directories
The following directories are the inputs and predictions from the BACD model. The before and after tifs were created from /split_tifs.py, and the scaling method that is the one the paper's authors used is percentile_perband. Other scaling methods are there for tests/comparisons.

### TQG_burn_area_minmax_perband
before and after tifs of the TQG_burn_area chips, created from using minmax_perband scaling method in split_tifs.py, and the resulting prediction pngs

### TQG_burn_area_percentile_perband
before and after tifs of the TQG_burn_area chips, created from using percentile_perband scaling method in split_tifs.py, and the resulting prediction pngs. This is the scaling method the authors of the BACD paper said they used for their data

### uint8_reversed_minmax_perband
before and after tifs of the 01_T29TQG_20180101_20211231 chips, created from using minmax_perband scaling method in split_tifs.py, and the resulting prediction pngs
