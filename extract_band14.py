"""
Script to extract band 14 from a 16-band GeoTIFF and save as a single-band int16 TIF.
Preserves all geospatial metadata (CRS, transform, bounds, etc.)

Input: 16-band TIF file
Output: Single-band TIF containing band 14 with int16 data type
"""

import os
import rasterio

# ============================================================================
# CONFIGURATION - CHANGE THESE PATHS
# ============================================================================

# Input TIF file path
INPUT_TIF = "/Users/domwelsh/BACDM_root/BACDM/chips_test/T29TQG_20180101_20211231_0-1280.tif"

# Output TIF file path
OUTPUT_TIF = "/Users/domwelsh/BACDM_root/BACDM/output/T29TQG_20180101_20211231_0-1280_band14.tif"

# Band to extract (1-indexed, so band 14)
BAND_TO_EXTRACT = 14

# ============================================================================
# PROCESSING
# ============================================================================

def extract_band(input_path, output_path, band_index, nodata_value=1000):
    """
    Extract a single band from a multi-band TIF and save as int16.
    Replaces source nodata values (65535) with the specified nodata value.

    Args:
        input_path: Path to input multi-band TIF
        output_path: Path for output single-band TIF
        band_index: Band number to extract (1-indexed)
        nodata_value: Value to use for nodata in output (default: 1000)
    """
    with rasterio.open(input_path) as src:
        # Read metadata
        meta = src.meta.copy()

        # Verify the band exists
        if band_index > src.count:
            raise ValueError(f"Requested band {band_index}, but input file has only {src.count} bands")

        # Update metadata for single-band int16 output with nodata value
        meta.update(
            count=1,
            dtype='int16',
            nodata=nodata_value
        )

        # Read the specified band
        band_data = src.read(band_index)

        # Replace nodata values (65535) with the specified nodata value
        nodata_mask = band_data == 65535
        band_data = band_data.astype('int16')
        band_data[nodata_mask] = nodata_value

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Write output file
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(band_data, 1)

            # Copy band description if it exists
            desc = src.descriptions[band_index - 1]
            if desc:
                dst.set_band_description(1, desc)

        num_nodata = nodata_mask.sum()
        print(f"Successfully extracted band {band_index}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Data type: int16")
        print(f"Nodata value: {nodata_value} ({num_nodata:,} pixels replaced)")


def main():
    """Extract band 14 from the input TIF."""

    # Check if input file exists
    if not os.path.exists(INPUT_TIF):
        print(f"Error: Input file not found: {INPUT_TIF}")
        return

    try:
        extract_band(INPUT_TIF, OUTPUT_TIF, BAND_TO_EXTRACT)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
