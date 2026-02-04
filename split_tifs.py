"""
Script to split 16-band GeoTIFFs into separate before/after 6-band GeoTIFFs.
Preserves all geospatial metadata (CRS, transform, bounds, etc.)

Input: 16-band TIF files where:
  - Bands 1-6 (indices 0-5): Pre-change image (B2, B3, B4, B8, B11, B12)
  - Band 7 (index 6): Unused
  - Bands 8-13 (indices 7-12): Post-change image (B2, B3, B4, B8, B11, B12)
  - Bands 14-16 (indices 13-15): Unused

Output: Two 6-band GeoTIFFs per input file (before and after)
"""

import os
import glob
import rasterio

# ============================================================================
# CONFIGURATION - CHANGE THESE PATHS
# ============================================================================

# Input directory containing 16-band TIF files
INPUT_DIR = "/Users/domwelsh/BACDM_root/BACDM/chips_test"

# Output directories for before and after images
OUTPUT_BEFORE_DIR = "/Users/domwelsh/BACDM_root/BACDM/test_data/before_uint8_reversed"
OUTPUT_AFTER_DIR = "/Users/domwelsh/BACDM_root/BACDM/test_data/after_uint8_reversed"

# Band indices to extract (reversed order for output to match BACDM data)
BEFORE_BANDS = [6, 5, 4, 3, 2, 1]
AFTER_BANDS = [13, 12, 11, 10, 9, 8]

# ============================================================================
# PROCESSING
# ============================================================================

def split_tif(input_path, output_before_path, output_after_path):
    """
    Split a 16-band TIF into two 6-band TIFs (before and after).

    Args:
        input_path: Path to input 16-band TIF
        output_before_path: Path for output before TIF (bands 1-6)
        output_after_path: Path for output after TIF (bands 8-13)
    """
    with rasterio.open(input_path) as src:
        # Read metadata
        meta = src.meta.copy()

        # Verify we have enough bands
        if src.count < 13:
            raise ValueError(f"Input file has only {src.count} bands, expected at least 13")

        # Update metadata for 6-band output with uint8 dtype and nodata value
        meta.update(count=6, dtype='uint8', nodata=255)

        # Read before bands (1-6) and convert from 0-10000 range to 0-255 uint8
        before_data = src.read(BEFORE_BANDS)
        # Create mask for nodata pixels (value 65535)
        nodata_mask = before_data == 65535
        # Scale data from 0-10000 to 0-255
        before_data = (before_data / 10000.0 * 255.0).clip(0, 255).astype('uint8')
        # Set nodata pixels to 255
        before_data[nodata_mask] = 255

        # Read after bands (8-13) and convert from 0-10000 range to 0-255 uint8
        after_data = src.read(AFTER_BANDS)
        # Create mask for nodata pixels (value 65535)
        nodata_mask = after_data == 65535
        # Scale data from 0-10000 to 0-255
        after_data = (after_data / 10000.0 * 255.0).clip(0, 255).astype('uint8')
        # Set nodata pixels to 255
        after_data[nodata_mask] = 255

        # Write before image
        with rasterio.open(output_before_path, 'w', **meta) as dst:
            dst.write(before_data)
            # Copy band descriptions if they exist
            for i, band_idx in enumerate(BEFORE_BANDS, start=1):
                desc = src.descriptions[band_idx - 1]
                if desc:
                    dst.set_band_description(i, desc)

        # Write after image
        with rasterio.open(output_after_path, 'w', **meta) as dst:
            dst.write(after_data)
            # Copy band descriptions if they exist
            for i, band_idx in enumerate(AFTER_BANDS, start=1):
                desc = src.descriptions[band_idx - 1]
                if desc:
                    dst.set_band_description(i, desc)

        print(f"  ✓ Created before: {os.path.basename(output_before_path)}")
        print(f"  ✓ Created after:  {os.path.basename(output_after_path)}")


def main():
    """Process all TIF files in the input directory."""

    # Create output directories if they don't exist
    os.makedirs(OUTPUT_BEFORE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_AFTER_DIR, exist_ok=True)

    # Find all TIF files in input directory
    tif_pattern = os.path.join(INPUT_DIR, "*.tif")
    tif_files = glob.glob(tif_pattern)

    if not tif_files:
        print(f"No TIF files found in {INPUT_DIR}")
        return

    print(f"Found {len(tif_files)} TIF file(s) to process\n")

    # Process each file
    success_count = 0
    error_count = 0

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        print(f"Processing: {filename}")

        # Create output paths
        output_before = os.path.join(OUTPUT_BEFORE_DIR, filename)
        output_after = os.path.join(OUTPUT_AFTER_DIR, filename)

        try:
            split_tif(tif_path, output_before, output_after)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1

        print()

    # Summary
    print("="*70)
    print(f"Processing complete!")
    print(f"  Success: {success_count} files")
    print(f"  Errors:  {error_count} files")
    print(f"\nOutput directories:")
    print(f"  Before: {OUTPUT_BEFORE_DIR}")
    print(f"  After:  {OUTPUT_AFTER_DIR}")


if __name__ == "__main__":
    main()
