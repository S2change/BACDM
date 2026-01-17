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
INPUT_DIR = "/Users/domwelsh/BACDM/chips_test"

# Output directories for before and after images
OUTPUT_BEFORE_DIR = "/Users/domwelsh/BACDM/test_data/before"
OUTPUT_AFTER_DIR = "/Users/domwelsh/BACDM/test_data/after"

# Band indices to extract
BEFORE_BANDS = [1, 2, 3, 4, 5, 6]  # Bands 1-6 (rasterio uses 1-based indexing)
AFTER_BANDS = [8, 9, 10, 11, 12, 13]  # Bands 8-13

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

        # Update metadata for 6-band output
        meta.update(count=6)

        # Read before bands (1-6)
        before_data = src.read(BEFORE_BANDS)

        # Read after bands (8-13)
        after_data = src.read(AFTER_BANDS)

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
