import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Add the parent directory to Python path to import utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import compute_indices

def check_compute_indices():
    """
    Test compute_indices function with real data from a sample Sentinel-2 image.
    """
    # Path to your test mosaic
    mosaic_path = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test/mosaics/s2_202101.tif"
    
    print(f"Testing with mosaic: {os.path.basename(mosaic_path)}")
    
    with rasterio.open(mosaic_path) as src:
        # Read all required bands for a small window
        window = rasterio.windows.Window(0, 0, 1024, 1024)  # Adjust size if needed
        data = src.read([1,2,3,4,9,10], window=window)  # B2,B3,B4,B8,B11,B12
        
        # Get nodata value
        nodata = src.profile.get('nodata', np.nan)
        print(f"\nMosaic information:")
        print(f"  Shape: {data.shape}")
        print(f"  Nodata value: {nodata}")
        
        # Create mask for nodata
        if np.isnan(nodata):
            valid_mask = ~np.isnan(data)
        else:
            valid_mask = data != nodata
            
        # Convert to reflectance
        data = np.where(valid_mask, data.astype(np.float32) / 10000.0, np.nan)
        
        # Split bands
        b2, b3, b4, b8, b11, b12 = data
        
        print("\nBand value ranges after conversion:")
        for i, band in enumerate([b2, b3, b4, b8, b11, b12]):
            valid_data = band[~np.isnan(band)]
            if len(valid_data) > 0:
                print(f"Band {i+1}:")
                print(f"  Min: {np.min(valid_data):.4f}")
                print(f"  Max: {np.max(valid_data):.4f}")
                print(f"  Mean: {np.mean(valid_data):.4f}")
                print(f"  Valid pixels: {len(valid_data)}")
            else:
                print(f"Band {i+1}: No valid data")
        
        # Compute indices with debugging
        print("\nComputing spectral indices...")
        ndvi, evi, nbr, crswir = compute_indices(b2, b3, b4, b8, b11, b12, debug=True)
        
        # Print final statistics
        print("\nFinal index statistics:")
        for name, index in [('NDVI', ndvi), ('EVI', evi), ('NBR', nbr), ('CRSWIR', crswir)]:
            valid_data = index[~np.isnan(index)]
            print(f"\n{name}:")
            print(f"  Min: {np.min(valid_data):.4f}")
            print(f"  Max: {np.max(valid_data):.4f}")
            print(f"  Mean: {np.mean(valid_data):.4f}")
            print(f"  Std: {np.std(valid_data):.4f}")
            print(f"  Valid pixels: {len(valid_data)}")

if __name__ == "__main__":
    check_compute_indices()