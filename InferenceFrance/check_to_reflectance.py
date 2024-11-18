import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Add the parent directory to Python path to import utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import to_reflectance, compute_indices

def check_reflectance_and_indices():
    """
    Test to_reflectance and compute_indices functions with real data.
    """
    # Path to your test mosaic
    mosaic_path = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test/mosaics/s2_202101.tif"
    
    print(f"\nTesting with mosaic: {os.path.basename(mosaic_path)}")
    
    with rasterio.open(mosaic_path) as src:
        # Read all required bands for a small window
        window = rasterio.windows.Window(0, 0, 1024, 1024)
        data = src.read([1,2,3,4,9,10], window=window)  # B2,B3,B4,B8,B11,B12
        nodata = src.profile.get('nodata', np.nan)
        
        print("\nInput data information:")
        print(f"  Shape: {data.shape}")
        print(f"  Nodata value: {nodata}")
        print(f"  Data type: {data.dtype}")
        
        # Show statistics of raw data
        print("\nRaw data statistics:")
        for i in range(data.shape[0]):
            valid_data = data[i][~np.isnan(data[i])] if np.isnan(nodata) else data[i][data[i] != nodata]
            if len(valid_data) > 0:
                print(f"\nBand {i+1}:")
                print(f"  Min: {np.min(valid_data)}")
                print(f"  Max: {np.max(valid_data)}")
                print(f"  Mean: {np.mean(valid_data)}")
                print(f"  Valid pixels: {len(valid_data)}")
                print(f"  Invalid pixels: {data[i].size - len(valid_data)}")
            else:
                print(f"\nBand {i+1}: No valid data")
                
        # Plot histograms of raw data
        plt.figure(figsize=(15, 5))
        plt.suptitle("Raw Data Histograms")
        for i in range(data.shape[0]):
            plt.subplot(2, 3, i+1)
            valid_data = data[i][~np.isnan(data[i])] if np.isnan(nodata) else data[i][data[i] != nodata]
            if len(valid_data) > 0:
                plt.hist(valid_data, bins=50)
                plt.title(f'Band {i+1}')
        plt.tight_layout()
        plt.show()
        
        # Convert to reflectance
        print("\nConverting to reflectance...")
        reflectance_data = to_reflectance(data, nodata=nodata)
        
        # Show reflectance statistics
        print("\nReflectance statistics:")
        for i in range(reflectance_data.shape[0]):
            valid_data = reflectance_data[i][~np.isnan(reflectance_data[i])]
            if len(valid_data) > 0:
                print(f"\nBand {i+1}:")
                print(f"  Min: {np.min(valid_data):.4f}")
                print(f"  Max: {np.max(valid_data):.4f}")
                print(f"  Mean: {np.mean(valid_data):.4f}")
                print(f"  Valid pixels: {len(valid_data)}")
                print(f"  Invalid pixels: {reflectance_data[i].size - len(valid_data)}")
            else:
                print(f"\nBand {i+1}: No valid data")
        
        # Plot histograms of reflectance data
        plt.figure(figsize=(15, 5))
        plt.suptitle("Reflectance Data Histograms")
        for i in range(reflectance_data.shape[0]):
            plt.subplot(2, 3, i+1)
            valid_data = reflectance_data[i][~np.isnan(reflectance_data[i])]
            if len(valid_data) > 0:
                plt.hist(valid_data, bins=50)
                plt.title(f'Band {i+1}')
        plt.tight_layout()
        plt.show()
        
        # Compute indices
        print("\nComputing spectral indices...")
        ndvi, evi, nbr, crswir = compute_indices(*reflectance_data, debug=True)
        
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
    check_reflectance_and_indices()