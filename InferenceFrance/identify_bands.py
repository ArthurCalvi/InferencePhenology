import rasterio
import numpy as np
import matplotlib.pyplot as plt

def inspect_raster(raster_path, window_size=1024):
    """
    Inspect a raster file by reading a window and analyzing band statistics
    
    Parameters:
    raster_path: str, path to the raster file
    window_size: int, size of the window to read (default: 1024)
    """
    with rasterio.open(raster_path) as src:
        # Print basic raster information
        print(f"Raster dimensions: {src.width}x{src.height}")
        print(f"Number of bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"Coordinate Reference System: {src.crs}")
        
        # Calculate window position (center of the raster)
        window_col = (src.width - window_size) // 2
        window_row = (src.height - window_size) // 2
        
        # Define the window
        window = rasterio.windows.Window(
            window_col, 
            window_row, 
            window_size, 
            window_size
        )
        
        # Read all bands in the window
        data = src.read(window=window) / 10000 
        
        # Analyze each band
        print("\nBand Statistics:")
        print("Band | Min     | Max     | Mean    | Std     | # Zeros | # Valid")
        print("-" * 65)
        
        for band_idx in range(src.count):
            band_data = data[band_idx]
            valid_mask = band_data != 0
            
            if valid_mask.any():
                band_stats = {
                    'min': np.min(band_data[valid_mask]),
                    'max': np.max(band_data[valid_mask]),
                    'mean': np.mean(band_data[valid_mask]),
                    'std': np.std(band_data[valid_mask]),
                    'zeros': np.sum(band_data == 0),
                    'valid': np.sum(valid_mask)
                }
            else:
                band_stats = {
                    'min': 0, 'max': 0, 'mean': 0, 'std': 0,
                    'zeros': window_size * window_size,
                    'valid': 0
                }
            
            print(f"{band_idx + 1:4d} | {band_stats['min']:7.1f} | {band_stats['max']:7.1f} | "
                  f"{band_stats['mean']:7.1f} | {band_stats['std']:7.1f} | {band_stats['zeros']:7d} | "
                  f"{band_stats['valid']:7d}")
        
        # Plot histograms for each band
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        for band_idx in range(src.count):
            band_data = data[band_idx]
            valid_data = band_data[band_data != 0]
            
            if len(valid_data) > 0:
                axes[band_idx].hist(valid_data.ravel(), bins=50, alpha=0.5)
                axes[band_idx].set_title(f'Band {band_idx + 1}')
                axes[band_idx].set_xlabel('Pixel Value')
                axes[band_idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Return the window data for further analysis if needed
        return data

def identify_sentinel2_bands(data):
    """
    Attempt to identify Sentinel-2 bands based on value ranges and patterns
    
    Parameters:
    data: numpy array of shape (bands, height, width)
    """
    # Typical value ranges and characteristics of Sentinel-2 bands
    band_characteristics = []
    
    for band_idx in range(data.shape[0]):
        band_data = data[band_idx]
        valid_data = band_data[band_data != 0]
        
        if len(valid_data) == 0:
            continue
            
        stats = {
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
        }
        
        # Look for patterns that might indicate specific bands
        if stats['max'] > 0.8:  # Possible blue, green, red, or NIR bands
            if stats['mean'] < 0.1:
                band_characteristics.append(f"Band {band_idx + 1}: Possibly Blue (B2) or Green (B3)")
            elif stats['mean'] < 0.15:
                band_characteristics.append(f"Band {band_idx + 1}: Possibly Red (B4)")
            else:
                band_characteristics.append(f"Band {band_idx + 1}: Possibly NIR (B8 or B8A)")
        elif stats['max'] < 0.5:  # Possible SWIR bands
            band_characteristics.append(f"Band {band_idx + 1}: Possibly SWIR (B11 or B12)")
        else:
            band_characteristics.append(f"Band {band_idx + 1}: Uncertain")
    
    print("\nPossible Band Identifications:")
    for characteristic in band_characteristics:
        print(characteristic)

import os 
# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define model path relative to project root
# DIR_RASTER = os.path.join(PROJECT_ROOT, 'extract_test', 'mosaics', 's2_20190115_4096centralwindow.tif')
DIR_RASTER = os.path.join(PROJECT_ROOT, 'geefetch_test', 's2_EPSG2154_450000_6950000.tif')
# Inspect the raster
data = inspect_raster(DIR_RASTER)

# Try to identify the bands
identify_sentinel2_bands(data)