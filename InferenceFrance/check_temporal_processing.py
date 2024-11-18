import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path to import utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import to_reflectance, compute_indices, get_paired_files

def plot_time_series(dates, values, title, band_names=None):
    """Helper function to plot time series."""
    plt.figure(figsize=(15, 5))
    if band_names is None:
        plt.plot(dates, values, 'o-')
    else:
        for i, band_name in enumerate(band_names):
            plt.plot(dates, values[:, i], 'o-', label=band_name)
        plt.legend()
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def check_temporal_processing():
    """
    Test to_reflectance and compute_indices functions across multiple dates.
    """
    base_dir = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test"
    
    # Get all mosaic files
    mosaic_paths, _ = get_paired_files(base_dir)
    dates = [datetime.strptime(os.path.basename(p).split('.tif')[0].split('_')[1], '%Y%m')
             for p in mosaic_paths]
    
    print(f"\nFound {len(mosaic_paths)} mosaic files:")
    for date, path in zip(dates, mosaic_paths):
        print(f"  {date.strftime('%Y-%m')}: {os.path.basename(path)}")
    
    # Define a window to process
    window = rasterio.windows.Window(0, 0, 1024, 1024)
    print(f"\nProcessing window: {window}")
    
    # Store temporal data
    reflectance_stats = []
    indices_stats = []
    
    # Process each date
    for date, mosaic_path in zip(dates, mosaic_paths):
        print(f"\n=== Processing {date.strftime('%Y-%m')} ===")
        
        with rasterio.open(mosaic_path) as src:
            # Read data
            data = src.read([1,2,3,4,9,10], window=window)
            nodata = src.profile.get('nodata', np.nan)
            
            # Convert to reflectance
            reflectance = to_reflectance(data, nodata=nodata)
            
            # Store reflectance statistics
            band_stats = []
            for i in range(reflectance.shape[0]):
                valid_data = reflectance[i][~np.isnan(reflectance[i])]
                if len(valid_data) > 0:
                    stats = {
                        'min': np.min(valid_data),
                        'max': np.max(valid_data),
                        'mean': np.mean(valid_data),
                        'std': np.std(valid_data),
                        'valid_pixels': len(valid_data),
                        'nan_percentage': np.mean(np.isnan(reflectance[i])) * 100
                    }
                else:
                    stats = {
                        'min': np.nan, 'max': np.nan, 'mean': np.nan, 
                        'std': np.nan, 'valid_pixels': 0, 'nan_percentage': 100
                    }
                band_stats.append(stats)
            reflectance_stats.append(band_stats)
            
            # Compute indices
            ndvi, evi, nbr, crswir = compute_indices(*reflectance, debug=False)
            
            # Store indices statistics
            indices_data = {
                'ndvi': ndvi,
                'evi': evi,
                'nbr': nbr,
                'crswir': crswir
            }
            
            date_indices_stats = {}
            for name, index in indices_data.items():
                valid_data = index[~np.isnan(index)]
                if len(valid_data) > 0:
                    stats = {
                        'min': np.min(valid_data),
                        'max': np.max(valid_data),
                        'mean': np.mean(valid_data),
                        'std': np.std(valid_data),
                        'valid_pixels': len(valid_data),
                        'nan_percentage': np.mean(np.isnan(index)) * 100
                    }
                else:
                    stats = {
                        'min': np.nan, 'max': np.nan, 'mean': np.nan,
                        'std': np.nan, 'valid_pixels': 0, 'nan_percentage': 100
                    }
                date_indices_stats[name] = stats
            indices_stats.append(date_indices_stats)
    
    # Plot temporal evolution of reflectance statistics
    band_names = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    for stat in ['mean', 'nan_percentage']:
        values = np.array([[band[stat] for band in date_stats] 
                          for date_stats in reflectance_stats])
        plot_time_series(dates, values, 
                        f'Temporal evolution of reflectance {stat}',
                        band_names)
    
    # Plot temporal evolution of indices
    for stat in ['mean', 'nan_percentage']:
        values = np.array([[date_stats[index][stat] for index in ['ndvi', 'evi', 'nbr', 'crswir']]
                          for date_stats in indices_stats])
        plot_time_series(dates, values,
                        f'Temporal evolution of indices {stat}',
                        ['NDVI', 'EVI', 'NBR', 'CRSWIR'])
    
    # Print detailed statistics
    print("\n=== Detailed Statistics ===")
    for i, date in enumerate(dates):
        print(f"\nDate: {date.strftime('%Y-%m')}")
        
        print("\nReflectance:")
        for j, band in enumerate(band_names):
            stats = reflectance_stats[i][j]
            print(f"\n{band}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Valid pixels: {stats['valid_pixels']}")
            print(f"  NaN%: {stats['nan_percentage']:.1f}%")
        
        print("\nIndices:")
        for index in ['ndvi', 'evi', 'nbr', 'crswir']:
            stats = indices_stats[i][index]
            print(f"\n{index.upper()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Valid pixels: {stats['valid_pixels']}")
            print(f"  NaN%: {stats['nan_percentage']:.1f}%")

if __name__ == "__main__":
    check_temporal_processing()