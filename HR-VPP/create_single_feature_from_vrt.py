import os
import glob
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_reference_info(rgb_file):
    """Get CRS and bounds from a reference RGB file."""
    with rasterio.open(rgb_file) as src:
        bounds = src.bounds
        if (bounds.left >= bounds.right) or (bounds.bottom >= bounds.top):
            raise ValueError(f"Invalid bounds in {rgb_file}: {bounds}")
            
        return {
            'crs': src.crs,
            'bounds': bounds,
            'transform': src.transform,
            'height': src.height,
            'width': src.width
        }

def check_data_validity(data, source_name):
    """Check if the data array contains valid values."""
    if data is None:
        raise ValueError(f"No data read from {source_name}")
    
    if data.size == 0:
        raise ValueError(f"Empty data array from {source_name}")
    
    if np.all(data == 0):
        raise ValueError(f"Data from {source_name} contains only zeros")
    
    if np.all(np.isnan(data)):
        raise ValueError(f"Data from {source_name} contains only NaN values")
    
    # Print some statistics
    print(f"\nData statistics for {source_name}:")
    print(f"Min: {np.nanmin(data)}")
    print(f"Max: {np.nanmax(data)}")
    print(f"Mean: {np.nanmean(data)}")
    print(f"Number of zeros: {np.sum(data == 0)}")
    print(f"Number of NaNs: {np.sum(np.isnan(data))}")
    print(f"Unique values: {len(np.unique(data))}")
    
    return True

from rasterio.warp import transform_bounds

def process_vpp_feature(vrt_path, ref_info, output_path):
    """Process a VPP feature using VRT with data validation."""
    with rasterio.open(vrt_path) as src:
        # Print source and target information
        print(f"\nProcessing Information:")
        print(f"Source CRS: {src.crs}")
        print(f"Target CRS: {ref_info['crs']}")
        
        # Transform bounds from target CRS to source CRS
        vrt_bounds = transform_bounds(
            ref_info['crs'],  # from target CRS
            src.crs,          # to VRT CRS
            *ref_info['bounds']
        )
        
        print(f"Original bounds (in target CRS): {ref_info['bounds']}")
        print(f"Transformed bounds (in VRT CRS): {vrt_bounds}")
        
        # Create output profile based on reference info
        profile = src.profile.copy()
        profile.update({
            'crs': ref_info['crs'],
            'transform': ref_info['transform'],
            'width': ref_info['width'],
            'height': ref_info['height'],
            'driver': 'GTiff',
            'nodata': src.nodata if src.nodata is not None else None,
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        })
        
        # First create a VRT window in source CRS
        with WarpedVRT(src,
                      crs=src.crs,  # Keep source CRS
                      transform=src.transform,  # Keep source transform
                      width=src.width,
                      height=src.height,
                      window=from_bounds(*vrt_bounds, src.transform)) as window_vrt:
            
            # Now create a second VRT to reproject the window
            with WarpedVRT(window_vrt,
                          crs=ref_info['crs'],
                          transform=ref_info['transform'],
                          width=ref_info['width'],
                          height=ref_info['height'],
                          resampling=Resampling.bilinear) as warped_vrt:
                
                print("\nReading and reprojecting data...")
                data = warped_vrt.read(1)
                
                try:
                    check_data_validity(data, "warped VRT")
                except ValueError as e:
                    print(f"Warning in warped data: {str(e)}")
                
                # Write to output file
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data, 1)
                    
                    # Verify output
                    print("\nChecking output data...")
                    try:
                        output_data = dst.read(1)
                        check_data_validity(output_data, "output file")
                    except ValueError as e:
                        print(f"Warning in output data: {str(e)}")
                        print("You might want to check if the input data is properly aligned with your reference tiles.")
                        
def main(dataset_dir, vrt_path, feature_name):
    """
    Main function to process VPP features for all tiles.
    
    Args:
        dataset_dir (str): Path to the dataset directory containing tiles
        vrt_path (str): Path to the VRT file
        feature_name (str): Name of the feature (e.g., 'EOSD')
    """
    # Check if VRT exists
    if not os.path.exists(vrt_path):
        raise ValueError(f"VRT file not found: {vrt_path}")
    
    # Get all tile directories
    tile_dirs = [d for d in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith('tile_')]
    
    if not tile_dirs:
        raise ValueError(f"No tile directories found in {dataset_dir}")
    
    # Sort tile directories for consistent processing order
    tile_dirs.sort()
    
    for tile_dir in tqdm(tile_dirs, desc="Processing tiles"):
        try:
            print(f"\nProcessing tile: {tile_dir}")
            
            # Get path to RGB directory
            rgb_dir = os.path.join(dataset_dir, tile_dir, 'rgb')
            if not os.path.exists(rgb_dir):
                print(f"Warning: RGB directory not found for {tile_dir}")
                continue
            
            # Get first RGB file
            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.tif')))
            if not rgb_files:
                print(f"Warning: No RGB files found in {tile_dir}")
                continue
            
            # Get reference information from first RGB file
            ref_info = get_reference_info(rgb_files[0])
            
            # Create features directory if it doesn't exist
            features_dir = os.path.join(dataset_dir, tile_dir, 'features')
            os.makedirs(features_dir, exist_ok=True)
            
            # Process VPP feature
            output_path = os.path.join(features_dir, f"{feature_name}.tif")
            try:
                process_vpp_feature(vrt_path, ref_info, output_path)
            except Exception as e:
                print(f"Error processing VPP feature for {tile_dir}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Error processing tile {tile_dir}: {str(e)}")
            continue

if __name__ == "__main__":
    # Example usage
    dataset_dir = "/Users/arthurcalvi/Data/species/validation/tiles_2_5_km"
    vrt_path = "/Users/arthurcalvi/Data/species/HR-VPP/Results-2/VRT/france_mosaic.vrt"
    feature_name = "EOSD"  # Replace with your feature name
    
    main(dataset_dir, vrt_path, feature_name)


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Process VPP features for tiles.')
#     parser.add_argument('dataset_dir', help='Path to the dataset directory containing tiles')
#     parser.add_argument('vrt_path', help='Path to the VRT file')
#     parser.add_argument('feature_name', help='Name of the feature (e.g., EOSD)')
    
#     args = parser.parse_args()
    
#     main(args.dataset_dir, args.vrt_path, args.feature_name)