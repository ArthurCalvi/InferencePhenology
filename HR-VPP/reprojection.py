import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import subprocess

def check_file_permissions(filepath):
    """Check file permissions and readability."""
    print(f"\nChecking file permissions for: {filepath}")
    print(f"File exists: {os.path.exists(filepath)}")
    print(f"File is readable: {os.access(filepath, os.R_OK)}")
    print(f"File size: {os.path.getsize(filepath)} bytes")
    
    # Try to read file info using gdalinfo
    try:
        result = subprocess.run(['gdalinfo', filepath], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            print("GDAL can read the file")
            # Print first few lines of gdalinfo output
            print("\nGDAL file info:")
            print('\n'.join(result.stdout.split('\n')[:10]))
        else:
            print("GDAL cannot read the file")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error running gdalinfo: {str(e)}")

def reproject_raster(src_path, dst_path, dst_crs='EPSG:2154'):
    """
    Reproject a raster file to a new CRS
    
    Parameters:
    src_path : str, path to source file
    dst_path : str, path to destination file
    dst_crs : str, target CRS
    """
    # Check file before opening
    check_file_permissions(src_path)
    
    # Read source data and metadata
    with rasterio.open(src_path) as src:
        print(f"\nSource file: {src_path}")
        print(f"Source CRS: {src.crs}")
        print(f"Source dtype: {src.dtypes[0]}")
        print(f"Source nodata: {src.nodata}")
        print(f"Source driver: {src.driver}")
        print(f"Source profile: {src.profile}")
        
        # Read the data
        print("\nReading source data...")
        source_data = src.read(1)
        print(f"Source data shape: {source_data.shape}")
        print(f"Source data range: [{np.min(source_data)}, {np.max(source_data)}]")
        print(f"Number of unique values: {len(np.unique(source_data))}")

        # Calculate transform
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata,
            'driver': 'GTiff',
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        })
        
        print(f"\nOutput profile: {kwargs}")
        
        # Create destination dataset
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            print("\nReprojecting data...")
            
            # Reproject
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                nodata=src.nodata
            )
    
    # Verify output in a separate operation
    print("\nVerifying output...")
    with rasterio.open(dst_path) as dst:
        out_data = dst.read(1)
        print(f"Output data shape: {out_data.shape}")
        print(f"Output data range: [{np.min(out_data)}, {np.max(out_data)}]")
        print(f"Number of unique values: {len(np.unique(out_data))}")

def main(directory):
    """
    Main function to process all tif files in a directory.
    
    Parameters:
    directory : str, path to directory containing tif files
    """
    # Convert to absolute path
    directory = os.path.abspath(directory)
    
    # Create tmp directory inside the input directory
    tmp_dir = os.path.join(directory, 'tmp_vrt')
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Get all tif files in directory
    tif_files = glob.glob(os.path.join(directory, '*.tif'))
    
    if not tif_files:
        print(f"No .tif files found in directory: {directory}")
        return
    
    print(f"Found {len(tif_files)} .tif files in {directory}")
    
    # Process each file
    for tif_file in tqdm(tif_files, desc="Processing files"):
        try:
            src_path = tif_file
            dst_path = os.path.join(tmp_dir, f'warped_{os.path.basename(tif_file)}')
            
            print(f"\nProcessing: {os.path.basename(tif_file)}")
            reproject_raster(src_path, dst_path)
            
        except Exception as e:
            print(f"Error processing {tif_file}: {str(e)}")
            continue

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Reproject TIF files in a directory')
    # parser.add_argument('directory', help='Directory containing TIF files')
    
    # args = parser.parse_args()
    
    # main(args.directory)
    directory = '/Users/arthurcalvi/Data/species/HR-VPP/Results-2'
    main(directory)