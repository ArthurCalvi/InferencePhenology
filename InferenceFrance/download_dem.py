import os
import rasterio
import multidem
import warnings
from rasterio.warp import transform_bounds, reproject
import numpy as np

def generate_dem_from_mosaic(mosaics_dir):
    """
    Generate a DEM file based on the extent of the first mosaic tile found.
    
    Parameters:
    mosaics_dir (str): Path to the directory containing mosaic TIF files
    
    Returns:
    bool: True if successful, False otherwise
    """
    warnings.filterwarnings("ignore")
    
    # Get the first .tif file from the mosaics directory
    try:
        tif_files = [f for f in os.listdir(mosaics_dir) if f.endswith('.tif')]
        if not tif_files:
            print("No TIF files found in the mosaics directory.")
            return False
            
        # Use the first TIF file as reference
        reference_tif = os.path.join(mosaics_dir, tif_files[0])
        print(f"Using reference file: {reference_tif}")
        
        # Open the reference raster
        with rasterio.open(reference_tif) as raster:
            profile = raster.profile
            target_transform = raster.transform
            
            # Transform bounds to WGS84 (EPSG:4326) for multidem
            target_bounds = transform_bounds(
                raster.crs, 
                {'init': 'EPSG:4326'}, 
                *raster.bounds
            )
            
            # Get DEM data using multidem
            dem, transform, crs = multidem.crop(
                target_bounds, 
                source="SRTM30", 
                datum="orthometric"
            )
            
            # Reproject DEM to match the reference raster
            dem, transform = reproject(
                dem,
                np.zeros((1, *raster.shape)),
                src_transform=transform,
                src_crs=crs,
                dst_crs={'init': str(raster.crs)},
                dst_transform=target_transform,
                dst_shape=raster.shape
            )
            
            # Update profile for the output DEM
            profile.update({
                'count': 1,
                'dtype': 'float32',
                'transform': target_transform
            })
            
            # Write the DEM file
            dem_path = os.path.join(os.path.dirname(mosaics_dir), 'dem.tif')
            with rasterio.open(dem_path, 'w', **profile) as dest:
                dest.write(dem.astype('float32'))
            
            print(f"DEM file successfully created at: {dem_path}")
            return True
            
    except Exception as e:
        print(f"Error generating DEM: {str(e)}")
        return False

if __name__ == "__main__":
    # Assuming the script is run from the same directory as the mosaics folder
    mosaics_dir = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test"
    
    if not os.path.exists(mosaics_dir):
        print(f"Error: Directory '{mosaics_dir}' not found.")
    else:
        success = generate_dem_from_mosaic(mosaics_dir)
        if success:
            print("DEM generation completed successfully.")
        else:
            print("DEM generation failed.")