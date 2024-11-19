import os
import rasterio
import multidem
import warnings
from rasterio.warp import transform_bounds, reproject
import numpy as np

def generate_dem_from_mosaic(reference_dir, output_dir):
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
        # Use the first TIF file as reference
        print(f"Using reference file: {reference_dir}")
        tile_id = os.path.basename(reference_dir).split('s2')[1].split('.tif')[0][1:]
        
        # Open the reference raster
        with rasterio.open(reference_dir) as raster:
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
            dem_path = os.path.join(output_dir, f'dem_{tile_id}.tif')
            with rasterio.open(dem_path, 'w', **profile) as dest:
                dest.write(dem.astype('float32'))
            
            print(f"DEM file successfully created at: {dem_path}")
            return True
            
    except Exception as e:
        print(f"Error generating DEM: {str(e)}")
        return False

if __name__ == "__main__":
    # Assuming the script is run from the same directory as the mosaics folder
    reference_dir = "/Users/arthurcalvi/Repo/InferencePhenology/data/mosaics/2021/01-01_plus_minus_30_days/s2/s2_EPSG2154_750000_6650000.tif"
    output_dir = "/Users/arthurcalvi/Repo/InferencePhenology/data/dem"
    
    if not os.path.exists(reference_dir):
        print(f"Error: Directory '{reference_dir}' not found.")
    else:
        success = generate_dem_from_mosaic(reference_dir, output_dir)
        if success:
            print("DEM generation completed successfully.")
        else:
            print("DEM generation failed.")