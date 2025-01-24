import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Define input and output directories
input_dir = Path('/Users/arthurcalvi/Repo/InferencePhenology/jeanzay')
output_dir = Path('/Users/arthurcalvi/Data/Disturbances_maps/phenology/classification')

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Get list of all .tif files
tif_files = list(input_dir.glob('*.tif'))

# Process each file with progress bar
for tif_file in tqdm(tif_files, desc="Processing files"):
    # Read the input raster
    with rasterio.open(tif_file) as src:
        # Read the data
        data = src.read(1)  # Read first band
        
        # Create binary classification (1: deciduous, 2: evergreen)
        classified = np.where(data >= 128, 2, 1).astype('uint8')
        
        # Prepare the output file path
        output_file = output_dir / tif_file.name
        
        # Copy the metadata from the source
        profile = src.profile.copy()
        
        # Update the metadata for the output
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'deflate',
            'predictor': 2,
            'zlevel': 9
        })
        
        # Write the output file
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(classified, 1)