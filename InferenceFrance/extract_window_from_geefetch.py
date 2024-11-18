import os
import rasterio
from pathlib import Path
from datetime import datetime

# Set up input and output directories
dir_ = os.getenv('SCRATCH')
input_base_dir = Path(dir_) / 'all_year_france'
output_dir = '/linkhome/rech/gennjv01/uyr48jk/work/InferencePhenology'

def create_output_dirs():
    """Create output directory structure if it doesn't exist."""
    base_dir = Path(output_dir) / "extract_test"
    mosaics_dir = base_dir / "mosaics"
    flg_dir = base_dir / "FLG"
    
    # Create directories
    for dir_path in [base_dir, mosaics_dir, flg_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return mosaics_dir, flg_dir

def get_central_window(width, height, window_size=4096):
    """Calculate the central window coordinates."""
    start_x = (width - window_size) // 2
    start_y = (height - window_size) // 2
    
    return rasterio.windows.Window(
        start_x,
        start_y,
        window_size,
        window_size
    )

def extract_window(input_path, output_path, window):
    """Extract and save a window from the input raster."""
    with rasterio.open(input_path) as src:
        # Read the data only for the specified window
        data = src.read(window=window)
        
        # Update the transform for the new window
        transform = src.window_transform(window)
        
        # Create output profile based on input
        profile = src.profile.copy()
        profile.update({
            'height': window.height,
            'width': window.width,
            'transform': transform
        })
        
        # Write the windowed data
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

def get_period_date(period_dir):
    """Extract date from period directory name."""
    # Convert directory name format to date
    month = int(period_dir.split('-')[0])
    return f"{month:02d}"

def process_mosaics(window_size=4096):
    """Process all mosaic files for 2022 and 2023."""
    # Create output directories
    mosaics_output_dir, flg_output_dir = create_output_dirs()
    
    # Process years 2022 and 2023
    years = ['2022', '2023']
    
    # Get window dimensions from first available file
    window = None
    
    for year in years:
        year_dir = input_base_dir / year
        if not year_dir.exists():
            print(f"Directory for year {year} not found")
            continue
            
        # Process each period
        for period in sorted(year_dir.iterdir()):
            if not period.is_dir():
                continue
                
            period_date = get_period_date(period.name)
            s2_dir = period / 's2'
            
            if not s2_dir.exists():
                print(f"S2 directory not found for {period}")
                continue
            
            # Process main mosaic files
            for file_path in s2_dir.glob('*.tif'):
                if file_path.name.startswith('.'):
                    continue
                    
                # Get window dimensions from first file if not already done
                if window is None:
                    with rasterio.open(file_path) as src:
                        window = get_central_window(src.width, src.height, window_size)
                
                # Create output filename with year and period info
                output_filename = f"{file_path.stem}_{period_date}_{year}_{window_size}centralwindow.tif"
                output_path = mosaics_output_dir / output_filename
                
                print(f"Processing {year}/{period.name}/{file_path.name}...")
                extract_window(str(file_path), str(output_path), window)
            
            # Process FLG files if they exist
            flg_dir = s2_dir / 'FLG'
            if flg_dir.exists():
                for file_path in flg_dir.glob('*.tif'):
                    if file_path.name.startswith('.'):
                        continue
                        
                    output_filename = f"{file_path.stem}_{period_date}_{year}_{window_size}centralwindow.tif"
                    output_path = flg_output_dir / output_filename
                    
                    print(f"Processing FLG/{year}/{period.name}/{file_path.name}...")
                    extract_window(str(file_path), str(output_path), window)

if __name__ == "__main__":
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_dir}")
    process_mosaics()