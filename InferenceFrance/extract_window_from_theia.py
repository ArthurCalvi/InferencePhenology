import os
import rasterio
from pathlib import Path

# Set up input and output directories
dir_ = os.getenv('DSDIR')
folder_dir = 'S2L3A_France2019'
input_dir = os.path.join(dir_, folder_dir)
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
    # Calculate the starting coordinates for the central window
    start_x = (width - window_size) // 2
    start_y = (height - window_size) // 2
    
    # Create the window
    window = rasterio.windows.Window(
        start_x, 
        start_y, 
        window_size, 
        window_size
    )
    
    return window

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

def process_mosaics(window_size=4096):
    """Process all mosaic files in the input directory."""
    # Create output directories
    mosaics_output_dir, flg_output_dir = create_output_dirs()
    
    # Process main mosaics
    main_files = [f for f in os.listdir(input_dir) if f.endswith('.tif') and not f.startswith('.')]
    
    # Get first file to determine dimensions
    with rasterio.open(os.path.join(input_dir, main_files[0])) as src:
        window = get_central_window(src.width, src.height, window_size)
    
    # Process main mosaics
    for filename in main_files:
        if filename.endswith('.tif') and not filename.startswith('.'):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace('.tif', f'_{window_size}centralwindow.tif')
            output_path = os.path.join(mosaics_output_dir, output_filename)
            
            print(f"Processing {filename}...")
            extract_window(input_path, output_path, window)
    
    # Process FLG mosaics
    flg_dir = os.path.join(input_dir, 'FLG')
    if os.path.exists(flg_dir):
        for filename in os.listdir(flg_dir):
            if filename.endswith('.tif') and not filename.startswith('.'):
                input_path = os.path.join(flg_dir, filename)
                output_filename = filename.replace('.tif', f'_{window_size}centralwindow.tif')
                output_path = os.path.join(flg_output_dir, output_filename)
                
                print(f"Processing FLG/{filename}...")
                extract_window(input_path, output_path, window)

if __name__ == "__main__":
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    process_mosaics()