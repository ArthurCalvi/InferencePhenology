import rasterio
import os

def inspect_tif_file(filepath, block_size=1000):
    """
    Inspect a .tif file with block reading for large files
    block_size: size of the square block to read (e.g., 1000x1000 pixels)
    """
    print("="*80)
    print(f"ANALYZING FILE: {filepath}")
    print("="*80)
    
    with rasterio.open(filepath) as raster:
        # Basic metadata
        print("\n1. BASIC METADATA:")
        print("-"*40)
        print(f"Driver: {raster.driver}")
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        print(f"Number of bands: {raster.count}")
        print(f"Width: {raster.width}")
        print(f"Height: {raster.height}")
        print(f"Coordinate Reference System (CRS): {raster.crs}")
        
        # Read a block from the center of the image
        center_x = raster.width // 2
        center_y = raster.height // 2
        half_block = block_size // 2
        
        window = rasterio.windows.Window(
            center_x - half_block,
            center_y - half_block,
            min(block_size, raster.width - center_x + half_block),
            min(block_size, raster.height - center_y + half_block)
        )
        
        print(f"\n2. ANALYZING CENTRAL BLOCK ({block_size}x{block_size} pixels):")
        print("-"*40)
        
        # Band information
        print("\n3. BAND INFORMATION:")
        print("-"*40)
        for band_idx in range(1, raster.count + 1):
            block_data = raster.read(band_idx, window=window)
            print(f"\nBand {band_idx}:")
            print(f"  Description: {raster.descriptions[band_idx-1] if raster.descriptions[band_idx-1] else 'No description'}")
            print(f"  Band tags: {raster.tags(band_idx) if raster.tags(band_idx) else 'No tags'}")
            print(f"  Data type: {block_data.dtype}")
            print(f"  Sample block statistics:")
            print(f"    Min value: {block_data.min()}")
            print(f"    Max value: {block_data.max()}")
            print(f"    Mean value: {block_data.mean():.2f}")
            print(f"    Standard deviation: {block_data.std():.2f}")

if __name__ == "__main__":
    import rasterio
    dir_ = os.getenv('DSDIR')
    folder_dir = 'S2L3A_France2019'
    filename = 's2_20190115.tif'
    filepath = os.path.join(dir_, folder_dir, filename)
    
    try:
        inspect_tif_file(filepath, block_size=1000)  # Analyze a 1000x1000 pixel block
    except Exception as e:
        print(f"ERROR: An error occurred while inspecting the file:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")