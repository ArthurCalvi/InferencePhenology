# test_rasterizer_visual.py

import sys
from pathlib import Path
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from shapely.geometry import box
import logging
from typing import Tuple, Optional
from tqdm import tqdm

# Add the comparison directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'comparison'))
from geodataframe_to_raster import GeoDataFrameRasterizer

def setup_logger() -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger('test_rasterizer')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def find_window_with_data(
    gdf: gpd.GeoDataFrame,
    reference_raster: Path,
    window_size: int = 1024
) -> Tuple[Window, gpd.GeoDataFrame]:
    """Find a window that contains vector data using spatial indexing.
    
    Args:
        gdf: Input GeoDataFrame
        reference_raster: Path to reference raster
        window_size: Size of the window to search
        
    Returns:
        Tuple of (window, clipped GeoDataFrame)
    """
    logger = logging.getLogger('test_rasterizer')
    
    # Create spatial index for the GeoDataFrame if it doesn't exist
    if not hasattr(gdf, 'sindex'):
        logger.info("Creating spatial index...")
        gdf.sindex
    
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        
        # Start from the center of the raster
        center_x = src.width // 2
        center_y = src.height // 2
        
        # Calculate total iterations for progress bar
        max_dim = max(src.width, src.height)
        total_iterations = sum(8 for offset in range(0, max_dim, window_size))  # 8 points per offset
        
        # Search in expanding squares from the center
        with tqdm(total=total_iterations, desc="Searching for window with data") as pbar:
            for offset in range(0, max_dim, window_size):
                for y in [center_y + offset, center_y - offset]:
                    for x in [center_x + offset, center_x - offset]:
                        if 0 <= x < src.width - window_size and 0 <= y < src.height - window_size:
                            # Create window
                            window = Window(x, y, window_size, window_size)
                            
                            # Get window bounds
                            window_transform = rasterio.windows.transform(window, transform)
                            bounds = rasterio.windows.bounds(window, window_transform)
                            window_box = box(*bounds)
                            
                            # Find intersecting features using spatial index
                            possible_matches_idx = list(gdf.sindex.intersection(bounds))
                            
                            if possible_matches_idx:
                                # Get the intersecting features
                                window_gdf = gdf.iloc[possible_matches_idx].copy()
                                # Clip to get precise intersection
                                window_gdf = window_gdf.clip(window_box)
                                
                                if not window_gdf.empty:
                                    logger.info(f"Found window with {len(window_gdf)} features at ({x}, {y})")
                                    return window, window_gdf
                        pbar.update(1)
                        
                            
    raise ValueError("No window with data found")

def plot_comparison(
    window: Window,
    window_gdf: gpd.GeoDataFrame,
    raster_data: np.ndarray,
    bounds: Tuple[float, float, float, float],
    value_map: dict
) -> None:
    """Plot vector and raster data side by side.
    
    Args:
        window: Rasterio window
        window_gdf: GeoDataFrame clipped to window
        raster_data: Rasterized data array
        bounds: Geographic bounds of the window
        value_map: Mapping of categories to raster values
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Create a color map for phenology types
    color_map = {
        'evergreen': 'darkgreen',
        'deciduous': 'lightgreen'
    }
    
    # Plot vector data
    window_gdf.plot(
        ax=ax1,
        column='phen_en',
        legend=True,
        categorical=True,
        cmap='RdYlGn',
        legend_kwds={'title': 'Phenology'}
    )
    ax1.set_title('Vector Data')
    ax1.set_xlim(bounds[0], bounds[2])
    ax1.set_ylim(bounds[1], bounds[3])
    
    # Plot raster data
    custom_cmap = plt.cm.colors.ListedColormap(['white', 'lightgreen', 'darkgreen'])
    bounds_cmap = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds_cmap, custom_cmap.N)
    
    im = ax2.imshow(
        raster_data,
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        cmap=custom_cmap,
        norm=norm
    )
    ax2.set_title('Rasterized Data')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['No Data', 'Deciduous', 'Evergreen'])
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to test and visualize rasterization."""
    logger = setup_logger()
    
    # Input data paths
    bdforet_path = Path('/Users/arthurcalvi/Data/Disturbances_maps/BDForet/bdforet_10_FF1_FF2_EN_year.parquet')
    reference_raster = Path('/Users/arthurcalvi/Data/species/DLT_2018_010m_fr_03035_v020/DLT_Dominant_Leaf_Type_France.tif')
    
    # Load data
    logger.info("Loading data...")
    gdf = gpd.read_parquet(bdforet_path)
    
    # Configuration
    value_map = {
        'evergreen': 2,
        'deciduous': 1
    }
    window_size = 1024
    
    # Find a window with data
    logger.info("Finding suitable window...")
    window, window_gdf = find_window_with_data(gdf, reference_raster, window_size)

    logger.info(f"Window bounds: {window}")
    #gdf head
    logger.info(f"Window GeoDataFrame head: {window_gdf.head()}")
    
    # Initialize rasterizer
    logger.info("Initializing rasterizer...")
    rasterizer = GeoDataFrameRasterizer(
        gdf=gdf,
        reference_raster=reference_raster,
        value_column='phen_en',
        value_map=value_map,
        window_size=window_size,
        logger=logger
    )
    
    # Process window
    logger.info("Processing window...")
    raster_data = rasterizer.process_window(window)
    
    # Get window bounds for plotting
    with rasterio.open(reference_raster) as src:
        bounds = rasterio.windows.bounds(window, src.transform)
    
    # Plot comparison
    logger.info("Creating comparison plot...")
    plot_comparison(window, window_gdf, raster_data, bounds, value_map)
    
    # Print statistics for both vector and raster data
    logger.info("\nVector data statistics:")
    logger.info(window_gdf['phen_en'].value_counts())
    
    logger.info("\nRaster statistics:")
    unique, counts = np.unique(raster_data, return_counts=True)
    stats = dict(zip(unique, counts))
    for value, count in stats.items():
        category = 'No Data' if value == 0 else ('Deciduous' if value == 1 else 'Evergreen')
        percentage = (count / raster_data.size) * 100
        logger.info(f"{category} (value {value}): {count} pixels ({percentage:.2f}%)")

if __name__ == '__main__':
    main()