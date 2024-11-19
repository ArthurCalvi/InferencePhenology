from __future__ import annotations
import os
import logging
import numpy as np
import rasterio
from datetime import datetime
from pathlib import Path
import sys
import re
from typing import List
import matplotlib.pyplot as plt
import shutil

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_inference.inference import FolderInference, TileData
from src_inference.plotting import plot_phenology_prediction
from src_inference.utils import get_aspect

def setup_logger() -> logging.Logger:
    """Configure logging for the test script."""
    logger = logging.getLogger('folder_inference_test')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def get_tile_data(data_dir: Path, tile_id: str, years: List[str]) -> TileData:
    """Get all paths and dates for a specific tile."""
    pattern = r'(\d{2})-(\d{2})_plus_minus_30_days'
    mosaic_paths = []
    dates = []
    
    # Get mosaic paths and dates
    for year in years:
        for item in sorted((data_dir / "mosaics" / year).glob('*/s2')):
            match = re.search(pattern, str(item))
            if match:
                month, day = map(int, match.groups())
                mosaic_path = item / f"s2_{tile_id}.tif"
                
                if mosaic_path.exists():
                    mosaic_paths.append(mosaic_path)
                    dates.append(datetime(int(year), month, day))
    
    # Get DEM path
    dem_path = data_dir / "dem" / f"dem_{tile_id}.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
    return TileData(
        tile_id=tile_id,
        mosaic_paths=mosaic_paths,
        dates=dates,
        dem_path=dem_path
    )

def create_visualization(tile_id: str, output_dir: Path, data_dir: Path, logger: logging.Logger) -> None:
    """Create visualization for a processed tile."""
    logger.info(f"Creating visualization for tile {tile_id}")
    
    # Find the output probability map
    prob_maps = list(output_dir.glob(f"prob_map_tile_H*_{tile_id}.tif"))
    if not prob_maps:
        raise FileNotFoundError(f"No probability map found for tile {tile_id}")
    prob_map_path = prob_maps[0]
    
    # Load probability map
    with rasterio.open(prob_map_path) as src:
        prob_map = src.read(1).astype(np.float32) / 255.0
        
    # Find a winter image for RGB visualization
    winter_months = [12, 1, 2]
    winter_path = None
    for year in ["2021", "2022"]:
        if winter_path is not None:
            break
        for month in winter_months:
            for day in range(1, 32):
                potential_path = (data_dir / "mosaics" / year / 
                                f"{month:02d}-{day:02d}_plus_minus_30_days" / 
                                "s2" / f"s2_{tile_id}.tif")
                if potential_path.exists():
                    winter_path = potential_path
                    break
    
    if winter_path is None:
        # Use first available image if no winter image found
        winter_path = next(data_dir.glob(f"**/*s2_{tile_id}.tif"))
        
    # Read RGB bands
    with rasterio.open(winter_path) as src:
        rgb = src.read([3,2,1])  # Read RGB bands
        
    # Normalize RGB to [0,1] using 2-98 percentile
    rgb = rgb.astype(np.float32)
    for i in range(3):
        band = rgb[i]
        valid = ~np.isnan(band)
        if np.any(valid):
            min_val = np.percentile(band[valid], 2)
            max_val = np.percentile(band[valid], 98)
            rgb[i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
            rgb[i] = np.nan_to_num(rgb[i], 0)
            
    # Read DEM and compute aspect
    dem_path = data_dir / "dem" / f"dem_{tile_id}.tif"
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    aspect_data = get_aspect(dem, resolution=10.0)
    
    # Create and save visualization
    fig, axes = plot_phenology_prediction(rgb, prob_map, aspect_data, logger=logger)
    fig.savefig(output_dir / f"visualization_{tile_id}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main test function."""
    logger = setup_logger()
    logger.info("Starting FolderInference test")
    
    # Setup paths
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    data_dir = Path("data")  # Update with actual path
    model_path = project_dir / "model" / "best_model_with_bdforet_no_resampled_weights_h2_y1_iter10_scaled01_featuresfromRFECV_nf10_f1_0.9601.pkl"
    output_dir = current_dir / "test_outputs"
    
    # Clean output directory if it exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test parameters
    years = ["2021", "2022"]
    tile_ids = [
        "EPSG2154_750000_6650000",
        "EPSG2154_750000_6700000"
    ]
    
    try:
        # Prepare tile data
        tiles_data = []
        for tile_id in tile_ids:
            try:
                tile_data = get_tile_data(data_dir, tile_id, years)
                tiles_data.append(tile_data)
                logger.info(f"Found {len(tile_data.dates)} dates for tile {tile_id}")
            except Exception as e:
                logger.error(f"Error preparing data for tile {tile_id}: {str(e)}")
                continue
        
        if not tiles_data:
            raise ValueError("No valid tile data found")
        
        # Initialize FolderInference
        logger.info("Initializing FolderInference")
        folder_inference = FolderInference(
            tiles_data=tiles_data,
            output_dir=output_dir,
            model_path=model_path,
            window_size=512,  # Small window size for testing
            workers_per_tile=4,  # Limit workers for testing
            num_harmonics=2,
            max_iter=1,
            logger=logger
        )
        
        # Save configurations
        folder_inference.save_configs()
        
        # Process each tile
        for i in range(folder_inference.get_num_tiles()):
            logger.info(f"Processing tile {i+1}/{folder_inference.get_num_tiles()}")
            folder_inference.process_single_tile(i)
            
            # Check progress
            progress = folder_inference.get_progress()
            logger.info(f"Progress: {progress['progress_percentage']:.1f}% complete "
                       f"({progress['completed_tiles']}/{progress['total_tiles']} tiles)")
        
        # Create visualizations
        for tile_id in tile_ids:
            try:
                create_visualization(tile_id, output_dir, data_dir, logger)
                logger.info(f"Created visualization for tile {tile_id}")
            except Exception as e:
                logger.error(f"Error creating visualization for tile {tile_id}: {str(e)}")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()