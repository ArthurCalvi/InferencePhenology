from __future__ import annotations
import os
import logging
import numpy as np
import rasterio
from datetime import datetime
from pathlib import Path
import sys
import re
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_inference.inference import TileInference
from src_inference.plotting import plot_phenology_prediction
from src_inference.utils import get_aspect


def setup_logger() -> logging.Logger:
    """Configure logging for the test script."""
    logger = logging.getLogger('tile_inference_test')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def get_mosaic_paths_and_dates(base_path: Path, years: List[str]) -> Tuple[List[Path], List[datetime]]:
    """Get mosaic paths and dates from the DL phenology directory structure.
    
    Args:
        base_path (Path): Base path to the dl_phenology directory
        tile_id (str): ID of the tile to process
        years (List[str]): List of years to process
    
    Returns:
        Tuple[List[Path], List[datetime]]: Lists of mosaic paths and corresponding dates
    
    Raises:
        ValueError: If no files are found matching the pattern
    """
    # Pattern for DL phenology files: YYYY-MM-DD_[some number]_S2[A or B]_DL_PHENOLOGY.tif
    pattern = r'(\d{4})-(\d{2})-(\d{2})_.*_S2[AB]_DL_PHENOLOGY\.tif'
    mosaic_paths = []
    dates = []

    
    for year in years:
        # Look in the dl_phenology directory for files matching the pattern
        for file_path in sorted(base_path.glob(f"{year}*.tif")):
            print(file_path)
            match = re.search(pattern, file_path.name)
            if match:
                year_str, month_str, day_str = match.groups()
                
                # Convert strings to integers for datetime
                year = int(year_str)
                month = int(month_str)
                day = int(day_str)
                
                # Create datetime object and append to lists
                date = datetime(year, month, day)
                dates.append(date)
                mosaic_paths.append(file_path)
    
    # Sort both lists by date
    sorted_pairs = sorted(zip(dates, mosaic_paths))
    dates = [date for date, _ in sorted_pairs]
    mosaic_paths = [path for _, path in sorted_pairs]
    
    if not mosaic_paths:
        raise ValueError(f"No files found matching pattern in {base_path}")
        
    return mosaic_paths, dates

def test_tile_inference():
    """Main test function for tile inference processing."""
    logger = setup_logger()
    logger.info("Starting TileInference test")
    tile_id = 'test_chronos_tile'
    
    # Setup paths
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    tile_dir = Path("/Users/arthurcalvi/Data/species/validation_inference_france_1/Region12_20210101_20230101_fr-Corse-HauteCorse_Lat42.15_Lon9.38")  # Update with actual path to tile directory
    data_dir = tile_dir / "dl_phenology"
    model_path = project_dir / "model" / "best_model_with_bdforet_no_resampled_weights_h2_y1_iter10_scaled01_featuresfromRFECV_nf10_f1_0.9601.pkl"
    output_dir = tile_dir / "phenology" 
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Test parameters
    years = ["2021", "2022"]
    
    try:
        # Get mosaic paths and dates
        mosaic_paths, dates = get_mosaic_paths_and_dates(data_dir, years)
        if not mosaic_paths:
            raise ValueError("No mosaic files found")
        logger.info(f"Found {len(dates)} dates for processing")
        for date in dates:
            logger.info(f"Processing date: {date}")
        
        # Get DEM path
        dem_path = tile_dir / "dem" / f"dem.tif"
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        # Initialize TileInference with the correct bands order
        logger.info("Initializing TileInference")
        tile_inference = TileInference(
            mosaic_paths=mosaic_paths,
            dates=dates,
            dem_path=dem_path,
            output_dir=output_dir,
            tile_id=tile_id,
            model_path=model_path,
            window_size=1024,
            num_harmonics=2,
            max_iter=1,
            max_workers=4,
            bands_order_raster=[1, 2, 3, 4, 5],  # Updated bands order for DL phenology data
            logger=logger
        )
        
        # Run inference
        logger.info("Running inference")
        tile_inference.run()
        
        # Create visualization
        output_file = next(output_dir.glob(f"prob_map_tile_H*_{tile_id}.tif"))
        with rasterio.open(output_file) as src:
            prob_map = src.read(1).astype(np.float32) / 255.0
            
        # Get a winter image for visualization
        winter_dates = [d for d in dates if d.month in [12, 1, 2]]
        if winter_dates:
            winter_date = winter_dates[len(winter_dates)//2]
            winter_idx = dates.index(winter_date)
            winter_path = mosaic_paths[winter_idx]
        else:
            winter_path = mosaic_paths[0]
            
        # Read and process RGB bands
        with rasterio.open(winter_path) as src:
            rgb = src.read([1, 2, 3])  # Read first three bands for RGB
            
        # Normalize RGB
        rgb = rgb.astype(np.float32)
        for i in range(3):
            band = rgb[i]
            valid = ~np.isnan(band)
            if np.any(valid):
                min_val = np.percentile(band[valid], 2)
                max_val = np.percentile(band[valid], 98)
                rgb[i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
                rgb[i] = np.nan_to_num(rgb[i], 0)
                
        # Get aspect for terrain shading
        dem_data = rasterio.open(dem_path).read(1)
        aspect_data = get_aspect(dem_data, resolution=10.0)
        
        # Create and save visualization
        fig, axes = plot_phenology_prediction(rgb, prob_map, aspect_data, logger=logger)
        fig.savefig(output_dir / f"visualization_tile_{tile_id}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_tile_inference()