from __future__ import annotations
import os
import logging
import numpy as np
import rasterio
from datetime import datetime
from pathlib import Path
import sys
import re
from typing import List, Tuple
import shutil
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

def get_mosaic_paths_and_dates(base_path: Path, tile_id: str, years: List[str]) -> Tuple[List[Path], List[datetime]]:
    """Get mosaic paths and dates from the directory structure."""
    pattern = r'(\d{2})-(\d{2})_plus_minus_30_days'
    mosaic_paths = []
    dates = []
    
    for year in years:
        for item in sorted((base_path / year).glob('*/s2')):
            match = re.search(pattern, str(item))
            if match:
                month, day = map(int, match.groups())
                date = datetime(int(year), month, day)
                
                # Construct mosaic path
                mosaic_path = (base_path / year / 
                             f"{month:02d}-{day:02d}_plus_minus_30_days" / 
                             "s2" / f"s2_{tile_id}.tif")
                
                if mosaic_path.exists():
                    mosaic_paths.append(mosaic_path)
                    dates.append(date)
                
    return mosaic_paths, dates

def test_tile_inference():
    """Main test function."""
    logger = setup_logger()
    logger.info("Starting TileInference test")
    
    # Setup paths
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    data_dir = Path("data")  # Update with actual path
    model_path = project_dir / "model" / "best_model_with_bdforet_no_resampled_weights_h2_y1_iter10_scaled01_featuresfromRFECV_nf10_f1_0.9601.pkl"
    output_dir = current_dir / "test_outputs"
    
    # Clean output directory if it exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Test parameters
    years = ["2021", "2022"]
    tile_id = "EPSG2154_750000_6700000"
    
    try:
        # Get mosaic paths and dates
        mosaic_paths, dates = get_mosaic_paths_and_dates(data_dir / "mosaics", tile_id, years)
        if not mosaic_paths:
            raise ValueError("No mosaic files found")
        logger.info(f"Found {len(dates)} dates for processing")
        for date in dates:
            logger.info(f"Processing date: {date}")
        
        # Get DEM path
        dem_path = data_dir / "dem" / f"dem_{tile_id}.tif"
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        # Initialize TileInference
        logger.info("Initializing TileInference")
        tile_inference = TileInference(
            mosaic_paths=mosaic_paths,
            dates=dates,
            dem_path=dem_path,
            output_dir=output_dir,
            tile_id=tile_id,
            model_path=model_path,
            window_size=1024,  # Small window size for testing
            num_harmonics=2,
            max_iter=1,
            max_workers=4,  # Limit workers for testing
            logger=logger
        )
        
        # Run inference
        logger.info("Running inference")
        tile_inference.run()
        
        # Verify output and create visualization
        output_file = next(output_dir.glob(f"prob_map_tile_H*_{tile_id}.tif"))
        assert output_file.exists(), "Output file not created"
        
        # Read output probability map
        with rasterio.open(output_file) as src:
            prob_map = src.read(1).astype(np.float32) / 255.0  # Convert to [0,1]
            
            # Check output properties
            assert src.dtypes[0] == 'uint8', "Wrong output data type"
            assert prob_map.shape[0] > 0 and prob_map.shape[1] > 0, "Invalid dimensions"
            assert np.any(prob_map > 0), "Output appears to be empty"
            
            logger.info(f"Output statistics - Shape: {prob_map.shape}, "
                       f"Range: [{np.min(prob_map):.3f}, {np.max(prob_map):.3f}], "
                       f"Mean: {np.mean(prob_map):.3f}")
        
        # Create visualization
        logger.info("Creating visualization")
        
        # Get winter RGB image (using January data if available)
        winter_dates = [d for d in dates if d.month in [12, 1, 2]]
        if winter_dates:
            winter_date = winter_dates[len(winter_dates)//2]
            winter_idx = dates.index(winter_date)
            winter_path = mosaic_paths[winter_idx]
            with rasterio.open(winter_path) as src:
                rgb = src.read([3,2,1])  # Read RGB bands
                
        else:
            logger.warning("No winter dates available, using first available image")
            with rasterio.open(mosaic_paths[0]) as src:
                rgb = src.read([3,2,1])
        
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
                
        # Compute aspect for terrain shading
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