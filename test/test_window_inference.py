from __future__ import annotations
import os
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re
from typing import List, Tuple

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_inference.inference import WindowInference, BandData
from src_inference.plotting import plot_phenology_prediction
from src_inference.utils import get_aspect

def setup_logger() -> logging.Logger:
    """Configure logging for the test script."""
    logger = logging.getLogger('window_inference_test')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def get_time_series_dates(base_path: Path, years: List[str]) -> List[datetime]:
    """
    Get list of available dates from the directory structure for multiple years.
    
    Args:
        base_path: Base path to data directory
        years: List of years to process
        
    Returns:
        Sorted list of dates
    """
    dates = []
    pattern = r'(\d{2})-(\d{2})_plus_minus_30_days'
    
    for year in years:
        for item in sorted(base_path.glob(f'{year}/*/s2')):
            match = re.search(pattern, str(item))
            if match:
                day, month = map(int, match.groups())
                dates.append(datetime(int(year), month, day))
                
    return sorted(dates)

def get_mosaic_path(date: datetime, base_path: Path, tile_id: str) -> Path:
    """Construct mosaic path for a given date."""
    return base_path / str(date.year) / f"{date.strftime('%d-%m')}_plus_minus_30_days" / "s2" / f"s2_{tile_id}.tif"

def read_bands_window(mosaic_path: Path, window: Window) -> np.ndarray:
    """
    Read specific bands from a mosaic file within a window.
    
    Args:
        mosaic_path: Path to the mosaic file
        window: Rasterio Window object defining the region to read
    
    Returns:
        Array of shape (bands, window_height, window_width)
    """
    with rasterio.open(mosaic_path) as src:
        # Read bands 1,3,4,9,10 (corresponding to B2,B4,B8,B11,B12)
        return src.read([1,3,4,9,10], window=window)

def read_dem_window(dem_path: Path, window: Window) -> np.ndarray:
    """Read DEM data within a window."""
    with rasterio.open(dem_path) as src:
        return src.read(1, window=window)

def read_all_bands(base_path: Path, tile_id: str, dates: List[datetime], 
                  window: Window, logger: logging.Logger) -> Tuple[np.ndarray, ...]:
    """
    Read all band data for given dates within a window.
    
    Args:
        base_path: Base path to data directory
        tile_id: ID of the tile to process
        dates: List of dates to process
        window: Window to read
        logger: Logger instance
        
    Returns:
        Tuple of arrays for each band (b2, b4, b8, b11, b12)
    """
    bands_shape = (len(dates), window.height, window.width)
    b2 = np.zeros(bands_shape, dtype=np.float32)
    b4 = np.zeros(bands_shape, dtype=np.float32)
    b8 = np.zeros(bands_shape, dtype=np.float32)
    b11 = np.zeros(bands_shape, dtype=np.float32)
    b12 = np.zeros(bands_shape, dtype=np.float32)
    
    for t, date in enumerate(dates):
        mosaic_path = get_mosaic_path(date, base_path, tile_id)
        if not mosaic_path.exists():
            logger.warning(f"Missing mosaic for date {date}")
            continue
            
        mosaic_data = read_bands_window(mosaic_path, window)
        b2[t] = mosaic_data[0] / 10000.0  # Convert to reflectance
        b4[t] = mosaic_data[1] / 10000.0
        b8[t] = mosaic_data[2] / 10000.0
        b11[t] = mosaic_data[3] / 10000.0
        b12[t] = mosaic_data[4] / 10000.0
        
    return b2, b4, b8, b11, b12

def main():
    """Main test function."""
    logger = setup_logger()
    logger.info("Starting WindowInference test")
    
    # Setup paths
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    data_dir = Path("data")  # Update with actual path on JeanZay
    mosaics_dir = data_dir / "mosaics"
    model_path = project_dir / "model" / "best_model_with_bdforet_no_resampled_weights_h2_y1_iter10_scaled01_featuresfromRFECV_nf10_f1_0.9601.pkl"
    
    # Test parameters
    years = ["2021", "2022"]  # Use two years of data
    tile_id = "EPSG2154_750000_6700000"
    window_size = 1024
    x_start, y_start = 0, 0  # Starting from top-left corner
    
    # Define window for reading
    window = Window(x_start, y_start, window_size, window_size)
    
    try:
        # Get available dates for both years
        dates = get_time_series_dates(mosaics_dir, years)
        logger.info(f"Found {len(dates)} dates in time series across {len(years)} years")
        for date in dates:
            logger.info(date)
        
        # Read all band data
        b2, b4, b8, b11, b12 = read_all_bands(mosaics_dir, tile_id, dates, window, logger)
            
        # Read DEM within window
        logger.info("Reading DEM data")
        dem_path = data_dir / "dem" / f"dem_{tile_id}.tif"
        dem = read_dem_window(dem_path, window)
        
        logger.info(f"Testing on {window_size}x{window_size} window")
        band_data = BandData(
            b2=b2,
            b4=b4,
            b8=b8,
            b11=b11,
            b12=b12,
            dates=dates,
            dem=dem
        )
        
        # Initialize WindowInference
        window_inference = WindowInference(
            band_data=band_data,
            model_path=model_path,
            num_harmonics=2,
            max_iter=1,
            logger=logger
        )
        
        # Test pipeline steps
        logger.info("Testing compute_indices")
        indices = window_inference.compute_indices()
        for name, index in indices.items():
            logger.info(f"{name} shape: {index.shape}, range: [{np.nanmin(index):.3f}, {np.nanmax(index):.3f}]")
        
        logger.info("Testing compute_features")
        features = window_inference.compute_features()
        for name in window_inference.REQUIRED_FEATURES:
            if name in features:
                feature = features[name]
                logger.info(f"{name} shape: {feature.shape}, range: [{np.nanmin(feature):.3f}, {np.nanmax(feature):.3f}]")
        
        logger.info("Testing full inference")
        prob_map = window_inference.run_inference()
        logger.info(f"Probability map shape: {prob_map.shape}")
        logger.info(f"Probability range: [{np.nanmin(prob_map):.3f}, {np.nanmax(prob_map):.3f}]")
        
        # Save probability map
        output_dir = current_dir / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Get reference mosaic for profile
        ref_mosaic_path = get_mosaic_path(dates[0], mosaics_dir, tile_id)
        
        # Save with proper profile
        with rasterio.open(ref_mosaic_path) as src:
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'dtype': 'uint8',
                'nodata': None,
                'width': window_size,
                'height': window_size,
                'transform': src.window_transform(window)
            })
            
            output_path = output_dir / f"test_prob_map_{tile_id}.tif"
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write((prob_map * 255).astype('uint8'), 1)
                
        logger.info(f"Saved probability map to {output_path} as GeoTIFF with dtype: {profile['dtype']}") 
        
        # Create visualization
        logger.info("Creating visualization")
        
        # Get winter RGB image (using January data if available)
        winter_dates = [d for d in dates if d.month in [12, 1, 2]]
        if winter_dates:
            winter_date = winter_dates[len(winter_dates)//2]  # Middle of summer
            winter_path = get_mosaic_path(winter_date, mosaics_dir, tile_id)
            with rasterio.open(winter_path) as src:
                # Read RGB bands (2,3,4) for visualization
                rgb = src.read([3,2,1], window=window)
        else:
            logger.warning("No winter dates available, using first available image for RGB")
            with rasterio.open(ref_mosaic_path) as src:
                rgb = src.read([3,2,1])  # Read RGB bands

        # Normalize RGB to [0,1]
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
        aspect_data = get_aspect(dem, resolution=10.0)

        # Create and save visualization
        fig, axes = plot_phenology_prediction(rgb, prob_map, aspect_data, logger=logger)
        fig.savefig(output_dir / f"visualization_{tile_id}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()