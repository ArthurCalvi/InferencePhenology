#!/usr/bin/env python3
"""
Script to prepare configurations for phenology inference on JeanZay.
"""
import argparse
from pathlib import Path
import logging
import re
from datetime import datetime
from inference import FolderInference, TileData
from typing import List

def setup_logger() -> logging.Logger:
    """Setup logger with appropriate formatting."""
    logger = logging.getLogger('prepare_inference')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def find_tiles(mosaic_dir: Path, year: str) -> List[str]:
    """Find all available tile IDs in the mosaic directory."""
    # Look in first available date directory
    first_date_dir = next(mosaic_dir.glob(f"{year}/*/s2"))
    tile_pattern = re.compile(r's2_(EPSG\d+_\d+_\d+)\.tif')
    
    tiles = set()
    for f in first_date_dir.glob("s2_*.tif"):
        match = tile_pattern.search(f.name)
        if match:
            tiles.add(match.group(1))
    
    return sorted(list(tiles))

def get_tile_data(mosaic_dir: Path, dem_dir: Path, tile_id: str, years: List[str]) -> TileData:
    """Get all paths and dates for a specific tile."""
    pattern = r'(\d{2})-(\d{2})_plus_minus_30_days'
    mosaic_paths = []
    dates = []
    
    for year in years:
        for item in sorted((mosaic_dir / year).glob('*/s2')):
            match = re.search(pattern, str(item))
            if match:
                month, day = map(int, match.groups())
                mosaic_path = item / f"s2_{tile_id}.tif"
                
                if mosaic_path.exists():
                    mosaic_paths.append(mosaic_path)
                    dates.append(datetime(int(year), month, day))
    
    dem_path = dem_dir / f"dem_{tile_id}.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
    return TileData(
        tile_id=tile_id,
        mosaic_paths=mosaic_paths,
        dates=dates,
        dem_path=dem_path
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare phenology inference configurations')
    
    # Required arguments
    parser.add_argument('--mosaic-dir', type=Path, required=True,
                      help='Directory containing mosaic data')
    parser.add_argument('--dem-dir', type=Path, required=True,
                      help='Directory containing DEM files')
    parser.add_argument('--output-dir', type=Path, required=True,
                      help='Output directory for inference results')
    parser.add_argument('--model-path', type=Path, required=True,
                      help='Path to the model file')
    
    # Optional arguments with defaults
    parser.add_argument('--years', nargs='+', default=["2021", "2022"],
                      help='Years to process (default: 2021 2022)')
    parser.add_argument('--window-size', type=int, default=1024,
                      help='Window size for processing (default: 1024)')
    parser.add_argument('--num-harmonics', type=int, default=2,
                      help='Number of harmonics (default: 2)')
    parser.add_argument('--max-iter', type=int, default=1,
                      help='Maximum number of iterations (default: 1)')
    parser.add_argument('--workers-per-tile', type=int, default=4,
                      help='Number of workers per tile (default: 4)')
    parser.add_argument('--max-concurrent-jobs', type=int, default=20,
                      help='Maximum number of concurrent jobs (default: 20)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    logger = setup_logger()
    
    try:
        # Find all available tiles
        tiles = find_tiles(args.mosaic_dir, args.years[0])
        logger.info(f"Found {len(tiles)} tiles to process")
        
        # Prepare tile data
        tiles_data = []
        for tile_id in tiles:
            try:
                tile_data = get_tile_data(args.mosaic_dir, args.dem_dir, tile_id, args.years)
                tiles_data.append(tile_data)
                logger.info(f"Prepared data for tile {tile_id}: {len(tile_data.dates)} dates")
            except Exception as e:
                logger.error(f"Error preparing data for tile {tile_id}: {str(e)}")
                continue
        
        if not tiles_data:
            raise ValueError("No valid tile data found")
        
        # Initialize FolderInference
        inference_dir = args.output_dir / "phenology_inference"
        folder_inference = FolderInference(
            tiles_data=tiles_data,
            output_dir=inference_dir,
            model_path=args.model_path,
            window_size=args.window_size,
            workers_per_tile=args.workers_per_tile,
            num_harmonics=args.num_harmonics,
            max_iter=args.max_iter,
            logger=logger
        )
        
        # Save configurations
        folder_inference.save_configs()
        
        # Create a summary file
        with open(inference_dir / "job_summary.txt", 'w') as f:
            f.write("Configuration Summary:\n")
            f.write("--------------------\n")
            f.write(f"Mosaic directory: {args.mosaic_dir}\n")
            f.write(f"DEM directory: {args.dem_dir}\n")
            f.write(f"Output directory: {inference_dir}\n")
            f.write(f"Model path: {args.model_path}\n")
            f.write(f"Total tiles: {len(tiles_data)}\n")
            f.write(f"Years: {args.years}\n")
            f.write(f"Window size: {args.window_size}\n")
            f.write(f"Harmonics: {args.num_harmonics}\n")
            f.write(f"Max iterations: {args.max_iter}\n")
            f.write(f"Workers per tile: {args.workers_per_tile}\n")
            f.write("\nSLURM Configuration:\n")
            f.write("-------------------\n")
            f.write(f"#SBATCH --array=0-{len(tiles_data)-1}%{args.max_concurrent_jobs}\n")
        
        logger.info(f"Prepared configurations for {len(tiles_data)} tiles")
        logger.info(f"Configuration saved in {inference_dir}")
        logger.info(f"To run the inference, update the array parameter in the SLURM script to: "
                   f"--array=0-{len(tiles_data)-1}%{args.max_concurrent_jobs}")
        
    except Exception as e:
        logger.error(f"Preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()