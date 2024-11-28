# download_dem.py
from __future__ import annotations
import os
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, List
import rasterio
import numpy as np
from rasterio.warp import reproject
import argparse
import multiprocessing
from dataclasses import dataclass
from tqdm import tqdm
# import multidem

@dataclass
class DEMTileInfo:
    """Data class to store DEM tile information."""
    tile_id: str
    target_bounds: tuple[float, float, float, float]
    target_transform: tuple[float, float, float, float, float, float]
    crs: str
    shape: tuple[int, int]
    profile: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DEMTileInfo':
        """Create a DEMTileInfo instance from a dictionary."""
        return cls(
            tile_id=data['tile_id'],
            target_bounds=tuple(data['target_bounds']),
            target_transform=tuple(data['target_transform']),
            crs=data['crs'],
            shape=tuple(data['shape']),
            profile=data['profile']
        )

class DEMDownloader:
    """Class to download and process DEM tiles using multidem."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DEMDownloader.
        
        Args:
            logger: Optional logger for tracking progress
        """
        self.logger = logger or logging.getLogger(__name__)

    def download_dem(self, tile_info: DEMTileInfo, output_dir: Union[str, Path]) -> bool:
        """
        Download and process a DEM tile using multidem.
        
        Args:
            tile_info: DEMTileInfo object containing tile information
            output_dir: Directory to save the output DEM file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_dir = Path(output_dir)
            output_file = output_dir / f'dem_{tile_info.tile_id}.tif'
            
            self.logger.info(f"Processing DEM for tile: {tile_info.tile_id}")
            
            # Download DEM data using multidem
            dem, transform, crs = multidem.crop(
                tile_info.target_bounds,
                source="SRTM30",
                datum="orthometric"
            )
            
            self.logger.debug(f"Downloaded DEM data for tile: {tile_info.tile_id}")
            
            # Prepare for reprojection
            dst_shape = tile_info.shape
            dst_transform = tile_info.target_transform
            
            # Create destination array for reprojection
            dem_reprojected = np.zeros((1, *dst_shape), dtype='float32')
            
            # Reproject DEM to match the reference raster
            dem_reprojected, transform_reprojected = reproject(
                source=dem,
                destination=dem_reprojected,
                src_transform=transform,
                src_crs=crs,
                dst_crs={'init': str(tile_info.crs)},
                dst_transform=dst_transform,
                dst_shape=dst_shape
            )
            
            self.logger.debug(f"Reprojected DEM data for tile: {tile_info.tile_id}")
            
            # Update profile for output
            profile = tile_info.profile.copy()
            profile.update({
                'count': 1,
                'dtype': 'float32',
                'transform': dst_transform,
                'crs': tile_info.crs
            })
            
            # Save the reprojected DEM
            output_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_file, 'w', **profile) as dest:
                dest.write(dem_reprojected[0].astype('float32'), 1)
            
            self.logger.info(f"Successfully saved DEM to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing DEM for tile {tile_info.tile_id}: {str(e)}")
            return False

    @staticmethod
    def _worker_function(args: tuple[DEMTileInfo, Path]) -> bool:
        """
        Static worker function for parallel processing.
        
        Args:
            args: Tuple containing (tile_info, output_dir)
            
        Returns:
            bool: True if successful, False otherwise
        """
        tile_info, output_dir = args
        downloader = DEMDownloader()  # Create a new instance for each worker
        return downloader.download_dem(tile_info, output_dir)

    def process_tiles(
        self, 
        info_file: Union[str, Path], 
        output_dir: Union[str, Path],
        num_workers: int = 1,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None
    ) -> None:
        """
        Process multiple DEM tiles in parallel.
        
        Args:
            info_file: JSON file containing tile information
            output_dir: Directory to save output DEM files
            num_workers: Number of parallel workers
            start_idx: Starting index for tile processing (optional)
            end_idx: Ending index for tile processing (optional)
        """
        info_file = Path(info_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tile information
        with open(info_file) as f:
            tile_data = json.load(f)
        
        # Select tiles based on indices if provided
        if start_idx is not None or end_idx is not None:
            start_idx = start_idx or 0
            end_idx = end_idx or len(tile_data)
            tile_data = tile_data[start_idx:end_idx]
        
        tile_infos = [DEMTileInfo.from_dict(data) for data in tile_data]
        self.logger.info(f"Processing {len(tile_infos)} tiles using {num_workers} workers")
        
        # Prepare arguments for worker function
        work_args = [(info, output_dir) for info in tile_infos]
        
        if num_workers > 1:
            with multiprocessing.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self._worker_function, work_args),
                    total=len(work_args),
                    desc="Processing tiles"
                ))
        else:
            results = [
                self._worker_function(args) 
                for args in tqdm(work_args, desc="Processing tiles")
            ]
        
        success_count = sum(results)
        self.logger.info(f"Successfully processed {success_count}/{len(tile_infos)} tiles")

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def main():
    parser = argparse.ArgumentParser(description="Download DEM data for specified tiles")
    parser.add_argument("info_file", type=str, help="JSON file containing tile information")
    parser.add_argument("output_dir", type=str, help="Output directory for DEM files")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Process tiles
    downloader = DEMDownloader(logger)
    downloader.process_tiles(args.info_file, args.output_dir, args.workers)

if __name__ == "__main__":
    main()

