from __future__ import annotations
import os
from pathlib import Path
import logging
import rasterio
import numpy as np
from rasterio.warp import transform_bounds, reproject
import argparse
from tqdm import tqdm
import multidem

class DEMDownloader:
    """Class to download and process DEM tiles using multidem based on RGB reference."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize the DEMDownloader.
        
        Args:
            base_dir: Base directory containing region folders with rgb subdirectories
        """
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the class."""
        logger = logging.getLogger('dem_downloader')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _get_rgb_reference(self, region_dir: Path) -> Path:
        """
        Get reference RGB file from region directory.
        
        Args:
            region_dir: Path to region directory
            
        Returns:
            Path to first TIFF file in rgb directory
            
        Raises:
            FileNotFoundError: If no TIFF file is found
        """
        rgb_dir = region_dir / 'rgb'
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found in {region_dir}")
            
        # Get first .tif file
        tif_files = list(rgb_dir.glob('*.tif'))
        if not tif_files:
            raise FileNotFoundError(f"No TIFF files found in {rgb_dir}")
            
        return tif_files[0]

    def download_dem(self, region_name: str) -> bool:
        """
        Download and process DEM using RGB file as reference.
        
        Args:
            region_name: Name of the region folder
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            region_dir = self.base_dir / region_name
            
            # Get reference RGB file
            rgb_file = self._get_rgb_reference(region_dir)
            self.logger.info(f"Using reference file: {rgb_file}")
            
            # Create DEM directory
            dem_dir = region_dir / 'dem'
            dem_dir.mkdir(parents=True, exist_ok=True)
            output_file = dem_dir / 'dem.tif'
            
            # Read reference file metadata
            with rasterio.open(rgb_file) as src:
                # Transform bounds to WGS84 for DEM download
                bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                
                # Store original CRS and transform for reprojection
                dst_crs = src.crs
                dst_transform = src.transform
                dst_shape = src.shape
                profile = src.profile
            
            self.logger.info(f"Processing DEM for region: {region_name}")
            
            # Download DEM data
            dem, transform, crs = multidem.crop(
                bounds,
                source="SRTM30",
                datum="orthometric"
            )
            
            # Prepare for reprojection
            dem_reprojected = np.zeros((1, *dst_shape), dtype='float32')
            
            # Reproject DEM to match reference
            dem_reprojected, _ = reproject(
                source=dem,
                destination=dem_reprojected,
                src_transform=transform,
                src_crs=crs,
                dst_crs=dst_crs,
                dst_transform=dst_transform,
                dst_shape=dst_shape
            )
            
            # Update profile for DEM output
            profile.update({
                'count': 1,
                'dtype': 'float32',
                'nodata': None,
                'compress': 'lzw'
            })
            
            # Save DEM
            with rasterio.open(output_file, 'w', **profile) as dest:
                dest.write(dem_reprojected[0].astype('float32'), 1)
            
            self.logger.info(f"Saved DEM to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing region {region_name}: {e}")
            return False

    def process_regions(self) -> None:
        """Process all regions in the base directory."""
        # Find all region directories (those containing rgb subdirectory)
        regions = [d.name for d in self.base_dir.iterdir() 
                  if d.is_dir() and (d / 'rgb').exists()]
        
        if not regions:
            self.logger.error(f"No valid region directories found in {self.base_dir}")
            return
            
        self.logger.info(f"Found {len(regions)} regions to process")
        
        # Process each region
        for region in tqdm(sorted(regions), desc="Processing regions"):
            self.download_dem(region)

def main():
    parser = argparse.ArgumentParser(description="Download DEM data based on RGB references")
    parser.add_argument("base_dir", type=str, 
                       help="Base directory containing region folders with rgb subdirectories")
    args = parser.parse_args()

    downloader = DEMDownloader(Path(args.base_dir))
    downloader.process_regions()

if __name__ == "__main__":
    main()