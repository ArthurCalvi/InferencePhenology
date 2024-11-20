# collect_dem_info.py
from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import rasterio
from rasterio.warp import transform_bounds
from dataclasses import dataclass, asdict
import argparse

@dataclass
class DEMTileInfo:
    """Data class to store DEM tile information."""
    tile_id: str
    target_bounds: tuple[float, float, float, float]
    target_transform: tuple[float, float, float, float, float, float]
    crs: str
    shape: tuple[int, int]
    profile: Dict[str, Any]

    def _convert_value(self, value: Any) -> Any:
        """
        Convert a value to a JSON-serializable format.
        
        Args:
            value: Value to convert
            
        Returns:
            JSON-serializable version of the value
        """
        if hasattr(value, 'item'):  # numpy types
            value = value.item()
        
        if isinstance(value, (np.floating, float)) and np.isnan(value):
            return None
        
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        
        if isinstance(value, (list, tuple)):
            return [self._convert_value(v) for v in value]
        
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary with JSON-serializable values."""
        profile_dict = dict(self.profile)
        
        # Convert numpy dtypes to strings
        if 'dtype' in profile_dict:
            profile_dict['dtype'] = str(profile_dict['dtype'])
            
        # Convert CRS object to string representation
        if 'crs' in profile_dict:
            profile_dict['crs'] = str(profile_dict['crs'])
            
        # Convert transform to list
        if 'transform' in profile_dict:
            profile_dict['transform'] = list(profile_dict['transform'])
            
        # Convert any numpy objects and NaN values
        profile_dict = self._convert_value(profile_dict)
            
        return {
            'tile_id': self.tile_id,
            'target_bounds': self.target_bounds,
            'target_transform': self.target_transform,
            'crs': self.crs,
            'shape': self.shape,
            'profile': profile_dict
        }

class DEMInfoCollector:
    """Class to collect DEM information from reference rasters."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DEMInfoCollector.
        
        Args:
            logger: Optional logger for tracking progress
        """
        self.logger = logger or logging.getLogger(__name__)

    def collect_tile_info(self, raster_path: Union[str, Path]) -> DEMTileInfo:
        """
        Collect DEM tile information from a reference raster.
        
        Args:
            raster_path: Path to the reference raster file
            
        Returns:
            DEMTileInfo object containing tile information
            
        Raises:
            ValueError: If the raster file cannot be opened or processed
        """
        try:
            raster_path = Path(raster_path)
            self.logger.info(f"Processing raster: {raster_path}")
            
            with rasterio.open(raster_path) as raster:
                # Extract tile ID from filename
                tile_id = raster_path.name.split('s2')[1].split('.tif')[0][1:]
                
                # Transform bounds to WGS84
                target_bounds = transform_bounds(
                    raster.crs,
                    {'init': 'EPSG:4326'},
                    *raster.bounds
                )
                
                # Create tile info object
                tile_info = DEMTileInfo(
                    tile_id=tile_id,
                    target_bounds=target_bounds,
                    target_transform=tuple(raster.transform)[:6],
                    crs=str(raster.crs),
                    shape=raster.shape,
                    profile=raster.profile
                )
                
                self.logger.debug(f"Successfully collected info for tile: {tile_id}")
                return tile_info
                
        except Exception as e:
            self.logger.error(f"Error processing raster {raster_path}: {str(e)}")
            raise

    def process_directory(self, input_dir: Union[str, Path], output_file: Union[str, Path]) -> None:
        """
        Process all raster files in a directory and save their information.
        
        Args:
            input_dir: Directory containing raster files
            output_file: Path to save the JSON output
            
        Raises:
            FileNotFoundError: If input directory doesn't exist
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
            
        self.logger.info(f"Processing directory: {input_dir}")
        
        # Collect information for all raster files
        tile_info_list = []
        for raster_file in input_dir.rglob("*s2*.tif"):
            try:
                tile_info = self.collect_tile_info(raster_file)
                tile_info_list.append(tile_info.to_dict())  # Convert to dict before serialization
            except Exception as e:
                self.logger.error(f"Failed to process {raster_file}: {str(e)}")
                continue
        
        # Save to JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(tile_info_list, f, indent=2)
            
        self.logger.info(f"Successfully saved info for {len(tile_info_list)} tiles to {output_file}")

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
    parser = argparse.ArgumentParser(description="Collect DEM information from raster files")
    parser.add_argument("input_dir", type=str, help="Directory containing raster files")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Process files
    collector = DEMInfoCollector(logger)
    collector.process_directory(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()