"""A module for efficiently rasterizing GeoDataFrames using windowed processing and spatial indexing."""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import Window, get_data_window
from shapely.geometry import box
from typing import Dict, Optional, Union, List, Tuple, Iterator
import logging
from pathlib import Path
from tqdm import tqdm

class GeoDataFrameRasterizer:
    """Class to efficiently rasterize GeoDataFrame using windowed processing and spatial indexing."""
    
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        reference_raster: Union[str, Path],
        value_column: str,
        value_map: Dict[str, int],
        window_size: int = 1024,
        nodata_value: int = 0,
        compression: str = 'LZW',
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the rasterizer with input data and parameters.
        
        Args:
            gdf: Input GeoDataFrame to rasterize
            reference_raster: Path to reference raster for projection and extent
            value_column: Column name containing values to rasterize
            value_map: Dictionary mapping column values to raster values
            window_size: Size of processing windows
            nodata_value: Value to use for nodata pixels
            compression: Compression algorithm for output raster
            logger: Optional logger for progress tracking
            
        Raises:
            FileNotFoundError: If reference raster doesn't exist
            ValueError: If value_column is not in GeoDataFrame or value_map is invalid
        """
        self.gdf = gdf
        self.reference_raster = Path(reference_raster)
        self.value_column = value_column
        self.value_map = value_map
        self.window_size = window_size
        self.nodata_value = nodata_value
        self.compression = compression.upper()  # Standardize compression string
        self.logger = logger or logging.getLogger(__name__)
        
        self._validate_inputs()
        with rasterio.open(self.reference_raster) as src:
            self.profile = self._setup_profile(src)
            self.transform = src.transform
            self.crs = src.crs
            self.shape = src.shape
        
        self._setup_spatial_index()
        
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.reference_raster.exists():
            raise FileNotFoundError(f"Reference raster not found: {self.reference_raster}")
        
        if self.value_column not in self.gdf.columns:
            raise ValueError(f"Column '{self.value_column}' not found in GeoDataFrame")
            
        unique_values = set(self.gdf[self.value_column].dropna().unique())
        missing_values = unique_values - set(self.value_map.keys())
        if missing_values:
            raise ValueError(f"Values {missing_values} from {self.value_column} not found in value_map")
            
    def _setup_profile(self, src: rasterio.DatasetReader) -> Dict:
        """Set up output raster profile based on reference raster.
        
        Args:
            src: Open rasterio dataset reader
            
        Returns:
            Dict containing the raster profile
        """
        profile = src.profile.copy()
        profile.update({
            'dtype': 'uint8',
            'nodata': self.nodata_value,
            'compress': self.compression,
            'count': 1,
            'tiled': True,
            'blockxsize': self.window_size,
            'blockysize': self.window_size
        })
        return profile
        
    def _setup_spatial_index(self) -> None:
        """Create spatial index for the GeoDataFrame."""
        self.logger.info("Creating spatial index...")
        self.spatial_index = self.gdf.sindex
        
    def get_windows(self, num_windows: Optional[int] = None) -> List[Window]:
        """Get a list of windows for processing.
        
        Args:
            num_windows: Optional number of windows to return. If None, returns all windows.
            
        Returns:
            List of Window objects
        """
        windows = []
        rows = range(0, self.shape[0], self.window_size)
        cols = range(0, self.shape[1], self.window_size)
        
        for row in rows:
            for col in cols:
                height = min(self.window_size, self.shape[0] - row)
                width = min(self.window_size, self.shape[1] - col)
                windows.append(Window(col, row, width, height))
                if num_windows and len(windows) >= num_windows:
                    return windows
        return windows
        
    def _get_window_bounds(self, window: Window) -> Tuple[float, float, float, float]:
        """Get the geographic bounds for a window.
        
        Args:
            window: Rasterio Window object
            
        Returns:
            Tuple of (minx, miny, maxx, maxy)
        """
        window_transform = rasterio.windows.transform(window, self.transform)
        return rasterio.windows.bounds(window, window_transform)
        
    def process_window(self, window: Window) -> np.ndarray:
        """Process a single window of the output raster using spatial indexing.
        
        Args:
            window: Rasterio Window object defining the region to process
            
        Returns:
            Numpy array containing rasterized data for the window
        """
        # Get window bounds and transform
        bounds = self._get_window_bounds(window)
        window_transform = rasterio.windows.transform(window, self.transform)
        window_box = box(*bounds)
        
        # Query spatial index
        possible_matches_idx = list(self.spatial_index.intersection(bounds))
        if not possible_matches_idx:
            return np.full((window.height, window.width), self.nodata_value, dtype='uint8')
        
        # Get and process matching features
        window_gdf = self.gdf.iloc[possible_matches_idx]
        window_gdf = window_gdf.clip(window_box)
        
        if window_gdf.empty:
            return np.full((window.height, window.width), self.nodata_value, dtype='uint8')
        
        # Create shapes for rasterization
        shapes = [
            (geom, self.value_map[value])
            for geom, value in zip(window_gdf.geometry, window_gdf[self.value_column])
            if value in self.value_map  # Skip any null values
        ]
        
        if not shapes:
            return np.full((window.height, window.width), self.nodata_value, dtype='uint8')
        
        # Rasterize window
        return features.rasterize(
            shapes=shapes,
            out_shape=(window.height, window.width),
            transform=window_transform,
            fill=self.nodata_value,
            dtype='uint8',
            all_touched=True
        )
        
    def process_windows(self, windows: List[Window]) -> List[Tuple[Window, np.ndarray]]:
        """Process multiple windows and return their data.
        
        Args:
            windows: List of Window objects to process
            
        Returns:
            List of tuples containing (window, data array)
        """
        results = []
        for window in tqdm(windows, desc="Processing windows"):
            data = self.process_window(window)
            results.append((window, data))
        return results
        
    def rasterize(self, output_path: Union[str, Path], num_windows: Optional[int] = None) -> None:
        """Rasterize the GeoDataFrame to a GeoTIFF.
        
        Args:
            output_path: Path for output GeoTIFF
            num_windows: Optional number of windows to process. If None, processes all windows.
        """
        try:
            output_path = Path(output_path)
            self.logger.info(f"Starting rasterization to {output_path}")
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get windows to process
            windows = self.get_windows(num_windows)
            self.logger.info(f"Processing {len(windows)} windows")
            
            # Process all windows
            processed_windows = self.process_windows(windows)
            
            # Write to file
            with rasterio.open(output_path, 'w', **self.profile) as dst:
                for window, data in processed_windows:
                    dst.write(data, 1, window=window)
                    
            self.logger.info(f"Successfully created raster: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Rasterization failed: {str(e)}")
            raise