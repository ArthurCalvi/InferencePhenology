# mask_inference.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window, union, from_bounds
from rasterio.transform import Affine
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class RasterMasker:
    """Class to handle masking of one raster with another."""
    
    def __init__(
        self,
        inference_path: Union[str, Path],
        mask_path: Union[str, Path],
        output_path: Union[str, Path],
        window_size: int = 1024,
        max_workers: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize RasterMasker.

        Args:
            inference_path: Path to inference raster
            mask_path: Path to mask raster
            output_path: Path for output raster
            window_size: Size of processing windows (default: 1024)
            max_workers: Maximum number of parallel workers (default: CPU count)
            logger: Optional logger instance
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If rasters have different CRS
        """
        self.inference_path = Path(inference_path)
        self.mask_path = Path(mask_path)
        self.output_path = Path(output_path)
        self.window_size = window_size
        self.max_workers = max_workers
        self.logger = logger or self._setup_logger()
        
        self._validate_inputs()
        self.common_extent = self._compute_common_extent()
        self.processing_windows = self._compute_processing_windows()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a default logger if none provided."""
        logger = logging.getLogger('raster_masker')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _validate_inputs(self) -> None:
        """
        Validate input rasters.
        
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If rasters have different CRS
        """
        if not self.inference_path.exists():
            raise FileNotFoundError(f"Inference raster not found: {self.inference_path}")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask raster not found: {self.mask_path}")
            
        with rasterio.open(self.inference_path) as src1, \
             rasterio.open(self.mask_path) as src2:
            if src1.crs != src2.crs:
                raise ValueError("Rasters must have the same CRS")
                
    def _compute_common_extent(self) -> Tuple[Window, Affine]:
        """
        Compute the common extent and transform between rasters.
        
        Returns:
            Tuple containing:
            - Window object for the common extent
            - Affine transform for the output raster
        """
        with rasterio.open(self.inference_path) as src1, \
             rasterio.open(self.mask_path) as src2:
            
            # Get bounds of both rasters
            bounds1 = src1.bounds
            bounds2 = src2.bounds
            
            # Compute intersection
            intersection = (
                max(bounds1.left, bounds2.left),
                max(bounds1.bottom, bounds2.bottom),
                min(bounds1.right, bounds2.right),
                min(bounds1.top, bounds2.top)
            )
            
            # Use the finest resolution
            resolution = min(abs(src1.transform.a), abs(src2.transform.a))
            
            # Compute dimensions
            width = int((intersection[2] - intersection[0]) / resolution)
            height = int((intersection[3] - intersection[1]) / resolution)
            
            # Create transform
            transform = Affine(
                resolution, 0, intersection[0],
                0, -resolution, intersection[3]
            )
            
            return Window(0, 0, width, height), transform
            
    def _compute_processing_windows(self) -> List[Window]:
        """
        Compute list of windows for processing.
        
        Returns:
            List of Window objects for processing
        """
        windows_list = []
        for y in range(0, self.common_extent[0].height, self.window_size):
            for x in range(0, self.common_extent[0].width, self.window_size):
                window = Window(
                    col_off=x,
                    row_off=y,
                    width=min(self.window_size, self.common_extent[0].width - x),
                    height=min(self.window_size, self.common_extent[0].height - y)
                )
                windows_list.append(window)
        return windows_list
    
    def _process_window(self, window: Window) -> Optional[np.ndarray]:
        """
        Process a single window.
        
        Args:
            window: Window object defining the region to process
            
        Returns:
            Masked array for the window or None if error occurs
        """
        try:
            with rasterio.open(self.inference_path) as src1, \
                 rasterio.open(self.mask_path) as src2:
                
                # Get window bounds
                window_bounds = (
                    self.common_extent[1].c + window.col_off * self.common_extent[1].a,
                    self.common_extent[1].f + (window.row_off + window.height) * self.common_extent[1].e,
                    self.common_extent[1].c + (window.col_off + window.width) * self.common_extent[1].a,
                    self.common_extent[1].f + window.row_off * self.common_extent[1].e
                )
                
                # Transform to windows in input rasters
                window1 = from_bounds(*window_bounds, src1.transform)
                window2 = from_bounds(*window_bounds, src2.transform)
                
                # Read and resample data
                target_shape = (window.height, window.width)
                inference_data = src1.read(1, window=window1, out_shape=target_shape)
                mask_data = src2.read(1, window=window2, out_shape=target_shape)
                
                # Apply mask
                masked_data = np.where(mask_data == 1, inference_data, 0)
                
                return masked_data
                
        except Exception as e:
            self.logger.error(f"Error processing window: {str(e)}")
            return None
    
    def process_raster(self) -> None:
        """Process the complete raster using parallel window processing."""
        try:
            self.logger.info("Starting raster masking...")
            
            # Setup output raster
            profile = {
                'driver': 'GTiff',
                'dtype': 'uint8',
                'nodata': 0,
                'width': self.common_extent[0].width,
                'height': self.common_extent[0].height,
                'count': 1,
                'crs': rasterio.open(self.inference_path).crs,
                'transform': self.common_extent[1],
                'compress': 'lzw'
            }
            
            with rasterio.open(self.output_path, 'w', **profile) as dst:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self._process_window, window) 
                             for window in self.processing_windows]
                    
                    for window, future in tqdm(zip(self.processing_windows, futures),
                                            total=len(self.processing_windows),
                                            desc="Processing windows"):
                        result = future.result()
                        if result is not None:
                            dst.write(result, 1, window=window)
                            
            self.logger.info(f"Successfully created masked raster: {self.output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to process raster: {str(e)}")
            raise


def main():
    """Main function."""
    # Setup paths
    inference_path = Path("/Users/arthurcalvi/Data/Disturbances_maps/phenology/classification_map.tif")
    mask_path = Path("/Users/arthurcalvi/Data/Disturbances_maps/BDForet/mask_forest.tif")
    output_path = Path("/Users/arthurcalvi/Data/Disturbances_maps/phenology/classification_map_forest_only.tif")
    
    # Initialize and run masker
    masker = RasterMasker(
        inference_path=inference_path,
        mask_path=mask_path,
        output_path=output_path,
        window_size=5120  # Adjust based on your memory constraints
    )
    
    masker.process_raster()


if __name__ == "__main__":
    main()