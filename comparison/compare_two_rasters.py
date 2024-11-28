from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window, union, from_bounds
from rasterio import windows
from rasterio.coords import BoundingBox
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


@dataclass
class CommonExtent:
    """Data class to store common extent information between two rasters."""
    bounds: BoundingBox
    width: int
    height: int
    transform: rasterio.Affine


class RasterComparison:
    """Class to handle comparison between two rasters with potentially different extents."""
    
    def __init__(
        self,
        raster1_path: Union[str, Path],
        raster2_path: Union[str, Path],
        map1_classes: Dict[int, str],
        map2_classes: Dict[int, str],
        window_size: int = 1024,
        max_workers: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize RasterComparison.

        Args:
            raster1_path: Path to first raster
            raster2_path: Path to second raster
            map1_classes: Dictionary mapping raster1 values to class names
            map2_classes: Dictionary mapping raster2 values to class names
            window_size: Size of processing windows (default: 1024)
            max_workers: Maximum number of parallel workers (default: CPU count)
            logger: Optional logger instance

        Raises:
            ValueError: If rasters have different CRS
            FileNotFoundError: If raster files don't exist
        """
        self.raster1_path = Path(raster1_path)
        self.raster2_path = Path(raster2_path)
        self.map1_classes = map1_classes
        self.map2_classes = map2_classes
        self.window_size = window_size
        self.max_workers = max_workers or os.cpu_count()
        self.logger = logger or self._setup_logger()
        
        self._validate_inputs()
        self.common_extent = self._compute_common_extent()
        self.processing_windows = self._compute_processing_windows()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up a default logger if none provided."""
        logger = logging.getLogger('raster_comparison')
        
        # Only add handler if logger doesn't already have one
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        logger.setLevel(logging.DEBUG)
        
        # Set other common loggers to WARNING to reduce noise
        logging.getLogger('rasterio').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('fiona').setLevel(logging.WARNING)
        
        return logger

    def _validate_inputs(self) -> None:
        """
        Validate input rasters.

        Raises:
            FileNotFoundError: If raster files don't exist
            ValueError: If rasters have different CRS
        """
        if not self.raster1_path.exists():
            raise FileNotFoundError(f"Raster 1 not found: {self.raster1_path}")
        if not self.raster2_path.exists():
            raise FileNotFoundError(f"Raster 2 not found: {self.raster2_path}")

        with rasterio.open(self.raster1_path) as src1, rasterio.open(self.raster2_path) as src2:
            if src1.crs != src2.crs:
                raise ValueError(f"Different CRS: {src1.crs} vs {src2.crs}")

    def _compute_common_extent(self) -> CommonExtent:
        """
        Compute the common extent between the two rasters.

        Returns:
            CommonExtent object containing bounds and transform information
        """
        with rasterio.open(self.raster1_path) as src1, rasterio.open(self.raster2_path) as src2:
            # Log raster properties
            self.logger.debug(f"\nRaster 1 properties:")
            self.logger.debug(f"CRS: {src1.crs}")
            self.logger.debug(f"Bounds: {src1.bounds}")
            self.logger.debug(f"Resolution: {src1.res}")
            self.logger.debug(f"Transform: {src1.transform}")
            
            self.logger.debug(f"\nRaster 2 properties:")
            self.logger.debug(f"CRS: {src2.crs}")
            self.logger.debug(f"Bounds: {src2.bounds}")
            self.logger.debug(f"Resolution: {src2.res}")
            self.logger.debug(f"Transform: {src2.transform}")
            
            # Get bounds in the same CRS
            bounds1 = src1.bounds
            bounds2 = src2.bounds
            
            # Compute intersection
            intersection_bounds = BoundingBox(
                max(bounds1.left, bounds2.left),
                max(bounds1.bottom, bounds2.bottom),
                min(bounds1.right, bounds2.right),
                min(bounds1.top, bounds2.top)
            )
            
            self.logger.debug(f"\nIntersection bounds: {intersection_bounds}")
            
            # Use the finer resolution between the two rasters
            resolution = min(abs(src1.transform.a), abs(src2.transform.a))
            self.logger.debug(f"Using resolution: {resolution}")
            
            # Compute dimensions based on bounds and resolution
            width = int((intersection_bounds.right - intersection_bounds.left) / resolution)
            height = int((intersection_bounds.top - intersection_bounds.bottom) / resolution)
            
            # Create transform for common extent
            transform = rasterio.transform.from_bounds(
                intersection_bounds.left,
                intersection_bounds.bottom,
                intersection_bounds.right,
                intersection_bounds.top,
                width,
                height
            )
            
            self.logger.debug(f"Computed transform: {transform}")
            self.logger.debug(f"Computed dimensions: {width}x{height}")
            
            return CommonExtent(
                bounds=intersection_bounds,
                width=width,
                height=height,
                transform=transform
            )

    def _compute_processing_windows(self) -> List[Window]:
        """
        Compute list of smaller windows for processing.

        Returns:
            List of Window objects for processing
        """
        windows_list = []
        for y in range(0, self.common_extent.height, self.window_size):
            for x in range(0, self.common_extent.width, self.window_size):
                window = Window(
                    col_off=x,
                    row_off=y,
                    width=min(self.window_size, self.common_extent.width - x),
                    height=min(self.window_size, self.common_extent.height - y)
                )
                windows_list.append(window)
        return windows_list


    def _process_window(self, window: Window) -> np.ndarray:
        """
        Process a single window and compute confusion matrix.

        Args:
            window: Window object defining the region to process

        Returns:
            Confusion matrix for the window
        """
        try:
            with rasterio.open(self.raster1_path) as src1, rasterio.open(self.raster2_path) as src2:
                # Log transforms and initial window info
                self.logger.debug(f"Raster 1 transform: {src1.transform}")
                self.logger.debug(f"Raster 2 transform: {src2.transform}")
                self.logger.debug(f"Common extent transform: {self.common_extent.transform}")
                self.logger.debug(f"Input window: {window}")
                
                # Get window bounds in common extent coordinates
                window_bounds = windows.bounds(window, self.common_extent.transform)
                self.logger.debug(f"Window bounds in common extent: {window_bounds}")
                
                # Transform to pixel coordinates in each raster
                window1 = windows.from_bounds(*window_bounds, src1.transform)
                window2 = windows.from_bounds(*window_bounds, src2.transform)
                
                self.logger.debug(f"Window in raster 1: {window1}")
                self.logger.debug(f"Window in raster 2: {window2}")
                
                # Read data
                data1 = src1.read(1, window=Window(int(window1.col_off), int(window1.row_off), 
                                                 int(window1.width), int(window1.height)))
                data2 = src2.read(1, window=Window(int(window2.col_off), int(window2.row_off), 
                                                 int(window2.width), int(window2.height)))
                
                self.logger.debug(f"Read data shapes - Raster 1: {data1.shape}, Raster 2: {data2.shape}")

                # Apply class mappings
                data1_mapped = np.vectorize(self.map1_classes.get)(data1)
                data2_mapped = np.vectorize(self.map2_classes.get)(data2)

                # Filter valid pixels
                valid_pixels = (data1_mapped != 'nodata') & (data2_mapped != 'nodata')
                y_true = data2_mapped[valid_pixels]
                y_pred = data1_mapped[valid_pixels]

                # Get unique classes
                classes = sorted(set(self.map1_classes.values()) - {'nodata'})

                # Compute confusion matrix
                return confusion_matrix(y_true, y_pred, labels=classes)

        except Exception as e:
            self.logger.error(f"Error processing window: {str(e)}")
            self.logger.exception("Full traceback:")
            return None

    def compute_metrics(self) -> Dict:
        """
        Compute agreement metrics and confusion matrix.

        Returns:
            Dictionary containing metrics and confusion matrix
        """
        self.logger.info("Starting metrics computation...")
        
        # Initialize confusion matrix
        n_classes = len(set(self.map1_classes.values()) - {'nodata'})
        global_cm = np.zeros((n_classes, n_classes))

        # Process windows in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_window, window)
                for window in self.processing_windows
            ]
            
            for future in tqdm(futures, desc="Processing windows"):
                cm = future.result()
                if cm is not None:
                    global_cm += cm

        # Calculate metrics
        overall_agreement = np.diag(global_cm).sum() / global_cm.sum() if global_cm.sum() > 0 else 0

        return {
            'confusion_matrix': global_cm,
            'overall_agreement': overall_agreement,
            'class_names': sorted(set(self.map1_classes.values()) - {'nodata'})
        }


def save_results(results: Dict, output_dir: Union[str, Path], prefix: str = "") -> None:
    """
    Save comparison results to files.

    Args:
        results: Dictionary containing comparison results
        output_dir: Directory to save results
        prefix: Optional prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save confusion matrix
    np.save(
        output_dir / f"{prefix}confusion_matrix.npy",
        results['confusion_matrix']
    )
    
    # Save metrics
    with open(output_dir / f"{prefix}metrics.txt", 'w') as f:
        f.write(f"Overall Agreement: {results['overall_agreement']:.4f}\n")
        f.write(f"Class Names: {', '.join(results['class_names'])}\n")