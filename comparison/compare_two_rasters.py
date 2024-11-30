
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
    """Data class to store common extent information between rasters."""
    bounds: BoundingBox
    width: int
    height: int
    transform: rasterio.Affine

@dataclass
class WindowMetrics:
    """
    Data class to store metrics for a window.
    
    Attributes:
        confusion_matrix: np.ndarray
            Confusion matrix for the window
        raster1_coverage: float
            Coverage of raster1 in forest mask
        raster2_coverage: float
            Coverage of raster2 in forest mask
        forest_pixels: int
            Total forest pixels in window
        eco_region_metrics: Dict[int, WindowMetrics]
            Metrics broken down by eco-region
    """
    confusion_matrix: np.ndarray
    raster1_coverage: float
    raster2_coverage: float
    forest_pixels: int
    eco_region_metrics: Dict[int, 'WindowMetrics'] = None

class RasterComparison:
    """Class to handle comparison between two rasters with forest mask and eco-region consideration."""
    
    def __init__(
        self,
        raster1_path: Union[str, Path],
        raster2_path: Union[str, Path],
        forest_mask_path: Union[str, Path],
        eco_region_path: Union[str, Path],
        map1_classes: Dict[int, str],
        map2_classes: Dict[int, str],
        eco_region_classes: Dict[int, str],
        window_size: int = 1024,
        max_workers: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize RasterComparison with forest mask and eco-region support.

        Args:
            raster1_path: Path to first raster
            raster2_path: Path to second raster
            forest_mask_path: Path to forest mask raster
            eco_region_path: Path to eco-region raster
            map1_classes: Dictionary mapping raster1 values to class names
            map2_classes: Dictionary mapping raster2 values to class names
            eco_region_classes: Dictionary mapping eco-region values to region names
            window_size: Size of processing windows (default: 1024)
            max_workers: Maximum number of parallel workers (default: CPU count)
            logger: Optional logger instance

        Raises:
            ValueError: If rasters have different CRS
            FileNotFoundError: If any raster files don't exist
        """
        self.raster1_path = Path(raster1_path)
        self.raster2_path = Path(raster2_path)
        self.forest_mask_path = Path(forest_mask_path)
        self.eco_region_path = Path(eco_region_path)
        self.map1_classes = map1_classes
        self.map2_classes = map2_classes
        self.eco_region_classes = eco_region_classes
        self.window_size = window_size
        self.max_workers = max_workers or os.cpu_count()
        self.logger = logger or self._setup_logger()

        # Initialize class mappings
        self.classes = sorted(set(self.map1_classes.values()).union(set(self.map2_classes.values())) - {'nodata'})
        self.class_name_to_int = {class_name: i for i, class_name in enumerate(self.classes)}
        self.int_to_class_name = {i: class_name for class_name, i in self.class_name_to_int.items()}
        
        self._validate_inputs()
        self.common_extent = self._compute_common_extent()
        self.processing_windows = self._compute_processing_windows()

    def _validate_inputs(self) -> None:
        """
        Validate input rasters including forest mask and eco-region.

        Raises:
            FileNotFoundError: If raster files don't exist
            ValueError: If rasters have different CRS
        """
        raster_paths = [
            self.raster1_path, 
            self.raster2_path, 
            self.forest_mask_path,
            self.eco_region_path
        ]
        
        for path in raster_paths:
            if not path.exists():
                raise FileNotFoundError(f"Raster not found: {path}")

        # Check CRS consistency
        with rasterio.open(self.raster1_path) as src1, \
             rasterio.open(self.raster2_path) as src2, \
             rasterio.open(self.forest_mask_path) as src_mask, \
             rasterio.open(self.eco_region_path) as src_eco:
            
            if not (src1.crs == src2.crs == src_mask.crs == src_eco.crs):
                raise ValueError("All rasters must have the same CRS")

    def _compute_common_extent(self) -> CommonExtent:
        """
        Compute the common extent between all rasters including forest mask and eco-region.

        Returns:
            CommonExtent object containing bounds and transform information
        """
        with rasterio.open(self.raster1_path) as src1, \
             rasterio.open(self.raster2_path) as src2, \
             rasterio.open(self.forest_mask_path) as src_mask, \
             rasterio.open(self.eco_region_path) as src_eco:
            
            # Get bounds of all rasters
            bounds1 = src1.bounds
            bounds2 = src2.bounds
            bounds_mask = src_mask.bounds
            bounds_eco = src_eco.bounds
            
            # Compute intersection of all bounds
            intersection_bounds = BoundingBox(
                max(bounds1.left, bounds2.left, bounds_mask.left, bounds_eco.left),
                max(bounds1.bottom, bounds2.bottom, bounds_mask.bottom, bounds_eco.bottom),
                min(bounds1.right, bounds2.right, bounds_mask.right, bounds_eco.right),
                min(bounds1.top, bounds2.top, bounds_mask.top, bounds_eco.top)
            )
            
            # Use the finest resolution among all rasters
            resolution = min(
                abs(src1.transform.a),
                abs(src2.transform.a),
                abs(src_mask.transform.a),
                abs(src_eco.transform.a)
            )
            
            # Compute dimensions
            width = int((intersection_bounds.right - intersection_bounds.left) / resolution)
            height = int((intersection_bounds.top - intersection_bounds.bottom) / resolution)
            
            # Create transform
            transform = rasterio.transform.from_bounds(
                intersection_bounds.left,
                intersection_bounds.bottom,
                intersection_bounds.right,
                intersection_bounds.top,
                width,
                height
            )
            
            return CommonExtent(
                bounds=intersection_bounds,
                width=width,
                height=height,
                transform=transform
            )

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

    def _process_window(self, window: Window) -> Optional[WindowMetrics]:
        """
        Process a single window and compute metrics considering forest mask and eco-regions.

        Args:
            window: Window object defining the region to process

        Returns:
            WindowMetrics object containing confusion matrix, coverage metrics,
            and eco-region specific metrics
        """
        try:
            with rasterio.open(self.raster1_path) as src1, \
                rasterio.open(self.raster2_path) as src2, \
                rasterio.open(self.forest_mask_path) as src_mask, \
                rasterio.open(self.eco_region_path) as src_eco:
                
                # Get window bounds
                window_bounds = windows.bounds(window, self.common_extent.transform)
                self.logger.debug(f"Processing window bounds: {window_bounds}")
                
                # Transform to pixel coordinates in each raster
                window1 = windows.from_bounds(*window_bounds, src1.transform)
                window2 = windows.from_bounds(*window_bounds, src2.transform)
                window_mask = windows.from_bounds(*window_bounds, src_mask.transform)
                window_eco = windows.from_bounds(*window_bounds, src_eco.transform)
                
                # Convert to integer coordinates
                window1 = Window(int(window1.col_off), int(window1.row_off), 
                            int(window1.width), int(window1.height))
                window2 = Window(int(window2.col_off), int(window2.row_off), 
                            int(window2.width), int(window2.height))
                window_mask = Window(int(window_mask.col_off), int(window_mask.row_off),
                                int(window_mask.width), int(window_mask.height))
                window_eco = Window(int(window_eco.col_off), int(window_eco.row_off),
                                int(window_eco.width), int(window_eco.height))
                
                # Read data and resample to common shape
                target_shape = (window.height, window.width)
                
                data1 = src1.read(1, window=window1, out_shape=target_shape)
                data2 = src2.read(1, window=window2, out_shape=target_shape)
                forest_mask = src_mask.read(1, window=window_mask, out_shape=target_shape)
                eco_region = src_eco.read(1, window=window_eco, out_shape=target_shape)
                
                # Apply class mappings with default 'nodata' for unmapped values
                data1_mapped = np.vectorize(lambda x: self.map1_classes.get(x, 'nodata'))(data1)
                data2_mapped = np.vectorize(lambda x: self.map2_classes.get(x, 'nodata'))(data2)
                
                # Consider only forest pixels
                forest_pixels = forest_mask == 1
                total_forest = np.sum(forest_pixels)

                # Early return if no forest pixels
                if total_forest == 0:
                    self.logger.info("No forest pixels in window, returning empty metrics")
                    n_classes = len(set(self.map1_classes.values()).union(set(self.map2_classes.values())) - {'nodata'})
                    return WindowMetrics(
                        confusion_matrix=np.zeros((n_classes, n_classes), dtype=np.int32),
                        raster1_coverage=0.0,
                        raster2_coverage=0.0,
                        forest_pixels=0,
                        eco_region_metrics={}
                    )

                # Map class names to integer labels
                data1_labels = np.vectorize(lambda x: self.class_name_to_int.get(x, -1))(data1_mapped)
                data2_labels = np.vectorize(lambda x: self.class_name_to_int.get(x, -1))(data2_mapped)
            
                # Valid pixels are those where both rasters have valid class labels and are in forest pixels
                valid_pixels = (data1_labels != -1) & (data2_labels != -1) & forest_pixels

                # Compute global metrics
                raster1_coverage = np.sum((data1_labels != -1) & forest_pixels) / total_forest
                raster2_coverage = np.sum((data2_labels != -1) & forest_pixels) / total_forest

                # If no valid pixels, return empty metrics
                if not np.any(valid_pixels):
                    self.logger.info("No valid pixels in window, returning empty metrics")
                    n_classes = len(self.classes)
                    return WindowMetrics(
                        confusion_matrix=np.zeros((n_classes, n_classes), dtype=np.int32),
                        raster1_coverage=raster1_coverage,
                        raster2_coverage=raster2_coverage,
                        forest_pixels=total_forest,
                        eco_region_metrics={}
                    )

                y_true = data2_labels[valid_pixels]
                y_pred = data1_labels[valid_pixels]
                conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(self.classes))))

                # Process metrics for each eco-region
                eco_region_metrics = {}
                for eco_id in self.eco_region_classes.keys():
                    # Create mask for current eco-region
                    eco_mask = eco_region == eco_id
                    eco_forest_pixels = np.sum(forest_pixels & eco_mask)
                    
                    if eco_forest_pixels > 0:
                        # Compute eco-region specific valid pixels
                        eco_valid_pixels = valid_pixels & eco_mask
                        
                        # Skip if no valid pixels for this eco-region
                        if not np.any(eco_valid_pixels):
                            continue
                        
                        # Compute coverage metrics for eco-region
                        eco_r1_coverage = np.sum((data1_labels != -1) & forest_pixels & eco_mask) / eco_forest_pixels
                        eco_r2_coverage = np.sum((data2_labels != -1) & forest_pixels & eco_mask) / eco_forest_pixels
                        
                        # Compute confusion matrix for eco-region
                        eco_y_true = data2_labels[eco_valid_pixels]
                        eco_y_pred = data1_labels[eco_valid_pixels]
                        eco_conf_matrix = confusion_matrix(eco_y_true, eco_y_pred, labels=list(range(len(self.classes))))
                        
                        eco_region_metrics[eco_id] = WindowMetrics(
                            confusion_matrix=eco_conf_matrix,
                            raster1_coverage=eco_r1_coverage,
                            raster2_coverage=eco_r2_coverage,
                            forest_pixels=eco_forest_pixels,
                            eco_region_metrics=None  # No nested eco-regions
                        )

                return WindowMetrics(
                    confusion_matrix=conf_matrix,
                    raster1_coverage=raster1_coverage,
                    raster2_coverage=raster2_coverage,
                    forest_pixels=total_forest,
                    eco_region_metrics=eco_region_metrics
                )

        except Exception as e:
            self.logger.error(f"Error processing window: {str(e)}")
            self.logger.exception("Full traceback:")
            return None


    def compute_metrics(self) -> Dict:
        """
        Compute agreement metrics and coverage statistics for global and eco-region specific results.

        Returns:
            Dictionary containing metrics, confusion matrices, and coverage statistics
            both globally and per eco-region
        """
        self.logger.info("Starting metrics computation...")
        
        # Initialize global metrics
        # n_classes = len(set(self.map1_classes.values()) - {'nodata'})
        # global_cm = np.zeros((n_classes, n_classes))
        n_classes = len(self.classes)
        global_cm = np.zeros((n_classes, n_classes))

        total_forest_pixels = 0
        weighted_r1_coverage = 0.0
        weighted_r2_coverage = 0.0

        # Initialize eco-region metrics
        eco_region_results = {eco_id: {
            'confusion_matrix': np.zeros((n_classes, n_classes)),
            'forest_pixels': 0,
            'weighted_r1_coverage': 0.0,
            'weighted_r2_coverage': 0.0
        } for eco_id in self.eco_region_classes.keys()}

        # Process windows in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_window, window) 
                        for window in self.processing_windows]
            
            for future in tqdm(futures, desc="Processing windows"):
                metrics = future.result()
                if metrics is None:
                    continue
                    
                # Update global metrics
                global_cm += metrics.confusion_matrix
                weighted_r1_coverage += metrics.raster1_coverage * metrics.forest_pixels
                weighted_r2_coverage += metrics.raster2_coverage * metrics.forest_pixels
                total_forest_pixels += metrics.forest_pixels

                # Update eco-region specific metrics
                if metrics.eco_region_metrics:
                    for eco_id, eco_metrics in metrics.eco_region_metrics.items():
                        eco_results = eco_region_results[eco_id]
                        eco_results['confusion_matrix'] += eco_metrics.confusion_matrix
                        eco_results['forest_pixels'] += eco_metrics.forest_pixels
                        eco_results['weighted_r1_coverage'] += eco_metrics.raster1_coverage * eco_metrics.forest_pixels
                        eco_results['weighted_r2_coverage'] += eco_metrics.raster2_coverage * eco_metrics.forest_pixels

        # Calculate final global metrics
        if total_forest_pixels > 0:
            overall_agreement = np.diag(global_cm).sum() / global_cm.sum() if global_cm.sum() > 0 else 0
            final_r1_coverage = weighted_r1_coverage / total_forest_pixels
            final_r2_coverage = weighted_r2_coverage / total_forest_pixels
        else:
            overall_agreement = final_r1_coverage = final_r2_coverage = 0.0

        # Calculate final eco-region metrics
        eco_region_final = {}
        for eco_id, eco_results in eco_region_results.items():
            if eco_results['forest_pixels'] > 0:
                eco_cm = eco_results['confusion_matrix']
                eco_region_final[eco_id] = {
                    'confusion_matrix': eco_cm,
                    'overall_agreement': np.diag(eco_cm).sum() / eco_cm.sum() if eco_cm.sum() > 0 else 0,
                    'raster1_coverage': eco_results['weighted_r1_coverage'] / eco_results['forest_pixels'],
                    'raster2_coverage': eco_results['weighted_r2_coverage'] / eco_results['forest_pixels'],
                    'forest_pixels': eco_results['forest_pixels'],
                    'region_name': self.eco_region_classes[eco_id]
                }

        return {
            'global': {
                'confusion_matrix': global_cm,
                'overall_agreement': overall_agreement,
                'class_names': sorted(set(self.map1_classes.values()) - {'nodata'}),
                'raster1_coverage': final_r1_coverage,
                'raster2_coverage': final_r2_coverage,
                'total_forest_pixels': total_forest_pixels
            },
            'eco_regions': eco_region_final
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
        f.write(f"Raster 1 Forest Coverage: {results['raster1_coverage']:.4f}\n")
        f.write(f"Raster 2 Forest Coverage: {results['raster2_coverage']:.4f}\n")
        f.write(f"Total Forest Pixels: {results['total_forest_pixels']}\n")