#!/usr/bin/env python3
"""Test module for RasterComparison metrics computation on a single window."""

import os
import sys
import unittest
import logging
from pathlib import Path
import rasterio
from rasterio import windows
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
import seaborn as sns

# Add parent directory to Python path to import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison.compare_two_rasters import RasterComparison, CommonExtent

class TestWindowMetrics(unittest.TestCase):
    """Test class for computing metrics on a single window."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Define paths
        cls.bdforet_path = Path("/Users/arthurcalvi/Data/Disturbances_maps/BDForet/bdforet_10_FF1_FF2_EN_year_raster.tif")
        cls.dlt_path = Path("/Users/arthurcalvi/Data/species/DLT_2018_010m_fr_03035_v020/DLT_Dominant_Leaf_Type_France.tif")
        
        # Define class mappings
        cls.bdforet_classes = {
            1: 'deciduous',
            2: 'evergreen',
            0: 'nodata'
        }
        
        cls.dlt_classes = {
            1: 'deciduous',
            2: 'evergreen',
            0: 'nodata'
        }
        
        # Setup logging
        cls.logger = cls._setup_logger()
        
        # Create output directory
        cls.output_dir = Path("test/test_outputs/window_metrics")
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _setup_logger():
        """Set up test logger."""
        # Create logger
        logger = logging.getLogger('test_window_metrics')
        logger.setLevel(logging.DEBUG)
        
        # Set other loggers to INFO or higher to reduce noise
        logging.getLogger('rasterio').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('fiona').setLevel(logging.WARNING)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def test_process_single_window(self):
        """Test processing of a single window and visualize results."""
        # Initialize comparison with window size
        comparison = RasterComparison(
            raster1_path=self.bdforet_path,
            raster2_path=self.dlt_path,
            map1_classes=self.bdforet_classes,
            map2_classes=self.dlt_classes,
            window_size=10240,  # Changed to 10240
            logger=self.logger
        )

        # Get the first window
        first_window = comparison.processing_windows[len(comparison.processing_windows) // 2]
        window_bounds = windows.bounds(first_window, comparison.common_extent.transform)
        
        self.logger.info(f"Processing window at bounds: {window_bounds}")
        
        # Process the window
        confusion_matrix = comparison._process_window(first_window)
        
        if confusion_matrix is not None:
            # Create confusion matrix visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, 
                       annot=True, 
                       fmt='d',
                       xticklabels=['deciduous', 'evergreen'],
                       yticklabels=['deciduous', 'evergreen'])
            plt.title('Confusion Matrix for First Window')
            plt.xlabel('BDForet')
            plt.ylabel('DLT')
            plt.savefig(self.output_dir / "window_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate and log metrics
            total_pixels = confusion_matrix.sum()
            correct_pixels = np.diag(confusion_matrix).sum()
            accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
            
            self.logger.info(f"\nMetrics for window:")
            self.logger.info(f"Total pixels: {total_pixels}")
            self.logger.info(f"Correct pixels: {correct_pixels}")
            self.logger.info(f"Overall accuracy: {accuracy:.4f}")
            
            # Save visual comparison of the window
            self._visualize_window_comparison(comparison, first_window)
        else:
            self.logger.error("Failed to compute confusion matrix for window")
            
        return confusion_matrix

    def _visualize_window_comparison(self, comparison: RasterComparison, window: Window) -> None:
        """Create a visual comparison of both maps for the given window."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Get window bounds in common extent coordinates
        window_bounds = windows.bounds(window, comparison.common_extent.transform)
        self.logger.debug(f"Visualizing window bounds: {window_bounds}")
        
        # Read and plot BDForet data
        with rasterio.open(self.bdforet_path) as src1:
            window1 = windows.from_bounds(*window_bounds, src1.transform)
            window1 = Window(int(window1.col_off), int(window1.row_off),
                           int(window1.width), int(window1.height))
            self.logger.debug(f"BDForet window: {window1}")
            data1 = src1.read(1, window=window1)
            im1 = ax1.imshow(data1, cmap='RdYlBu', vmin=1, vmax=2)
            ax1.set_title(f'BDForet Window\nShape: {data1.shape}')
            plt.colorbar(im1, ax=ax1, label='Class')
        
        # Read and plot DLT data
        with rasterio.open(self.dlt_path) as src2:
            window2 = windows.from_bounds(*window_bounds, src2.transform)
            window2 = Window(int(window2.col_off), int(window2.row_off),
                           int(window2.width), int(window2.height))
            self.logger.debug(f"DLT window: {window2}")
            data2 = src2.read(1, window=window2)
            im2 = ax2.imshow(data2, cmap='RdYlBu', vmin=1, vmax=2)
            ax2.set_title(f'DLT Window\nShape: {data2.shape}')
            plt.colorbar(im2, ax=ax2, label='Class')
        
        # Add overall title with coordinate info
        plt.suptitle(f'Visual Comparison of Window Data\nBounds: {window_bounds}')
        
        # Save figure
        plt.savefig(self.output_dir / "window_visual_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save some statistics about the data
        total_pixels = data1.size
        valid_pixels1 = np.sum(data1 != 0)
        valid_pixels2 = np.sum(data2 != 0)
        
        self.logger.info(f"\nWindow statistics:")
        self.logger.info(f"Total pixels: {total_pixels}")
        self.logger.info(f"Valid pixels in BDForet: {valid_pixels1} ({valid_pixels1/total_pixels*100:.2f}%)")
        self.logger.info(f"Valid pixels in DLT: {valid_pixels2} ({valid_pixels2/total_pixels*100:.2f}%)")
        
        # Also save individual arrays for deeper analysis if needed
        np.save(self.output_dir / "bdforet_window.npy", data1)
        np.save(self.output_dir / "dlt_window.npy", data2)

if __name__ == '__main__':
    unittest.main(verbosity=2)