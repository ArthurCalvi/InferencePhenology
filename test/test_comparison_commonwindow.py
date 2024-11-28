#!/usr/bin/env python3
"""Test module for RasterComparison common extent computation."""

import os
import sys
import unittest
import logging
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.plot import show

# Add parent directory to Python path to import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison.compare_two_rasters import RasterComparison, CommonExtent

class TestRasterComparison(unittest.TestCase):
    """Test class for RasterComparison."""

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
        
    @staticmethod
    def _setup_logger():
        """Set up test logger."""
        logger = logging.getLogger('test_raster_comparison')
        logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def test_file_existence(self):
        """Test if input files exist."""
        self.assertTrue(self.bdforet_path.exists(), "BDForet file not found")
        self.assertTrue(self.dlt_path.exists(), "DLT file not found")

    def test_raster_properties(self):
        """Test and print basic properties of both rasters."""
        with rasterio.open(self.bdforet_path) as bdforet, rasterio.open(self.dlt_path) as dlt:
            # Log basic properties
            self.logger.info("\nBDForet properties:")
            self.logger.info(f"CRS: {bdforet.crs}")
            self.logger.info(f"Bounds: {bdforet.bounds}")
            self.logger.info(f"Shape: {bdforet.shape}")
            self.logger.info(f"Transform: {bdforet.transform}")
            
            self.logger.info("\nDLT properties:")
            self.logger.info(f"CRS: {dlt.crs}")
            self.logger.info(f"Bounds: {dlt.bounds}")
            self.logger.info(f"Shape: {dlt.shape}")
            self.logger.info(f"Transform: {dlt.transform}")

    def test_common_extent_computation(self):
        """Test computation of common extent."""
        try:
            comparison = RasterComparison(
                raster1_path=self.bdforet_path,
                raster2_path=self.dlt_path,
                map1_classes=self.bdforet_classes,
                map2_classes=self.dlt_classes,
                logger=self.logger
            )
            
            common_extent = comparison.common_extent
            
            # Log common extent properties
            self.logger.info("\nCommon extent properties:")
            self.logger.info(f"Bounds: {common_extent.bounds}")
            self.logger.info(f"Shape: ({common_extent.width}, {common_extent.height})")
            self.logger.info(f"Transform: {common_extent.transform}")
            
            # Verify that common extent is within both rasters' bounds
            with rasterio.open(self.bdforet_path) as bdforet, rasterio.open(self.dlt_path) as dlt:
                self.assertGreaterEqual(common_extent.bounds.left, min(bdforet.bounds.left, dlt.bounds.left))
                self.assertGreaterEqual(common_extent.bounds.bottom, min(bdforet.bounds.bottom, dlt.bounds.bottom))
                self.assertLessEqual(common_extent.bounds.right, max(bdforet.bounds.right, dlt.bounds.right))
                self.assertLessEqual(common_extent.bounds.top, max(bdforet.bounds.top, dlt.bounds.top))
            
            return common_extent
            
        except Exception as e:
            self.logger.error(f"Error computing common extent: {str(e)}")
            raise

    def test_visualize_extents(self):
        """Create a visualization of the extents and processing windows."""
        # Initialize RasterComparison with larger window size (50*1024)
        comparison = RasterComparison(
            raster1_path=self.bdforet_path,
            raster2_path=self.dlt_path,
            map1_classes=self.bdforet_classes,
            map2_classes=self.dlt_classes,
            window_size=10240,  # 50*1024
            logger=self.logger
        )
        
        common_extent = comparison.common_extent
        windows = comparison.processing_windows
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot extents
        with rasterio.open(self.bdforet_path) as bdforet, rasterio.open(self.dlt_path) as dlt:
            # Plot BDForet bounds
            bdforet_bounds = bdforet.bounds
            ax.plot([bdforet_bounds.left, bdforet_bounds.right, bdforet_bounds.right, 
                    bdforet_bounds.left, bdforet_bounds.left],
                   [bdforet_bounds.bottom, bdforet_bounds.bottom, bdforet_bounds.top, 
                    bdforet_bounds.top, bdforet_bounds.bottom],
                   'b-', label='BDForet', alpha=0.5)
            
            # Plot DLT bounds
            dlt_bounds = dlt.bounds
            ax.plot([dlt_bounds.left, dlt_bounds.right, dlt_bounds.right, 
                    dlt_bounds.left, dlt_bounds.left],
                   [dlt_bounds.bottom, dlt_bounds.bottom, dlt_bounds.top, 
                    dlt_bounds.top, dlt_bounds.bottom],
                   'r-', label='DLT', alpha=0.5)
            
            # Plot common extent
            ax.plot([common_extent.bounds.left, common_extent.bounds.right, common_extent.bounds.right,
                    common_extent.bounds.left, common_extent.bounds.left],
                   [common_extent.bounds.bottom, common_extent.bounds.bottom, common_extent.bounds.top,
                    common_extent.bounds.top, common_extent.bounds.bottom],
                   'g--', label='Common Extent', linewidth=2)
            
            # Plot processing windows
            self.logger.info(f"Plotting {len(windows)} processing windows...")
            for i, window in enumerate(windows):
                # Convert window pixels to coordinates
                window_bounds = rasterio.windows.bounds(window, common_extent.transform)
                # window_bounds returns (left, bottom, right, top)
                ax.plot([window_bounds[0], window_bounds[2], window_bounds[2],
                        window_bounds[0], window_bounds[0]],
                       [window_bounds[1], window_bounds[1], window_bounds[3],
                        window_bounds[3], window_bounds[1]],
                       'k:', alpha=1, linewidth=1)
                
                # Add window number at center (only for first few windows)
                if i < 5:  # Limit labels to avoid cluttering
                    center_x = (window_bounds[0] + window_bounds[2]) / 2
                    center_y = (window_bounds[1] + window_bounds[3]) / 2
                    ax.text(center_x, center_y, str(i), 
                           horizontalalignment='center',
                           verticalalignment='center')
        
        ax.set_title('Raster Extents Comparison with Processing Windows')
        ax.legend()
        ax.grid(lw=0.2)
        
        # Save plot
        output_dir = Path("test/test_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "extents_comparison_with_windows.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Total number of windows: {len(windows)}")
        
        # Log some statistics about the windows
        window_sizes = [(w.width * common_extent.transform[0], w.height * abs(common_extent.transform[4])) 
                       for w in windows]
        avg_width = sum(w[0] for w in window_sizes) / len(window_sizes)
        avg_height = sum(w[1] for w in window_sizes) / len(window_sizes)
        
        self.logger.info(f"Average window dimensions: {avg_width:.2f}m x {avg_height:.2f}m")

    def test_processing_windows(self):
        """Test computation of processing windows."""
        comparison = RasterComparison(
            raster1_path=self.bdforet_path,
            raster2_path=self.dlt_path,
            map1_classes=self.bdforet_classes,
            map2_classes=self.dlt_classes,
            window_size=51200,  # 50*1024
            logger=self.logger
        )
        
        windows = comparison.processing_windows
        
        # Log windows information
        self.logger.info(f"\nNumber of processing windows: {len(windows)}")
        self.logger.info("First few windows:")
        for i, window in enumerate(windows[:5]):
            self.logger.info(f"Window {i}: {window}")
        
        # Verify windows cover the entire common extent
        total_pixels = sum(w.width * w.height for w in windows)
        expected_pixels = comparison.common_extent.width * comparison.common_extent.height
        self.assertAlmostEqual(total_pixels, expected_pixels, delta=comparison.window_size)

if __name__ == '__main__':
    unittest.main(verbosity=2)