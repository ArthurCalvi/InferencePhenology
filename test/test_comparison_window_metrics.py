#!/usr/bin/env python3
"""Test module for RasterComparison metrics computation on a single window with forest mask."""

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
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to Python path to import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison.compare_two_rasters import RasterComparison, CommonExtent


class TestWindowMetrics(unittest.TestCase):
    """Test class for computing metrics on a single window with forest mask."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Define paths
        cls.bdforet_path = Path("/Users/arthurcalvi/Data/Disturbances_maps/BDForet/bdforet_10_FF1_FF2_EN_year_raster.tif")
        cls.dlt_path = Path("/Users/arthurcalvi/Data/species/DLT_2018_010m_fr_03035_v020/DLT_Dominant_Leaf_Type_France.tif")
        cls.forest_mask_path = Path("/Users/arthurcalvi/Data/Disturbances_maps/BDForet/mask_forest.tif")
        
        # Define class mappings
        cls.bdforet_classes = {
            1: 'deciduous',
            2: 'evergreen',
            0: 'nodata'
        }
        
        cls.dlt_classes = {
            1: 'deciduous',
            2: 'evergreen',
            255: 'nodata'
        }
        
        # Setup logging
        cls.logger = cls._setup_logger()
        
        # Create output directory
        cls.output_dir = Path("test/test_outputs/window_metrics")
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """
        Set up test logger.
        
        Returns:
            logging.Logger: Configured logger instance
        """
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
        """Test processing of a single window and visualize results with forest mask."""
        # Initialize comparison with window size
        comparison = RasterComparison(
            raster1_path=self.bdforet_path,
            raster2_path=self.dlt_path,
            forest_mask_path=self.forest_mask_path,
            map1_classes=self.bdforet_classes,
            map2_classes=self.dlt_classes,
            window_size=10240,
            logger=self.logger
        )

        # Get the first window
        first_window = comparison.processing_windows[len(comparison.processing_windows) // 4]
        window_bounds = windows.bounds(first_window, comparison.common_extent.transform)
        
        self.logger.info(f"Processing window at bounds: {window_bounds}")
        
        # Process the window
        window_metrics = comparison._process_window(first_window)
        
        if window_metrics is not None:
            # Create confusion matrix visualization
            plt.figure(figsize=(10, 8))
            
            # Convert to percentages
            cm_sum = window_metrics.confusion_matrix.sum()
            if cm_sum > 0:
                cm_percentages = window_metrics.confusion_matrix / cm_sum * 100
            else:
                cm_percentages = window_metrics.confusion_matrix
            
            # Create heatmap with both counts and percentages
            sns.heatmap(cm_percentages, 
                       annot=np.array([[f'{val:.1f}%\n({int(count)})' 
                                      for val, count in zip(row_pct, row_count)] 
                                     for row_pct, row_count in zip(cm_percentages, window_metrics.confusion_matrix)]),
                       fmt='',
                       xticklabels=['deciduous', 'evergreen'],
                       yticklabels=['deciduous', 'evergreen'])
            plt.title('Confusion Matrix for Window (Forest Areas Only)\nPercentages and (Counts)')
            plt.xlabel('BDForet')
            plt.ylabel('DLT')
            plt.savefig(self.output_dir / "window_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate and log metrics
            total_pixels = window_metrics.confusion_matrix.sum()
            correct_pixels = np.diag(window_metrics.confusion_matrix).sum()
            accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
            
            self.logger.info(f"\nMetrics for window:")
            self.logger.info(f"Total forest pixels: {window_metrics.forest_pixels}")
            self.logger.info(f"Valid pixels in confusion matrix: {total_pixels}")
            self.logger.info(f"Correct pixels: {correct_pixels}")
            self.logger.info(f"Overall accuracy: {accuracy:.4f}")
            self.logger.info(f"Raster1 (BDForet) coverage: {window_metrics.raster1_coverage:.4f}")
            self.logger.info(f"Raster2 (DLT) coverage: {window_metrics.raster2_coverage:.4f}")
            
            # Save visual comparison of the window
            self._visualize_window_comparison(comparison, first_window)
        else:
            self.logger.error("Failed to compute metrics for window")

 
    # def _visualize_window_comparison(self, comparison: RasterComparison, window: Window) -> None:
    #     """
    #     Create a visual comparison of both maps and forest mask for the given window.
        
    #     Args:
    #         comparison: RasterComparison instance
    #         window: Window object defining the region to visualize
    #     """
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
    #     # Get window bounds in common extent coordinates
    #     window_bounds = windows.bounds(window, comparison.common_extent.transform)
    #     self.logger.debug(f"Visualizing window bounds: {window_bounds}")
        
    #     # Create custom colormap for classification maps
    #     colors_class = ['#1f77b4', '#ff7f0e']  # Blue for deciduous, Orange for evergreen
    #     class_cmap = LinearSegmentedColormap.from_list('custom', colors_class)
        
    #     # Define target shape based on the window
    #     target_shape = (window.height, window.width)
    #     self.logger.debug(f"Visualization target shape: {target_shape}")

    #     # Read and plot BDForet data
    #     with rasterio.open(self.bdforet_path) as src1:
    #         window1 = windows.from_bounds(*window_bounds, src1.transform)
    #         window1 = Window(int(window1.col_off), int(window1.row_off),
    #                        int(window1.width), int(window1.height))
    #         data1 = src1.read(1, window=window1, out_shape=target_shape)
    #         self.logger.debug(f"BDForet data shape: {data1.shape}")
    #         # Mask nodata values
    #         data1_masked = np.ma.masked_where(data1 == 0, data1)
    #         self.logger.debug(f"BDForet data shape after masking: {data1_masked.shape}")
    #         im1 = axes[0, 0].imshow(data1_masked, cmap=class_cmap, vmin=1, vmax=2)
    #         axes[0, 0].set_title('BDForet Classification')
    #         plt.colorbar(im1, ax=axes[0, 0], label='Class (1: Deciduous, 2: Evergreen)')
        
    #     # Read and plot DLT data with masked nodata
    #     with rasterio.open(self.dlt_path) as src2:
    #         window2 = windows.from_bounds(*window_bounds, src2.transform)
    #         window2 = Window(int(window2.col_off), int(window2.row_off),
    #                        int(window2.width), int(window2.height))
    #         data2 = src2.read(1, window=window2, out_shape=target_shape)
    #         data2_masked = np.ma.masked_where(data2 == 0, data2)
    #         self.logger.debug(f"DLT data shape after masking: {data2_masked.shape}")
    #         im2 = axes[0, 1].imshow(data2_masked, cmap=class_cmap, vmin=1, vmax=2)
    #         axes[0, 1].set_title('DLT Classification')
    #         plt.colorbar(im2, ax=axes[0, 1], label='Class (1: Deciduous, 2: Evergreen)')
        
    #     # Read and plot forest mask
    #     with rasterio.open(self.forest_mask_path) as src_mask:
    #         window_mask = windows.from_bounds(*window_bounds, src_mask.transform)
    #         window_mask = Window(int(window_mask.col_off), int(window_mask.row_off),
    #                            int(window_mask.width), int(window_mask.height))
    #         mask_data = src_mask.read(1, window=window_mask, out_shape=target_shape)
    #         self.logger.debug(f"Forest mask data shape: {mask_data.shape}")
            
    #         # Handle potential floating point values in mask
    #         mask_data = mask_data.astype(np.float32)
    #         mask_data_masked = np.ma.masked_where(mask_data == 0, mask_data)
    #         unique_vals = np.unique(mask_data)
    #         self.logger.debug(f"Forest mask unique values: {unique_vals}")
            
    #         # Count occurrences of each value safely
    #         value_counts = {val: np.sum(mask_data == val) for val in unique_vals}
    #         self.logger.debug(f"Forest mask value counts: {value_counts}")
            
    #         # Warning if no forest pixels
    #         if 1 not in value_counts:
    #             self.logger.warning("No forest pixels found in this window! "
    #                               "Consider choosing a different window location.")
            
    #         im_mask = axes[1, 0].imshow(mask_data_masked, cmap='Greens', vmin=0, vmax=1)
    #         axes[1, 0].set_title('Forest Mask')
    #         plt.colorbar(im_mask, ax=axes[1, 0], label='Forest (0: No, 1: Yes)')
        
    #     # Create and plot difference map
    #     diff_data = np.zeros_like(data1, dtype=np.float32)
    #     diff_data.fill(np.nan)
        
    #     # Apply masks
    #     forest_pixels = mask_data == 1
    #     self.logger.debug(f"Number of forest pixels in visualization: {np.sum(forest_pixels)}")
        
    #     valid_pixels = (data1 != 0) & (data2 != 0) & forest_pixels
    #     self.logger.debug(f"Number of valid pixels in visualization: {np.sum(valid_pixels)}")
        
    #     # Calculate agreement only for valid forest pixels
    #     if np.any(valid_pixels):
    #         diff_data[valid_pixels] = (data1[valid_pixels] == data2[valid_pixels]).astype(float)
        
    #     self.logger.debug(f"Difference map shape: {diff_data.shape}")
        
    #     im_diff = axes[1, 1].imshow(diff_data, cmap='RdYlGn', vmin=0, vmax=1)
    #     axes[1, 1].set_title('Agreement Map (Forest Only)')
    #     plt.colorbar(im_diff, ax=axes[1, 1], label='Agreement (0: Disagree, 1: Agree)')
        
    #     # Add overall title with coordinate info
    #     plt.suptitle(f'Visual Comparison of Window Data\nBounds: {window_bounds}')
        
    #     # Adjust layout and save
    #     plt.tight_layout()
    #     plt.savefig(self.output_dir / "window_visual_comparison.png", dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     # Save statistics
    #     total_pixels = mask_data.size
    #     forest_pixels_count = np.sum(forest_pixels)
    #     valid_pixels_count = np.sum(valid_pixels)
        
    #     self.logger.info(f"\nWindow statistics:")
    #     self.logger.info(f"Total pixels: {total_pixels}")
    #     self.logger.info(f"Forest pixels: {forest_pixels_count} ({forest_pixels_count/total_pixels*100:.2f}%)")
    #     self.logger.info(f"Valid forest pixels: {valid_pixels_count} "
    #                     f"({valid_pixels_count/forest_pixels_count*100:.2f}% of forest)")
        
    #     # Save arrays for deeper analysis
    #     np.save(self.output_dir / "bdforet_window.npy", data1)
    #     np.save(self.output_dir / "dlt_window.npy", data2)
    #     np.save(self.output_dir / "forest_mask_window.npy", mask_data)
    def _visualize_window_comparison(self, comparison: RasterComparison, window: Window) -> None:
        """
        Create a visual comparison of both maps and forest mask for the given window.
        
        Args:
            comparison: RasterComparison instance
            window: Window object defining the region to visualize
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Get window bounds in common extent coordinates
        window_bounds = windows.bounds(window, comparison.common_extent.transform)
        self.logger.debug(f"Visualizing window bounds: {window_bounds}")
        
        # Create custom colormap for classification maps
        colors_class = ['#1f77b4', '#ff7f0e']  # Blue for deciduous, Orange for evergreen
        class_cmap = LinearSegmentedColormap.from_list('custom', colors_class)
        
        # Define target shape based on the window
        target_shape = (window.height, window.width)
        self.logger.debug(f"Visualization target shape: {target_shape}")

        # Get valid values from class mappings (excluding 'nodata')
        bdforet_valid_values = [k for k, v in self.bdforet_classes.items() if v != 'nodata']
        dlt_valid_values = [k for k, v in self.dlt_classes.items() if v != 'nodata']
        
        # Read and plot BDForet data with proper masking
        with rasterio.open(self.bdforet_path) as src1:
            window1 = windows.from_bounds(*window_bounds, src1.transform)
            window1 = Window(int(window1.col_off), int(window1.row_off),
                           int(window1.width), int(window1.height))
            data1 = src1.read(1, window=window1, out_shape=target_shape)
            
            # Log unique values and their counts
            unique_vals = np.unique(data1)
            self.logger.info(f"BDForet unique values: {unique_vals}")
            value_counts = {val: np.sum(data1 == val) for val in unique_vals}
            self.logger.info(f"BDForet value counts: {value_counts}")
            
            # Mask values not in valid class mapping
            data1_masked = np.ma.masked_where(~np.isin(data1, bdforet_valid_values), data1)
            im1 = axes[0, 0].imshow(data1_masked, cmap=class_cmap, vmin=min(bdforet_valid_values), vmax=max(bdforet_valid_values))
            axes[0, 0].set_title('BDForet Classification')
            plt.colorbar(im1, ax=axes[0, 0], label='Class (1: Deciduous, 2: Evergreen)')
        
        # Read and plot DLT data with proper masking
        with rasterio.open(self.dlt_path) as src2:
            window2 = windows.from_bounds(*window_bounds, src2.transform)
            window2 = Window(int(window2.col_off), int(window2.row_off),
                           int(window2.width), int(window2.height))
            data2 = src2.read(1, window=window2, out_shape=target_shape)
            
            # Log unique values and their counts
            unique_vals = np.unique(data2)
            self.logger.info(f"DLT unique values: {unique_vals}")
            value_counts = {val: np.sum(data2 == val) for val in unique_vals}
            self.logger.info(f"DLT value counts: {value_counts}")
            
            # Mask values not in valid class mapping
            data2_masked = np.ma.masked_where(~np.isin(data2, dlt_valid_values), data2)
            im2 = axes[0, 1].imshow(data2_masked, cmap=class_cmap, vmin=min(dlt_valid_values), vmax=max(dlt_valid_values))
            axes[0, 1].set_title('DLT Classification')
            plt.colorbar(im2, ax=axes[0, 1], label='Class (1: Deciduous, 2: Evergreen)')
        
        # Read and plot forest mask with proper handling of values
        with rasterio.open(self.forest_mask_path) as src_mask:
            window_mask = windows.from_bounds(*window_bounds, src_mask.transform)
            window_mask = Window(int(window_mask.col_off), int(window_mask.row_off),
                               int(window_mask.width), int(window_mask.height))
            mask_data = src_mask.read(1, window=window_mask, out_shape=target_shape)
            
            # Log unique values and their counts
            unique_vals = np.unique(mask_data)
            self.logger.info(f"Forest mask unique values: {unique_vals}")
            value_counts = {val: np.sum(mask_data == val) for val in unique_vals}
            self.logger.info(f"Forest mask value counts: {value_counts}")
            
            # Only show valid forest pixels (value 1)
            mask_data_masked = np.ma.masked_where(mask_data != 1, mask_data)
            im_mask = axes[1, 0].imshow(mask_data_masked, cmap='Greens', vmin=0, vmax=1)
            axes[1, 0].set_title('Forest Mask')
            plt.colorbar(im_mask, ax=axes[1, 0], label='Forest (1: Yes)')
        
        # Create and plot difference map for valid pixels only
        diff_data = np.zeros_like(data1, dtype=np.float32)
        diff_data.fill(np.nan)
        
        # Apply masks using valid values from mappings
        forest_pixels = mask_data == 1
        valid_pixels = (np.isin(data1, bdforet_valid_values) & 
                       np.isin(data2, dlt_valid_values) & 
                       forest_pixels)
        
        self.logger.debug(f"Number of forest pixels in visualization: {np.sum(forest_pixels)}")
        self.logger.debug(f"Number of valid pixels in visualization: {np.sum(valid_pixels)}")
        
        if np.any(valid_pixels):
            diff_data[valid_pixels] = (data1[valid_pixels] == data2[valid_pixels]).astype(float)
        
        im_diff = axes[1, 1].imshow(diff_data, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 1].set_title('Agreement Map (Forest Only)')
        plt.colorbar(im_diff, ax=axes[1, 1], label='Agreement (0: Disagree, 1: Agree)')
        
        # Add overall title with coordinate info
        plt.suptitle(f'Visual Comparison of Window Data\nBounds: {window_bounds}')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / "window_visual_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics
        total_pixels = mask_data.size
        forest_pixels_count = np.sum(forest_pixels)
        valid_pixels_count = np.sum(valid_pixels)
        
        self.logger.info(f"\nWindow statistics:")
        self.logger.info(f"Total pixels: {total_pixels}")
        self.logger.info(f"Forest pixels: {forest_pixels_count} ({forest_pixels_count/total_pixels*100:.2f}%)")
        if forest_pixels_count > 0:
            self.logger.info(f"Valid forest pixels: {valid_pixels_count} "
                           f"({valid_pixels_count/forest_pixels_count*100:.2f}% of forest)")
        
        # Save arrays for deeper analysis
        np.save(self.output_dir / "bdforet_window.npy", data1)
        np.save(self.output_dir / "dlt_window.npy", data2)
        np.save(self.output_dir / "forest_mask_window.npy", mask_data)


if __name__ == '__main__':
    unittest.main(verbosity=2)