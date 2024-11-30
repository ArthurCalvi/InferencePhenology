#!/usr/bin/env python3
"""Test module for RasterComparison metrics computation on multiple windows with forest mask and eco-regions."""

import os
import sys
import unittest
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.windows import Window
from rasterio import windows
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to Python path to import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison.compare_two_rasters import RasterComparison
from tqdm import tqdm


class TestWindowMetrics(unittest.TestCase):
    """Test class for computing metrics on multiple windows with eco-regions and forest mask."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Define paths
        cls.bdforet_path = Path("/Users/arthurcalvi/Data/Disturbances_maps/BDForet/bdforet_10_FF1_FF2_EN_year_raster.tif")
        cls.dlt_path = Path("/Users/arthurcalvi/Data/species/DLT_2018_010m_fr_03035_v020/DLT_Dominant_Leaf_Type_France.tif")
        cls.forest_mask_path = Path("/Users/arthurcalvi/Data/Disturbances_maps/BDForet/mask_forest.tif")
        cls.eco_region_path = Path("/Users/arthurcalvi/Data/eco-regions/France/greco.tif")
        
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
        
        cls.eco_region_classes = {
            1: 'Alps',
            2: 'Central Massif',
            3: 'Corsica',
            4: 'Greater Crystalline and Oceanic West',
            5: 'Greater Semi-Continental East',
            6: 'Jura',
            7: 'Mediterranean',
            8: 'Oceanic Southwest',
            9: 'Pyrenees',
            10: 'Semi-Oceanic North Center',
            11: 'Vosges'
        }
        
        # Setup logging
        cls.logger = cls._setup_logger()
        
        # Create output directory
        cls.output_dir = Path("test/test_outputs/window_metrics_with_eco_regions")
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

    def test_process_multiple_windows_with_eco_regions(self):
        """Test processing of multiple windows and aggregate metrics by eco-region."""
        # Initialize comparison with window size
        comparison = RasterComparison(
            raster1_path=self.bdforet_path,
            raster2_path=self.dlt_path,
            forest_mask_path=self.forest_mask_path,
            eco_region_path=self.eco_region_path,
            map1_classes=self.bdforet_classes,
            map2_classes=self.dlt_classes,
            eco_region_classes=self.eco_region_classes,
            window_size=10240,
            logger=self.logger
        )

        # Select 10 windows evenly spaced throughout the dataset
        total_windows = len(comparison.processing_windows)
        indices = np.linspace(0, total_windows - 1, 10, dtype=int)
        selected_windows = [comparison.processing_windows[i] for i in indices]

        # Initialize metrics aggregation
        n_classes = len(comparison.classes)
        aggregated_cm = np.zeros((n_classes, n_classes), dtype=np.int32)
        eco_region_results = {eco_id: {
            'confusion_matrix': np.zeros((n_classes, n_classes), dtype=np.int32),
            'forest_pixels': 0,
            'weighted_r1_coverage': 0.0,
            'weighted_r2_coverage': 0.0
        } for eco_id in comparison.eco_region_classes.keys()}

        # Process each window
        for idx, window in enumerate(tqdm(selected_windows, desc="Processing windows")):
            self.logger.info(f"Processing window {idx + 1}/{len(selected_windows)}")
            window_metrics = comparison._process_window(window)

            if window_metrics is not None:
                # Aggregate global metrics
                aggregated_cm += window_metrics.confusion_matrix

                # Aggregate eco-region-specific metrics
                for eco_id, eco_metrics in window_metrics.eco_region_metrics.items():
                    eco_region_results[eco_id]['confusion_matrix'] += eco_metrics.confusion_matrix
                    eco_region_results[eco_id]['forest_pixels'] += eco_metrics.forest_pixels
                    eco_region_results[eco_id]['weighted_r1_coverage'] += eco_metrics.raster1_coverage * eco_metrics.forest_pixels
                    eco_region_results[eco_id]['weighted_r2_coverage'] += eco_metrics.raster2_coverage * eco_metrics.forest_pixels
            else:
                self.logger.error(f"Failed to compute metrics for window {idx + 1}")

        # Compute and log eco-region-specific metrics
        self.logger.info("\nEco-region-specific Metrics:")
        for eco_id, eco_data in eco_region_results.items():
            if eco_data['forest_pixels'] > 0:
                overall_agreement = np.diag(eco_data['confusion_matrix']).sum() / eco_data['confusion_matrix'].sum() \
                    if eco_data['confusion_matrix'].sum() > 0 else 0
                raster1_coverage = eco_data['weighted_r1_coverage'] / eco_data['forest_pixels']
                raster2_coverage = eco_data['weighted_r2_coverage'] / eco_data['forest_pixels']
                region_name = comparison.eco_region_classes[eco_id]

                self.logger.info(f"\n{region_name}:")
                self.logger.info(f"  Agreement: {overall_agreement:.4f}")
                self.logger.info(f"  Raster1 Coverage: {raster1_coverage:.4f}")
                self.logger.info(f"  Raster2 Coverage: {raster2_coverage:.4f}")
                self.logger.info(f"  Forest Pixels: {eco_data['forest_pixels']}")

        # Create confusion matrix visualization for global metrics
        plt.figure(figsize=(10, 8))
        sns.heatmap(aggregated_cm, annot=True, fmt='d',
                    xticklabels=comparison.classes, yticklabels=comparison.classes,
                    cmap='Blues', cbar_kws={'label': 'Pixel Count'})
        plt.title('Aggregated Confusion Matrix (Global)')
        plt.xlabel('BDForet')
        plt.ylabel('DLT')
        plt.savefig(self.output_dir / "global_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
