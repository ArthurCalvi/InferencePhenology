#!/usr/bin/env python3
"""Test module for RasterComparison metrics computation on a single window with forest mask and eco-regions."""

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

class TestWindowMetricsWithEcoRegions(unittest.TestCase):
    """Test class for computing metrics on a single window with forest mask and eco-regions."""

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
    
    def test_process_single_window_with_eco_regions(self):
        """Test processing of a single window and visualize results with forest mask and eco-regions."""
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

        # Get the first window
        first_window = comparison.processing_windows[len(comparison.processing_windows) // 4]
        window_bounds = windows.bounds(first_window, comparison.common_extent.transform)
        
        self.logger.info(f"Processing window at bounds: {window_bounds}")
        
        # Process the window
        window_metrics = comparison._process_window(first_window)
        
        if window_metrics is not None:
            self._visualize_confusion_matrices(window_metrics)
            self._visualize_window_data(comparison, first_window)
            self._log_metrics(window_metrics)

    def _visualize_confusion_matrices(self, metrics):
        """Create and save confusion matrix visualizations for global and eco-region results."""
        # Global confusion matrix
        self._plot_confusion_matrix(
            metrics.confusion_matrix,
            title='Global Confusion Matrix (Forest Areas Only)',
            output_path=self.output_dir / "global_confusion_matrix.png"
        )
        
        # Eco-region confusion matrices
        if metrics.eco_region_metrics:
            for eco_id, eco_metrics in metrics.eco_region_metrics.items():
                region_name = self.eco_region_classes[eco_id]
                self._plot_confusion_matrix(
                    eco_metrics.confusion_matrix,
                    title=f'Confusion Matrix for {region_name}',
                    output_path=self.output_dir / f"confusion_matrix_{region_name.lower().replace(' ', '_')}.png"
                )

    def _plot_confusion_matrix(self, cm, title, output_path):
        """Plot and save a single confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Convert to percentages
        cm_sum = cm.sum()
        if cm_sum > 0:
            cm_percentages = cm / cm_sum * 100
        else:
            cm_percentages = cm
            
        # Create heatmap
        sns.heatmap(
            cm_percentages,
            annot=np.array([[f'{val:.1f}%\n({int(count)})' 
                            for val, count in zip(row_pct, row_count)]
                           for row_pct, row_count in zip(cm_percentages, cm)]),
            fmt='',
            xticklabels=['deciduous', 'evergreen'],
            yticklabels=['deciduous', 'evergreen']
        )
        
        plt.title(title)
        plt.xlabel('BDForet')
        plt.ylabel('DLT')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_window_data(self, comparison, window):
        """Create and save visualizations of the window data including eco-regions."""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        axes = [
            fig.add_subplot(gs[0, 0]),  # BDForet
            fig.add_subplot(gs[0, 1]),  # DLT
            fig.add_subplot(gs[0, 2]),  # Forest Mask
            fig.add_subplot(gs[1, 0]),  # Eco-regions
            fig.add_subplot(gs[1, 1]),  # Agreement Map
            fig.add_subplot(gs[1, 2]),  # Legend
        ]
        
        window_bounds = windows.bounds(window, comparison.common_extent.transform)
        
        # Create custom colormaps
        colors_class = ['#1f77b4', '#ff7f0e']
        class_cmap = LinearSegmentedColormap.from_list('custom', colors_class)
        
        # Plot each layer
        self._plot_classification_layer(
            comparison.raster1_path, window_bounds, axes[0],
            'BDForet Classification', class_cmap, self.bdforet_classes
        )
        
        self._plot_classification_layer(
            comparison.raster2_path, window_bounds, axes[1],
            'DLT Classification', class_cmap, self.dlt_classes
        )
        
        self._plot_forest_mask(
            comparison.forest_mask_path, window_bounds, axes[2]
        )
        
        self._plot_eco_regions(
            comparison.eco_region_path, window_bounds, axes[3],
            self.eco_region_classes
        )
        
        self._plot_agreement_map(
            comparison, window, window_bounds, axes[4]
        )
        
        # Add eco-region legend
        axes[5].axis('off')
        self._add_eco_region_legend(axes[5])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "window_visual_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_classification_layer(self, path, bounds, ax, title, cmap, class_mapping):
        """Plot a classification layer with proper masking."""
        with rasterio.open(path) as src:
            window = windows.from_bounds(*bounds, src.transform)
            data = src.read(1, window=window)
            valid_values = [k for k, v in class_mapping.items() if v != 'nodata']
            masked_data = np.ma.masked_where(~np.isin(data, valid_values), data)
            im = ax.imshow(masked_data, cmap=cmap, vmin=min(valid_values), vmax=max(valid_values))
            ax.set_title(title)
            return im

    def _plot_forest_mask(self, path, bounds, ax):
        """Plot the forest mask."""
        with rasterio.open(path) as src:
            window = windows.from_bounds(*bounds, src.transform)
            mask_data = src.read(1, window=window)
            mask_data_masked = np.ma.masked_where(mask_data != 1, mask_data)
            im = ax.imshow(mask_data_masked, cmap='Greens', vmin=0, vmax=1)
            ax.set_title('Forest Mask')
            return im

    def _plot_eco_regions(self, path, bounds, ax, eco_region_classes):
        """Plot eco-regions with custom colormap."""
        with rasterio.open(path) as src:
            window = windows.from_bounds(*bounds, src.transform)
            eco_data = src.read(1, window=window)
            
            # Create colormap for eco-regions
            n_regions = len(eco_region_classes)
            colors = plt.cm.tab20(np.linspace(0, 1, n_regions))
            eco_cmap = LinearSegmentedColormap.from_list('eco_regions', colors)
            
            im = ax.imshow(eco_data, cmap=eco_cmap, vmin=1, vmax=len(eco_region_classes))
            ax.set_title('Eco-regions')
            return im

    def _plot_agreement_map(self, comparison, window, bounds, ax):
        """Plot the agreement map."""
        metrics = comparison._process_window(window)
        if metrics is not None:
            window_shape = (window.height, window.width)
            agreement_data = np.full(window_shape, np.nan)
            with rasterio.open(comparison.forest_mask_path) as src:
                window_mask = windows.from_bounds(*bounds, src.transform)
                forest_mask = src.read(1, window=window_mask, out_shape=window_shape)
                valid_mask = forest_mask == 1
                agreement_data[valid_mask] = 1 if metrics.confusion_matrix.sum() > 0 else 0
            
            im = ax.imshow(agreement_data, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title('Agreement Map (Forest Only)')
            return im

    def _add_eco_region_legend(self, ax):
        """Add a legend for eco-regions."""
        handles = []
        for eco_id, name in self.eco_region_classes.items():
            color = plt.cm.tab20(eco_id / len(self.eco_region_classes))
            handles.append(plt.Rectangle((0,0), 1, 1, fc=color))
        
        ax.legend(handles, self.eco_region_classes.values(),
                 loc='center', title='Eco-regions')

    def _log_metrics(self, metrics):
        """Log detailed metrics for the window."""
        self.logger.info("\nGlobal Metrics:")
        self._log_single_metrics(metrics)
        
        if metrics.eco_region_metrics:
            self.logger.info("\nEco-region Metrics:")
            for eco_id, eco_metrics in metrics.eco_region_metrics.items():
                region_name = self.eco_region_classes[eco_id]
                self.logger.info(f"\n{region_name}:")
                self._log_single_metrics(eco_metrics)

    def _log_single_metrics(self, metrics):
        """Log metrics for a single region (global or eco-region)."""
        total_pixels = metrics.confusion_matrix.sum()
        correct_pixels = np.diag(metrics.confusion_matrix).sum()
        accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
        
        self.logger.info(f"Forest pixels: {metrics.forest_pixels}")
        self.logger.info(f"Valid pixels: {total_pixels}")
        self.logger.info(f"Correct pixels: {correct_pixels}")
        self.logger.info(f"Overall accuracy: {accuracy:.4f}")
        self.logger.info(f"Raster1 (BDForet) coverage: {metrics.raster1_coverage:.4f}")
        self.logger.info(f"Raster2 (DLT) coverage: {metrics.raster2_coverage:.4f}")

if __name__ == '__main__':
    unittest.main(verbosity=2)