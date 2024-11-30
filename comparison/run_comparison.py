# run_comparison_metrics.py
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any

from compare_two_rasters import RasterComparison

def setup_logger(output_dir: Path) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log file
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger('comparison_metrics')
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    file_handler = logging.FileHandler(output_dir / 'comparison_metrics.log')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def save_metrics_json(metrics: Dict[str, Any], output_path: Path) -> None:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary containing metrics
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(i) for i in obj]
        return obj
    
    metrics_json = convert_numpy(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)

def save_metrics_csv(metrics: Dict[str, Any], output_dir: Path) -> None:
    """
    Save metrics to CSV files for easy analysis.
    
    Args:
        metrics: Dictionary containing metrics
        output_dir: Directory to save CSV files
    """
    # Global metrics
    global_data = metrics['global']
    global_df = pd.DataFrame({
        'metric': ['overall_agreement', 'raster1_coverage', 'raster2_coverage', 'total_forest_pixels'],
        'value': [
            global_data['overall_agreement'],
            global_data['raster1_coverage'],
            global_data['raster2_coverage'],
            global_data['total_forest_pixels']
        ]
    })
    global_df.to_csv(output_dir / 'global_metrics.csv', index=False)
    
    # Confusion matrix
    conf_matrix_df = pd.DataFrame(
        global_data['confusion_matrix'],
        columns=global_data['class_names'],
        index=global_data['class_names']
    )
    conf_matrix_df.to_csv(output_dir / 'confusion_matrix.csv')
    
    # Eco-region metrics
    eco_metrics = []
    for eco_id, eco_data in metrics['eco_regions'].items():
        eco_metrics.append({
            'eco_region': eco_data['region_name'],
            'overall_agreement': eco_data['overall_agreement'],
            'raster1_coverage': eco_data['raster1_coverage'],
            'raster2_coverage': eco_data['raster2_coverage'],
            'forest_pixels': eco_data['forest_pixels']
        })
    
    eco_df = pd.DataFrame(eco_metrics)
    eco_df.to_csv(output_dir / 'eco_region_metrics.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Compute comparison metrics between two raster datasets')
    
    parser.add_argument('--bdforet-path', type=Path, required=True,
                       help='Path to BDForet raster')
    parser.add_argument('--dlt-path', type=Path, required=True,
                       help='Path to DLT raster')
    parser.add_argument('--forest-mask-path', type=Path, required=True,
                       help='Path to forest mask raster')
    parser.add_argument('--eco-region-path', type=Path, required=True,
                       help='Path to eco-region raster')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--window-size', type=int, default=10240,
                       help='Size of processing windows (default: 10240)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir / f'comparison_results_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(output_dir)
    logger.info(f"Starting comparison metrics computation")
    logger.info(f"Output directory: {output_dir}")
    
    # Class mappings
    bdforet_classes = {
        1: 'deciduous',
        2: 'evergreen',
        0: 'nodata'
    }
    
    dlt_classes = {
        1: 'deciduous',
        2: 'evergreen',
        255: 'nodata'
    }
    
    eco_region_classes = {
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
    
    try:
        # Initialize comparison object
        comparison = RasterComparison(
            raster1_path=args.bdforet_path,
            raster2_path=args.dlt_path,
            forest_mask_path=args.forest_mask_path,
            eco_region_path=args.eco_region_path,
            map1_classes=bdforet_classes,
            map2_classes=dlt_classes,
            eco_region_classes=eco_region_classes,
            window_size=args.window_size,
            max_workers=args.max_workers,
            logger=logger
        )
        
        logger.info("Computing metrics...")
        metrics = comparison.compute_metrics()
        
        # Save results
        logger.info("Saving results...")
        
        # Save as JSON
        json_path = output_dir / 'metrics.json'
        save_metrics_json(metrics, json_path)
        logger.info(f"Saved metrics to {json_path}")
        
        # Save as CSV
        save_metrics_csv(metrics, output_dir)
        logger.info(f"Saved detailed metrics as CSV files in {output_dir}")
        
        # Log summary statistics
        logger.info("\nSummary Statistics:")
        logger.info(f"Overall Agreement: {metrics['global']['overall_agreement']:.4f}")
        logger.info(f"Total Forest Pixels: {metrics['global']['total_forest_pixels']:,}")
        logger.info(f"BDForet Coverage: {metrics['global']['raster1_coverage']:.4f}")
        logger.info(f"DLT Coverage: {metrics['global']['raster2_coverage']:.4f}")
        
        logger.info("\nEco-region Statistics:")
        for eco_id, eco_data in metrics['eco_regions'].items():
            logger.info(f"\n{eco_data['region_name']}:")
            logger.info(f"  Agreement: {eco_data['overall_agreement']:.4f}")
            logger.info(f"  Forest Pixels: {eco_data['forest_pixels']:,}")
        
        logger.info("\nComparison metrics computation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during computation: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == '__main__':
    main()