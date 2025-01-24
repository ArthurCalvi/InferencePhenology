#!/usr/bin/env python3
"""
Comparison script between MyInference and DLT, and MyInference and BDForet.
Optimized for running on JeanZay supercomputer.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm

from compare_two_rasters import RasterComparison

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare phenology classification maps')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Directory to save outputs')
    parser.add_argument('--window-size', type=int, default=5120,
                       help='Size of processing windows')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of worker processes')
    return parser.parse_args()

def setup_logger(output_dir: Path) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('comparison_metrics')
    logger.setLevel(logging.INFO)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(output_dir / 'comparison_metrics.log')
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def save_metrics_json(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save metrics dictionary to JSON file."""
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

def save_metrics_csv(metrics: Dict[str, Any], output_dir: Path, prefix: str = "") -> None:
    """Save metrics to CSV files for easy analysis."""
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
    global_df.to_csv(output_dir / f'{prefix}global_metrics.csv', index=False)
    
    # Confusion matrix
    conf_matrix_df = pd.DataFrame(
        global_data['confusion_matrix'],
        columns=global_data['class_names'],
        index=global_data['class_names']
    )
    conf_matrix_df.to_csv(output_dir / f'{prefix}confusion_matrix.csv')
    
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
    eco_df.to_csv(output_dir / f'{prefix}eco_region_metrics.csv', index=False)

def main():
    args = parse_args()
    
    # Setup paths
    dlt_path = Path("/linkhome/rech/gennjv01/uyr48jk/work/InferencePhenology/data/DLT_2018_010m_fr_03035_v020/DLT_Dominant_Leaf_Type_France.tif")
    bdforet_path = Path("/linkhome/rech/gennjv01/uyr48jk/work/InferencePhenology/data/bdforet_10_FF1_FF2_EN_year_raster.tif")
    myinference_path = Path("/linkhome/rech/gennjv01/uyr48jk/work/InferencePhenology/data/classification_map.tif")
    forest_mask_path = Path("/linkhome/rech/gennjv01/uyr48jk/work/InferencePhenology/data/mask_forest.tif")
    eco_region_path = Path("/linkhome/rech/gennjv01/uyr48jk/work/InferencePhenology/data/greco.tif")

    # Setup logging
    logger = setup_logger(args.output_dir)
    logger.info("Starting comparison metrics computation")
    
    # Class mappings
    myinference_classes = {
        1: 'deciduous',
        2: 'evergreen'
    }
    
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
    
    # Comparison configurations
    comparisons = [
        {
            'raster1_path': myinference_path,
            'raster2_path': dlt_path,
            'map1_classes': myinference_classes,
            'map2_classes': dlt_classes,
            'prefix': 'MyInference_vs_DLT_'
        },
        {
            'raster1_path': myinference_path,
            'raster2_path': bdforet_path,
            'map1_classes': myinference_classes,
            'map2_classes': bdforet_classes,
            'prefix': 'MyInference_vs_BDForet_'
        }
    ]
    
    for comp in comparisons:
        try:
            logger.info(f"\nStarting comparison: {comp['prefix'].strip('_')}")
            comparison = RasterComparison(
                raster1_path=comp['raster1_path'],
                raster2_path=comp['raster2_path'],
                forest_mask_path=forest_mask_path,
                eco_region_path=eco_region_path,
                map1_classes=comp['map1_classes'],
                map2_classes=comp['map2_classes'],
                eco_region_classes=eco_region_classes,
                window_size=args.window_size,
                max_workers=args.max_workers,
                logger=logger
            )
            
            logger.info("Computing metrics...")
            metrics = comparison.compute_metrics()
            
            # Save results
            logger.info("Saving results...")
            json_path = args.output_dir / f'{comp["prefix"]}metrics.json'
            save_metrics_json(metrics, json_path)
            save_metrics_csv(metrics, args.output_dir, prefix=comp['prefix'])
            
            # Log summary
            logger.info("\nSummary Statistics:")
            logger.info(f"Overall Agreement: {metrics['global']['overall_agreement']:.4f}")
            logger.info(f"Total Forest Pixels: {metrics['global']['total_forest_pixels']:,}")
            logger.info(f"Raster1 Coverage: {metrics['global']['raster1_coverage']:.4f}")
            logger.info(f"Raster2 Coverage: {metrics['global']['raster2_coverage']:.4f}")
            
        except Exception as e:
            logger.error(f"Error during computation for {comp['prefix'].strip('_')}: {str(e)}")
            logger.exception("Full traceback:")
            continue

if __name__ == '__main__':
    main()