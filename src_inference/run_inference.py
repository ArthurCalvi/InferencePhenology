#!/usr/bin/env python3
"""
Script to run inference for a single tile using configurations prepared by prepare_inference.py
"""
import argparse
from pathlib import Path
import logging
import json
from typing import Dict
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_inference.inference import FolderInference

def setup_logger() -> logging.Logger:
    """Setup logger with appropriate formatting."""
    logger = logging.getLogger('run_inference')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def load_metadata(config_dir: Path) -> Dict:
    """Load metadata from configuration directory."""
    metadata_file = config_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
    with open(metadata_file) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Run inference for a single tile')
    parser.add_argument('--config-dir', type=Path, required=True,
                      help='Directory containing configuration files')
    parser.add_argument('--tile-idx', type=int, required=True,
                      help='Index of the tile to process')
    
    args = parser.parse_args()
    logger = setup_logger()
    
    try:
        # Load metadata
        metadata = load_metadata(args.config_dir)
        
        # Validate tile index
        if args.tile_idx >= metadata['num_tiles']:
            raise ValueError(f"Invalid tile index {args.tile_idx}. "
                           f"Max index is {metadata['num_tiles']-1}")
        
        # Initialize FolderInference
        folder_inference = FolderInference.from_config(
            config_dir=args.config_dir,
            logger=logger
        )
        
        # Process tile and get progress
        logger.info(f"Processing tile {args.tile_idx}/{metadata['num_tiles']-1}")
        folder_inference.process_single_tile(args.tile_idx)
        
        # Log progress
        progress = folder_inference.get_progress()
        logger.info(f"Progress: {progress['progress_percentage']:.1f}% complete "
                   f"({progress['completed_tiles']}/{progress['total_tiles']} tiles)")
        
    except Exception as e:
        logger.error(f"Failed to process tile {args.tile_idx}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()