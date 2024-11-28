#!/usr/bin/env python3
"""
Test script for prepare_inference.py with local paths.
"""
import os
import sys
import logging
import shutil
from pathlib import Path
import subprocess

# Add the parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logger() -> logging.Logger:
    """Setup logger with appropriate formatting."""
    logger = logging.getLogger('test_prepare_inference')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def main():
    logger = setup_logger()
    
    # Define local paths
    base_dir = Path("/Users/arthurcalvi/Repo/InferencePhenology")
    mosaic_dir = base_dir / "data/mosaics"
    dem_dir = base_dir / "data/dem"
    output_dir = base_dir / "test/test_outputs"
    model_path = base_dir / "model/best_model_with_bdforet_no_resampled_weights_h2_y1_iter10_scaled01_featuresfromRFECV_nf10_f1_0.9601.pkl"
    
    # Path to prepare_inference.py
    prepare_inference_script = base_dir / "src_inference/prepare_inference.py"
    
    # Clean output directory if it exists
    if output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input paths
    for path, name in [(mosaic_dir, "Mosaic directory"), 
                      (dem_dir, "DEM directory"),
                      (model_path, "Model file"),
                      (prepare_inference_script, "Prepare inference script")]:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            sys.exit(1)
            
    logger.info("Starting prepare_inference.py test")
    
    try:
        # Construct command
        cmd = [
            sys.executable,  # Current Python interpreter
            str(prepare_inference_script),
            "--mosaic-dir", str(mosaic_dir),
            "--dem-dir", str(dem_dir),
            "--output-dir", str(output_dir),
            "--model-path", str(model_path),
            "--num-harmonics", "2",
            "--max-iter", "1",
            "--workers-per-tile", "8",
            "--window-size", "1024",
            "--years", "2021", "2022"
        ]
        
        # Run prepare_inference.py
        logger.info("Running prepare_inference.py with command:")
        logger.info(" ".join(cmd))
        
        result = subprocess.run(cmd, 
                              check=True,
                              capture_output=True,
                              text=True)
        
        # Print output
        logger.info("prepare_inference.py output:")
        for line in result.stdout.split('\n'):
            if line.strip():
                logger.info(line)
                
        # Check if configuration files were created
        inference_dir = output_dir / "phenology_inference"
        config_dir = inference_dir / "configs"
        summary_file = inference_dir / "job_summary.txt"
        metadata_file = config_dir / "metadata.json"
        
        # Verify directory structure
        if not config_dir.exists():
            raise RuntimeError(f"Config directory not created: {config_dir}")
        
        if not summary_file.exists():
            raise RuntimeError(f"Summary file not created: {summary_file}")
            
        if not metadata_file.exists():
            raise RuntimeError(f"Metadata file not created: {metadata_file}")
            
        # Verify config files
        config_files = list(config_dir.glob("tile_config_*.json"))
        if not config_files:
            raise RuntimeError("No tile configuration files created")
            
        logger.info(f"Created {len(config_files)} tile configuration files")
        
        # Print summary file content
        logger.info("\nJob Summary:")
        with open(summary_file) as f:
            print(f.read())
            
        logger.info("Test completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error("prepare_inference.py failed with error:")
        logger.error(e.stderr)
        raise
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()