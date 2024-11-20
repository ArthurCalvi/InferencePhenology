#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
import subprocess

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logger() -> logging.Logger:
    logger = logging.getLogger('test_run_inference')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def main():
    logger = setup_logger()
    
    # Define paths
    base_dir = Path("/Users/arthurcalvi/Repo/InferencePhenology")
    config_dir = base_dir / "test/test_outputs/phenology_inference/configs"
    run_inference_script = base_dir / "src_inference/run_inference.py"

    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        sys.exit(1)

    try:
        # Test processing first tile (index 0)
        cmd = [
            sys.executable,
            str(run_inference_script),
            "--config-dir", str(config_dir),
            "--tile-idx", "0"
        ]
        
        logger.info("Running inference with command:")
        logger.info(" ".join(cmd))
        
        result = subprocess.run(cmd, 
                              check=True,
                              capture_output=True,
                              text=True)
        
        logger.info("Inference output:")
        print(result.stdout)
        
        # Verify results
        output_dir = base_dir / "test/test_outputs/phenology_inference"
        prob_maps = list(output_dir.glob("prob_map_tile*.tif"))
        
        if not prob_maps:
            raise RuntimeError("No probability maps generated")
            
        logger.info(f"Generated probability maps: {[p.name for p in prob_maps]}")
        logger.info("Test completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error("Inference failed with error:")
        logger.error(e.stderr)
        raise
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()