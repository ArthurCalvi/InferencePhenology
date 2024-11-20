# test_dem_pipeline.py
from pathlib import Path
import sys
import unittest
import shutil
import json
import logging
import rasterio
import numpy as np

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from src_inference.collect_dem_info import DEMInfoCollector, setup_logging as setup_collect_logging
from src_inference.download_dem import DEMDownloader, setup_logging as setup_download_logging

class TestDEMPipeline(unittest.TestCase):
    """Test class for the DEM data collection and download pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Define paths
        cls.mosaic_dir = Path("/Users/arthurcalvi/Repo/InferencePhenology/data/mosaics/2021/01-01_plus_minus_30_days")
        cls.output_base = Path("/Users/arthurcalvi/Repo/InferencePhenology/test/test_outputs")
        cls.temp_dir = cls.output_base / "temp"
        
        # Create test directories
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        cls.info_file = cls.temp_dir / "dem_info.json"
        cls.dem_output_dir = cls.temp_dir / "dem_files"
        
        # Setup logging
        cls.collect_logger = setup_collect_logging(logging.DEBUG)
        cls.download_logger = setup_download_logging(logging.DEBUG)

    def test_01_collect_dem_info(self):
        """Test DEM information collection."""
        collector = DEMInfoCollector(self.collect_logger)
        
        # Process directory
        collector.process_directory(self.mosaic_dir, self.info_file)
        
        # Verify file was created and contains valid data
        self.assertTrue(self.info_file.exists())
        with open(self.info_file) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        
        # Log some statistics
        self.collect_logger.info(f"Collected information for {len(data)} tiles")
        return data

    def test_02_download_dem(self):
        """Test DEM download and processing."""
        # Skip if info file doesn't exist
        if not self.info_file.exists():
            self.skipTest("DEM info file not found. Run test_01_collect_dem_info first.")
        
        downloader = DEMDownloader(self.download_logger)
        
        # Process tiles
        downloader.process_tiles(
            self.info_file,
            self.dem_output_dir,
            num_workers=1  # Use single worker for testing
        )
        
        # Verify output files
        dem_files = list(self.dem_output_dir.glob("*.tif"))
        self.assertTrue(len(dem_files) > 0)
        
        # Check first DEM file
        with rasterio.open(dem_files[0]) as src:
            data = src.read(1)
            self.assertIsInstance(data, np.ndarray)
            self.assertTrue(data.size > 0)
            self.assertTrue(np.any(~np.isnan(data)))
        
        self.download_logger.info(f"Successfully created {len(dem_files)} DEM files")

    def test_03_validate_dem_files(self):
        """Validate the generated DEM files."""
        # Skip if output directory doesn't exist
        if not self.dem_output_dir.exists():
            self.skipTest("DEM output directory not found. Run test_02_download_dem first.")
            
        dem_files = list(self.dem_output_dir.glob("*.tif"))
        
        for dem_file in dem_files:
            with rasterio.open(dem_file) as src:
                # Check basic properties
                self.assertEqual(src.count, 1)  # Should be single band
                self.assertEqual(src.dtypes[0], 'float32')
                
                # Read data and check for valid values
                data = src.read(1)
                self.assertTrue(np.any(~np.isnan(data)))
                
                # Log some statistics
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    self.download_logger.info(
                        f"DEM file {dem_file.name} stats - "
                        f"Min: {np.min(valid_data):.2f}, "
                        f"Max: {np.max(valid_data):.2f}, "
                        f"Mean: {np.mean(valid_data):.2f}"
                    )

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Uncomment the following line to clean up test files
        # shutil.rmtree(cls.temp_dir, ignore_errors=True)
        pass

if __name__ == "__main__":
    unittest.main(verbosity=2)