from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
from joblib import load
from pathlib import Path
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import yaml

# Add the parent directory to Python path to import the module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_inference.utils import compute_indices, fit_periodic_function_with_harmonics_robust, calculate_optimal_windows

@dataclass
class BandData:
    """
    Data class to hold band reflectance time series and metadata.
    
    Attributes:
        b2: Blue band time series
        b4: Red band time series
        b8: NIR band time series
        b11: SWIR1 band time series
        b12: SWIR2 band time series
        dates: List of acquisition dates
        dem: Optional Digital Elevation Model
        cloud_mask: Optional cloud probability mask (0-1 range, 1 means clear)
    """
    b2: np.ndarray  # Blue band
    b4: np.ndarray  # Red band
    b8: np.ndarray  # NIR band
    b11: np.ndarray  # SWIR1 band
    b12: np.ndarray  # SWIR2 band
    dates: List[datetime]  # Acquisition dates
    dem: Optional[np.ndarray] = None  # Digital Elevation Model
    cloud_mask: Optional[np.ndarray] = None  # Cloud probability mask

    def __post_init__(self):
        """Validate input data dimensions and types."""
        # Check that all bands have the same shape
        shapes = {
            'b2': self.b2.shape,
            'b4': self.b4.shape,
            'b8': self.b8.shape,
            'b11': self.b11.shape,
            'b12': self.b12.shape
        }
        if len(set(shapes.values())) > 1:
            raise ValueError(f"All bands must have the same shape. Got shapes: {shapes}")
        
        # Check temporal dimension matches dates
        if self.b2.shape[0] != len(self.dates):
            raise ValueError(f"Temporal dimension ({self.b2.shape[0]}) must match "
                           f"number of dates ({len(self.dates)})")
        
        # Check DEM shape if provided
        if self.dem is not None:
            if self.dem.shape != self.b2.shape[1:]:
                raise ValueError(f"DEM shape {self.dem.shape} must match spatial "
                               f"dimensions of bands {self.b2.shape[1:]}")
                
        # Check cloud mask shape and values if provided
        if self.cloud_mask is not None:
            if self.cloud_mask.shape != self.b2.shape:
                raise ValueError(f"Cloud mask shape {self.cloud_mask.shape} must match "
                               f"band data shape {self.b2.shape}")
            if not ((self.cloud_mask >= 0) & (self.cloud_mask <= 1)).all():
                raise ValueError("Cloud mask values must be between 0 and 1")

## WINDOW INFERENCE CLASS ## 

class WindowInference:
    """Class to perform inference on a single window of time series data."""
    
    # List of required features for the model
    REQUIRED_FEATURES = [
        'amplitude_evi_h1', 'amplitude_nbr_h1', 'amplitude_ndvi_h1',
        'cos_phase_crswir_h1', 'cos_phase_nbr_h1', 'cos_phase_ndvi_h1',
        'offset_crswir', 'offset_evi', 'offset_nbr', 'elevation'
    ]
    
    def __init__(
        self,
        band_data: BandData,
        model: Any = None,
        num_harmonics: int = 2,
        max_iter: int = 1,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize the WindowInference class.
        
        Args:
            band_data: BandData object containing all required band time series
            model_path: Path to the pickled model file
            num_harmonics: Number of harmonics for periodic function fitting
            max_iter: Maximum iterations for harmonic fitting
            logger: Optional logger for tracking progress and debugging
        """
        self.band_data = band_data
        self.num_harmonics = num_harmonics
        self.max_iter = max_iter
        self.logger = logger or logging.getLogger(__name__)
        
        # Load model if path provided
        self.model = model
                
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        if self.num_harmonics < 1:
            raise ValueError("num_harmonics must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
            
    # def _get_quality_weights(self) -> np.ndarray:
    #     """
    #     Get quality weights for feature computation.
        
    #     Returns:
    #         Array of quality weights (1 for good quality, 0 for bad quality)
    #     """
    #     if self.band_data.cloud_mask is not None:
    #         self.logger.info("Using provided cloud mask as quality weights")
    #         return self.band_data.cloud_mask
    #     else:
    #         self.logger.info("No cloud mask provided, using default weights of 1.0")
    #         return np.ones_like(self.band_data.b2)
    def _get_quality_weights(self) -> np.ndarray:
        """
        Get quality weights for feature computation.
        NaN values in any band set the weight to 0.
        
        Returns:
            Array of quality weights (1 for good quality, 0 for bad quality or NaN)
        """
        try:
            if self.logger:
                self.logger.info("Computing quality weights with NaN masking")
                
            # Start with provided cloud mask if available
            if self.band_data.cloud_mask is not None:
                weights = self.band_data.cloud_mask
                if self.logger:
                    self.logger.debug("Using provided cloud mask as initial weights")
            else:
                weights = np.ones_like(self.band_data.b2)
                if self.logger:
                    self.logger.debug("No cloud mask provided, using default weights of 1.0")
                    
            # Create NaN mask for each band
            nan_masks = [
                np.isnan(band) for band in [
                    self.band_data.b2,
                    self.band_data.b4,
                    self.band_data.b8,
                    self.band_data.b11,
                    self.band_data.b12
                ]
            ]
            
            # Combine NaN masks (if any band has NaN, weight should be 0)
            combined_nan_mask = np.any(nan_masks, axis=0)
            
            # Update weights: multiply by (1 - NaN mask) to set weights to 0 where there are NaNs
            weights = weights * (~combined_nan_mask)
            
            if self.logger:
                self.logger.debug(f"NaN-masked pixels: {np.sum(combined_nan_mask)} out of {combined_nan_mask.size}")
                self.logger.debug(f"Final zero weights: {np.sum(weights == 0)} out of {weights.size}")
                
            return weights
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to compute quality weights: {str(e)}")
            raise
            
    def compute_indices(self) -> Dict[str, np.ndarray]:
        """
        Compute spectral indices from reflectance bands.
        
        Returns:
            Dictionary containing computed indices (ndvi, evi, nbr, crswir)
        """
        try:
            self.logger.info("Computing spectral indices")
            
            ndvi, evi, nbr, crswir = compute_indices(
                self.band_data.b2,
                self.band_data.b4,
                self.band_data.b8,
                self.band_data.b11,
                self.band_data.b12,
                logger=self.logger
            )
            
            return {
                'ndvi': ndvi,
                'evi': evi,
                'nbr': nbr,
                'crswir': crswir
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compute indices: {str(e)}")
            raise
            
    def compute_features(self) -> Dict[str, np.ndarray]:
        """
        Compute temporal features using periodic function fitting.
        
        Returns:
            Dictionary containing all computed features
        """
        try:
            self.logger.info("Computing temporal features")
            
            # Get quality weights and indices
            qa_weights = self._get_quality_weights()
            indices = self.compute_indices()
            
            # Compute harmonic features for each index
            features = {}
            for index_name, index_data in indices.items():
                results = fit_periodic_function_with_harmonics_robust(
                    index_data,
                    qa_weights,
                    self.band_data.dates,
                    num_harmonics=self.num_harmonics,
                    max_iter=self.max_iter,
                    logger=self.logger
                )
                features[index_name] = results
                
            # Organize features into final format
            feature_data = {}
            
            # Process each index's results
            for index_name, result in features.items():
                # Store amplitudes
                for i in range(self.num_harmonics):
                    feature_name = f'amplitude_{index_name}_h{i+1}'
                    feature_data[feature_name] = result[i].reshape(-1)
                    
                # Store phases as cosine
                for i in range(self.num_harmonics):
                    cos_name = f'cos_phase_{index_name}_h{i+1}'
                    phase = result[self.num_harmonics + i]
                    feature_data[cos_name] = np.cos(phase).reshape(-1)
                    
                # Store offset
                feature_data[f'offset_{index_name}'] = result[-2].reshape(-1)  # -2 because last is variance
                
            # Add DEM if available
            if self.band_data.dem is not None:
                feature_data['elevation'] = self.band_data.dem.reshape(-1)
            else:
                self.logger.warning("DEM not provided, using zeros for elevation")
                feature_data['elevation'] = np.zeros(feature_data['offset_ndvi'].shape)
                
            self.logger.info(f"Computed {len(feature_data)} features")
            return feature_data
            
        except Exception as e:
            self.logger.error(f"Failed to compute features: {str(e)}")
            raise
            
    def run_inference(self) -> np.ndarray:
        """
        Run full inference pipeline: compute features and apply model.
        
        Returns:
            Array containing probability map for each pixel
            
        Raises:
            RuntimeError: If model is not loaded
            Exception: If inference fails
        """
        try:
            if self.model is None:
                raise RuntimeError("No model loaded for inference")
                
            # Compute all features
            feature_data = self.compute_features()
            
            # Create DataFrame with required features
            df = pd.DataFrame(feature_data)
            df = df[self.REQUIRED_FEATURES]  # Select only required features
            df = df.fillna(0)  # Handle any missing values
            
            # Run model inference
            self.logger.info("Running model inference")
            probabilities = self.model.predict_proba(df)
            
            # Reshape probabilities back to spatial dimensions
            spatial_shape = self.band_data.b2.shape[1:]  # Get spatial dimensions from input data
            prob_map = probabilities[:, 1].reshape(*spatial_shape)  # Use class 1 probabilities
            
            self.logger.info("Inference completed successfully")
            return prob_map
            
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            raise

## TILE INFERENCE CLASS ##

class TileInference:
    """Class to perform inference on a complete tile using windows."""
    
    def __init__(
        self,
        mosaic_paths: List[Path],
        dates: List[datetime],
        dem_path: Path,
        output_dir: Path,
        tile_id: str,
        model_path: Path,
        cloud_mask_paths: Optional[List[Path]] = None,
        window_size: int = 1024,
        num_harmonics: int = 2,
        max_iter: int = 1,
        max_workers: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize TileInference with explicit paths and dates.
        
        Args:
            mosaic_paths: List of paths to Sentinel-2 mosaic files (bands order should be B02:1, B04:3, B08:4, B11:9, B12:10)
            dates: List of dates corresponding to the mosaic files
            dem_path: Path to the DEM file
            output_dir: Directory to save outputs
            tile_id: ID of the tile to process
            model_path: Path to the model file
            cloud_mask_paths: Optional list of paths to cloud mask files
            window_size: Size of processing windows
            num_harmonics: Number of harmonics for periodic function fitting
            max_iter: Maximum iterations for harmonic fitting
            max_workers: Maximum number of parallel workers
            logger: Optional logger instance
        """
        # Validate inputs
        if len(mosaic_paths) != len(dates):
            raise ValueError("Number of mosaic paths must match number of dates")
        if cloud_mask_paths and len(cloud_mask_paths) != len(dates):
            raise ValueError("Number of cloud mask paths must match number of dates")
            
        self.mosaic_paths = [Path(p) for p in mosaic_paths]
        self.dates = dates
        self.dem_path = Path(dem_path)
        self.output_dir = Path(output_dir)
        self.tile_id = tile_id
        self.model_path = Path(model_path)
        self.cloud_mask_paths = [Path(p) for p in cloud_mask_paths] if cloud_mask_paths else None
        self.window_size = window_size
        self.num_harmonics = num_harmonics
        self.max_iter = max_iter
        self.weights_ = '0'
        self.max_workers = min(os.cpu_count(), 4) if max_workers is None else max_workers
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if os.path.exists(self.model_path):
            self.model = load(self.model_path)
            self.logger.info(f"Loaded model from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        self._validate_setup()
        
    def _validate_setup(self) -> None:
        """Validate input data and paths."""
        # Check if all mosaic files exist
        missing_mosaics = [p for p in self.mosaic_paths if not p.exists()]
        if missing_mosaics:
            raise FileNotFoundError(f"Missing mosaic files: {missing_mosaics}")
            
        # Check if all cloud mask files exist (if provided)
        if self.cloud_mask_paths:
            missing_masks = [p for p in self.cloud_mask_paths if not p.exists()]
            if missing_masks:
                raise FileNotFoundError(f"Missing cloud mask files: {missing_masks}")
                
        # Check DEM file
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")
            
        # Get profile from first mosaic
        with rasterio.open(self.mosaic_paths[0]) as src:
            self.profile = src.profile.copy()
            
    def _process_window(self, window: Window) -> Optional[np.ndarray]:
        """
        Process a single window.
        
        Args:
            window: Rasterio Window object defining the region to process
        
        Returns:
            Probability map for the window or None if processing failed
        """
        try:
            # Initialize band arrays
            window_shape = (len(self.dates), window.height, window.width)
            b2 = np.zeros(window_shape, dtype=np.float32)
            b4 = np.zeros(window_shape, dtype=np.float32)
            b8 = np.zeros(window_shape, dtype=np.float32)
            b11 = np.zeros(window_shape, dtype=np.float32)
            b12 = np.zeros(window_shape, dtype=np.float32)
            
            # Initialize cloud mask if provided
            cloud_mask = None
            if self.cloud_mask_paths:
                cloud_mask = np.zeros(window_shape, dtype=np.float32)
            
            # Read time series data for window
            for t, mosaic_path in enumerate(self.mosaic_paths):
                with rasterio.open(mosaic_path) as src:
                    # Read bands 1,3,4,9,10 (B2,B4,B8,B11,B12)
                    data = src.read([1,3,4,9,10], window=window)
                    b2[t] = data[0] / 10000.0
                    b4[t] = data[1] / 10000.0
                    b8[t] = data[2] / 10000.0
                    b11[t] = data[3] / 10000.0
                    b12[t] = data[4] / 10000.0
                    
                # Read cloud mask if provided
                if self.cloud_mask_paths:
                    with rasterio.open(self.cloud_mask_paths[t]) as src:
                        cloud_mask[t] = src.read(1, window=window)
                    self.weights_ = '1'
                    
            # Read DEM for window
            with rasterio.open(self.dem_path) as src:
                dem = src.read(1, window=window)
                
            # Create BandData object
            band_data = BandData(
                b2=b2, b4=b4, b8=b8, b11=b11, b12=b12,
                dates=self.dates,
                dem=dem,
                cloud_mask=cloud_mask
            )
            
            # Initialize and run window inference
            window_inference = WindowInference(
                band_data=band_data,
                model=self.model,
                num_harmonics=self.num_harmonics,
                max_iter=self.max_iter
            )
            
            return window_inference.run_inference()
            
        except Exception as e:
            self.logger.error(f"Error processing window {window}: {str(e)}")
            return None
            
    def run(self) -> None:
        """Run inference on the complete tile using parallel processing."""
        try:
            self.logger.info(f"Processing tile {self.tile_id}")
            
            # Calculate optimal windows
            windows = calculate_optimal_windows(
                self.mosaic_paths[0],
                window_size=self.window_size,
                logger=self.logger
            )
            
            # Prepare output file
            output_path = self.output_dir / f"prob_map_tile_H{self.num_harmonics}_W{self.weights_}_IRLS{self.max_iter}_{self.tile_id}.tif"
            profile = self.profile.copy()
            #add compression
            profile.update({
                'count': 1,
                'dtype': 'uint8',
                'nodata': 0, 
                'compress': 'lzw'
            })
            
            # Process windows in parallel
            self.logger.info(f"Using {self.max_workers} workers for parallel processing")
            with rasterio.open(output_path, 'w', **profile) as dst:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_window = {}
                    
                    # Submit all windows for processing
                    for window in windows:
                        future = executor.submit(self._process_window, window)
                        future_to_window[future] = window
                        
                    # Process results as they complete
                    for future in tqdm(as_completed(future_to_window),
                                   total=len(windows),
                                   desc="Processing windows"):
                        window = future_to_window[future]
                        try:
                            prob_map = future.result()
                            if prob_map is not None:
                                # Scale to uint8 and write
                                result = (prob_map * 255).astype(np.uint8)
                                dst.write(result, 1, window=window)
                        except Exception as e:
                            self.logger.error(f"Error processing window {window}: {str(e)}")
                            continue
                            
            self.logger.info(f"Successfully processed tile {self.tile_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process tile {self.tile_id}: {str(e)}")
            raise
            
    def __str__(self) -> str:
        """String representation of the TileInference instance."""
        mask_status = "with" if self.cloud_mask_paths else "without"
        return (f"TileInference(tile_id={self.tile_id}, "
                f"dates={len(self.dates)} dates, "
                f"{mask_status} cloud masks)")

@dataclass
class TileData:
    """Data class to hold all paths and dates for a single tile."""
    tile_id: str
    mosaic_paths: List[Path]
    dates: List[datetime]
    dem_path: Path
    cloud_mask_paths: Optional[List[Path]] = None

class FolderInference:
    """
    Class to manage inference over multiple tiles using SLURM job arrays.
    Supports flexible input structure and resuming interrupted processing.
    """
    @classmethod
    def from_config(
        cls,
        config_dir: Path,
        logger: Optional[logging.Logger] = None
    ) -> 'FolderInference':
        """
        Create FolderInference instance from saved configuration files.
        
        Args:
            config_dir: Directory containing configuration files
            logger: Optional logger for tracking progress
            
        Returns:
            FolderInference instance initialized from configs
            
        Raises:
            FileNotFoundError: If required config files are missing 
            ValueError: If config data is invalid
        """
        config_dir = Path(config_dir)
        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
            
        # Load metadata
        metadata_file = config_dir / "metadata.json" 
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        with open(metadata_file) as f:
            metadata = json.load(f)
            
        # Load all tile configs
        tiles_data = []
        for i in range(metadata['num_tiles']):
            config_file = config_dir / f"tile_config_{i:03d}.json"
            if not config_file.exists():
                raise FileNotFoundError(f"Missing tile config: {config_file}")
                
            with open(config_file) as f:
                tile_config = json.load(f)
                
            # Convert strings back to Paths
            tile_data = TileData(
                tile_id=tile_config['tile_id'],
                mosaic_paths=[Path(p) for p in tile_config['mosaic_paths']],
                dates=[datetime.strptime(d, "%Y-%m-%d") for d in tile_config['dates']],
                dem_path=Path(tile_config['dem_path']),
                cloud_mask_paths=[Path(p) for p in tile_config['cloud_mask_paths']] 
                    if tile_config.get('cloud_mask_paths') else None
            )
            tiles_data.append(tile_data)
            
        # Create instance
        return cls(
            tiles_data=tiles_data,
            output_dir=Path(metadata['output_dir']),
            model_path=Path(metadata['model_path']),
            window_size=metadata.get('window_size', 1024),
            workers_per_tile=metadata.get('workers_per_tile', 4),
            num_harmonics=metadata.get('num_harmonics', 2),
            max_iter=metadata.get('max_iter', 1),
            logger=logger
        )
    
    def __init__(
        self,
        tiles_data: List[TileData],
        output_dir: Path,
        model_path: Path,
        config_path: Optional[Path] = None,
        window_size: int = 1024,
        workers_per_tile: int = 4,
        num_harmonics: int = 2,
        max_iter: int = 1,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize FolderInference with explicit tile data.
        
        Args:
            tiles_data: List of TileData objects containing paths and dates
            output_dir: Directory to save outputs
            model_path: Path to the model file
            config_path: Optional path to YAML config file for overriding defaults
            window_size: Size of processing windows for TileInference
            workers_per_tile: Number of CPU cores to use per tile
            num_harmonics: Number of harmonics for periodic function fitting
            max_iter: Maximum iterations for harmonic fitting
            logger: Optional logger instance
        """
        self.tiles_data = tiles_data
        self.output_dir = Path(output_dir)
        self.model_path = Path(model_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
        else:
            self.window_size = window_size
            self.workers_per_tile = workers_per_tile
            self.num_harmonics = num_harmonics
            self.max_iter = max_iter
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = self.output_dir / "configs"
        self.config_dir.mkdir(exist_ok=True)
        
        # State tracking file
        self.state_file = self.output_dir / "processing_state.json"
        
        # Initialize
        self._validate_setup()
        self.tiles_info = self._prepare_tiles_info()
        
    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        self.window_size = config.get('window_size', 1024)
        self.workers_per_tile = config.get('workers_per_tile', 4)
        self.num_harmonics = config.get('num_harmonics', 2)
        self.max_iter = config.get('max_iter', 1)
        
    def _validate_setup(self) -> None:
        """Validate input data and paths."""
        if not self.tiles_data:
            raise ValueError("No tile data provided")
            
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        for tile_data in self.tiles_data:
            if len(tile_data.mosaic_paths) != len(tile_data.dates):
                raise ValueError(f"Mismatch between mosaic paths and dates for tile {tile_data.tile_id}")
            if tile_data.cloud_mask_paths and len(tile_data.cloud_mask_paths) != len(tile_data.dates):
                raise ValueError(f"Mismatch between cloud mask paths and dates for tile {tile_data.tile_id}")
            
    def _prepare_tiles_info(self) -> List[Dict]:
        """Prepare information for each tile to be processed."""
        tiles_info = []
        
        for tile_data in self.tiles_data:
            tiles_info.append({
                'tile_id': tile_data.tile_id,
                'mosaic_paths': [str(p) for p in tile_data.mosaic_paths],
                'dates': [d.strftime("%Y-%m-%d") for d in tile_data.dates],
                'dem_path': str(tile_data.dem_path),
                'cloud_mask_paths': [str(p) for p in tile_data.cloud_mask_paths] if tile_data.cloud_mask_paths else None,
                'config': {
                    'window_size': self.window_size,
                    'workers_per_tile': self.workers_per_tile,
                    'num_harmonics': self.num_harmonics,
                    'max_iter': self.max_iter
                }
            })
                    
        return tiles_info
        
    def save_configs(self) -> None:
        """Save configuration files for each tile."""
        # Save tile configurations
        for i, tile_info in enumerate(self.tiles_info):
            config_file = self.config_dir / f"tile_config_{i:03d}.json"
            with open(config_file, 'w') as f:
                json.dump(tile_info, f, indent=2)
                
        # Save metadata
        metadata = {
            'num_tiles': len(self.tiles_info),
            'model_path': str(self.model_path),
            'output_dir': str(self.output_dir)
        }
        with open(self.config_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Initialize processing state if it doesn't exist
        if not self.state_file.exists():
            self._save_processing_state(set())
            
    def _load_processing_state(self) -> set:
        """Load the set of completed tile indices."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            return set(state['completed_tiles'])
        return set()
        
    def _save_processing_state(self, completed_tiles: set) -> None:
        """Save the set of completed tile indices."""
        with open(self.state_file, 'w') as f:
            json.dump({
                'completed_tiles': sorted(list(completed_tiles)),
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
            
    def process_single_tile(self, tile_idx: int) -> None:
        """
        Process a single tile based on its index.
        Checks if tile has already been processed before starting.
        
        Args:
            tile_idx: Index of the tile to process
        """
        if tile_idx >= len(self.tiles_info):
            raise ValueError(f"Invalid tile index: {tile_idx}")
            
        # Check if tile has already been processed
        completed_tiles = self._load_processing_state()
        if tile_idx in completed_tiles:
            self.logger.info(f"Tile {tile_idx} already processed, skipping")
            return
            
        # Process tile
        try:
            tile_info = self.tiles_info[tile_idx]
            
            # Convert string dates back to datetime
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in tile_info['dates']]
            
            # Initialize and run TileInference
            tile_inference = TileInference(
                mosaic_paths=[Path(p) for p in tile_info['mosaic_paths']],
                dates=dates,
                dem_path=Path(tile_info['dem_path']),
                output_dir=self.output_dir,
                tile_id=tile_info['tile_id'],
                model_path=self.model_path,
                cloud_mask_paths=[Path(p) for p in tile_info['cloud_mask_paths']] if tile_info['cloud_mask_paths'] else None,
                window_size=tile_info['config']['window_size'],
                num_harmonics=tile_info['config']['num_harmonics'],
                max_iter=tile_info['config']['max_iter'],
                max_workers=tile_info['config']['workers_per_tile'],
                logger=self.logger
            )
            
            tile_inference.run()
            
            # Update processing state
            completed_tiles.add(tile_idx)
            self._save_processing_state(completed_tiles)
            
        except Exception as e:
            self.logger.error(f"Error processing tile {tile_idx}: {str(e)}")
            raise
        
    def get_remaining_tiles(self) -> List[int]:
        """Get list of tile indices that still need to be processed."""
        completed_tiles = self._load_processing_state()
        all_tiles = set(range(len(self.tiles_info)))
        return sorted(list(all_tiles - completed_tiles))
        
    def get_num_tiles(self) -> int:
        """Get total number of tiles to process."""
        return len(self.tiles_info)
        
    def get_progress(self) -> Dict:
        """Get processing progress information."""
        completed_tiles = self._load_processing_state()
        total_tiles = self.get_num_tiles()
        return {
            'total_tiles': total_tiles,
            'completed_tiles': len(completed_tiles),
            'remaining_tiles': total_tiles - len(completed_tiles),
            'progress_percentage': (len(completed_tiles) / total_tiles) * 100 if total_tiles > 0 else 0
        }
        
    def __str__(self) -> str:
        """String representation of the FolderInference instance."""
        progress = self.get_progress()
        return (f"FolderInference(tiles={progress['total_tiles']}, "
                f"completed={progress['completed_tiles']}, "
                f"progress={progress['progress_percentage']:.1f}%)")