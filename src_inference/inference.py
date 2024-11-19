from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import logging
from datetime import datetime
from joblib import load
from pathlib import Path

# Add the parent directory to Python path to import the module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_inference.utils import compute_indices, fit_periodic_function_with_harmonics_robust

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
        model_path: Optional[Path] = None,
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
        self.model = None
        if model_path is not None:
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            with open(model_path, 'rb') as f:
                self.model = load(f)
                
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        if self.num_harmonics < 1:
            raise ValueError("num_harmonics must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
            
    def _get_quality_weights(self) -> np.ndarray:
        """
        Get quality weights for feature computation.
        
        Returns:
            Array of quality weights (1 for good quality, 0 for bad quality)
        """
        if self.band_data.cloud_mask is not None:
            self.logger.info("Using provided cloud mask as quality weights")
            return self.band_data.cloud_mask
        else:
            self.logger.info("No cloud mask provided, using default weights of 1.0")
            return np.ones_like(self.band_data.b2)
            
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