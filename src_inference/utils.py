from __future__ import annotations
import numpy as np
import rasterio
from rasterio.windows import Window
from datetime import datetime
import os
from skimage.morphology import dilation, disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from typing import List, Tuple, Optional, Union, Callable
import multidem
from rasterio.warp import transform_bounds, reproject
import warnings
from scipy.ndimage import gaussian_filter
import logging
from pathlib import Path
import matplotlib.pyplot as plt

def get_aspect(
    dem: np.ndarray,
    resolution: float = 10.0,
    sigma: int = 1,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Calculate aspect (slope direction) from a Digital Elevation Model (DEM).
    
    Args:
        dem: Input DEM array
        resolution: Spatial resolution of the DEM in meters
        sigma: Gaussian smoothing sigma parameter
        logger: Optional logger instance for tracking progress
    
    Returns:
        aspect: Array containing aspect values in degrees (0-360, clockwise from north)
    """
    try:
        dem = gaussian_filter(dem, sigma=sigma)
        dy, dx = np.gradient(dem, resolution)
        aspect = np.degrees(np.arctan2(dy, dx))
        aspect = 90.0 - aspect
        aspect[aspect < 0] += 360.0
        
        if logger:
            logger.info("Aspect calculation completed successfully")
        return aspect
    except Exception as e:
        if logger:
            logger.error(f"Error calculating aspect: {str(e)}")
        raise

def write_dem_features(
    path: Union[str, Path],
    item: int = -2,
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Write DEM features to a file using SRTM30 data.
    
    Args:
        path: Path to the input raster file
        item: Index for output path construction
        logger: Optional logger instance for tracking progress
    
    Returns:
        int: 1 if successful, 0 if failed
    """
    try:
        if logger:
            logger.info(f"Processing DEM features for {path}")
            
        raster = rasterio.open(path)
        pr = raster.profile
        target_transform = raster.transform
        target_bounds = transform_bounds(raster.crs, {'init':'EPSG:4326'}, *raster.bounds)
        
        dem, transform, crs = multidem.crop(target_bounds, source="SRTM30", datum="orthometric")
        dem, transform = reproject(
            dem, np.zeros((1,*raster.shape)),
            src_transform=transform,
            src_crs=crs,
            dst_crs={'init':str(raster.crs)},
            dst_transform=target_transform,
            dst_shape=raster.shape
        )
        
        out_path = Path(path).parent.parent / 'dem.tif'
        pr.update({
            'transform': target_transform,
            'count': 1,
            'dtype': 'float32'
        })
        
        with rasterio.open(out_path, "w", **pr) as dest:
            dest.write(dem.astype('float32'))
            
        if logger:
            logger.info(f"Successfully wrote DEM features to {out_path}")
        return 1
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to write DEM features: {str(e)}")
        return 0

def fit_periodic_function_with_harmonics_robust(
    time_series: np.ndarray,
    qa: np.ndarray,
    dates: List[datetime],
    num_harmonics: int = 3,
    max_iter: int = 10,
    tol: float = 5e-2,
    percentile: float = 75.0,
    min_param_threshold: float = 1e-5,
    callback: Optional[Callable[[int, float], None]] = None,
    logger: Optional[logging.Logger] = None, 
    debug: bool = False
) -> Tuple[np.ndarray]:
    """
    Fit a robust periodic function with harmonics to time series data.
    
    Args:
        time_series: Input time series array (shape: [time, height, width])
        qa: Quality assessment array
        dates: List of datetime objects for each time point
        num_harmonics: Number of harmonics to use in the fit
        max_iter: Maximum number of iterations for IRLS
        tol: Convergence tolerance for relative parameter change
        percentile: Percentile for convergence criterion
        min_param_threshold: Threshold for considering parameters significant
        callback: Optional callback function(iteration, relative_change)
        logger: Optional logger instance for tracking progress
        debug: Enable debug mode for additional plots and information
    
    Returns:
        Tuple containing:
        - List of amplitude maps for each harmonic
        - List of phase maps for each harmonic
        - Offset map
        - Variance residual map
    """
    try:
        if logger:
            logger.info(f"Starting periodic function fitting with {num_harmonics} harmonics")
        
        # Convert dates and compute normalized time
        times_datetime64 = np.array(dates, dtype='datetime64[D]')
        days_since_start = (times_datetime64 - times_datetime64[0]).astype(int)
        t_normalized = days_since_start / 365.25
        
        # Setup design matrix
        harmonics = []
        for k in range(1, num_harmonics + 1):
            t_radians = 2 * np.pi * k * t_normalized
            harmonics.extend([np.cos(t_radians), np.sin(t_radians)])
        A = np.stack(harmonics + [np.ones_like(t_normalized)], axis=-1)
        
        # Initialize parameters
        pixels = time_series.reshape(time_series.shape[0], -1)
        weights = qa.reshape(qa.shape[0], -1)
       
       # Initialize delta
        delta = 1.345

        # Compute the pseudoinverse of the design matrix
        A_pinv = np.linalg.pinv(A)  # Shape: (num_params, time)
        # Initial least squares fit to estimate parameters
        initial_params = np.dot(A_pinv, pixels).T  # Shape: (n_pixels, num_params)
        params = initial_params.copy()
        # Initialize parameters
        num_params = 2 * num_harmonics + 1

        # Calculate initial residuals
        initial_fitted_values = np.dot(A, params.T)
        initial_residuals = pixels - initial_fitted_values

        # Estimate initial sigma
        sigma_initial = np.std(initial_residuals, axis=0)
        sigma_initial[sigma_initial == 0] = np.finfo(float).eps  # Avoid division by zero

        # Set delta based on initial residuals and do not update it
        delta = 1.345 * sigma_initial
        delta[delta == 0] = np.finfo(float).eps  # Avoid zero delta

        epsilon = 1e-8
        
        for iteration in range(max_iter):
            params_old = params.copy()

            # Broadcasting for weighted design matrix
            A_expanded = np.expand_dims(A, 2)
            weights_expanded = np.expand_dims(weights, 1)
            A_weighted = A_expanded * weights_expanded

            # Compute the normal equation components
            ATA = np.einsum('ijk,ilk->jlk', A_weighted, A_expanded)
            ATb = np.einsum('ijk,ik->jk', A_weighted, pixels)

            # Solve for parameters
            ATA_reshaped = ATA.transpose(2, 0, 1)
            ATb_reshaped = ATb.T

            
            params = np.array([solve_params(ATA_reshaped[i], ATb_reshaped[i]) for i in range(ATA_reshaped.shape[0])])
            params = np.nan_to_num(params)  # Replace NaNs with zero

            # Calculate fitted values and residuals
            fitted_values = np.dot(A, params.T)  # Shape: (time, n_pixels)
            residuals = pixels - fitted_values

            # Estimate sigma (standard deviation of residuals)
            sigma_residuals = np.std(residuals, axis=0)
            sigma_residuals[sigma_residuals == 0] = np.finfo(float).eps  # Avoid division by zero

            # Update weights based on residuals using Huber loss
            residuals_abs = np.abs(residuals)
            delta = 1.345 * sigma_residuals  # Update delta based on residuals
            delta[delta == 0] = epsilon  # Avoid zero delta
            mask = residuals_abs <= delta
            weights_update = np.where(mask, 1, delta / (residuals_abs + epsilon))
            weights = weights * weights_update

            # Compute relative change, avoiding division by small numbers
            min_param_threshold = 1e-5  # Or another appropriate small value
            param_diff = np.abs(params - params_old)
            relative_change = param_diff / (np.maximum(np.abs(params_old), min_param_threshold))
            relative_change_flat = relative_change.flatten()

            if debug:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                _ = ax.hist(param_diff.flatten(), bins=100)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_title(f"Params diff - Iteration {iteration + 1}")
                plt.show()

            # Compute the desired percentile of relative change
            percentile_value = np.percentile(relative_change_flat, percentile)

            if callback:
                callback(iteration, percentile_value)
                
            if logger:
                logger.debug(f"Iteration {iteration + 1}: relative change = {percentile_value:.6f}")
            
            if percentile_value < tol:
                if logger:
                    logger.info(f"Converged after {iteration + 1} iterations (relative change: {percentile_value:.6f})")
                break
        else:
            if logger:
                logger.warning(f"Maximum iterations ({max_iter}) reached without convergence")
        
        # Extract results
        params_reshaped = params.reshape(time_series.shape[1], time_series.shape[2], -1).transpose(2, 0, 1)
        amplitude_maps = []
        phase_maps = []
        
        for i in range(num_harmonics):
            A_params = params_reshaped[2 * i]
            B_params = params_reshaped[2 * i + 1]
            amplitude_map = np.sqrt(A_params ** 2 + B_params ** 2)
            phase_map = np.arctan2(B_params, A_params)
            
            phase_adjusted = (phase_map - (2 * np.pi * (i + 1) * t_normalized[0])) % (2 * np.pi)
            phase_normalized = np.where(phase_adjusted > np.pi, phase_adjusted - 2 * np.pi, phase_adjusted)
            
            amplitude_maps.append(amplitude_map)
            phase_maps.append(phase_normalized)
            
        offset_map = params_reshaped[-1]

        # Var residual map reshaped to the tile shape
        var_residual = np.var(residuals, axis=0)
        var_residual_map = var_residual.reshape(time_series.shape[1], time_series.shape[2])
            
        if logger:
            logger.info("Successfully completed periodic function fitting")
        return (*amplitude_maps, *phase_maps, offset_map, var_residual_map)
        
    except Exception as e:
        if logger:
            logger.error(f"Error in periodic function fitting: {str(e)}")
        raise

def solve_params(ATA: np.ndarray, ATb: np.ndarray) -> np.ndarray:
    """ Solve linear equations with error handling for non-invertible cases. """
    try:
        return np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return np.full(ATb.shape, np.nan)  # Return NaN for non-invertible matrices

def compute_indices(
    b2: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
    b11: np.ndarray,
    b12: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectral indices (NDVI, EVI, NBR, CRSWIR) from reflectance values.
    
    Args:
        b2-b12: Reflectance arrays for different spectral bands
        logger: Optional logger instance for tracking progress
    
    Returns:
        Tuple containing (NDVI, EVI, NBR, CRSWIR) arrays
    """
    try:
        if logger:
            logger.info("Computing spectral indices")
        
        # Constants
        crswir_coeff = (1610 - 842) / (2190 - 842)
        epsilon = 1e-8
        
        # Compute denominators
        ndvi_denom = b8 + b4 + epsilon
        evi_denom = b8 + 6 * b4 - 7.5 * b2 + 1 + epsilon
        nbr_denom = b8 + b12 + epsilon
        crswir_denom = ((b12 - b8) * crswir_coeff + b8) + epsilon
        
        # Compute indices
        ndvi = (b8 - b4) / ndvi_denom
        evi = 2.5 * (b8 - b4) / evi_denom
        nbr = (b8 - b12) / nbr_denom
        crswir = b11 / crswir_denom
        
        # Scale between 0 and 1
        ndvi = np.clip(ndvi, -1, 1) / 2 + 0.5
        evi = np.clip(evi, -1, 1) / 2 + 0.5
        nbr = np.clip(nbr, -1, 1) / 2 + 0.5
        crswir = np.clip(crswir, 0, 5) / 5
        
        # Clean indices
        indices = {
            'NDVI': ndvi,
            'EVI': evi,
            'NBR': nbr,
            'CRSWIR': crswir
        }
        
        for name, index in indices.items():
            index = np.nan_to_num(index, nan=0.0, posinf=0.0, neginf=0.0)
            if logger:
                valid_data = index[~np.isnan(index) & ~np.isinf(index)]
                if len(valid_data) > 0:
                    logger.debug(f"{name} stats - Mean: {np.mean(valid_data):.3f}, "
                               f"Min: {np.min(valid_data):.3f}, Max: {np.max(valid_data):.3f}")
        
        if logger:
            logger.info("Successfully computed spectral indices")
        return ndvi, evi, nbr, crswir
    
    except Exception as e:
        if logger:
            logger.error(f"Error computing spectral indices: {str(e)}")
        raise

def compute_qa_weights(
    flg: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Compute quality assessment weights based on FLG band.
    
    Args:
        flg: Input FLG array
        logger: Optional logger instance for tracking progress
    
    Returns:
        Array of computed weights
    """
    try:
        if logger:
            logger.info("Computing QA weights")
        
        # Create mask for clouds (1), snow (2), and water (3)
        mask = (flg == 1) | (flg == 2) | (flg == 3)
        
        if logger:
            logger.debug(f"Masked pixels - Cloud: {np.sum(flg == 1)}, "
                        f"Snow: {np.sum(flg == 2)}, Water: {np.sum(flg == 3)}")
        
        # Convert to float32 and process
        weights = (~mask).astype(np.float32)
        weights = postprocess_cloud_mask(weights, 5, 25)
        
        if logger:
            logger.info("Successfully computed QA weights")
            logger.debug(f"Weight stats - Mean: {np.mean(weights):.3f}, "
                        f"Zero weights: {np.sum(weights == 0)} pixels")
        
        return weights
    
    except Exception as e:
        if logger:
            logger.error(f"Error computing QA weights: {str(e)}")
        raise

def calculate_optimal_windows(
    raster_path: Union[str, Path],
    window_size: int = 1024,
    logger: Optional[logging.Logger] = None
) -> List[Window]:
    """
    Calculate optimal windows for processing based on the raster dimensions.
    
    Args:
        raster_path: Path to the raster file
        window_size: Size of processing windows
        logger: Optional logger instance for tracking progress
    
    Returns:
        List of Window objects for optimal processing
    """
    try:
        if logger:
            logger.info(f"Calculating optimal windows for {raster_path}")
        
        with rasterio.open(raster_path) as src:
            width = src.width
            height = src.height
        
        windows = []
        for y in range(0, height, window_size):
            for x in range(0, width, window_size):
                window = Window(
                    col_off=x,
                    row_off=y,
                    width=min(window_size, width - x),
                    height=min(window_size, height - y)
                )
                windows.append(window)
        
        if logger:
            logger.info(f"Created {len(windows)} processing windows")
            
        return windows
    
    except Exception as e:
        if logger:
            logger.error(f"Error calculating optimal windows: {str(e)}")
        raise

from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from skimage.morphology import dilation, disk
from skimage.filters import rank
from skimage.util import img_as_ubyte

def postprocess_cloud_mask(
    cloud_mask: np.ndarray,
    dilation_radius: int = 5,
    mean_radius: int = 20,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Postprocess a cloud mask using dilation and mean filtering.
    
    Args:
        cloud_mask: Input cloud mask array
        dilation_radius: Radius for dilation operation
        mean_radius: Radius for mean filtering
        logger: Optional logger instance
    
    Returns:
        Processed cloud mask as float32 array
    """
    try:
        cloud_mask_uint8 = img_as_ubyte(cloud_mask)
        dilated = dilation(cloud_mask_uint8, disk(dilation_radius))
        mean = rank.mean(dilated, disk(mean_radius)) / 255
        result = mean.astype('float32')
        
        if logger:
            logger.info(f"Cloud mask processed: {np.sum(result > 0)} affected pixels")
            
        return result
    except Exception as e:
        if logger:
            logger.error(f"Cloud mask processing failed: {str(e)}")
        raise

def get_paired_files(
    base_dir: Union[str, Path],
    suffix: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Path], List[Path]]:
    """
    Get paired mosaic and FLG files ensuring they match by date.
    
    Args:
        base_dir: Base directory containing 'mosaics' and 'FLG' subdirectories
        suffix: File suffix to filter (default: '.tif')
        logger: Optional logger instance
    
    Returns:
        Tuple of (mosaic_paths, flg_paths) lists, sorted by date
    """
    try:
        base_dir = Path(base_dir)
        mosaic_dir = base_dir / 'mosaics'
        flg_dir = base_dir / 'FLG'
        suffix = suffix or '.tif'
        
        # Get and sort files
        mosaic_files = sorted(
            [f for f in mosaic_dir.glob(f"*{suffix}")],
            key=lambda x: x.name.split('_')[1]
        )
        flg_files = sorted(
            [f for f in flg_dir.glob(f"*{suffix}")],
            key=lambda x: x.name.split('_')[1]
        )
        
        # Verify files exist
        if not mosaic_files or not flg_files:
            raise FileNotFoundError("No matching files found")
        for flg_path in flg_files:
            if not flg_path.exists():
                raise FileNotFoundError(f"Missing FLG file: {flg_path}")
        
        if logger:
            logger.info(f"Found {len(mosaic_files)} paired files ({mosaic_files[0].name.split('_')[1]} to {mosaic_files[-1].name.split('_')[1]})")
        
        return mosaic_files, flg_files
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to get paired files: {str(e)}")
        raise

def to_reflectance(
    data: np.ndarray,
    nodata: Optional[Union[float, int]] = None,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Convert mosaic data to reflectance values.
    
    Args:
        data: Input mosaic data array
        nodata: Value to treat as no data (if None, NaN values are used)
        logger: Optional logger instance
    
    Returns:
        Array of reflectance values
    """
    try:
        # Create mask
        if nodata is None or np.isnan(nodata):
            mask = ~np.isnan(data)
        else:
            mask = data != nodata
            
        # Convert data
        result = np.full_like(data, np.nan, dtype=np.float32)
        valid_pixels = np.sum(mask)
        
        if valid_pixels > 0:
            valid_data = data[mask].astype(np.float32) / 10000.0
            result[mask] = valid_data
            
            # Check for problematic values
            problematic = np.sum((valid_data < 0) | (valid_data > 1))
            
            if logger:
                logger.info(f"Converted {valid_pixels:,} pixels to reflectance "
                          f"({problematic} outside valid range)")
        else:
            if logger:
                logger.warning("No valid pixels found")
                
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"Reflectance conversion failed: {str(e)}")
        raise