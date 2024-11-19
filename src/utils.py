#import necessary libraries
import numpy as np
import rasterio
from rasterio.windows import Window
from datetime import datetime
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.morphology import dilation, disk
from skimage.filters import rank
import numpy as np
from datetime import datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
import multidem
from rasterio.warp import transform_bounds, reproject
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter

def get_aspect(dem: np.ndarray, resolution: float = 10.0, sigma: int = 1) -> np.ndarray:
    """
    Calculate aspect from DEM.
    """
    dem = gaussian_filter(dem, sigma=sigma)

    # Calculate gradients
    dy, dx = np.gradient(dem, resolution)
    
    # Calculate aspect
    aspect = np.degrees(np.arctan2(dy, dx))
    
    # Convert to compass bearing (clockwise from north)
    aspect = 90.0 - aspect
    aspect[aspect < 0] += 360.0
    
    return aspect


def write_dem_features(path, item=-2):
    print(path)
    raster = rasterio.open(path)
    # src = raster.read().clip(0,10000)
    pr = raster.profile
    target_transform = raster.transform
    target_bounds = transform_bounds(raster.crs, {'init':'EPSG:4326'}, *raster.bounds)
    try:
        dem, transform, crs = multidem.crop(target_bounds, source="SRTM30", datum="orthometric")
        dem, transform = reproject(dem, np.zeros((1,*raster.shape)), src_transform=transform, src_crs=crs, dst_crs={'init':str(raster.crs)}, dst_transform=target_transform, dst_shape=raster.shape)
        out_path = os.path.join( "/".join(path.split('/')[:item]), 'dem.tif')
        pr['transform'] = target_transform
        pr['count'] = 1
        pr['dtype'] = 'float32'
        with rasterio.open(out_path, "w", **pr) as dest:
            dest.write(dem.astype('float32'))
        return 1
    except :
        return 0

def fit_periodic_function_with_harmonics_robust(
    time_series: np.ndarray,
    qa: np.ndarray,
    dates: List[datetime],
    num_harmonics: int = 3,
    max_iter: int = 10,
    tol: float = 5e-2,  # Adjusted tolerance for relative change
    percentile: float = 75.0,  # Percentile for convergence criterion
    min_param_threshold: float = 1e-5,  # Threshold to consider parameter significant
    verbose=0,
    debug=False
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    # Convert dates to 'datetime64' and compute normalized time as fraction of year
    times_datetime64 = np.array(dates, dtype='datetime64[D]')
    start_date = times_datetime64[0]
    days_since_start = (times_datetime64 - start_date).astype(int)
    t_normalized = days_since_start / 365.25  # Normalize to fraction of year

    # Initial design matrix with harmonics and constant term
    harmonics = []
    for k in range(1, num_harmonics + 1):
        t_radians = 2 * np.pi * k * t_normalized
        harmonics.extend([np.cos(t_radians), np.sin(t_radians)])

    A = np.stack(harmonics + [np.ones_like(t_normalized)], axis=-1)  # Design matrix

    # Reshape time_series and qa for vectorized operations
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

        if verbose > 0:
            print(f"Iteration {iteration + 1}: {percentile}th percentile of relative parameter change = {percentile_value}")

        # Check for convergence
        if percentile_value < tol:
            if verbose > 0:
                print(f"Converged after {iteration + 1} iterations.")
            break

    # Reshape parameters
    params_reshaped = params.reshape(time_series.shape[1], time_series.shape[2], num_params).transpose(2, 0, 1)

    # Extract amplitude and phase maps
    amplitude_maps = []
    phase_maps = []

    for i in range(num_harmonics):
        A_params = params_reshaped[2 * i]
        B_params = params_reshaped[2 * i + 1]
        amplitude_map = np.sqrt(A_params ** 2 + B_params ** 2)
        phase_map = np.arctan2(B_params, A_params)

        # Adjust and normalize phases
        phase_adjusted = (phase_map - (2 * np.pi * (i + 1) * t_normalized[0])) % (2 * np.pi)
        phase_normalized = np.where(phase_adjusted > np.pi, phase_adjusted - 2 * np.pi, phase_adjusted)

        amplitude_maps.append(amplitude_map)
        phase_maps.append(phase_normalized)

    # Offset map
    offset_map = params_reshaped[-1]

    return (*amplitude_maps, *phase_maps, offset_map)

def solve_params(ATA: np.ndarray, ATb: np.ndarray) -> np.ndarray:
    """ Solve linear equations with error handling for non-invertible cases. """
    try:
        return np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return np.full(ATb.shape, np.nan)  # Return NaN for non-invertible matrices

from skimage.util import img_as_ubyte

def postprocess_cloud_mask(cloud_mask: np.array, n: int = 5, nm: int = 20) -> np.array:
    # Convert cloud_mask to uint8 image
    cloud_mask_uint8 = img_as_ubyte(cloud_mask)
    dilated = dilation(cloud_mask_uint8, disk(n))
    mean = rank.mean(dilated, disk(nm)) / 255
    return mean.astype('float32')


def get_paired_files(base_dir: str, suffix=None) -> tuple[list, list]:
    """
    Get paired mosaic and FLG files ensuring they match by date.
    
    Args:
        base_dir: Base directory containing 'mosaics' and 'FLG' subdirectories
        
    Returns:
        Tuple of (mosaic_paths, flg_paths) lists, sorted by date
    """
    mosaic_dir = os.path.join(base_dir, 'mosaics')
    flg_dir = os.path.join(base_dir, 'FLG')
    
    # Get all mosaic files and extract dates
    if suffix is None:
        suffix = '.tif'
    mosaic_files = [f for f in os.listdir(mosaic_dir) 
                   if f.endswith(suffix)]
    
    flg_files = [f for f in os.listdir(flg_dir) 
                   if f.endswith(suffix)]
    
    # Sort files by date
    mosaic_files.sort(key=lambda x: x.split('_')[1])
    flg_files.sort(key=lambda x: x.split('_')[1])
    
    mosaic_paths = [os.path.join(mosaic_dir, f) for f in mosaic_files]
    flg_paths = [os.path.join(flg_dir, f) for f in flg_files]
    
    # Verify all FLG files exist
    for flg_path in flg_paths:
        if not os.path.exists(flg_path):
            raise FileNotFoundError(f"Missing FLG file: {flg_path}")
            
    return mosaic_paths, flg_paths

def to_reflectance(data: np.ndarray, nodata=None, logger=None) -> np.ndarray:
    """
    Convert mosaic data to reflectance values.
    """
    def _print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    _print("\n=== Reflectance Conversion ===")
    _print(f"Input data shape: {data.shape}")
    _print(f"Input data type: {data.dtype}")
    _print(f"Nodata type: {type(nodata)}, value: {nodata}")
    
    # Print raw data stats
    _print("\nRaw data statistics:")
    _print(f"  Min: {np.nanmin(data)}")
    _print(f"  Max: {np.nanmax(data)}")
    _print(f"  Mean: {np.nanmean(data)}")
    _print(f"  % of NaNs: {np.mean(np.isnan(data)) * 100:.2f}%")
    
    # Create mask based on nodata type
    if nodata is None or np.isnan(nodata):
        mask = ~np.isnan(data)
        nodata_pixels = np.sum(np.isnan(data))
        _print("\nUsing NaN-based masking")
    else:
        mask = data != nodata
        nodata_pixels = np.sum(data == nodata)
        _print(f"\nUsing value-based masking with nodata={nodata}")
    
    total_pixels = data.size
    nodata_percentage = (nodata_pixels / total_pixels) * 100
    _print(f"Nodata pixels: {nodata_pixels} ({nodata_percentage:.2f}%)")
    
    # Create result array and convert data
    result = np.full_like(data, np.nan, dtype=np.float32)
    
    if np.any(mask):
        valid_data = data[mask].astype(np.float32) / 10000.0
        result[mask] = valid_data
        
        _print("\nReflectance statistics:")
        _print(f"  Valid pixels: {np.sum(mask)}")
        _print(f"  Min: {np.nanmin(valid_data):.4f}")
        _print(f"  Max: {np.nanmax(valid_data):.4f}")
        _print(f"  Mean: {np.nanmean(valid_data):.4f}")
        _print(f"  Median: {np.nanmedian(valid_data):.4f}")
        
        # Check problematic values
        very_low = np.sum(valid_data < 0) / len(valid_data) * 100
        very_high = np.sum(valid_data > 1) / len(valid_data) * 100
        if very_low > 0:
            _print(f"\nWARNING: {very_low:.2f}% of valid pixels have reflectance < 0")
        if very_high > 0:
            _print(f"WARNING: {very_high:.2f}% of valid pixels have reflectance > 1")
    else:
        _print("\nWARNING: No valid pixels found in input data")
    
    return result

def compute_indices(b2: np.ndarray, b3: np.ndarray, b4: np.ndarray, b8: np.ndarray,
                   b11: np.ndarray, b12: np.ndarray, debug: bool = False, logger=None) -> tuple:
    """
    Compute spectral indices from reflectance values.
    """
    def _print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    _print("\n=== Computing Spectral Indices ===")
    
    # Print band statistics
    bands = {'B2': b2, 'B3': b3, 'B4': b4, 'B8': b8, 'B11': b11, 'B12': b12}
    _print("\nBand Statistics:")
    for name, band in bands.items():
        valid_data = band[~np.isnan(band)]
        if len(valid_data) > 0:
            _print(f"\n{name}:")
            _print(f"  Min: {np.min(valid_data):.4f}")
            _print(f"  Max: {np.max(valid_data):.4f}")
            _print(f"  Mean: {np.mean(valid_data):.4f}")
            _print(f"  NaN%: {np.mean(np.isnan(band))*100:.1f}%")
        else:
            _print(f"\nWARNING: {name} contains all NaN values")
    
    # Compute indices
    _print("\nComputing indices...")
    crswir_coeff = (1610 - 842) / (2190 - 842)
    epsilon = 1e-8
    
    # Compute and check denominators
    ndvi_denom = b8 + b4 + epsilon
    evi_denom = b8 + 6 * b4 - 7.5 * b2 + 1 + epsilon
    nbr_denom = b8 + b12 + epsilon
    crswir_denom = ((b12 - b8) * crswir_coeff + b8) + epsilon
    
    _print("\nDenominator Statistics:")
    denoms = {
        'NDVI_denom': ndvi_denom,
        'EVI_denom': evi_denom,
        'NBR_denom': nbr_denom,
        'CRSWIR_denom': crswir_denom
    }
    for name, denom in denoms.items():
        _print(f"\n{name}:")
        _print(f"  Min: {np.nanmin(denom):.4f}")
        _print(f"  Max: {np.nanmax(denom):.4f}")
        _print(f"  Near zero (<1e-6): {np.mean(np.abs(denom) < 1e-6)*100:.2f}%")
    
    # Compute indices
    ndvi = (b8 - b4) / ndvi_denom
    evi = 2.5 * (b8 - b4) / evi_denom
    nbr = (b8 - b12) / nbr_denom
    crswir = b11 / crswir_denom

    # Scaling between 0 and 1
    ndvi = np.clip(ndvi, -1, 1) / 2 + 0.5
    evi = np.clip(evi, -1, 1) / 2 + 0.5
    nbr = np.clip(nbr, -1, 1) / 2 + 0.5
    crswir = np.clip(crswir, 0, 5) / 5 
    
    # Print raw statistics
    _print("\nRaw Index Statistics (before cleaning):")
    indices = {'NDVI': ndvi, 'EVI': evi, 'NBR': nbr, 'CRSWIR': crswir}
    for name, index in indices.items():
        valid_data = index[~np.isnan(index) & ~np.isinf(index)]
        if len(valid_data) > 0:
            _print(f"\n{name}:")
            _print(f"  Min: {np.min(valid_data):.4f}")
            _print(f"  Max: {np.max(valid_data):.4f}")
            _print(f"  Mean: {np.mean(valid_data):.4f}")
            _print(f"  NaN%: {np.mean(np.isnan(index))*100:.1f}%")
            _print(f"  Inf%: {np.mean(np.isinf(index))*100:.1f}%")
        else:
            _print(f"\nWARNING: {name} contains no valid values before cleaning")
    
    # Clean indices
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    evi = np.nan_to_num(evi, nan=0.0, posinf=0.0, neginf=0.0)
    nbr = np.nan_to_num(nbr, nan=0.0, posinf=0.0, neginf=0.0)
    crswir = np.nan_to_num(crswir, nan=0.0, posinf=0.0, neginf=0.0)
    
    if debug:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Spectral Indices Histograms')
            
            indices = {
                'NDVI': (ndvi, axes[0,0], (-1, 1)),
                'EVI': (evi, axes[0,1], (-1, 1)),
                'NBR': (nbr, axes[1,0], (-1, 1)),
                'CRSWIR': (crswir, axes[1,1], (0, 2))
            }
            
            for name, (data, ax, (vmin, vmax)) in indices.items():
                valid_data = data[(data >= vmin) & (data <= vmax)]
                if len(valid_data) > 0:
                    ax.hist(valid_data, bins=50, range=(vmin, vmax))
                    ax.set_title(f"{name}\n(values between {vmin} and {vmax})")
                    ax.grid(True, alpha=0.3)
                    
                    # Add stats to plot
                    stats = f"Mean: {np.mean(valid_data):.2f}\n"
                    stats += f"Std: {np.std(valid_data):.2f}\n"
                    stats += f"Valid%: {len(valid_data)/data.size*100:.1f}%"
                    ax.text(0.95, 0.95, stats,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            _print(f"\nWARNING: Could not create debug plots: {str(e)}")
    
    return ndvi, evi, nbr, crswir

def compute_qa_weights(flg: np.ndarray, logger=None) -> np.ndarray:
    """
    Compute weights based on FLG band.
    """
    def _print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
    _print("\n=== Computing QA Weights ===")
    _print(f"Input FLG shape: {flg.shape}")
    _print(f"FLG unique values: {np.unique(flg)}")
    
    # Create mask for clouds (1), snow (2), and water (3)
    mask = (flg == 1) | (flg == 2) | (flg == 3)
    
    # Log mask statistics
    _print("\nMask Statistics:")
    _print(f"  Cloud pixels: {np.sum(flg == 1)}")
    _print(f"  Snow pixels: {np.sum(flg == 2)}")
    _print(f"  Water pixels: {np.sum(flg == 3)}")
    
    # Convert to float32 and invert boolean
    weights = (~mask).astype(np.float32)
    weights = postprocess_cloud_mask(weights, 5, 25)
    
    _print("\nWeight statistics:")
    _print(f"  Min: {np.min(weights):.3f}")
    _print(f"  Max: {np.max(weights):.3f}")
    _print(f"  Mean: {np.mean(weights):.3f}")
    _print(f"  Zero weights: {np.sum(weights == 0)} pixels")
    
    return weights

def calculate_optimal_windows(raster_path: str, window_size: int = 1024) -> list:
    """
    Calculate optimal windows for processing based on the raster dimensions.
    """
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

    return windows

def write_features(output_path: str, template_path: str, features: dict, window: Window) -> None:
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()

    # Count total number of feature layers
    num_bands = sum(
        [len(val) if isinstance(val, list) else 1
         for val in features.values()]
    )

    # Update profile for feature raster - match window size
    profile.update(
        width=window.width,
        height=window.height,
        dtype=rasterio.uint16,
        count=num_bands,
        nodata=65535,
        tiled=False,  # Disable internal tiling for temp files
        compress=None  # No compression for temp files
    )

    # Create new temp file
    with rasterio.open(output_path, 'w', **profile) as dst:
        band_idx = 1
        for feature_data in features.values():
            if isinstance(feature_data, list):
                for data in feature_data:
                    # Verify data shape matches window
                    if data.shape != (window.height, window.width):
                        print(f"Warning: Data shape {data.shape} doesn't match window {(window.height, window.width)}")
                        continue
                        
                    data = np.nan_to_num(data, nan=65535)
                    # Write to full extent since this is a temp file matching window size
                    dst.write(data, band_idx)
                    band_idx += 1
            else:
                if feature_data.shape != (window.height, window.width):
                    print(f"Warning: Data shape {feature_data.shape} doesn't match window {(window.height, window.width)}")
                    continue
                    
                data = np.nan_to_num(feature_data, nan=65535)
                dst.write(data, band_idx)
                band_idx += 1

def process_window(window: Window, mosaic_paths: list, flg_paths: list, 
                  dates: list, num_harmonics: int = 2, max_iter: int = 1) -> dict:
    """
    Process a single window across all temporal mosaics.
    Modified to use explicit FLG paths.
    """
    time_series = {
        'ndvi': [], 'evi': [], 'nbr': [], 'crswir': []
    }
    weights = []

    # Read data for each date
    for mosaic_path, flg_path in zip(mosaic_paths, flg_paths):
        with rasterio.open(mosaic_path) as src:
            # Read all required bands
            data = src.read([1,2,3,4,9,10], window=window)  # B2,B3,B4,B8,B11,B12
            
            # Verify data shape
            if data.shape[1:] != (window.height, window.width):
                raise ValueError(f"Data shape {data.shape} doesn't match window {(window.height, window.width)}")

            # Convert to reflectance
            data = to_reflectance(data)

            # Compute indices
            ndvi, evi, nbr, crswir = compute_indices(*data)

            # Store indices
            time_series['ndvi'].append(ndvi)
            time_series['evi'].append(evi)
            time_series['nbr'].append(nbr)
            time_series['crswir'].append(crswir)

        # Read and process FLG data
        if os.path.exists(flg_path):
            with rasterio.open(flg_path) as src:
                flg = src.read(1, window=window)
                if flg.shape != (window.height, window.width):
                    raise ValueError(f"FLG shape {flg.shape} doesn't match window {(window.height, window.width)}")
                
                # Check for all-nodata or all-zero window
                if np.all(flg == 0) or np.all(flg == 65535):
                    print(f"Warning: Window {window} appears to be all nodata/zero in FLG")
                    
                weight = compute_qa_weights(flg)
                weights.append(weight)
        else:
            print(f"Warning: Missing FLG file for window {window} - setting all weights to 1")
            weight = np.ones((window.height, window.width), dtype=np.float32)
            weights.append(weight)

    # Convert lists to arrays
    for key in time_series:
        time_series[key] = np.array(time_series[key])
        if np.all(np.isnan(time_series[key])):
            raise ValueError(f"All NaN values for {key}")
            
    weights = np.array(weights)
    if np.all(weights == 0):
        raise ValueError("All zero weights")

    # Extract features for each index
    features = {}
    for index_name, index_data in time_series.items():
        if np.all(np.isnan(index_data)):
            print(f"Warning: All NaN values for {index_name}")
            continue
            
        result = fit_periodic_function_with_harmonics_robust(
            index_data, weights, dates,
            num_harmonics=num_harmonics,
            max_iter=max_iter
        )

        # Unpack results
        amplitudes = result[:num_harmonics]
        phases = result[num_harmonics:2*num_harmonics]
        offset = result[2*num_harmonics]

        # Scale and validate data
        amp_scaled = [np.clip(np.nan_to_num(amp * 65535, nan=0.0), 0, 65535).astype(np.uint16)
                    for amp in amplitudes]
        phase_scaled = [np.clip(((phase + np.pi) / (2 * np.pi) * 65535), 0, 65535).astype(np.uint16)
                        for phase in phases]
        offset_scaled = np.clip(np.nan_to_num(offset * 65535, nan=0.0), 0, 65535).astype(np.uint16)

        # Validate each feature doesn't contain all nodata
        for i, amp in enumerate(amp_scaled):
            if np.all(amp == 65535):
                print(f"Warning: All nodata in amplitude {i} for {index_name}")
        for i, phase in enumerate(phase_scaled):
            if np.all(phase == 65535):
                print(f"Warning: All nodata in phase {i} for {index_name}")
        if np.all(offset_scaled == 65535):
            print(f"Warning: All nodata in offset for {index_name}")

        features[f'{index_name}_amplitude'] = amp_scaled
        features[f'{index_name}_phase'] = phase_scaled
        features[f'{index_name}_offset'] = offset_scaled

    return features

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_window_wrapper(args):
    window, mosaic_paths, flg_paths, dates, output_dir, num_harmonics, max_iter = args
    temp_output_path = os.path.join(output_dir, f'harmonic_features_{window.row_off}_{window.col_off}.tif')
    try:
        logger.info(f"Processing window at row {window.row_off}, col {window.col_off}")
        features = process_window(
            window, mosaic_paths, flg_paths, dates,
            num_harmonics=num_harmonics,
            max_iter=max_iter
        )
        if features is None:
            logger.error(f"No features returned for window {window}")
            return None
        write_features(temp_output_path, mosaic_paths[0], features, window)
        return temp_output_path
    except Exception as e:
        logger.exception(f"Error processing window {window}: {e}")
        return None
    
class PhenologyClassifier:
    """
    A class to handle phenology classification for multiple tiles.
    Loads the model and parameters once and provides methods for inference.
    """

    def __init__(self, model_path: str, params_path: str, config: str):
        """
        Initialize the classifier with model and parameters.

        Parameters:
        model_path (str): Path to the saved model file
        params_path (str): Path to the model parameters JSON file
        config (str): Configuration string for feature extraction
        """
        from joblib import load
        import json

        # Load model and parameters
        self.model = load(model_path)
        with open(params_path, 'r') as f:
            self.params = json.load(f)

        # Store configuration and extract number of harmonics
        self.config = config
        self.n_harmonics = int(config.split('_')[-3][1])

        # Define features used in training
        self.features = [
            'amplitude_evi_h1', 'amplitude_nbr_h1', 'amplitude_ndvi_h1',
            'cos_phase_crswir_h1', 'cos_phase_nbr_h1', 'cos_phase_ndvi_h1',
            'offset_crswir', 'offset_evi', 'offset_nbr', 'elevation'
        ]

        # Define spectral indices used
        self.selected_indices = ['nbr', 'ndvi', 'crswir', 'evi']

    def process_tile(self, tile_path: str, mode: str = 'pred', return_raster: bool = False, return_aspect: bool = False) -> None:
        """
        Process a single tile and save the classification results.

        Parameters:
        mode (str): Mode of operation ('pred' for prediction, 'proba' for probability)
        tile_path (str): Path to the tile directory
        return_raster (bool): Whether to return the classification raster
        return_aspect (bool): Whether to return the aspect raster

        Returns:
        None
        """
        import os
        import numpy as np
        import rasterio
        import pandas as pd

        # Get reference file profile for output
        dem_path = os.path.join(tile_path, 'dem.tif')
        with rasterio.open(dem_path) as src:
            profile = src.profile
            height = profile['height']
            width = profile['width']

        # Load CHM for forest mask
        # chm = rasterio.open(os.path.join(tile_path, 'chm2020.tif')).read(1)
        # forest_mask = (chm > 500).astype(bool)

        # Load and prepare features
        path_features = os.path.join(tile_path, 'features')
        features_data = {}

        # Load selected indices
        for index in self.selected_indices:
            ap_file = rasterio.open(
                os.path.join(path_features, f'APO_{index}_{self.config}.tif')
            ).read()

            # Extract amplitude, phase, and offset for each harmonic
            for h in range(self.n_harmonics):
                # Amplitude
                features_data[f'amplitude_{index}_h{h+1}'] = (
                    ap_file[h] / 65535
                ).reshape(-1)

                # Phase
                phase = ap_file[self.n_harmonics+h] / 65535 * (2 * np.pi) - np.pi
                features_data[f'phase_{index}_h{h+1}'] = phase.reshape(-1)

                # Compute cos and sin of phase
                features_data[f'cos_phase_{index}_h{h+1}'] = np.cos(phase).reshape(-1)
                features_data[f'sin_phase_{index}_h{h+1}'] = np.sin(phase).reshape(-1)

            # Offset
            features_data[f'offset_{index}'] = (
                ap_file[2*self.n_harmonics] / 65535
            ).reshape(-1)

        # Add elevation
        dem = rasterio.open(os.path.join(tile_path, 'dem.tif')).read(1)
        aspect = get_aspect(dem, sigma=3)
        features_data['elevation'] = dem.reshape(-1)

        # Create DataFrame with only the features used in training
        df = pd.DataFrame(features_data)[self.features]

        # Perform inference
        try :
          if mode == 'pred':
              predictions = self.model.predict(df)
              # Reshape predictions to original dimensions and apply forest mask
              prediction_map = predictions.reshape(height, width)
              prediction_map = prediction_map.astype(np.uint8)
          elif mode == 'proba':
              predictions = self.model.predict_proba(df)
              prediction_map = predictions[:, 1].reshape(height, width)
              prodiction_map = (prediction_map * 255).astype(np.uint8)
        except :
          print('error in inference')
          print(df.columns)


        # Update profile for output
        profile.update(
            dtype='uint8',
            count=1,
            compress='lzw'
        )

        # Create results directory if it doesn't exist
        os.makedirs(os.path.join(tile_path, 'results'), exist_ok=True)

        # Save results
        output_path = os.path.join(
            tile_path, 'results', f'phenology_{self.config}_{mode}.tif'
        )
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction_map, 1)

        if return_raster and not return_aspect:
            return prediction_map
        elif return_raster and return_aspect:
            return prediction_map, aspect

    def process_multiple_tiles(self, tile_paths: list[str], n_jobs: int = -1) -> None:
        """
        Process multiple tiles in parallel.

        Parameters:
        tile_paths (list[str]): List of paths to tile directories
        n_jobs (int): Number of parallel jobs. -1 means using all processors.

        Returns:
        None
        """
        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs)(
            delayed(self.process_tile)(tile_path)
            for tile_path in tile_paths
        )

