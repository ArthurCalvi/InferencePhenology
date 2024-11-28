import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import logging
import sys
import os 
# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_inference.utils import fit_periodic_function_with_harmonics_robust
from src_inference.utils import reconstruct_signal

from typing import Callable, Tuple
# def fit_periodic_function_with_harmonics_robust(
#     time_series: np.ndarray,
#     qa: np.ndarray,
#     dates: List[datetime],
#     num_harmonics: int = 3,
#     max_iter: int = 10,
#     tol: float = 5e-2,
#     percentile: float = 75.0,
#     min_param_threshold: float = 1e-5,
#     callback: Optional[Callable[[int, float], None]] = None,
#     logger: Optional[logging.Logger] = None, 
#     debug: bool = False
# ) -> Tuple[np.ndarray]:
#     try:
#         if logger:
#             logger.info(f"Starting periodic function fitting with {num_harmonics} harmonics")
#             logger.info(f"Input shapes - time_series: {time_series.shape}, qa: {qa.shape}")
#             logger.info(f"Number of dates: {len(dates)}")
        
#         # Log initial data statistics
#         if logger:
#             logger.info(f"Initial time_series stats:")
#             logger.info(f"  NaN count: {np.sum(np.isnan(time_series))}")
#             logger.info(f"  Value range: [{np.nanmin(time_series)}, {np.nanmax(time_series)}]")
#             logger.info(f"Initial QA stats:")
#             logger.info(f"  Range: [{np.min(qa)}, {np.max(qa)}]")
#             logger.info(f"  Zero weights: {np.sum(qa == 0)}")

#         # Reshape and handle NaNs
#         pixels = time_series.reshape(time_series.shape[0], -1)
#         weights = qa.reshape(qa.shape[0], -1)
#         nan_mask = np.isnan(pixels)
        
#         if logger:
#             logger.info(f"Reshaped dimensions - pixels: {pixels.shape}, weights: {weights.shape}")
#             logger.info(f"NaN mask sum: {np.sum(nan_mask)} out of {pixels.size}")

#         # Compute and verify mean filling
#         pixel_means = np.nanmean(pixels, axis=0)
#         if logger:
#             logger.info(f"Pixel means stats:")
#             logger.info(f"  NaN in means: {np.sum(np.isnan(pixel_means))}")
#             logger.info(f"  Mean range: [{np.nanmin(pixel_means)}, {np.nanmax(pixel_means)}]")

#         # Fill NaNs
#         pixels_filled = np.where(nan_mask, 
#                                np.tile(pixel_means, (pixels.shape[0], 1)),
#                                pixels)
#         weights = np.where(nan_mask, 0, weights)
        
#         if logger:
#             logger.info(f"After filling:")
#             logger.info(f"  NaN count: {np.sum(np.isnan(pixels_filled))}")
#             logger.info(f"  Value range: [{np.min(pixels_filled)}, {np.max(pixels_filled)}]")
#             logger.info(f"  Zero weights count: {np.sum(weights == 0)}")

#         # Setup design matrix
#         times_datetime64 = np.array(dates, dtype='datetime64[D]')
#         days_since_start = (times_datetime64 - times_datetime64[0]).astype(int)
#         t_normalized = days_since_start / 365.25
        
#         harmonics = []
#         for k in range(1, num_harmonics + 1):
#             t_radians = 2 * np.pi * k * t_normalized
#             harmonics.extend([np.cos(t_radians), np.sin(t_radians)])
#         A = np.stack(harmonics + [np.ones_like(t_normalized)], axis=-1)
        
#         if logger:
#             logger.info(f"Design matrix A shape: {A.shape}")
#             logger.info(f"A stats - min: {np.min(A)}, max: {np.max(A)}")

#         # Initial parameter estimation
#         A_pinv = np.linalg.pinv(A)
#         initial_params = np.dot(A_pinv, pixels_filled).T

#         if logger:
#             logger.info(f"Initial parameter estimation:")
#             logger.info(f"  A_pinv shape: {A_pinv.shape}")
#             logger.info(f"  Initial params shape: {initial_params.shape}")
#             logger.info(f"  NaN in params: {np.sum(np.isnan(initial_params))}")
#             logger.info(f"  Param range: [{np.min(initial_params)}, {np.max(initial_params)}]")

#         params = initial_params.copy()
        
#         # Initial residuals check
#         initial_fitted_values = np.dot(A, params.T)
#         initial_residuals = pixels_filled - initial_fitted_values
        
#         if logger:
#             logger.info(f"Initial residuals stats:")
#             logger.info(f"  Shape: {initial_residuals.shape}")
#             logger.info(f"  Range: [{np.min(initial_residuals)}, {np.max(initial_residuals)}]")
#             logger.info(f"  NaN count: {np.sum(np.isnan(initial_residuals))}")

#         # Estimate initial sigma
#         sigma_initial = np.std(initial_residuals, axis=0)
#         sigma_initial[sigma_initial == 0] = np.finfo(float).eps  # Avoid division by zero

#         # Set delta based on initial residuals and do not update it
#         delta = 1.345 * sigma_initial
#         delta[delta == 0] = np.finfo(float).eps  # Avoid zero delta

#         epsilon = 1e-8
        
#         for iteration in range(max_iter):
#             params_old = params.copy()

#             # Broadcasting for weighted design matrix
#             A_expanded = np.expand_dims(A, 2)
#             weights_expanded = np.expand_dims(weights, 1)
#             A_weighted = A_expanded * weights_expanded

#             # Compute the normal equation components
#             ATA = np.einsum('ijk,ilk->jlk', A_weighted, A_expanded)
#             ATb = np.einsum('ijk,ik->jk', A_weighted, pixels_filled)

#             # Solve for parameters
#             ATA_reshaped = ATA.transpose(2, 0, 1)
#             ATb_reshaped = ATb.T

            
#             params = np.array([solve_params(ATA_reshaped[i], ATb_reshaped[i]) for i in range(ATA_reshaped.shape[0])])
#             params = np.nan_to_num(params)  # Replace NaNs with zero

#             # Calculate fitted values and residuals
#             fitted_values = np.dot(A, params.T)  # Shape: (time, n_pixels)
#             residuals = pixels_filled - fitted_values

#             # Estimate sigma (standard deviation of residuals)
#             sigma_residuals = np.std(residuals, axis=0)
#             sigma_residuals[sigma_residuals == 0] = np.finfo(float).eps  # Avoid division by zero

#             # Update weights based on residuals using Huber loss
#             residuals_abs = np.abs(residuals)
#             delta = 1.345 * sigma_residuals  # Update delta based on residuals
#             delta[delta == 0] = epsilon  # Avoid zero delta
#             mask = residuals_abs <= delta
#             weights_update = np.where(mask, 1, delta / (residuals_abs + epsilon))
#             weights = weights * weights_update

#             # Compute relative change, avoiding division by small numbers
#             min_param_threshold = 1e-5  # Or another appropriate small value
#             param_diff = np.abs(params - params_old)
#             relative_change = param_diff / (np.maximum(np.abs(params_old), min_param_threshold))
#             relative_change_flat = relative_change.flatten()

#             if debug:
#                 fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#                 _ = ax.hist(param_diff.flatten(), bins=100)
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.grid(axis='y', linestyle='--', alpha=0.7)
#                 ax.set_title(f"Params diff - Iteration {iteration + 1}")
#                 plt.show()

#             # Compute the desired percentile of relative change
#             percentile_value = np.percentile(relative_change_flat, percentile)

#             if callback:
#                 callback(iteration, percentile_value)
                
#             if logger:
#                 logger.debug(f"Iteration {iteration + 1}: relative change = {percentile_value:.6f}")
            
#             if percentile_value < tol:
#                 if logger:
#                     logger.info(f"Converged after {iteration + 1} iterations (relative change: {percentile_value:.6f})")
#                 break
#         else:
#             if logger:
#                 logger.warning(f"Maximum iterations ({max_iter}) reached without convergence")
        
#         # Extract results
#         params_reshaped = params.reshape(time_series.shape[1], time_series.shape[2], -1).transpose(2, 0, 1)
#         amplitude_maps = []
#         phase_maps = []
        
#         for i in range(num_harmonics):
#             A_params = params_reshaped[2 * i]
#             B_params = params_reshaped[2 * i + 1]
#             amplitude_map = np.sqrt(A_params ** 2 + B_params ** 2)
#             phase_map = np.arctan2(B_params, A_params)
            
#             phase_adjusted = (phase_map - (2 * np.pi * (i + 1) * t_normalized[0])) % (2 * np.pi)
#             phase_normalized = np.where(phase_adjusted > np.pi, phase_adjusted - 2 * np.pi, phase_adjusted)
            
#             amplitude_maps.append(amplitude_map)
#             phase_maps.append(phase_normalized)
            
#         offset_map = params_reshaped[-1]

#         # Var residual map reshaped to the tile shape
#         var_residual = np.var(residuals, axis=0)
#         var_residual_map = var_residual.reshape(time_series.shape[1], time_series.shape[2])
            
#         if logger:
#             logger.info("Successfully completed periodic function fitting")
#         return (*amplitude_maps, *phase_maps, offset_map, var_residual_map)
        
#     except Exception as e:
#         if logger:
#             logger.error(f"Error in periodic function fitting: {str(e)}")
#             logger.error(f"Error location: {str(e.__traceback__.tb_lineno)}")
#         raise

def solve_params(ATA: np.ndarray, ATb: np.ndarray) -> np.ndarray:
    """ Solve linear equations with error handling for non-invertible cases. """
    try:
        return np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return np.full(ATb.shape, np.nan)  # Return NaN for non-invertible matrices

def setup_logger() -> logging.Logger:
    """Set up a logger for testing."""
    logger = logging.getLogger('test_periodic_fitting')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def generate_test_data(
    num_timesteps: int = 36,
    spatial_shape: Tuple[int, int] = (10, 10),
    noise_level: float = 0.1,
    nan_ratio: float = 0.2,
    harmonics: List[Tuple[float, float, float]] = [(0.3, 0.2, 0.5), (0.15, 0.1, 1.0)],
    base_level: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[datetime], np.ndarray]:
    """Generate synthetic time series data with NaN values for testing."""
    np.random.seed(seed)
    
    # Generate dates (one every 10 days)
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i*10) for i in range(num_timesteps)]
    
    # Generate signal
    t = np.linspace(0, 2*np.pi, num_timesteps).reshape(-1, 1, 1)
    signal = np.full((num_timesteps, *spatial_shape), base_level)
    for amplitude, phase, freq in harmonics:
        signal += amplitude * np.sin(freq * t + phase)
    
    # Add noise and NaNs
    signal += np.random.normal(0, noise_level, signal.shape)
    nan_mask = np.random.random(signal.shape) < nan_ratio
    noisy_signal = signal.copy()
    noisy_signal[nan_mask] = np.nan
    
    # Create QA weights (1 for valid data, 0 for NaN)
    qa_weights = (~np.isnan(noisy_signal)).astype(float)
    
    return noisy_signal, qa_weights, dates, signal

def test_periodic_fitting_with_reconstruction(
    logger: Optional[logging.Logger] = None
) -> None:
    """Test the periodic function fitting with reconstruction visualization."""
    if logger is None:
        logger = setup_logger()
    
    # Generate test data
    time_series, qa_weights, dates, ground_truth = generate_test_data()
    
    results = fit_periodic_function_with_harmonics_robust(
        time_series=time_series,
        qa=qa_weights,
        dates=dates,
        num_harmonics=2,
        max_iter=10,
        logger=logger
    )
    
    # Extract components
    num_harmonics = 2
    amplitudes = results[:num_harmonics]
    phases = results[num_harmonics:2*num_harmonics]
    offset = results[2*num_harmonics]
    
    # Choose a pixel with some NaN values
    nan_pixels = np.any(np.isnan(time_series), axis=0)
    pixel_coords = tuple(map(lambda x: x[0], np.where(nan_pixels)))[:2]
    
    # Reconstruct signal
    reconstructed, dense_dates = reconstruct_signal(
        amplitudes=amplitudes,
        phases=phases,
        offset_map=offset,
        dates=dates,
        pixel_coords=pixel_coords
    )
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    fig.suptitle(f'Periodic Function Fitting Results (Pixel {pixel_coords})')
    
    # Top panel: Data and fits
    ax1.plot(dates, ground_truth[:, pixel_coords[0], pixel_coords[1]], 
            'g-', label='Ground truth', alpha=0.5)
    ax1.plot(dates, time_series[:, pixel_coords[0], pixel_coords[1]], 
            'k.', label='Observed data', alpha=0.5)
    ax1.plot(dense_dates, reconstructed, 'r-', label='Reconstructed signal')
    
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Bottom panel: Weights
    ax2.stem(dates, qa_weights[:, pixel_coords[0], pixel_coords[1]],
             basefmt=' ', label='Weights (0=NaN, 1=Valid)',
             linefmt='b-', markerfmt='bo')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Weight')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('test/test_outputs/periodic_fitting_reconstruction_test.png')
    plt.close()
    
    # Compute error metrics
    valid_mask = ~np.isnan(time_series[:, pixel_coords[0], pixel_coords[1]])
    original_dates = np.array(dates, dtype='datetime64[D]')
    dense_dates_idx = np.searchsorted(dense_dates, original_dates[valid_mask])
    
    rmse = np.sqrt(np.mean(
        (time_series[valid_mask, pixel_coords[0], pixel_coords[1]] - 
         reconstructed[dense_dates_idx])**2
    ))
    
    logger.info('\nReconstruction Quality:')
    logger.info(f'RMSE on valid points: {rmse:.4f}')
    logger.info(f'Number of valid points: {np.sum(valid_mask)}')
    logger.info(f'Reconstructed signal length: {len(reconstructed)}')
    
    # Parameter analysis
    logger.info('\nFitted Parameters:')
    for i in range(num_harmonics):
        logger.info(f'Harmonic {i+1}:')
        logger.info(f'  Amplitude: {amplitudes[i][pixel_coords[0], pixel_coords[1]]:.4f}')
        logger.info(f'  Phase: {phases[i][pixel_coords[0], pixel_coords[1]]:.4f}')
    logger.info(f'Offset: {offset[pixel_coords[0], pixel_coords[1]]:.4f}')

if __name__ == "__main__":
    logger = setup_logger()
    test_periodic_fitting_with_reconstruction(logger)