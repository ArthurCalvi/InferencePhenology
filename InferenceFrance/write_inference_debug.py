import sys
import rasterio 
import os 
from datetime import datetime 
from tqdm import tqdm 
import numpy as np
import logging

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define model path relative to project root
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model', 'best_model_no_resampled_weights_h2_y1_iter1_f1_0.9605.pkl')
# Add src to Python path
sys.path.append(PROJECT_ROOT)
from src.utils import to_reflectance, compute_indices, compute_qa_weights, get_paired_files, calculate_optimal_windows


def setup_logger():
    """Configure and return a logger"""
    logger = logging.getLogger('inference')
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    file_handler = logging.FileHandler('inference.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def process_single_date(mosaic_path: str, flg_path: str, window: rasterio.windows.Window, logger) -> tuple:
    """
    Process a single date's mosaic and FLG data.
    """
    logger.info(f"\n=== Processing {os.path.basename(mosaic_path)} ===")
    
    try:
        with rasterio.open(mosaic_path) as src:
            # Read all required bands
            data = src.read([1,2,3,4,9,10], window=window)  # B2,B3,B4,B8,B11,B12
            logger.info(f"Raw data shape: {data.shape}, window: {window}")
            
            # Check for all-nodata windows (nan)
            nodata_percentage = np.mean(np.isnan(data)) * 100
            logger.info(f"Nodata percentage: {nodata_percentage}%")
            if nodata_percentage > 95:
                logger.warning(f"Window contains {nodata_percentage}% nodata values")
            
            # Convert to reflectance
            nodata = src.profile.get('nodata', np.nan)
            data = to_reflectance(data, nodata=nodata)
            logger.info(f"Reflectance range: {np.nanmin(data)} to {np.nanmax(data)}")
            
            # Compute indices
            ndvi, evi, nbr, crswir = compute_indices(*data, debug=True, logger=logger)
            
            # Log statistics for each index
            logger.info("\nIndex Statistics:")
            for name, index in [('NDVI', ndvi), ('EVI', evi), ('NBR', nbr), ('CRSWIR', crswir)]:
                valid_data = index[~np.isnan(index)]
                if len(valid_data) > 0:
                    logger.info(f"{name}:")
                    logger.info(f"  min: {np.min(valid_data):.3f}")
                    logger.info(f"  max: {np.max(valid_data):.3f}")
                    logger.info(f"  mean: {np.mean(valid_data):.3f}")
                    logger.info(f"  NaN%: {np.mean(np.isnan(index))*100:.1f}%")
                else:
                    logger.warning(f"{name} contains all NaN values")

        # Read and process FLG data
        with rasterio.open(flg_path) as src:
            flg = src.read(1, window=window)
            weight = compute_qa_weights(flg, logger=logger)
            logger.info("\nWeight statistics:")
            logger.info(f"  min: {np.min(weight):.3f}")
            logger.info(f"  max: {np.max(weight):.3f}")
            logger.info(f"  mean: {np.mean(weight):.3f}")
            
        return ndvi, evi, nbr, crswir, weight
        
    except Exception as e:
        logger.error(f"Error processing {mosaic_path}: {str(e)}")
        raise

def process_window_with_inference(window, mosaic_paths, flg_paths, dates, model, logger, num_harmonics=2, max_iter=1):
    """
    Process a window and perform inference.
    """
    logger.info(f"\n=== Processing window: {window} ===")
    
    # Initialize arrays to store time series
    time_series = {
        'ndvi': [], 'evi': [], 'nbr': [], 'crswir': []
    }
    weights = []
    
    # Process each date
    for mosaic_path, flg_path in zip(mosaic_paths, flg_paths):
        try:
            ndvi, evi, nbr, crswir, weight = process_single_date(mosaic_path, flg_path, window, logger)
            time_series['ndvi'].append(ndvi)
            time_series['evi'].append(evi)
            time_series['nbr'].append(nbr)
            time_series['crswir'].append(crswir)
            weights.append(weight)
        except Exception as e:
            logger.error(f"Failed to process {mosaic_path}: {str(e)}")
            return None
    
    # Convert lists to arrays and check for validity
    logger.info("\nChecking time series validity:")
    for key in time_series:
        time_series[key] = np.array(time_series[key])
        valid_percentage = np.mean(~np.isnan(time_series[key])) * 100
        logger.info(f"{key} valid data percentage: {valid_percentage:.2f}%")
        if np.all(np.isnan(time_series[key])):
            logger.error(f"All NaN values for {key} in entire time series")
            return None
    
    weights = np.array(weights)
    if np.all(weights == 0):
        logger.error("All zero weights in time series")
        return None
        
    # Compute harmonic features
    try:
        from src.utils import fit_periodic_function_with_harmonics_robust
        feature_data = {}
        
        for index_name, index_data in time_series.items():
            logger.info(f"\nProcessing {index_name}:")
            
            result = fit_periodic_function_with_harmonics_robust(
                index_data, weights, dates,
                num_harmonics=num_harmonics,
                max_iter=max_iter,
                verbose=1
            )
            
            # Unpack results
            amplitudes = result[:num_harmonics]
            phases = result[num_harmonics:2*num_harmonics]
            offset = result[2*num_harmonics]
            
            # Store amplitudes
            for i, amp in enumerate(amplitudes):
                feature_name = f'amplitude_{index_name}_h{i+1}'
                feature_data[feature_name] = amp.reshape(-1)
                logger.info(f"{feature_name}:")
                logger.info(f"  min: {np.nanmin(amp):.4f}")
                logger.info(f"  max: {np.nanmax(amp):.4f}")
                logger.info(f"  mean: {np.nanmean(amp):.4f}")
            
            # Store phases and compute cos/sin
            for i, phase in enumerate(phases):
                cos_name = f'cos_phase_{index_name}_h{i+1}'
                feature_data[cos_name] = np.cos(phase).reshape(-1)
                logger.info(f"{cos_name}:")
                logger.info(f"  min: {np.nanmin(feature_data[cos_name]):.4f}")
                logger.info(f"  max: {np.nanmax(feature_data[cos_name]):.4f}")
            
            # Store offset
            offset_name = f'offset_{index_name}'
            feature_data[offset_name] = offset.reshape(-1)
            logger.info(f"{offset_name}:")
            logger.info(f"  min: {np.nanmin(offset):.4f}")
            logger.info(f"  max: {np.nanmax(offset):.4f}")
            logger.info(f"  mean: {np.nanmean(offset):.4f}")
            
    except Exception as e:
        logger.error(f"Error computing harmonic features: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
    # Prepare features for model inference
    try:
        import pandas as pd
        
        # Add DEM data
        dem_path = os.path.join(os.path.dirname(os.path.dirname(mosaic_paths[0])), 'dem.tif')
        with rasterio.open(dem_path) as src:
            dem = src.read(1, window=window)
        feature_data['elevation'] = dem.reshape(-1)
        
        # Create DataFrame with all features
        df = pd.DataFrame(feature_data)
        
        # List required features
        required_features = [
            'amplitude_evi_h1', 'amplitude_nbr_h1', 'amplitude_ndvi_h1',
            'cos_phase_crswir_h1', 'cos_phase_nbr_h1', 'cos_phase_ndvi_h1',
            'offset_crswir', 'offset_evi', 'offset_nbr', 'elevation'
        ]
        
        # Check for missing features
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return None
        
        # Select required features
        df = df[required_features]
        
        # Print feature statistics
        logger.info("\nFeature statistics for model input:")
        for col in df.columns:
            logger.info(f"{col}:")
            logger.info(f"  min: {df[col].min():.4f}")
            logger.info(f"  max: {df[col].max():.4f}")
            logger.info(f"  mean: {df[col].mean():.4f}")
            logger.info(f"  NaN%: {df[col].isna().mean()*100:.2f}%")
        
        # Replace NaNs with 0
        df = df.fillna(0)
        
        # Get predictions
        probabilities = model.predict_proba(df)
        prob_map = probabilities[:, 1].reshape(window.height, window.width)
        
        return (prob_map * 255).astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    base_dir = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test"
    output_dir = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test"
    window_size = 1024
    num_harmonics = 2
    max_iter = 1
    
    # Setup logger
    logger = setup_logger()
    logger.info("\n=== Starting inference process ===")
    
    # Load model
    from joblib import load
    model = load(DIR_MODEL)
    logger.info("Model loaded successfully")
    
    # Get paired files
    try:
        mosaic_paths, flg_paths = get_paired_files(base_dir)
        if len(flg_paths) == 0:
            flg_paths = ['None'] * len(mosaic_paths)
        logger.info(f"\nFound {len(mosaic_paths)} mosaics:")
        for p in mosaic_paths:
            logger.info(f"  - {p}")
    except FileNotFoundError as e:
        logger.error(f"Error finding files: {str(e)}")
        return
    
    # Extract dates
    dates = [datetime.strptime(os.path.basename(p).split('.tif')[0].split('_')[1], '%Y%m')
             for p in mosaic_paths]
    
    # Calculate windows
    windows = calculate_optimal_windows(mosaic_paths[0], window_size)
    logger.info(f"\nCreated {len(windows)} windows for processing")
    
    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'phenology_probabilities.tif')
    logger.info(f"\nOutput will be saved to: {output_path}")
    
    # Get profile
    with rasterio.open(mosaic_paths[0]) as src:
        profile = src.profile.copy()
    profile.update(dtype='uint8', count=1, compress='lzw', nodata=255)
    
    # Process windows
    with rasterio.open(output_path, 'w', **profile) as dst:
        for window in tqdm(windows, desc="Processing windows"):
            try:
                result = process_window_with_inference(
                    window, mosaic_paths, flg_paths, dates, model, logger,
                    num_harmonics, max_iter
                )
                if result is not None:
                    dst.write(result, 1, window=window)
            except Exception as e:
                logger.error(f"Error processing window {window}: {str(e)}")
                continue

    logger.info("\nProcessing complete")

if __name__ == "__main__":
    main()