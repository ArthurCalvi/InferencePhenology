
import sys
import rasterio 
import os 
from datetime import datetime 
from tqdm import tqdm 


# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define model path relative to project root
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model', 'best_model_no_resampled_weights_h2_y1_iter1_f1_0.9605.pkl')
# Add src to Python path
sys.path.append(PROJECT_ROOT)

from src.utils import process_window, get_paired_files, calculate_optimal_windows

def process_window_with_inference(window, mosaic_paths, flg_paths, dates, model, num_harmonics=2, max_iter=1):
    """
    Process a window and perform inference directly.
    """
    features = process_window(
        window, mosaic_paths, flg_paths, dates,
        num_harmonics=num_harmonics,
        max_iter=max_iter
    )
    
    if features is None:
        return None
        
    # Convert features to format expected by model
    import pandas as pd
    import numpy as np
    
    # Initialize feature dictionary
    feature_data = {}
    
    # Process each index
    for index in ['nbr', 'ndvi', 'crswir', 'evi']:
        # Amplitude
        amp_key = f'amplitude_{index}_h1'
        if f'{index}_amplitude' in features and len(features[f'{index}_amplitude']) > 0:
            feature_data[amp_key] = (features[f'{index}_amplitude'][0] / 65535).reshape(-1)
            
        # Phase
        phase = (features[f'{index}_phase'][0] / 65535 * 2 * np.pi - np.pi).reshape(-1)
        feature_data[f'cos_phase_{index}_h1'] = np.cos(phase)
        feature_data[f'sin_phase_{index}_h1'] = np.sin(phase)
        
        # Offset
        feature_data[f'offset_{index}'] = (features[f'{index}_offset'] / 65535).reshape(-1)
    
    # Add elevation (assuming it's available in the first mosaic)
    with rasterio.open(mosaic_paths[0]) as src:
        dem = rasterio.open(os.path.join(os.path.dirname(os.path.dirname(mosaic_paths[0])), 'dem.tif')).read(1, window=window)
        feature_data['elevation'] = dem.reshape(-1)
    
    # Create DataFrame with only the features used by the model
    required_features = [
        'amplitude_evi_h1', 'amplitude_nbr_h1', 'amplitude_ndvi_h1',
        'cos_phase_crswir_h1', 'cos_phase_nbr_h1', 'cos_phase_ndvi_h1',
        'offset_crswir', 'offset_evi', 'offset_nbr', 'elevation'
    ]
    
    df = pd.DataFrame(feature_data)
    df = df[required_features]
    
    # Get probability predictions
    probabilities = model.predict_proba(df)
    
    # Reshape probabilities back to window shape
    prob_map = probabilities[:, 1].reshape(window.height, window.width)
    
    # Scale to uint8 (0-255)
    return (prob_map * 255).astype(np.uint8)

def main():
    # Configuration
    base_dir = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test"
    output_dir = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test"
    window_size = 1024
    num_harmonics = 2
    max_iter = 1

    # Load the model
    from joblib import load
    import json
    
    model = load(DIR_MODEL)

    # Get paired files
    try:
        mosaic_paths, flg_paths = get_paired_files(base_dir)
        print(f"Found {len(mosaic_paths)} mosaics :")
        for p in mosaic_paths:
            print(f"  - {p}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Extract dates from filenames
    dates = [datetime.strptime(os.path.basename(p).split('.tif')[0].split('_')[1], '%Y%m')
             for p in mosaic_paths]

    # Calculate windows from first mosaic
    windows = calculate_optimal_windows(mosaic_paths[0], window_size)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'phenology_probabilities.tif')

    # Get reference profile from first mosaic
    with rasterio.open(mosaic_paths[0]) as src:
        profile = src.profile.copy()
        
    # Update profile for probability output
    profile.update(
        dtype='uint8',
        count=1,
        compress='lzw',
        nodata=255  # Using 255 as nodata for uint8
    )

    # Create output file
    with rasterio.open(output_path, 'w', **profile) as dst:
        # Process windows in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        max_workers = min(os.cpu_count(), 4)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for window in windows:
                future = executor.submit(
                    process_window_with_inference,
                    window, mosaic_paths, flg_paths, dates, model,
                    num_harmonics, max_iter
                )
                futures.append((future, window))

            # Process results as they complete
            for future, window in tqdm(futures, desc="Processing windows"):
                try:
                    result = future.result()
                    if result is not None:
                        dst.write(result, 1, window=window)
                except Exception as e:
                    print(f"Error processing window {window}: {e}")
                    import traceback
                    traceback.print_exc()

    print("Processing complete")

if __name__ == "__main__":
    main()