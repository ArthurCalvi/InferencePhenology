import sys
import os
import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path to import utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import (to_reflectance, compute_indices, get_paired_files, 
                      fit_periodic_function_with_harmonics_robust, compute_qa_weights)

from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
#import gaussian filter
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

def plot_phenology_prediction(rgb: np.ndarray,
                            prediction: np.ndarray,
                            aspect: np.ndarray,
                            figsize: tuple = (12, 6),
                            alpha: float = 0.5) -> None:
    """
    Plot RGB image and phenology prediction map side by side with terrain shading.
    """
    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 25, figure=fig)
    
    # Create three axes: RGB, prediction, and colorbar
    ax1 = fig.add_subplot(gs[0, :12])
    ax2 = fig.add_subplot(gs[0, 12:24])
    cax = fig.add_subplot(gs[0, 24:])
    
    # Plot RGB image
    rgb_display = np.moveaxis(rgb, 0, -1)
    ax1.imshow(rgb_display)
    
    # Add scalebar to RGB image
    scalebar1 = ScaleBar(10, "m", length_fraction=0.5,
                        location='lower center',
                        color='k',
                        box_color='w',
                        box_alpha=0.75,
                        sep=5)
    ax1.add_artist(scalebar1)
    ax1.set_title('RGB Image', pad=20)
    
    # Create custom colormap for probability
    colors_probability = [
        (0.8, 0.4, 0.2),  # Brown-orange (deciduous)
        (0.6, 0.5, 0.2),  # Brown-yellow transition
        (0.4, 0.6, 0.2),  # Yellow-green transition
        (0.1, 0.6, 0.1)   # Green (evergreen)
    ]
    cmap_probability = mcolors.LinearSegmentedColormap.from_list(
        'deciduous_evergreen', colors_probability
    )
    
    # Create hillshade effect from aspect
    aspect_rad = np.radians(aspect)
    azimuth = 5 * np.pi / 4
    altitude = np.pi / 4
    shading = np.cos(aspect_rad - azimuth) * np.sin(altitude)
    shading = (shading + 1) / 2
    shading = 0.3 + 0.7 * shading
    
    # Plot prediction with hillshade
    masked_prediction = np.ma.masked_array(prediction, mask=np.isnan(prediction))
    phenology = ax2.imshow(masked_prediction, cmap=cmap_probability, vmin=0, vmax=1)
    ax2.imshow(shading, cmap='gray', alpha=0.1)
    
    # Add colorbar in separate axis
    cbar = plt.colorbar(phenology, cax=cax)
    cbar.ax.set_ylabel('Probability of Evergreen', rotation=270, labelpad=25)
    cbar.ax.set_yticklabels(['Deciduous', 'Evergreen'],
                           rotation=270, va='center')
    cbar.ax.set_yticks([0.1, 0.9])
    
    # Add scalebar to phenology map
    scalebar2 = ScaleBar(10, "m", length_fraction=0.5,
                        location='lower center',
                        color='k',
                        box_color='w',
                        box_alpha=0.75,
                        sep=5)
    ax2.add_artist(scalebar2)
    ax2.set_title('Phenology Classification', pad=20)
    
    # Remove axes
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    return fig, (ax1, ax2, cax)

def plot_all_harmonic_features(features, window_shape, num_harmonics):
    """
    Plot all harmonic features in a comprehensive way.
    """
    # Plot amplitudes for all indices together
    fig, axes = plt.subplots(len(features), num_harmonics + 1, 
                            figsize=(4*(num_harmonics + 1), 4*len(features)))
    fig.suptitle('Harmonic Features for All Indices', fontsize=16)
    
    # Create a colormap for each type of feature
    vmin_amp = np.inf
    vmax_amp = -np.inf
    vmin_offset = np.inf
    vmax_offset = -np.inf
    
    # Find global min/max for better comparison
    for index_name, result in features.items():
        # Amplitudes
        for i in range(num_harmonics):
            amp = result[i]
            vmin_amp = min(vmin_amp, np.nanmin(amp))
            vmax_amp = max(vmax_amp, np.nanmax(amp))
        # Offset
        offset = result[-1]
        vmin_offset = min(vmin_offset, np.nanmin(offset))
        vmax_offset = max(vmax_offset, np.nanmax(offset))
    
    # Plot each index's features
    for idx, (index_name, result) in enumerate(features.items()):
        # Plot amplitudes
        for i in range(num_harmonics):
            ax = axes[idx, i]
            im = ax.imshow(result[i], vmin=vmin_amp, vmax=vmax_amp, cmap='viridis')
            plt.colorbar(im, ax=ax)
            
            if idx == 0:
                ax.set_title(f'Harmonic {i+1}\nAmplitude')
            if i == 0:
                ax.set_ylabel(index_name.upper())
        
        # Plot offset
        ax = axes[idx, -1]
        im = ax.imshow(result[-1], vmin=vmin_offset, vmax=vmax_offset, cmap='viridis')
        plt.colorbar(im, ax=ax)
        if idx == 0:
            ax.set_title('Offset')
    
    plt.tight_layout()
    plt.show()
    
    # Plot phase relationships
    if num_harmonics > 0:
        fig, axes = plt.subplots(len(features), num_harmonics, 
                                figsize=(4*num_harmonics, 4*len(features)))
        if num_harmonics == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle('Phase-Amplitude Relationships', fontsize=16)
        
        for idx, (index_name, result) in enumerate(features.items()):
            for i in range(num_harmonics):
                ax = axes[idx, i]
                amplitude = result[i].flatten()
                phase = result[num_harmonics + i].flatten()
                
                # Remove NaN values
                valid = ~np.isnan(amplitude) & ~np.isnan(phase)
                amplitude = amplitude[valid]
                phase = phase[valid]
                
                if len(amplitude) > 0:
                    # Create 2D histogram
                    h = ax.hist2d(phase, amplitude, bins=50, cmap='viridis')
                    plt.colorbar(h[3], ax=ax)
                    
                    ax.set_xlabel('Phase')
                    ax.set_ylabel('Amplitude')
                    
                    if idx == 0:
                        ax.set_title(f'Harmonic {i+1}')
                    if i == 0:
                        ax.set_ylabel(f'{index_name.upper()}\nAmplitude')
        
        plt.tight_layout()
        plt.show()

def check_harmonic_features():
    """
    Test the complete pipeline including harmonic feature extraction.
    """
    base_dir = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test"
    num_harmonics = 2
    max_iter = 1
    
    print("\n=== Starting Harmonic Features Check ===")
    
    # Get all mosaic and FLG files
    mosaic_paths, flg_paths = get_paired_files(base_dir)
    if len(flg_paths) == 0:
        flg_paths = ['None'] * len(mosaic_paths)

    dates = [datetime.strptime(os.path.basename(p).split('.tif')[0].split('_')[1], '%Y%m')
             for p in mosaic_paths]
    
    print(f"\nProcessing {len(mosaic_paths)} files:")
    for i, (date, mosaic_path, flg_path) in enumerate(zip(dates, mosaic_paths, flg_paths)):
        print(f"\n{i+1}. Date: {date.strftime('%Y-%m')}")
        print(f"   Mosaic: {os.path.basename(mosaic_path)}")
        print(f"   FLG: {os.path.basename(flg_path)}")
    
    # Define a window to process
    window = rasterio.windows.Window(0, 0, 1024, 1024)
    print(f"\nProcessing window: {window}")
    
    # Store temporal data
    all_indices = []
    weights = []
    
    # Process each date
    for i, (date, mosaic_path, flg_path) in enumerate(zip(dates, mosaic_paths, flg_paths)):
        print(f"\n=== Processing date {i+1}/{len(dates)}: {date.strftime('%Y-%m')} ===")
        print(f"Reading mosaic: {os.path.basename(mosaic_path)}")
        
        # Process mosaic
        with rasterio.open(mosaic_path) as src:
            # Read data
            data = src.read([1,2,3,4,9,10], window=window)
            print(f"Raw data shape: {data.shape}")
            print(f"Raw data range: [{np.min(data)}, {np.max(data)}]")
            
            # Get nodata value
            nodata = src.profile.get('nodata', np.nan)
            print(f"Nodata value: {nodata}")
            
            # Convert to reflectance
            reflectance = to_reflectance(data, nodata=nodata)
            print(f"Reflectance shape: {reflectance.shape}")
            print(f"Reflectance range: [{np.nanmin(reflectance)}, {np.nanmax(reflectance)}]")
            
            # Compute indices
            ndvi, evi, nbr, crswir = compute_indices(*reflectance, debug=True)
            
            # Print indices statistics
            print("\nIndices statistics:")
            for name, index in [('NDVI', ndvi), ('EVI', evi), ('NBR', nbr), ('CRSWIR', crswir)]:
                valid_data = index[~np.isnan(index)]
                if len(valid_data) > 0:
                    print(f"{name}:")
                    print(f"  Shape: {index.shape}")
                    print(f"  Range: [{np.min(valid_data):.4f}, {np.max(valid_data):.4f}]")
                    print(f"  Mean: {np.mean(valid_data):.4f}")
                    print(f"  NaN%: {np.mean(np.isnan(index))*100:.1f}%")
                else:
                    print(f"{name}: All NaN")
            
            indices_data = {
                'ndvi': ndvi,
                'evi': evi,
                'nbr': nbr,
                'crswir': crswir
            }
            
        # print(f"\nReading FLG: {os.path.basename(flg_path)}")
        # with rasterio.open(flg_path) as src:
        #     # Read FLG data
        #     flg = src.read(1, window=window)
        #     print(f"FLG shape: {flg.shape}")
        #     print(f"FLG unique values: {np.unique(flg)}")
            
        #     # Compute weights
        #     weight = compute_qa_weights(flg)
        #     print(f"Weight shape: {weight.shape}")
        #     print(f"Weight range: [{np.min(weight)}, {np.max(weight)}]")
        
        # Store data
        print("\nStoring data...")
        all_indices.append(indices_data)
        # weights.append(weight)
        
        print(f"Current number of stored dates: {len(all_indices)}")
    
    # Convert weights to array
    # weights = np.array(weights)
    weights = np.ones((len(all_indices), window.height, window.width))
    
    print(f"\nFinal data check:")
    print(f"Number of dates processed: {len(all_indices)}")
    print(f"Weights shape: {weights.shape}")
    
    # Stack and check time series
    print("\n=== Time Series Statistics ===")
    for index_name in ['ndvi', 'evi', 'nbr', 'crswir']:
        print(f"\nProcessing {index_name}...")
        # Stack all dates for this index
        print(f"Stacking {len(all_indices)} dates...")
        
        # Debug print for the first date
        first_date_data = all_indices[0][index_name]
        print(f"First date shape: {first_date_data.shape}")
        print(f"First date range: [{np.nanmin(first_date_data)}, {np.nanmax(first_date_data)}]")
        
        index_data = np.stack([date_data[index_name] for date_data in all_indices])
        
        print(f"{index_name.upper()}:")
        print(f"  Stacked shape: {index_data.shape}")
        print(f"  Valid data range: [{np.nanmin(index_data):.4f}, {np.nanmax(index_data):.4f}]")
        print(f"  Mean: {np.nanmean(index_data):.4f}")
        print(f"  NaN%: {np.mean(np.isnan(index_data))*100:.1f}%")
    
    print("\n=== Computing Harmonic Features ===")
    features = {}
    
    window_shape = (window.height, window.width)
    for index_name in ['ndvi', 'evi', 'nbr', 'crswir']:
        print(f"\nProcessing {index_name}...")
        
        # Stack time series
        index_data = np.stack([date_data[index_name] for date_data in all_indices])
        print(f"Input shape: {index_data.shape}")
        print(f"Input data range: [{np.nanmin(index_data):.4f}, {np.nanmax(index_data):.4f}]")
        print(f"Input NaN%: {np.mean(np.isnan(index_data))*100:.1f}%")
        
        # Compute features
        result = fit_periodic_function_with_harmonics_robust(
            index_data, weights, dates,
            num_harmonics=num_harmonics,
            max_iter=max_iter,
            verbose=1
        )
        features[index_name] = result
        
        # Check results
        print("\nFeature statistics:")
        # Amplitudes
        for i in range(num_harmonics):
            amp = result[i]
            print(f"\nHarmonic {i+1} Amplitude:")
            print(f"  Shape: {amp.shape}")
            print(f"  Range: [{np.nanmin(amp):.4f}, {np.nanmax(amp):.4f}]")
            print(f"  Mean: {np.nanmean(amp):.4f}")
            print(f"  NaN%: {np.mean(np.isnan(amp))*100:.1f}%")
        
        # Offset
        offset = result[-1]
        print("\nOffset:")
        print(f"  Shape: {offset.shape}")
        print(f"  Range: [{np.nanmin(offset):.4f}, {np.nanmax(offset):.4f}]")
        print(f"  Mean: {np.nanmean(offset):.4f}")
        print(f"  NaN%: {np.mean(np.isnan(offset))*100:.1f}%")

    # Add at the end:
    # After computing and plotting harmonic features, add:
    print("\n=== Running Model Inference ===")
    
    # Load model
    model_path = os.path.join(PROJECT_ROOT, 'model', 
                             'best_model_with_bdforet_no_resampled_weights_h2_y1_iter10_scaled01_featuresfromRFECV_nf10_f1_0.9601.pkl')
    from joblib import load
    model = load(model_path)
    print("Model loaded successfully")
    
    # Prepare features for model
    import pandas as pd
    feature_data = {}
    
    # Extract and reshape features
    for index_name, result in features.items():
        # Store amplitudes
        for i in range(num_harmonics):
            feature_name = f'amplitude_{index_name}_h{i+1}'
            feature_data[feature_name] = result[i].reshape(-1)
        
        # Store phases as cosine
        for i in range(num_harmonics):
            cos_name = f'cos_phase_{index_name}_h{i+1}'
            phase = result[num_harmonics + i]
            feature_data[cos_name] = np.cos(phase).reshape(-1)
        
        # Store offset
        feature_data[f'offset_{index_name}'] = result[-1].reshape(-1)
    
    # Add DEM data
    dem_path = os.path.join(base_dir, 'dem.tif')
    with rasterio.open(dem_path) as src:
        dem = src.read(1, window=window)
    feature_data['elevation'] = dem.reshape(-1)
    
    # Create DataFrame with required features
    required_features = [
        'amplitude_evi_h1', 'amplitude_nbr_h1', 'amplitude_ndvi_h1',
        'cos_phase_crswir_h1', 'cos_phase_nbr_h1', 'cos_phase_ndvi_h1',
        'offset_crswir', 'offset_evi', 'offset_nbr', 'elevation'
    ]
    
    df = pd.DataFrame(feature_data)
    df = df[required_features]  # Select and order features
    
    print("\nFeature statistics for model:")
    for col in df.columns:
        print(f"{col}:")
        print(f"  Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
        print(f"  Mean: {df[col].mean():.4f}")
        print(f"  NaN%: {df[col].isna().mean()*100:.2f}%")
    
    # Replace any remaining NaNs with 0
    df = df.fillna(0)
    
    # Get probability predictions
    print("\nRunning inference...")
    probabilities = model.predict_proba(df)
    prob_map = probabilities[:, 1].reshape(window.height, window.width)
    
    print("\nProbability map statistics:")
    print(f"Range: [{np.min(prob_map):.4f}, {np.max(prob_map):.4f}]")
    print(f"Mean: {np.mean(prob_map):.4f}")
    print(f"Std: {np.std(prob_map):.4f}")

     # Add at the end:
    print("\n=== Plotting Features ===")
    plot_all_harmonic_features(features, (window.height, window.width), num_harmonics)
    
    # After computing probabilities, add:
    print("\n=== Preparing Visualization ===")
    
    # Read RGB image
    with rasterio.open(mosaic_paths[0]) as src:
        rgb = src.read([3,2,1], window=window)  # Read RGB bands
        # Normalize RGB to [0,1]
        rgb = rgb.astype(np.float32)
        for i in range(3):
            band = rgb[i]
            valid = ~np.isnan(band)
            if np.any(valid):
                min_val = np.percentile(band[valid], 2)
                max_val = np.percentile(band[valid], 98)
                rgb[i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
            rgb[i] = np.nan_to_num(rgb[i], 0)
    
    # Read DEM and compute aspect
    dem_path = os.path.join(base_dir, 'dem.tif')
    with rasterio.open(dem_path) as src:
        dem = src.read(1, window=window)
        aspect = get_aspect(dem)
    
    # Plot results
    print("\n=== Plotting Results ===")
    plot_phenology_prediction(rgb, prob_map, aspect)
    plt.show()

if __name__ == "__main__":
    check_harmonic_features()