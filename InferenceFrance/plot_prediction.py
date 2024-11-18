import os 
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

# Add the parent directory to Python path to import utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src.utils import get_aspect

def load_forest_mask(reference_path: str, chm_path: str) -> np.ndarray:
    """
    Load and reproject the CHM to create a forest mask based on the reference raster.
    
    Args:
        reference_path (str): Path to the reference raster (for projection and extent)
        chm_path (str): Path to the CHM raster
        
    Returns:
        np.ndarray: Forest mask matching the reference raster's extent and projection
    """
    try:
        # Open the reference raster and get bounds and CRS
        with rasterio.open(reference_path) as ref_raster:
            ref_bounds = ref_raster.bounds
            ref_crs = ref_raster.crs
            ref_transform = ref_raster.transform
            ref_shape = ref_raster.shape

        # Open the CHM raster
        with rasterio.open(chm_path) as chm_raster:
            chm_crs = chm_raster.crs
            dtype = chm_raster.profile['dtype']
            
            # Convert reference bounds to CHM CRS
            chm_bounds = transform_bounds(ref_crs, chm_crs, *ref_bounds)
            
            # Define the window in the CHM based on transformed bounds
            window = from_bounds(*chm_bounds, transform=chm_raster.transform)
            chm_window_data = chm_raster.read(1, window=window)
            chm_window_transform = chm_raster.window_transform(window)

        # Create an empty array for reprojected data
        reprojected_chm = np.zeros(ref_shape, dtype=dtype)
        
        # Reproject the cropped CHM data
        reproject(
            source=chm_window_data,
            destination=reprojected_chm,
            src_transform=chm_window_transform,
            src_crs=chm_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )

        # Create forest mask (values >= 250)
        forest_mask = reprojected_chm >= 0
        
        return forest_mask

    except Exception as e:
        print(f"Error creating forest mask: {e}")
        return None

def plot_phenology_prediction(rgb: np.ndarray,
                            prediction: np.ndarray,
                            aspect: np.ndarray,
                            forest_mask: np.ndarray = None,
                            figsize: tuple = (12, 6),
                            alpha: float = 0.5) -> None:
    """
    Plot RGB image and phenology prediction map side by side with terrain shading and forest mask.
    
    Args:
        rgb (np.ndarray): RGB image array
        prediction (np.ndarray): Phenology prediction array
        aspect (np.ndarray): Aspect array for terrain shading
        forest_mask (np.ndarray): Boolean mask for forest areas
        figsize (tuple): Figure size
        alpha (float): Transparency for terrain shading
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
    
    # Apply forest mask to prediction if provided
    if forest_mask is not None:
        masked_prediction = np.ma.masked_array(prediction, 
                                             mask=np.logical_or(np.isnan(prediction), ~forest_mask))
    else:
        masked_prediction = np.ma.masked_array(prediction, mask=np.isnan(prediction))
    
    # Plot prediction with hillshade
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
    ax2.set_title('Phenology Classification (Forest Areas)', pad=20)
    
    # Remove axes
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    
    # Print statistics for forest areas only
    print("\nPhenology Map Statistics (Forest Areas Only):")
    valid_pred = masked_prediction[~masked_prediction.mask]
    print(f"Valid forest pixels: {len(valid_pred)}")
    print(f"Range: [{np.min(valid_pred):.4f}, {np.max(valid_pred):.4f}]")
    print(f"Mean: {np.mean(valid_pred):.4f}")
    print(f"Std: {np.std(valid_pred):.4f}")
    
    return fig, (ax1, ax2, cax)

# Main execution
output_dir = "/Users/arthurcalvi/Repo/InferencePhenology/geefetch_test"
output_path = os.path.join(output_dir, 'phenology_map.tif')
big_chm_path = '/Users/arthurcalvi/Data/Disturbances_maps/FORMS/int2/2020_l93.tif'

print("\n=== Preparing Visualization ===")

# Read RGB image
mosaics = os.path.join(output_dir, 'mosaics')
rgb_path = os.path.join(mosaics, os.listdir(mosaics)[0])
with rasterio.open(rgb_path) as src:
    rgb = src.read([3,2,1])  # Read RGB bands
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
dem_path = os.path.join(output_dir, 'dem.tif')
with rasterio.open(dem_path) as src:
    dem = src.read(1)
    aspect = get_aspect(dem)

# Read phenology prediction
with rasterio.open(output_path) as src:
    prediction = src.read(1) / 255  # Scale back to [0,1]

# Load and create forest mask
forest_mask = load_forest_mask(output_path, big_chm_path)

# Plot RGB image and phenology prediction with forest mask
fig, axes = plot_phenology_prediction(rgb, prediction, aspect, forest_mask)
fig.savefig('figures/test_phenology_map.png', dpi=300)
plt.show()