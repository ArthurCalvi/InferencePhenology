#make necessary imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as mcolors



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

def plot_phenology_prediction(rgb: np.ndarray,
                                  prediction: np.ndarray,
                                  aspect: np.ndarray,
                                  figsize: tuple = (12, 6),
                                  alpha: float = 0.5, 
                                  logger = None) -> None:
    """Plot RGB image and phenology prediction map side by side with terrain shading."""
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
    
    # Print statistics
    if logger is not None:
        logger.info("\nPhenology Map Statistics:")
        valid_pred = masked_prediction[~masked_prediction.mask]
        logger.info(f"Valid pixels: {len(valid_pred)}")
        logger.info(f"Range: [{np.min(valid_pred):.4f}, {np.max(valid_pred):.4f}]")
        logger.info(f"Mean: {np.mean(valid_pred):.4f}")
        logger.info(f"Std: {np.std(valid_pred):.4f}")
    
    return fig, (ax1, ax2, cax)