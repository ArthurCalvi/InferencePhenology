import json
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Dict 

greco_regions_fr_en = {
    "Grand Ouest cristallin et océanique": "Greater Crystalline and Oceanic West",
    "Centre Nord semi-océanique": "Semi-Oceanic North Center",
    "Grand Est semi-continental": "Greater Semi-Continental East",
    "Vosges": "Vosges",
    "Jura": "Jura",
    "Sud-Ouest océanique": "Oceanic Southwest",
    "Massif central": "Central Massif",
    "Alpes": "Alps",
    "Pyrénées": "Pyrenees",
    "Méditerranée": "Mediterranean",
    "Corse": "Corsica"
}

mapping_real_greco = {
    'Côtes_et_plateaux_de_la_Manche': 'Centre Nord semi-océanique',
    'Ardenne_primaire': 'Grand Est semi-continental',
    'Préalpes_du_Nord': 'Alpes',
    'Garrigues': 'Méditerranée',
    'Massif_vosgien_central': 'Vosges',
    'Premier_plateau_du_Jura': 'Jura',
    'Piémont_pyrénéen': 'Pyrénées',
    'Terres_rouges': 'Sud-Ouest océanique',
    'Corse_occidentale': 'Corse',
    "Châtaigneraie_du_Centre_et_de_l'Ouest": 'Massif central',
    'Ouest-Bretagne_et_Nord-Cotentin': 'Grand Ouest cristallin et océanique',
    'Total': 'Total'
}

eco_region_classes = {
 1: 'Alps',
2: 'Central Massif',
3: 'Corsica',
4: 'Greater Crystalline and Oceanic West',
5: 'Greater Semi-Continental East',
6: 'Jura',
7: 'Mediterranean',
8: 'Oceanic Southwest',
9: 'Pyrenees',
10: 'Semi-Oceanic North Center',
11: 'Vosges'
 }

def load_greco_regions(greco_filepath: str, mapping_real_greco: Dict[str, str]) -> gpd.GeoDataFrame:
    """Load and preprocess eco-regions GeoDataFrame."""
    greco = gpd.read_file(greco_filepath)
    greco['greco'] = greco.codeser.apply(lambda x: x[0])
    greco = greco.dissolve(by='greco', aggfunc='first')
    greco = greco.reset_index().iloc[1:].to_crs('EPSG:2154')
    greco['greco_name'] = greco['NomSER'].map({k.replace('_', ' '): v for k, v in mapping_real_greco.items()})
    greco['greco_name'] = greco['greco_name'].map(greco_regions_fr_en)

    # Assign colors explicitly
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(greco))]
    greco['color'] = colors
    return greco.sort_values(by='greco_name')

def plot_agreement_comparison(greco_gdf: gpd.GeoDataFrame, 
                            dlt_metrics: dict, 
                            bdforet_metrics: dict, 
                            output_path: str = './figures/agreement_comparison.png'):
    """
    Create a figure comparing agreement metrics between inference and both DLT/BDForet.
    
    Args:
        greco_gdf: GeoDataFrame containing eco-regions geometries
        dlt_metrics: Dictionary containing DLT comparison metrics
        bdforet_metrics: Dictionary containing BDForet comparison metrics
        output_path: Path to save the output figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), subplot_kw={'aspect': 'equal'})
    
    # Prepare comparison data
    comparisons = [
        ('BD Foret', bdforet_metrics, ax1),
        ('DLT Copernicus', dlt_metrics, ax2)
    ]
    
    # Create custom colormap from red to green
    colors = ['#ff7f7f', '#ffff7f', '#7fff7f']
    cmap = LinearSegmentedColormap.from_list('agreement', colors)
    
    vmin, vmax = 0.3, 1.0  # Agreement range
    
    for title, metrics, ax in comparisons:
        # Create mapping of regions to agreement values
        agreements = {
            metrics['eco_regions'][str(i)]['region_name']: 
            metrics['eco_regions'][str(i)]['overall_agreement']
            for i in range(1, 12)
        }
        
        # Merge with GeoDataFrame
        greco_data = greco_gdf.copy()
        greco_data['agreement'] = greco_data['greco_name'].map(agreements)
        
        # Plot the map
        greco_data.plot(column='agreement', 
                       ax=ax,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       legend=False,
                       edgecolor='white',
                       linewidth=0.5)
        
        # Add region numbers
        for idx, row in greco_data.iterrows():
            centroid = row['geometry'].centroid
            agreement = row['agreement']
            ax.text(centroid.x, centroid.y, 
                   f"{idx + 1}\n{agreement:.2%}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=8,
                   color='black',
                   weight='bold')
        
        # Set title and remove axes
        ax.set_title(f'Agreement with {title}\nOverall: {metrics["global"]["overall_agreement"]:.2%}',
                    fontsize=12,
                    pad=20)
        ax.axis('off')
    
    # Add colorbar
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cax, label='Agreement')
    cbar.ax.yaxis.set_label_position('right')
    
    # Add region legend
    legend_patches = [
        mpatches.Patch(color='none', 
                      label=f"{i}: {name}")
        for i, name in eco_region_classes.items()
    ]
    fig.legend(handles=legend_patches,
              loc='lower center',
              ncol=4,
              frameon=False,
              fontsize=8,
              bbox_to_anchor=(0.5, 0))
    
    # Adjust layout and save
    plt.subplots_adjust(left=0.05, right=0.9, bottom=0.15, top=0.95, wspace=0.1)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Load metrics data
with open('./results/MyInference_vs_DLT_metrics.json', 'r') as f:
    dlt_metrics = json.load(f)
    
with open('./results/MyInference_vs_BDForet_metrics.json', 'r') as f:
    bdforet_metrics = json.load(f)


greco_gdf = load_greco_regions('/Users/arthurcalvi/Data/eco-regions/France/ser_l93_new/ser_l93_new.shp', mapping_real_greco)
print(greco_gdf)

plot_agreement_comparison(greco_gdf, dlt_metrics, bdforet_metrics)