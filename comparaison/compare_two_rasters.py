import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import confusion_matrix
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

class MapComparison:
    def __init__(
        self,
        map1_path: str,
        map2_path: str,
        ecoregions_gdf: gpd.GeoDataFrame,
        map1_classes: dict,
        map2_classes: dict,
        window_size: int = 1024
    ):
        """
        Initialize the map comparison.
        
        Args:
            map1_path: Path to first map (your output)
            map2_path: Path to second map (reference)
            ecoregions_gdf: GeoDataFrame containing ecoregion polygons
            map1_classes: Dict mapping values in map1 to class names (e.g., {1: 'deciduous', 2: 'evergreen'})
            map2_classes: Dict mapping values in map2 to class names
            window_size: Size of processing windows
        """
        self.map1_path = map1_path
        self.map2_path = map2_path
        self.ecoregions_gdf = ecoregions_gdf
        self.map1_classes = map1_classes
        self.map2_classes = map2_classes
        self.window_size = window_size
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input files and data."""
        # Check if files exist
        if not os.path.exists(self.map1_path) or not os.path.exists(self.map2_path):
            raise FileNotFoundError("One or both map files not found")
            
        # Verify matching CRS and extent
        with rasterio.open(self.map1_path) as src1, rasterio.open(self.map2_path) as src2:
            if src1.crs != src2.crs:
                raise ValueError("Maps have different CRS")
            if src1.bounds != src2.bounds:
                raise ValueError("Maps have different bounds")
                
        # Ensure ecoregions GDF has same CRS as maps
        with rasterio.open(self.map1_path) as src:
            if self.ecoregions_gdf.crs != src.crs:
                self.ecoregions_gdf = self.ecoregions_gdf.to_crs(src.crs)

    def _process_window(self, window, eco_mask=None):
        """
        Process a single window and compute confusion matrix.
        
        Args:
            window: rasterio window object
            eco_mask: Optional mask for ecoregion
            
        Returns:
            Confusion matrix for the window
        """
        with rasterio.open(self.map1_path) as src1, rasterio.open(self.map2_path) as src2:
            data1 = src1.read(1, window=window)
            data2 = src2.read(1, window=window)
            
            # Apply mappings
            data1_mapped = np.vectorize(self.map1_classes.get)(data1)
            data2_mapped = np.vectorize(self.map2_classes.get)(data2)
            
            # Apply ecoregion mask if provided
            if eco_mask is not None:
                mask = eco_mask[window.row_off:window.row_off + window.height,
                              window.col_off:window.col_off + window.width]
                valid_pixels = mask & (data1_mapped != 'nodata') & (data2_mapped != 'nodata')
            else:
                valid_pixels = (data1_mapped != 'nodata') & (data2_mapped != 'nodata')
            
            # Flatten and filter valid pixels
            y_true = data2_mapped[valid_pixels]
            y_pred = data1_mapped[valid_pixels]
            
            # Get unique classes
            classes = sorted(set(self.map1_classes.values()) - {'nodata'})
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            return cm
            
    def compute_metrics(self):
        """
        Compute agreement metrics and confusion matrices globally and per ecoregion.
        
        Returns:
            dict: Contains global and per-ecoregion metrics
        """
        results = {
            'global': {},
            'per_ecoregion': {}
        }
        
        # Create windows
        with rasterio.open(self.map1_path) as src:
            windows = list(src.block_windows(1))
            
            # Create ecoregion masks
            eco_masks = {}
            for idx, row in self.ecoregions_gdf.iterrows():
                mask = rasterio.features.rasterize(
                    [(row.geometry, 1)],
                    out_shape=src.shape,
                    transform=src.transform,
                    fill=0,
                    dtype='uint8'
                )
                eco_masks[row['eco_id']] = mask
        
        # Process global metrics
        print("Computing global metrics...")
        global_cm = np.zeros((len(set(self.map1_classes.values())) - 1,
                            len(set(self.map1_classes.values())) - 1))
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_window = {
                executor.submit(self._process_window, window): window
                for window in windows
            }
            
            for future in tqdm(future_to_window):
                cm = future.result()
                if cm is not None:
                    global_cm += cm
        
        # Calculate global metrics
        results['global']['confusion_matrix'] = global_cm
        results['global']['overall_agreement'] = np.diag(global_cm).sum() / global_cm.sum()
        
        # Process per-ecoregion metrics
        print("Computing per-ecoregion metrics...")
        for eco_id, eco_mask in eco_masks.items():
            eco_cm = np.zeros_like(global_cm)
            
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_window = {
                    executor.submit(self._process_window, window, eco_mask): window
                    for window in windows
                }
                
                for future in tqdm(future_to_window, desc=f"Ecoregion {eco_id}"):
                    cm = future.result()
                    if cm is not None:
                        eco_cm += cm
            
            results['per_ecoregion'][eco_id] = {
                'confusion_matrix': eco_cm,
                'overall_agreement': np.diag(eco_cm).sum() / eco_cm.sum() if eco_cm.sum() > 0 else 0
            }
        
        return results

def format_results(results, class_names):
    """
    Format results into pandas DataFrames for easy viewing and export.
    
    Args:
        results: Results dictionary from compute_metrics
        class_names: List of class names in order
        
    Returns:
        tuple: (global_df, eco_df, global_cm_df, eco_cm_dict)
    """
    # Format global results
    global_df = pd.DataFrame({
        'Metric': ['Overall Agreement'],
        'Value': [results['global']['overall_agreement']]
    })
    
    # Format global confusion matrix
    global_cm_df = pd.DataFrame(
        results['global']['confusion_matrix'],
        columns=class_names,
        index=class_names
    )
    
    # Format ecoregion results
    eco_results = []
    eco_cm_dict = {}
    
    for eco_id, eco_data in results['per_ecoregion'].items():
        eco_results.append({
            'Ecoregion': eco_id,
            'Overall Agreement': eco_data['overall_agreement']
        })
        
        eco_cm_dict[eco_id] = pd.DataFrame(
            eco_data['confusion_matrix'],
            columns=class_names,
            index=class_names
        )
    
    eco_df = pd.DataFrame(eco_results)
    
    return global_df, eco_df, global_cm_df, eco_cm_dict

# Example usage:
if __name__ == "__main__":
    # Example mappings
    map1_classes = {
        1: 'deciduous',
        2: 'evergreen',
        255: 'nodata'
    }
    
    map2_classes = {
        10: 'deciduous',
        20: 'evergreen',
        0: 'nodata'
    }
    
    # Initialize comparison
    map1_path = 'path/to/your/output.tif'
    map2_path = 'path/to/reference.tif'
    comparison = MapComparison(
        map1_path=map1_path,
        map2_path=map2_path,
        ecoregions_gdf=your_ecoregions_gdf,
        map1_classes=map1_classes,
        map2_classes=map2_classes
    )
    
    # Compute metrics
    results = comparison.compute_metrics()
    
    # Format results
    class_names = ['deciduous', 'evergreen']
    global_df, eco_df, global_cm_df, eco_cm_dict = format_results(results, class_names)
    
    # Print results
    print("\nGlobal Metrics:")
    print(global_df)
    print("\nGlobal Confusion Matrix:")
    print(global_cm_df)
    print("\nPer-Ecoregion Metrics:")
    print(eco_df)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    filename1 = os.path.basename(map1_path).replace('.tif', '')
    filename2 = os.path.basename(map2_path).replace('.tif', '')
    global_df.to_parquet(f'results/global_metrics_{filename1}_vs_{filename2}.parquet', index=False)
    eco_df.to_parquet(f'results/ecoregion_metrics_{filename1}_vs_{filename2}.parquet', index=False)
    global_cm_df.to_parquet(f'results/global_confusion_matrix_{filename1}_vs_{filename2}.parquet')
    
    for eco_id, cm_df in eco_cm_dict.items():
        cm_df.to_parquet(f'confusion_matrix_eco_{eco_id}.parquet')