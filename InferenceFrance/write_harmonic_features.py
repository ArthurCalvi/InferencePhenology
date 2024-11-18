import os
import sys 
from datetime import datetime 
from tqdm import tqdm 
sys.path.append('./src')
from utils import get_paired_files, calculate_optimal_windows, process_window_wrapper 

from rasterio.merge import merge
def main():
    # Configuration
    base_dir = "/Users/arthurcalvi/Repo/InferencePhenology/extract_test"
    output_dir = "/Users/arthurcalvi/Repo/InferencePhenology/extract_test"
    window_size = 1024
    num_harmonics = 2
    max_iter = 1

    # Get paired files
    try:
        mosaic_paths, flg_paths = get_paired_files(base_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Extract dates from filenames
    dates = [datetime.strptime(os.path.basename(p).split('_')[1], '%Y%m%d')
             for p in mosaic_paths]

    # Calculate windows from first mosaic
    windows = calculate_optimal_windows(mosaic_paths[0], window_size)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'harmonic_features.tif')

    # Create arguments for each window
    process_args = [
    (window, mosaic_paths, flg_paths, dates, output_dir, num_harmonics, max_iter)
    for window in windows
    ]


    # Use ProcessPoolExecutor for better control
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Limit the number of processes to avoid memory issues
    max_workers = min(os.cpu_count(), 4)  # Adjust this number based on your system
    
    temp_files = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_window = {executor.submit(process_window_wrapper, args): args[0] for args in process_args}
        for future in tqdm(as_completed(future_to_window), total=len(windows), desc="Processing windows"):
            temp_file = future.result()
            if temp_file:
                temp_files.append(temp_file)

    # # Merge temporary files into final output
    # datasets = [rasterio.open(fp) for fp in temp_files]
    
    # # Print info about temp files before merging
    # print("\nChecking temporary files before merge:")
    # for i, ds in enumerate(datasets):
    #     nodata_percentage = np.mean(ds.read() == 65535) * 100
    #     print(f"Temp file {i}: shape={ds.shape}, nodata={nodata_percentage:.2f}%")
    
    # mosaic, out_trans = merge(datasets)
    
    # # Check merged result
    # nodata_percentage = np.mean(mosaic == 65535) * 100
    # print(f"\nMerged result: shape={mosaic.shape}, nodata={nodata_percentage:.2f}%")

    # # Update metadata and write final output
    # out_meta = datasets[0].meta.copy()
    # out_meta.update({
    #     "driver": "GTiff",
    #     "height": mosaic.shape[1],
    #     "width": mosaic.shape[2],
    #     "transform": out_trans,
    #     "compress": None,  # No compression for debugging
    #     "nodata": 65535,
    #     "count": mosaic.shape[0],
    #     "tiled": True,
    #     "blockxsize": 256,
    #     "blockysize": 256
    # })

    # output_path = os.path.join(output_dir, 'harmonic_features.tif')
    # with rasterio.open(output_path, "w", **out_meta) as dest:
    #     dest.write(mosaic)


if __name__ == "__main__":
    main()