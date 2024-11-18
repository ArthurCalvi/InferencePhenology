# import os
# import shutil
# import rasterio
# from pathlib import Path
# from tqdm import tqdm
# import numpy as np

# def get_metro_bounds_lambert93():
#     """
#     Return the approximate bounds of metropolitan France in Lambert 93
#     """
#     return {
#         'west': -1000000,
#         'east': 4000000,
#         'south': 2000000,
#         'north': 10000000
#     }

# def check_file_bounds(file_path, metro_bounds):
#     """
#     Check if a file's bounds intersect with metropolitan France
    
#     Returns:
#     - dict with intersection info and file stats
#     """
#     try:
#         with rasterio.open(file_path) as src:
#             # Get file bounds
#             bounds = src.bounds
            
#             # Check intersection
#             intersects = (
#                 bounds.left <= metro_bounds['east'] and
#                 bounds.right >= metro_bounds['west'] and
#                 bounds.bottom <= metro_bounds['north'] and
#                 bounds.top >= metro_bounds['south']
#             )
            
#             # Read a sample of data to check values
#             data = src.read(1)
            
#             return {
#                 'file': file_path,
#                 'intersects': intersects,
#                 'bounds': bounds,
#                 'crs': src.crs,
#                 'shape': data.shape,
#                 'min_val': float(np.min(data)),
#                 'max_val': float(np.max(data)),
#                 'mean_val': float(np.mean(data)),
#                 'non_zero': float(np.sum(data != 0)) / data.size * 100  # percentage of non-zero values
#             }
            
#     except Exception as e:
#         return {
#             'file': file_path,
#             'error': str(e)
#         }

# def move_file_to_trash(file_path, trash_dir, reason):
#     """
#     Move a file to the trash directory with reason subdirectory
#     """
#     # Create reason subdirectory
#     reason_dir = os.path.join(trash_dir, reason)
#     os.makedirs(reason_dir, exist_ok=True)
    
#     # Move file
#     filename = os.path.basename(file_path)
#     dest_path = os.path.join(reason_dir, filename)
    
#     try:
#         shutil.move(file_path, dest_path)
#         print(f"Moved {filename} to {reason} folder")
#     except Exception as e:
#         print(f"Error moving {filename}: {str(e)}")

# def main(tmp_vrt_dir):
#     """
#     Main function to check all files in tmp_vrt directory
#     """
#     # Create trash directory
#     trash_dir = os.path.join(os.path.dirname(tmp_vrt_dir), "tmp_trash_vrt")
#     os.makedirs(trash_dir, exist_ok=True)
    
#     # Get metro France bounds
#     metro_bounds = get_metro_bounds_lambert93()
#     print(f"Metropolitan France bounds (Lambert 93):")
#     print(f"West: {metro_bounds['west']}, East: {metro_bounds['east']}")
#     print(f"South: {metro_bounds['south']}, North: {metro_bounds['north']}\n")
    
#     # Get all .tif and .vrt files
#     files = list(Path(tmp_vrt_dir).glob('*.[tv][ir][tf]'))
#     if not files:
#         print(f"No .tif or .vrt files found in {tmp_vrt_dir}")
#         return
    
#     print(f"Found {len(files)} files to check")
    
#     # Check each file
#     results = []
#     outside_bounds = []
#     errors = []
#     empty_files = []
    
#     for file in tqdm(files, desc="Checking files"):
#         result = check_file_bounds(str(file), metro_bounds)
        
#         if 'error' in result:
#             errors.append(result)
#             move_file_to_trash(str(file), trash_dir, "errors")
#             continue
            
#         results.append(result)
        
#         if not result['intersects']:
#             outside_bounds.append(result)
#             move_file_to_trash(str(file), trash_dir, "outside_bounds")
            
#         elif result['non_zero'] < 0.01:  # less than 1% non-zero values
#             empty_files.append(result)
#             move_file_to_trash(str(file), trash_dir, "empty_files")
    
#     # Print summary
#     print("\nSummary:")
#     print(f"Total files checked: {len(files)}")
#     print(f"Files outside bounds: {len(outside_bounds)}")
#     print(f"Files with errors: {len(errors)}")
#     print(f"Files with <1% non-zero values: {len(empty_files)}")
#     print(f"Remaining valid files: {len(files) - len(outside_bounds) - len(errors) - len(empty_files)}")
    
#     # Print details of moved files
#     if outside_bounds:
#         print("\nFiles outside metropolitan France bounds (moved to trash/outside_bounds):")
#         for file in outside_bounds:
#             print(f"\nFile: {os.path.basename(file['file'])}")
#             print(f"Bounds: {file['bounds']}")
#             print(f"Data range: {file['min_val']} to {file['max_val']}")
#             print(f"Non-zero values: {file['non_zero']:.2f}%")
    
#     if empty_files:
#         print("\nFiles with <1% non-zero values (moved to trash/empty_files):")
#         for file in empty_files:
#             print(f"\nFile: {os.path.basename(file['file'])}")
#             print(f"Bounds: {file['bounds']}")
#             print(f"Non-zero values: {file['non_zero']:.2f}%")
    
#     if errors:
#         print("\nFiles with errors (moved to trash/errors):")
#         for file in errors:
#             print(f"{os.path.basename(file['file'])}: {file['error']}")
    
#     # Create a log file
#     log_file = os.path.join(trash_dir, "moved_files_log.txt")
#     with open(log_file, 'w') as f:
#         f.write("Files moved to trash:\n\n")
        
#         f.write("\nOutside Bounds:\n")
#         for file in outside_bounds:
#             f.write(f"{os.path.basename(file['file'])}: {file['bounds']}\n")
            
#         f.write("\nEmpty Files:\n")
#         for file in empty_files:
#             f.write(f"{os.path.basename(file['file'])}: {file['non_zero']:.2f}% non-zero\n")
            
#         f.write("\nErrors:\n")
#         for file in errors:
#             f.write(f"{os.path.basename(file['file'])}: {file['error']}\n")

# if __name__ == "__main__":
#     # import argparse
    
#     # parser = argparse.ArgumentParser(description='Verify files within Metropolitan France bounds')
#     # parser.add_argument('tmp_vrt_dir', help='Path to tmp_vrt directory')
    
#     # args = parser.parse_args()

#     directory = '/Users/arthurcalvi/Data/species/HR-VPP/Results-2/tmp_vrt'
#     main(directory)


import os
import shutil
import rasterio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import traceback

def get_metro_bounds_lambert93():
    """
    Return the approximate bounds of metropolitan France in Lambert 93
    """
    west = -100000
    east = 1500000
    south = 4050000
    north = 8150000
    
    # Return in the same format as rasterio bounds
    return {
        'left': west,
        'right': east,
        'bottom': south,
        'top': north
    }

def check_intersection(bounds1, bounds2):
    """
    Check if two bounding boxes intersect
    """
    try:
        # Debug print
        print(f"\nChecking intersection between:")
        print(f"Bounds1: {bounds1}")
        print(f"Bounds2: {bounds2}")
        
        # Check for required keys
        required_keys = ['left', 'right', 'bottom', 'top']
        if not all(key in bounds1 for key in required_keys):
            raise KeyError(f"Missing keys in bounds1. Required keys: {required_keys}. Available keys: {bounds1.keys()}")
        if not all(key in bounds2 for key in required_keys):
            raise KeyError(f"Missing keys in bounds2. Required keys: {required_keys}. Available keys: {bounds2.keys()}")
        
        intersects = not (bounds1['right'] < bounds2['left'] or
                         bounds1['left'] > bounds2['right'] or
                         bounds1['top'] < bounds2['bottom'] or
                         bounds1['bottom'] > bounds2['top'])
        
        print(f"Intersection result: {intersects}")
        return intersects
        
    except Exception as e:
        print(f"\nError in check_intersection:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        raise

def check_file_bounds(file_path, metro_bounds):
    """
    Check if a file's bounds intersect with metropolitan France
    """
    try:
        print(f"\nProcessing file: {file_path}")
        
        with rasterio.open(file_path) as src:
            # Get file bounds
            bounds = src.bounds
            print(f"File bounds from rasterio: {bounds}")
            
            # Format bounds for intersection check
            try:
                file_bounds = {
                    'left': bounds.left,
                    'right': bounds.right,
                    'bottom': bounds.bottom,
                    'top': bounds.top
                }
                print(f"Formatted file bounds: {file_bounds}")
            except AttributeError as e:
                print(f"Error formatting bounds. Bounds object: {bounds}")
                print(f"Bounds type: {type(bounds)}")
                raise
            
            # Check intersection
            try:
                intersects = check_intersection(file_bounds, metro_bounds)
            except Exception as e:
                print(f"Error in intersection check: {str(e)}")
                raise
            
            # Calculate overlap area
            if intersects:
                overlap_width = min(file_bounds['right'], metro_bounds['right']) - max(file_bounds['left'], metro_bounds['left'])
                overlap_height = min(file_bounds['top'], metro_bounds['top']) - max(file_bounds['bottom'], metro_bounds['bottom'])
                overlap_area = max(0, overlap_width) * max(0, overlap_height)
                
                file_area = (file_bounds['right'] - file_bounds['left']) * (file_bounds['top'] - file_bounds['bottom'])
                overlap_percentage = (overlap_area / file_area) * 100 if file_area > 0 else 0
            else:
                overlap_percentage = 0
            
            # Read data to check values
            try:
                data = src.read(1)
                print(f"Successfully read data with shape: {data.shape}")
            except Exception as e:
                print(f"Error reading data: {str(e)}")
                raise
            
            result = {
                'file': file_path,
                'intersects': intersects,
                'overlap_percentage': overlap_percentage,
                'bounds': bounds,
                'crs': src.crs,
                'shape': data.shape,
                'min_val': float(np.min(data)),
                'max_val': float(np.max(data)),
                'mean_val': float(np.mean(data)),
                'non_zero': float(np.sum(data != 0)) / data.size * 100
            }
            
            print(f"Successfully created result dictionary")
            return result
            
    except Exception as e:
        error_info = {
            'file': file_path,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"\nError processing file {file_path}:")
        print(f"Error type: {error_info['error_type']}")
        print(f"Error message: {error_info['error_message']}")
        print("Stack trace:")
        print(error_info['traceback'])
        return error_info

def move_file_to_trash(file_path, trash_dir, reason):
    """
    Move a file to the trash directory with reason subdirectory
    """
    try:
        reason_dir = os.path.join(trash_dir, reason)
        os.makedirs(reason_dir, exist_ok=True)
        
        filename = os.path.basename(file_path)
        dest_path = os.path.join(reason_dir, filename)
        
        print(f"\nMoving file:")
        print(f"From: {file_path}")
        print(f"To: {dest_path}")
        
        shutil.move(file_path, dest_path)
        print(f"Successfully moved {filename} to {reason} folder")
        
    except Exception as e:
        print(f"\nError moving file {file_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()

def main(tmp_vrt_dir):
    """
    Main function to check all files in tmp_vrt directory
    """
    print(f"\nStarting process with directory: {tmp_vrt_dir}")
    
    # Create trash directory
    trash_dir = os.path.join(os.path.dirname(tmp_vrt_dir), "tmp_trash_vrt")
    os.makedirs(trash_dir, exist_ok=True)
    print(f"Created trash directory: {trash_dir}")
    
    # Get metro France bounds
    metro_bounds = get_metro_bounds_lambert93()
    print(f"\nMetropolitan France bounds (Lambert 93):")
    print(f"Left: {metro_bounds['left']}, Right: {metro_bounds['right']}")
    print(f"Bottom: {metro_bounds['bottom']}, Top: {metro_bounds['top']}")
    
    # Get all .tif and .vrt files
    files = list(Path(tmp_vrt_dir).glob('*.[tv][ir][tf]'))
    if not files:
        print(f"No .tif or .vrt files found in {tmp_vrt_dir}")
        return
    
    print(f"\nFound {len(files)} files to check")
    
    # Check each file
    results = []
    outside_bounds = []
    errors = []
    empty_files = []
    
    for file in tqdm(files, desc="Checking files"):
        print(f"\nProcessing: {file}")
        result = check_file_bounds(str(file), metro_bounds)
        
        if 'error_type' in result:
            print(f"File processed with error")
            errors.append(result)
            move_file_to_trash(str(file), trash_dir, "errors")
            continue
            
        print(f"File processed successfully")
        results.append(result)
        
        if not result['intersects']:
            outside_bounds.append(result)
            move_file_to_trash(str(file), trash_dir, "outside_bounds")
            
        elif result['non_zero'] < 0.001:
            empty_files.append(result)
            move_file_to_trash(str(file), trash_dir, "empty_files")
    
    # Create summary
    create_summary(results, outside_bounds, errors, empty_files, trash_dir)

def create_summary(results, outside_bounds, errors, empty_files, trash_dir):
    """Create detailed summary and log file"""
    print("\nSummary:")
    print(f"Total files processed: {len(results) + len(outside_bounds) + len(errors) + len(empty_files)}")
    print(f"Files outside bounds: {len(outside_bounds)}")
    print(f"Files with errors: {len(errors)}")
    print(f"Files with <1% non-zero values: {len(empty_files)}")
    print(f"Remaining valid files: {len(results) - len(empty_files)}")
    
    # Create detailed log file
    log_file = os.path.join(trash_dir, "processing_log.txt")
    with open(log_file, 'w') as f:
        f.write("Processing Log\n\n")
        
        f.write("Error Files:\n")
        for file in errors:
            f.write(f"\nFile: {os.path.basename(file['file'])}\n")
            f.write(f"Error type: {file['error_type']}\n")
            f.write(f"Error message: {file['error_message']}\n")
            f.write("Stack trace:\n")
            f.write(file['traceback'])
            f.write("\n")
        
        f.write("\nOutside Bounds Files:\n")
        for file in outside_bounds:
            f.write(f"\nFile: {os.path.basename(file['file'])}\n")
            f.write(f"Bounds: {file['bounds']}\n")
            f.write(f"CRS: {file['crs']}\n")
        
        f.write("\nEmpty Files:\n")
        for file in empty_files:
            f.write(f"\nFile: {os.path.basename(file['file'])}\n")
            f.write(f"Non-zero values: {file['non_zero']:.2f}%\n")
        
        f.write("\nValid Files:\n")
        for file in results:
            if file['intersects'] and file['non_zero'] >= 1:
                f.write(f"\nFile: {os.path.basename(file['file'])}\n")
                f.write(f"Bounds: {file['bounds']}\n")
                f.write(f"Non-zero values: {file['non_zero']:.2f}%\n")

    print(f"\nDetailed log written to: {log_file}")

if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description='Verify files within Metropolitan France bounds')
    # parser.add_argument('tmp_vrt_dir', help='Path to tmp_vrt directory')
    
    # args = parser.parse_args()
    
    # main(args.tmp_vrt_dir)
    directory = '/Users/arthurcalvi/Data/species/HR-VPP/Results-2/tmp_vrt'
    main(directory)