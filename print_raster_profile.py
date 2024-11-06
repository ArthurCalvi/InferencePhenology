import rasterio 
import os 
folder_dir = '/gpfsdswork/dataset/S2L3A_France2019'
filename = 's2_20190115.tif'

raster = rasterio.open(os.path.join(folder_dir, filename))
print(raster.profile)