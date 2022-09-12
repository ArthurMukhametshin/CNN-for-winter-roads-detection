from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import numpy as np
import rasterio

raster_path_1 = 'D:/Practise/TIF_Sentinel2_images/7/RGB7_new.tif'
shapefile_path_1 = 'C:/Практика/Database/Roads7_buffer.shp'

with rasterio.open(raster_path_1, 'r') as src:
    raster_img = src.read()
    raster_meta = src.meta

def polygon_generation(polygon, transform):                             # Генерируем полигоны
    poly_pts = []
    poly = cascaded_union(polygon)
    for k in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(k))                          # Конвертируем полигоны в изображение

    new_poly = Polygon(poly_pts)                                        # Генерируем полигональные объекты
    return new_poly

shape = gpd.read_file(shapefile_path_1)

poly_shp = []                                                           # Генерируем бинарную маску
im_size = (src.meta['height'], src.meta['width'])
for num, row in shape.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        poly = polygon_generation(row['geometry'], src.meta['transform'])
        poly_shp.append(poly)
    else:
        for p in row['geometry']:
            poly = polygon_generation(p, src.meta['transform'])
            poly_shp.append(poly)

mask = rasterize(shapes=poly_shp, out_shape=im_size)

mask = mask.astype('uint16')

save_path = 'D:/Practise/TIF_Sentinel2_images/7/Mask7.tif'
bin_mask_meta = src.meta.copy()
bin_mask_meta.update({'count': 1})
with rasterio.open(save_path, 'w', **bin_mask_meta) as dst:
        dst.write(mask * 1, 1)
