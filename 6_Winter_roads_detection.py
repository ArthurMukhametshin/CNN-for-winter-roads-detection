''' Предсказание изображений с их последующей полигонизацией и слиянием в единый шейп-файл'''

import numpy as np
from tensorflow.keras.utils import normalize
import os
import glob
import cv2 as cv
from tensorflow.keras.models import load_model
import gdal
import pandas as pd
import geopandas as gpd
import pyproj.datadir
import ogr
import fiona
from shapely.ops import cascaded_union
from shapely.geometry import shape, mapping

model = load_model('CNN.hdf5')
model.load_weights('CNN.hdf5')
model.summary()

directory = 'D:/Practise/Learning/Original_images/' # Путь к изображениям  для предсказания

# Получаем список путей изображений
image_paths = []
basenames = []
for directory_path in glob.glob(directory):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        image_paths.append(img_path)
        base_name = os.path.basename(img_path)
        basenames.append(base_name)

# Читаем изображения и нормализуем их
images = []
for img_path in image_paths:
    img_bgr = cv.imread(img_path, cv.IMREAD_COLOR|cv.IMREAD_ANYDEPTH)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    images.append(img_rgb)
images = np.array(images)
images = normalize(images, axis=1)

# Считываем данные о привязке изображений
projection_list = []
geotransform_list = []
metadata_list = []
for img_path in image_paths:
    dataset1 = gdal.Open(img_path)
    projection = dataset1.GetProjection()
    geotransform = dataset1.GetGeoTransform()
    metadata = dataset1.GetMetadata()
    projection_list.append(projection)
    geotransform_list.append(geotransform)
    metadata_list.append(metadata)

# Сохраняем предсказанные изображения
i = 0
saved_images = []
for img in images:
    img_ed = np.expand_dims(img, 0)
    prediction = (model.predict(img_ed))
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]
    save_path = 'D:/Practise/Learning/Predicted_images/{0}'.format(basenames[i])
    saved_images.append(save_path)
    cv.imwrite(save_path, predicted_img)
    i += 1

# Присваиваем сохраненным изображениям привязку и метаданные
i = 0
for img in saved_images:
    geotransform = geotransform_list[i]
    projection = projection_list[i]
    metadata = metadata_list[i]
    dataset2 = gdal.Open(img, gdal.GA_Update)
    dataset2.SetGeoTransform(geotransform)
    dataset2.SetProjection(projection)
    dataset2.SetMetadata(metadata)
    i += 1

# Полигонизируем предсказанные растры
gdal.UseExceptions()
shapefiles_pol = []
i = 0
for img in saved_images:
    image = gdal.Open(img)
    band = image.GetRasterBand(1)
    save_path = 'D:/Practise/Learning/Shapefiles_pol/{0}.shp'.format(basenames[i])
    shapefiles_pol.append(save_path)
    dst_layername = "POLYGONIZED_STUFF"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(save_path)
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None)
    gdal.Polygonize(band, band, dst_layer, -1, [], callback=None)
    i += 1

# Объединяем шейп-файлы в один шейп-файл
shapefiles_pol_un = []
pyproj.datadir.set_data_dir('C:/Users/ARTHUR MUKHAMETSHIN/anaconda3/envs/python_3_6/Library/share/proj') # Устанавливаем директорию с папкой с данными о проекциях
for shapefile in shapefiles_pol:
    gdf = gpd.read_file(shapefile)
    shapefiles_pol_un.append(gdf)
un_shp = gpd.GeoDataFrame(pd.concat(shapefiles_pol_un))
un_shp.to_file('D:/Practise/Learning/Shapefiles_pol_un/Roads_pol.shp')

# Сливаем полигоны с общей границей
src = 'D:/Practise/Learning/Shapefiles_pol_un/Roads_pol.shp'
with fiona.open(src, 'r') as ds_in:
    crs = ds_in.crs
    drv = ds_in.driver

    geoms = [shape(x["geometry"]) for x in ds_in]
    dissolved = cascaded_union(geoms)

schema = {
    "geometry": "Polygon",
    "properties": {"id": "int"}
}

with fiona.open('D:/Practise/Learning/Shapefiles_pol_un/Roads_pol_merged.shp', 'w', driver=drv, schema=schema, crs=crs) as ds_dst:
    for i,g in enumerate(dissolved):
        ds_dst.write({"geometry": mapping(g), "properties": {"id": i}})
