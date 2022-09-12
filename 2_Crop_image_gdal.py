from osgeo import gdal

img_path = 'D:/Practise/TIF_Sentinel2_images/7/Mask7.tif' # Путь к изображению для разбиения

raster = gdal.Open(img_path)
gt = raster.GetGeoTransform()

xmin = gt[0]
ymax = gt[3]
res = gt[1]
xlen = res * raster.RasterXSize
ylen = res * raster.RasterYSize

div = 31
# xdiv =
# ydiv =

xsize = xlen/div    # xdiv
ysize = ylen/div    # ydiv

xsteps = [xmin + xsize * i for i in range(div+1)]   # xdiv
ysteps = [ymax - ysize * i for i in range(div+1)]   # ydiv

a = 1
for i in range(div):    # xdiv
    for j in range(div):    # ydiv
        xmin = xsteps[i]
        xmax = xsteps[i+1]
        ymax = ysteps[j]
        ymin = ysteps[j+1]

        gdal.Warp(str(a) + '.tif', raster, outputBounds = (xmin, ymin, xmax, ymax))
        a += 1
