import cv2 as cv
import glob
import os
import numpy as np

rd_masks = [] # список с масками, на которых присутствуют дороги
masks = [] # список с масками общий
for directory_path in glob.glob('D:/Practise/Learning/Original_masks/'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        masks.append(mask_path)
for mask in masks:
    m = cv.imread(mask, cv.IMREAD_GRAYSCALE|cv.IMREAD_ANYDEPTH)
    uv = np.unique(m)
    if len(uv) == 2:
        name = os.path.basename(mask)
        save_path = 'D:/Practise/Learning/Masks_with_roads/{0}.tif'.format(name)
        cv.imwrite(save_path, m)
        rd_masks.append(save_path)


images = [] # список с изображениями общий
for directory_path in glob.glob('D:/Practise/Learning/Original_images/'):
    for image_path in glob.glob(os.path.join(directory_path, "*.tif")):
        images.append(image_path)
for image in images:
    for mask in rd_masks:
        img_name = os.path.basename(image)
        mask_name = os.path.basename(mask)
        if img_name == mask_name:
            rd_image = cv.imread(image, cv.IMREAD_COLOR|cv.IMREAD_ANYDEPTH)
            save_path = 'D:/Practise/Learning/Images_with_roads/{0}.tif'.format(img_name)
            cv.imwrite(save_path, rd_image)
