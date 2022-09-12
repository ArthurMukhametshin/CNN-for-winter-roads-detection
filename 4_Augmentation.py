import tensorflow as tf
import cv2 as cv
import glob
import os
import numpy
original_images = []
for directory_path in glob.glob('D:/Practise/Learning/Images_with_roads/'):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        original_images.append(img_path)

original_masks = []
for directory_path in glob.glob('D:/Practise/Learning/Masks_with_roads/'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        original_masks.append(mask_path)

# 0 - ПЕРЕСОХРАНЕНИЕ
num = 1
for img in original_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    save_path = 'D:/Practise/Learning/Augmented_images/0_{0}.tif'.format(num)
    cv.imwrite(save_path, img_rgb)
    num += 1

num = 1
for mask in original_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/0_{0}.tif'.format(num)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 1 - ОТРАЖЕНИЕ
reflected_images = []
num = 1
for img in original_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    ref_img = tf.image.flip_left_right(img_rgb)
    ref_img_array = ref_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/1_{0}.tif'.format(num)
    reflected_images.append(save_path)
    cv.imwrite(save_path, ref_img_array)
    num += 1

reflected_masks = []
num = 1
for mask in original_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    mask_gs = numpy.expand_dims(mask_gs, axis=2)
    ref_mask = tf.image.flip_left_right(mask_gs)
    ref_mask_array = ref_mask.numpy()
    save_path = 'D:/Practise/Learning/Augmented_masks/1_{0}.tif'.format(num)
    reflected_masks.append(save_path)
    cv.imwrite(save_path, ref_mask_array)
    num += 1

# 2 - ПОВОРОТЫ НА 90
rotated_images_90 = []
num = 1
for img in original_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    rot_90 = tf.image.rot90(img_rgb)
    rot_90_array = rot_90.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/2_{0}.tif'.format(num)
    rotated_images_90.append(save_path)
    cv.imwrite(save_path, rot_90_array)
    num += 1

rotated_masks_90 = []
num = 1
for mask in original_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    mask_gs = numpy.expand_dims(mask_gs, axis=2)
    rot_90 = tf.image.rot90(mask_gs)
    rot_90_array = rot_90.numpy()
    save_path = 'D:/Practise/Learning/Augmented_masks/2_{0}.tif'.format(num)
    rotated_masks_90.append(save_path)
    cv.imwrite(save_path, rot_90_array)
    num += 1

# 3 - ПОВОРОТЫ НА 180
rotated_images_180 = []
num = 1
for img in rotated_images_90:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    rot_180 = tf.image.rot90(img_rgb)
    rot_180_array = rot_180.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/3_{0}.tif'.format(num)
    rotated_images_180.append(save_path)
    cv.imwrite(save_path, rot_180_array)
    num += 1

rotated_masks_180 = []
num = 1
for mask in rotated_masks_90:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    mask_gs = numpy.expand_dims(mask_gs, axis=2)
    rot_180 = tf.image.rot90(mask_gs)
    rot_180_array = rot_180.numpy()
    save_path = 'D:/Practise/Learning/Augmented_masks/3_{0}.tif'.format(num)
    rotated_masks_180.append(save_path)
    cv.imwrite(save_path, rot_180_array)
    num += 1

# 4 - ПОВОРОТЫ НА 270
rotated_images_270 = []
num = 1
for img in rotated_images_180:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    rot_270 = tf.image.rot90(img_rgb)
    rot_270_array = rot_270.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/4_{0}.tif'.format(num)
    rotated_images_270.append(save_path)
    cv.imwrite(save_path, rot_270_array)
    num += 1

rotated_masks_270 = []
num = 1
for mask in rotated_masks_180:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    mask_gs = numpy.expand_dims(mask_gs, axis=2)
    rot_270 = tf.image.rot90(mask_gs)
    rot_270_array = rot_270.numpy()
    save_path = 'D:/Practise/Learning/Augmented_masks/4_{0}.tif'.format(num)
    rotated_masks_270.append(save_path)
    cv.imwrite(save_path, rot_270_array)
    num += 1

# 5 - НАСЫЩЕННОСТЬ
saturated_images = []
num = 1
for img in original_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    sat_img = tf.image.adjust_saturation(img_rgb, 3)
    sat_img_array = sat_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/5_{0}.tif'.format(num)
    saturated_images.append(save_path)
    cv.imwrite(save_path, sat_img_array)
    num += 1

saturated_masks = []
num = 1
for mask in original_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/5_{0}.tif'.format(num)
    saturated_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 6 - ЯРКОСТЬ
bright_images = []
num = 1
for img in original_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    br_img = tf.image.adjust_brightness(img_rgb, 0.5)
    br_img_array = br_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/6_{0}.tif'.format(num)
    bright_images.append(save_path)
    cv.imwrite(save_path, br_img_array)
    num += 1

bright_masks = []
num = 1
for mask in original_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/6_{0}.tif'.format(num)
    bright_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 7 - ПОВОРОТЫ НА 90 - ЯРКОСТЬ
rotated_90_bright_images = []
num = 1
for img in rotated_images_90:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    br_img = tf.image.adjust_brightness(img_rgb, 0.5)
    br_img_array = br_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/7_{0}.tif'.format(num)
    rotated_90_bright_images.append(save_path)
    cv.imwrite(save_path, br_img_array)
    num += 1

rotated_90_bright_masks = []
num = 1
for mask in rotated_masks_90:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/7_{0}.tif'.format(num)
    rotated_90_bright_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 8 - ПОВОРОТЫ НА 90 - НАСЫЩЕННОСТЬ
rotated_90_saturated_images = []
num = 1
for img in rotated_images_90:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    sat_img = tf.image.adjust_saturation(img_rgb, 3)
    sat_img_array = sat_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/8_{0}.tif'.format(num)
    rotated_90_saturated_images.append(save_path)
    cv.imwrite(save_path, sat_img_array)
    num += 1

rotated_90_saturated_masks = []
num = 1
for mask in rotated_masks_90:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/8_{0}.tif'.format(num)
    rotated_90_saturated_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 9 - ПОВОРОТЫ НА 180 - ЯРКОСТЬ
rotated_180_bright_images = []
num = 1
for img in rotated_images_180:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    br_img = tf.image.adjust_brightness(img_rgb, 0.5)
    br_img_array = br_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/9_{0}.tif'.format(num)
    rotated_180_bright_images.append(save_path)
    cv.imwrite(save_path, br_img_array)
    num += 1

rotated_180_bright_masks = []
num = 1
for mask in rotated_masks_180:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/9_{0}.tif'.format(num)
    rotated_180_bright_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 10 - ПОВОРОТЫ НА 180 - НАСЫЩЕННОСТЬ
rotated_180_saturated_images = []
num = 1
for img in rotated_images_180:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    sat_img = tf.image.adjust_saturation(img_rgb, 3)
    sat_img_array = sat_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/10_{0}.tif'.format(num)
    rotated_180_saturated_images.append(save_path)
    cv.imwrite(save_path, sat_img_array)
    num += 1

rotated_180_saturated_masks = []
num = 1
for mask in rotated_masks_180:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/10_{0}.tif'.format(num)
    rotated_180_saturated_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 11 - ПОВОРОТЫ НА 270 - ЯРКОСТЬ
rotated_270_bright_images = []
num = 1
for img in rotated_images_270:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    br_img = tf.image.adjust_brightness(img_rgb, 0.5)
    br_img_array = br_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/11_{0}.tif'.format(num)
    rotated_270_bright_images.append(save_path)
    cv.imwrite(save_path, br_img_array)
    num += 1

rotated_270_bright_masks = []
num = 1
for mask in rotated_masks_270:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/11_{0}.tif'.format(num)
    rotated_270_bright_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 12 - ПОВОРОТЫ НА 270 - НАСЫЩЕННОСТЬ
rotated_270_saturated_images = []
num = 1
for img in rotated_images_270:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    sat_img = tf.image.adjust_saturation(img_rgb, 3)
    sat_img_array = sat_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/12_{0}.tif'.format(num)
    rotated_270_saturated_images.append(save_path)
    cv.imwrite(save_path, sat_img_array)
    num += 1

rotated_270_saturated_masks = []
num = 1
for mask in rotated_masks_270:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/12_{0}.tif'.format(num)
    rotated_270_saturated_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 13 - ОТРАЖЕНИЕ - ПОВОРОТ НА 90
reflected_rotated_90_images = []
num = 1
for img in reflected_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    rot_90 = tf.image.rot90(img_rgb)
    rot_90_array = rot_90.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/13_{0}.tif'.format(num)
    reflected_rotated_90_images.append(save_path)
    cv.imwrite(save_path, rot_90_array)
    num += 1

reflected_rotated_90_masks = []
num = 1
for mask in reflected_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    mask_gs = numpy.expand_dims(mask_gs, axis=2)
    rot_90 = tf.image.rot90(mask_gs)
    rot_90_array = rot_90.numpy()
    save_path = 'D:/Practise/Learning/Augmented_masks/13_{0}.tif'.format(num)
    reflected_rotated_90_masks.append(save_path)
    cv.imwrite(save_path, rot_90_array)
    num += 1

# 14 - ОТРАЖЕНИЕ - ПОВОРОТ НА 180
reflected_rotated_180_images = []
num = 1
for img in reflected_rotated_90_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    rot_180 = tf.image.rot90(img_rgb)
    rot_180_array = rot_180.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/14_{0}.tif'.format(num)
    reflected_rotated_180_images.append(save_path)
    cv.imwrite(save_path, rot_180_array)
    num += 1

reflected_rotated_180_masks = []
num = 1
for mask in reflected_rotated_90_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    mask_gs = numpy.expand_dims(mask_gs, axis=2)
    rot_180 = tf.image.rot90(mask_gs)
    rot_180_array = rot_180.numpy()
    save_path = 'D:/Practise/Learning/Augmented_masks/14_{0}.tif'.format(num)
    reflected_rotated_180_masks.append(save_path)
    cv.imwrite(save_path, rot_180_array)
    num += 1

# 15 - ОТРАЖЕНИЕ - ПОВОРОТ НА 270
reflected_rotated_270_images = []
num = 1
for img in reflected_rotated_180_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    rot_270 = tf.image.rot90(img_rgb)
    rot_270_array = rot_270.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/15_{0}.tif'.format(num)
    reflected_rotated_270_images.append(save_path)
    cv.imwrite(save_path, rot_270_array)
    num += 1

reflected_rotated_270_masks = []
num = 1
for mask in reflected_rotated_180_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    mask_gs = numpy.expand_dims(mask_gs, axis=2)
    rot_270 = tf.image.rot90(mask_gs)
    rot_270_array = rot_270.numpy()
    save_path = 'D:/Practise/Learning/Augmented_masks/15_{0}.tif'.format(num)
    reflected_rotated_270_masks.append(save_path)
    cv.imwrite(save_path, rot_270_array)
    num += 1

# 16 - ОТРАЖЕНИЕ - ЯРКОСТЬ
reflected_bright_images = []
num = 1
for img in reflected_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    br_img = tf.image.adjust_brightness(img_rgb, 0.5)
    br_img_array = br_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/16_{0}.tif'.format(num)
    bright_images.append(save_path)
    cv.imwrite(save_path, br_img_array)
    num += 1

reflected_bright_masks = []
num = 1
for mask in reflected_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/16_{0}.tif'.format(num)
    reflected_bright_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 17 - ОТРАЖЕНИЕ - НАСЫЩЕННОСТЬ
reflected_saturated_images = []
num = 1
for img in reflected_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    sat_img = tf.image.adjust_saturation(img_rgb, 3)
    sat_img_array = sat_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/17_{0}.tif'.format(num)
    saturated_images.append(save_path)
    cv.imwrite(save_path, sat_img_array)
    num += 1

reflected_saturated_masks = []
num = 1
for mask in original_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/17_{0}.tif'.format(num)
    saturated_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 18 - ОТРАЖЕНИЕ - ПОВОРОТ НА 90 - ЯРКОСТЬ
reflected_rotated_90_bright_images = []
num = 1
for img in reflected_rotated_90_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    br_img = tf.image.adjust_brightness(img_rgb, 0.5)
    br_img_array = br_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/18_{0}.tif'.format(num)
    reflected_rotated_90_bright_images.append(save_path)
    cv.imwrite(save_path, br_img_array)
    num += 1

reflected_rotated_90_bright_masks = []
num = 1
for mask in reflected_rotated_90_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/18_{0}.tif'.format(num)
    reflected_rotated_90_bright_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 19 - ОТРАЖЕНИЕ - ПОВОРОТ НА 90 - НАСЫЩЕННОСТЬ
reflected_rotated_90_saturated_images = []
num = 1
for img in reflected_rotated_90_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    sat_img = tf.image.adjust_saturation(img_rgb, 3)
    sat_img_array = sat_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/19_{0}.tif'.format(num)
    reflected_rotated_90_saturated_images.append(save_path)
    cv.imwrite(save_path, sat_img_array)
    num += 1

reflected_rotated_90_saturated_masks = []
num = 1
for mask in reflected_rotated_90_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/19_{0}.tif'.format(num)
    reflected_rotated_90_saturated_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 20 - ОТРАЖЕНИЕ - ПОВОРОТ НА 180 - ЯРКОСТЬ
reflected_rotated_180_bright_images = []
num = 1
for img in reflected_rotated_180_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    br_img = tf.image.adjust_brightness(img_rgb, 0.5)
    br_img_array = br_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/20_{0}.tif'.format(num)
    reflected_rotated_180_bright_images.append(save_path)
    cv.imwrite(save_path, br_img_array)
    num += 1

reflected_rotated_180_bright_masks = []
num = 1
for mask in reflected_rotated_180_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/20_{0}.tif'.format(num)
    reflected_rotated_180_bright_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 21 - ОТРАЖЕНИЕ - ПОВОРОТ НА 180 - НАСЫЩЕННОСТЬ
reflected_rotated_180_saturated_images = []
num = 1
for img in reflected_rotated_180_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    sat_img = tf.image.adjust_saturation(img_rgb, 3)
    sat_img_array = sat_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/21_{0}.tif'.format(num)
    reflected_rotated_180_saturated_images.append(save_path)
    cv.imwrite(save_path, sat_img_array)
    num += 1

reflected_rotated_180_saturated_masks = []
num = 1
for mask in reflected_rotated_180_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/21_{0}.tif'.format(num)
    reflected_rotated_180_saturated_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 22 - ОТРАЖЕНИЕ - ПОВОРОТ НА 270 - ЯРКОСТЬ
reflected_rotated_270_bright_images = []
num = 1
for img in reflected_rotated_270_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    br_img = tf.image.adjust_brightness(img_rgb, 0.5)
    br_img_array = br_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/22_{0}.tif'.format(num)
    reflected_rotated_270_bright_images.append(save_path)
    cv.imwrite(save_path, br_img_array)
    num += 1

reflected_rotated_270_bright_masks = []
num = 1
for mask in reflected_rotated_270_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/22_{0}.tif'.format(num)
    reflected_rotated_270_bright_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1

# 23 - ОТРАЖЕНИЕ - ПОВОРОТ НА 270 - НАСЫЩЕННОСТЬ
reflected_rotated_270_saturated_images = []
num = 1
for img in reflected_rotated_270_images:
    img_bgr = cv.imread(img, cv.IMREAD_ANYDEPTH | cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    sat_img = tf.image.adjust_saturation(img_rgb, 3)
    sat_img_array = sat_img.numpy()
    save_path = 'D:/Practise/Learning/Augmented_images/23_{0}.tif'.format(num)
    reflected_rotated_270_saturated_images.append(save_path)
    cv.imwrite(save_path, sat_img_array)
    num += 1

reflected_rotated_270_saturated_masks = []
num = 1
for mask in reflected_rotated_270_masks:
    mask_gs = cv.imread(mask, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    save_path = 'D:/Practise/Learning/Augmented_masks/23_{0}.tif'.format(num)
    reflected_rotated_270_saturated_masks.append(save_path)
    cv.imwrite(save_path, mask_gs)
    num += 1
