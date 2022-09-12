from Simple_multiclass_unet_model import multiclass_unet_model
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.metrics import MeanIoU
from tensorflow.keras.models import load_model

n_classes = 2                                      # Количество классов для сегментации
train_images = []                                  # Представляем информацию о тренировочных изображениях в виде списка

for directory_path in glob.glob('D:/Practise/Learning/Augmented_images/'):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR|cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        train_images.append(img)

train_images = np.array(train_images)               # Конвертируем список в массив для обучения

train_masks = []                                    # Представляем информацию о масках в виде списка
for directory_path in glob.glob('D:/Practise/Learning/Augmented_masks/'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE|cv2.IMREAD_ANYDEPTH)
        train_masks.append(mask)

train_masks = np.array(train_masks)                 # Конвертируем список в массив для обучения

labelencoder = LabelEncoder()                       # Начинаем кодировать метки, имея дело с многомерным массивом, поэтому необходимо его сгладить, кодировать и изменить форму
n, h, w = train_masks.shape                         # Определяем параметры массива
train_masks_reshaped = train_masks.reshape(-1, 1)   # Раскладываем массив в один столбец
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)     # Присваиваем метки для значений
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)  # Возвращаем массив в исходное состояние
np.unique(train_masks_encoded_original_shape)       # Находим уникальные значения массива

train_images = normalize(train_images, axis=1)      # Нормализуем массив вдоль оси 1
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)      # Увеличиваем количество осей массива масок до 3 // Скорее всего тут надо будет вместо 3 поставить 2

# Создаем подмножество данных для быстрого тестирования, отбираем 10% на тестирование и оставшиеся на обучение
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0)

weight_1 = np.where(y_train != 1, y_train, 50.09328783)
weight_2 = np.where(weight_1 !=0, weight_1, 0.505041)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multiclass_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH,
                                 IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], sample_weight_mode = 'temporal')
model.summary()

history = model.fit(X_train, y_train,
                    batch_size=8,
                    verbose=1,
                    epochs=6,
                    validation_data=(X_test, y_test),
                    sample_weight=weight_2,
                    shuffle=True)
model.save('CNN.hdf5')

_, acc = model.evaluate(X_test, y_test)     # Оцениваем модель
print("Точность = ", (acc * 100.0), "%")

loss = history.history['loss']                  # Строим график обучения и проверки точности и потерь для каждой эпохи
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model = load_model('CNN.hdf5')
model.load_weights('CNN.hdf5')
model.summary()

# IOU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:, :, :, 0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Тестирование

i = 1
while i < 100:
    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0])
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='binary')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='binary')
    plt.legend()
    plt.show()
    i += 1