import tensorflow as tf
import numpy as np
import os
import cv2 as cv
from sklearn.model_selection import train_test_split



def load_data(test_size=0.15):
    gray_path = sorted([os.path.join('/home/rasa/pix2pix/data/gray', path)
                    for path in os.listdir('/home/rasa/pix2pix/data/gray')
                    if path.endswith('.jpg')])

    color_path = sorted([os.path.join('/home/rasa/pix2pix/data/color', path)
                    for path in os.listdir('/home/rasa/pix2pix/data/color')
                    if path.endswith('.jpg')])
    print(gray_path)

    color_images = np.zeros((len(color_path)+5, 128, 128, 3))
    gray_images = np.zeros((len(gray_path)+5, 128, 128))
    for i in range(len(gray_path)):
        gray_img = cv.imread(gray_path[i], 0)
        color_img = cv.imread(color_path[i])
        gray_img = cv.resize(gray_img, (128, 128))
        color_img = cv.resize(color_img, (128, 128))
        color_images[i] = color_img
        gray_images[i] = gray_img

    x_train, x_test, y_train, y_test = train_test_split(gray_images, color_images, test_size=test_size)
    x_train = x_train / 127.5 -1
    x_test = x_test / 127.5 - 1
    y_train = y_train / 127.5 - 1
    y_test = y_test / 127.5 -1

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train[..., np.newaxis], y_train)) \
    .shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test[..., np.newaxis], y_test)) \
    .shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

    return (train_dataset, test_dataset)

