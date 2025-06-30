import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_generator():
    inputs = Input(shape=(128, 128, 1))

    d1 = Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    d1 = LeakyReLU()(d1)

    d2 = Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(d1)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU()(d2)

    d3 = Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(d2)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU()(d3)

    d4 = Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(d3)
    d4 = BatchNormalization()(d4)
    d4 = LeakyReLU()(d4)

    d5 = Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(d4)
    d5 = BatchNormalization()(d5)
    d5 = LeakyReLU()(d5)

    d6 = Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(d5)
    d6 = BatchNormalization()(d6)
    d6 = LeakyReLU()(d6)

    d7 = Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(d6)
    d7 = BatchNormalization()(d7)
    d7 = LeakyReLU()(d7)

    u1 = Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(d7)
    u1 = BatchNormalization()(u1)
    u1 = Dropout(0.5)(u1)
    u1 = ReLU()(u1)
    u1 = Concatenate()([u1, d6])

    u2 = Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(u1)
    u2 = BatchNormalization()(u2)
    u2 = Dropout(0.5)(u2)
    u2 = ReLU()(u2)
    u2 = Concatenate()([u2, d5])

    u3 = Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(u2)
    u3 = BatchNormalization()(u3)
    u3 = Dropout(0.5)(u3)
    u3 = ReLU()(u3)
    u3 = Concatenate()([u3, d4])

    u4 = Conv2DTranspose(256, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(u3)
    u4 = BatchNormalization()(u4)
    u4 = ReLU()(u4)
    u4 = Concatenate()([u4, d3])

    u5 = Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(u4)
    u5 = BatchNormalization()(u5)
    u5 = ReLU()(u5)
    u5 = Concatenate()([u5, d2])

    u6 = Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(u5)
    u6 = BatchNormalization()(u6)
    u6 = ReLU()(u6)
    u6 = Concatenate()([u6, d1])

    output = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(u6)

    return Model(inputs=inputs, outputs=output)
