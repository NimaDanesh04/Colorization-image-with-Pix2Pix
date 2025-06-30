import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_discriminator():
    inp = Input(shape=(128, 128, 1), name='input_image')
    tar = Input(shape=(128, 128, 3), name='target_image')

    x = Concatenate()([inp, tar])  # (128, 128, 4)

    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512, kernel_size=4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(x)  # (16x16x1)

    return Model(inputs=[inp, tar], outputs=x, name='PatchGAN_Discriminator')
