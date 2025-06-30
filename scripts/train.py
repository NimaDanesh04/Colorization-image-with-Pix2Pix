import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import sys
sys.path.append('/home/rasa/pix2pix')
from models.generator import build_generator
from models.discriminator import build_discriminator
from dataloder import load_data
from tqdm import tqdm
import os


loss_object = BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(disc_generated_output, gen_output, target):
    target = tf.cast(target, tf.float32)
    gen_output = tf.cast(gen_output, tf.float32)

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)

    return total_gen_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        bar = tqdm(dataset, desc='Training', unit='step')

        for input_image, target in bar:
            gen_loss, disc_loss = train_step(input_image, target)

            # Update the bar description with current losses
            bar.set_postfix({
                'Gen Loss': f"{gen_loss.numpy():.4f}",
                'Disc Loss': f"{disc_loss.numpy():.4f}"
            })

generator = build_generator()
discriminator = build_discriminator()
train_dataset, test_dataset = load_data()

EPOCHS = 50

train(train_dataset, EPOCHS)

generator.save('models/generator.h5')

for i, (input_img, _) in enumerate(test_dataset.take(10)):
    img = tf.squeeze(input_img)  

    if img.shape.rank == 3 and img.shape[0] == 1:
        img = tf.transpose(img, [1, 2, 0]) 
    elif img.shape.rank == 2:
        img = tf.expand_dims(img, axis=-1) 
    img = tf.expand_dims(img, axis=0) 
    prediction = generator(img, training=False)
    final_predict = ((prediction[0].numpy() + 1) * 127.5).astype(np.uint8)
    tf.keras.preprocessing.image.save_img(os.path.join('/home/rasa/pix2pix/predictions', f"{i}.jpg"), final_predict)
    