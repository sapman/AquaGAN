import os

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(80 * 35 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())  # Normalize and scale inputs or activations. See remark bellow
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((35, 80, 256)))

    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(4, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator_features_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same', input_shape=[140, 320, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(4096))
    model.add(layers.LeakyReLU())
    return model


def make_discriminator_final_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1, input_shape=[4096]))
    return model


noise_dim = 100

generator_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_fe_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_final_optimizer = tf.keras.optimizers.Adam(2e-4)

generator = make_generator_model()
discriminator_fe = make_discriminator_features_model()
discriminator_final = make_discriminator_final_model()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_fe_optimizer=discriminator_fe_optimizer,
                                 discriminator_final_optimizer=discriminator_final_optimizer,
                                 generator=generator,
                                 discriminator_fe=discriminator_fe,
                                 discriminator_final=discriminator_final)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise = tf.random.normal([100, noise_dim])
images = (generator(noise) + 1) * 0.5
for i in range(images.shape[0]):
    plt.imsave(f'output_images/image{i}.png', images[i, :, :, :].numpy().clip(0, 1))