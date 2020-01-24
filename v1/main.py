import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers


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


def make_discriminator_model():
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

    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # noise = seed
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return disc_loss, gen_loss


def train(epochs,start_epoch):
    for epoch in range(start_epoch, epochs):
        start = time.time()
        print("Starting epoch number %d" % (epoch + 1))
        for i in range(len(data_it)):
            start_batch = time.time()
            image_batch = data_it.next()
            disc_loss, gen_loss = train_step((image_batch / 127.5) - 1)
            print(f'epoch {epoch + 1}, batch [{i + 1}/{len(data_it)}] took {time.time() - start_batch}s')
            print("disc_loss: " + str(disc_loss))
            print("gen_loss: " + str(gen_loss))
            print('---------------------------')

        # Produce images for the GIF as we go

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        generate_and_save_images(generator, epoch + 1, seed)
        print('Saved Images :)')
    # Generate after the final epoch
    # display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(32, 16))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) * 0.5)
        plt.axis('off')
    try:
        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    except:
        pass
    plt.close(fig)


BATCH_SIZE = 32
EPOCHS = 1000
noise_dim = 100
num_examples_to_generate = 16
datagen = ImageDataGenerator()
data_it = datagen.flow_from_directory('../fishDataSets/', target_size=(140, 320), class_mode=None,
                                      batch_size=BATCH_SIZE)
if os.path.exists('./seed.npy'):
    seed = np.load('./seed.npy')
else:
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    np.save('./seed.npy', seed)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator = make_generator_model()
discriminator = make_discriminator_model()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# noise = tf.random.normal([100, noise_dim])
# generated = generator(noise, training=False)
# for i in range(noise.shape[0]):
#     img = (generated[i] + 0.5).numpy().clip(0, 1)
#     plt.imsave(f"./output_images/output{i}.png", img)

train(EPOCHS, 466)
# generate_and_save_images(generator,
#                          0000,
#                          seed)
