"""
FILE: tf_tm.py
AUTHOR: Daniel Philipov
DESCRIPTION: This file contains the code for the training of the model.
Old-Fashioned GAN coded in tensorflow only.
"""

import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
########################################################################################################################
#                                                       General                                                        #
########################################################################################################################

_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generate_and_save(model, epoch, test_input, figsize: tuple or int = (4, 4)):
    def max_factors(n):
        return [(i, n // i) for i in range(1, int(n ** 0.5) + 1) if n % i == 0][-1]
    if isinstance(figsize, int):
        figsize = max_factors(figsize)
    predictions = model(test_input, training=False)
    _fig = plt.figure(figsize=figsize)
    for i in range(predictions.shape[0]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(predictions[i, :, :] * 127.5 + 127.5)
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

########################################################################################################################
#                                                    Discriminator                                                     #
########################################################################################################################

def make_discriminator_model(shape: tuple, img_shape: tuple, dropout: float = 0.3, bias=False, T=tf.nn.leaky_relu):
    model = tf.keras.sequential()
    model.add(layers.Conv2D(shape[0], 4, strides=2, padding='same', input_shape=img_shape, use_bias=bias, activation=T))
    model.add(layers.Dropout(dropout))
    for p, s, q in zip(shape[:-1], shape[1:], shape[1:]):
        model.add(layers.Conv2D(q * 3, 4, strides=s//p, padding='same', use_bias=bias, activation=T))
        model.add(layers.Dropout(dropout))
        assert model.output_shape == (None, s, s, q * 3)
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = _cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = _cross_entropy(tf.zeros_like(disc_fake_output), disc_fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

########################################################################################################################
#                                                      Generator                                                       #
########################################################################################################################

def make_generator_model(shape: tuple, depth: tuple, input_shape: int, bias=False, T=tf.nn.leaky_relu):
    assert len(shape) == len(depth)
    model = tf.keras.Sequential()
    model.add(layers.Dense(shape[0] ** 2 * depth[0] * 3, use_bias=False, input_shape=input_shape, activation=T))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((shape[0], shape[0], depth[0] * 3)))
    for p, s, q in zip(shape[:-1], shape[1:], depth[1:]):
        model.add(layers.Conv2DTranspose(q * 3, 4, strides=s//p, padding='same', use_bias=bias, activation=T))
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(3, 4, strides=2, padding='same', use_bias=bias, activation=tf.nn.tanh))
    return model

def generator_loss(disc_fake_output):
    loss = _cross_entropy(tf.ones_like(disc_fake_output), disc_fake_output)
    return loss

########################################################################################################################
#                                                       Training                                                       #
########################################################################################################################

@tf.function
def train_step(images, info, generator, discriminator, gen_optimizer, disc_optimizer, noise_dim, batch_size):
    # generating noise from a normal distribution
    noise = tf.random.normal([batch_size, noise_dim])
    input_data = tf.concat([noise, info], axis=1)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate images from noise
        generated_images = generator(input_data, training=True)

        # Get the logits for the fake images from the discriminator
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate the loss
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    # Calculate the gradients for generator and discriminator
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Apply the gradients to the optimizer
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint,
          checkpoint_prefix, noise_dim, num_examples_to_generate, seed):
    for epoch in range(epochs):
        for info_batch, image_batch in dataset:
            train_step(image_batch, info_batch, generator, discriminator, generator_optimizer, discriminator_optimizer,
                        noise_dim, image_batch.shape[0])
        generate_and_save(generator, epoch + 1, seed, figsize=num_examples_to_generate)
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
    generate_and_save(generator, epochs, seed)
