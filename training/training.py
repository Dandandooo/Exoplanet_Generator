import tensorflow as tf, numpy as np, pandas as pd, pickle as pkl
import tensorflow.keras.layers as layers
from datetime import date

# Set PARAMETERS
BUFFER_SIZE = 64
BATCH_SIZE = 32
IMG_SIZE = 256
EPOCHS = 1000
noise_dim = 10
num_examples_to_generate = 9
generator_lr = .001
discriminator_lr = .0001

############################################################################################################
# Define the Data
############################################################################################################

planets = pkl.load(open('planets.pkl', 'rb'))
images_arrs = np.load('training_images.npy')
image_info = np.load('training_info.npy')
labels = np.load('training_labels.npy')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    rotation_range=15,
    width_shift_range=16,
    height_shift_range=16,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(images_arrs, augment=True)

############################################################################################################
# Define the Generator
############################################################################################################


