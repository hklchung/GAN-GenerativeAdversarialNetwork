import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, LeakyReLU, AveragePooling2D, Embedding
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ZeroPadding2D, Concatenate, Add
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
from keras import backend, Input
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from sklearn.utils import shuffle
from sklearn.utils import shuffle
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from tensorflow import keras

#=======================Discriminator model func===============================
# Function to create discriminator
def discriminator(nodes_per_layer=32):
    input_x = Input(shape=(256, 256, 3))

    x = Conv2D(nodes_per_layer, (3, 3), strides=(2, 2))(input_x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(nodes_per_layer, (3, 3), strides=(2, 2))(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(nodes_per_layer, (3, 3), strides=(2, 2))(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(nodes_per_layer, (3, 3), strides=(2, 2))(x)
    x = LeakyReLU(0.2)(x)

    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    output_x = Activation('sigmoid')(x)
    return Model(input_x, output_x)

#=======================Generator model funcs==================================
# Function to create full residual blocks
def full_residual(nodes_per_layer, x):
  res_x = Conv2D(nodes_per_layer, (3, 3), padding='same')(x)
  res_x = Activation('relu')(res_x)
  res_x = Conv2D(nodes_per_layer, (3, 3), padding='same')(res_x)
  res_x = Activation('relu')(res_x)
  x = Add()([x, res_x])
  return(x)

# Function to create half residual blocks
def half_residual(nodes_per_layer, x_half):
  res_x = Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(x_half)
  res_x = Activation('relu')(res_x)
  res_x = Conv2D(nodes_per_layer * 2, (3, 3), padding='same')(res_x)
  res_x = Activation('relu')(res_x)
  x_half = Add()([x_half, res_x])
  return(x_half)

# Function to create quarter residual blocks
def quarter_residual(nodes_per_layer, x_quarter):
  res_x = Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(x_quarter)
  res_x = Activation('relu')(res_x)
  res_x = Conv2D(nodes_per_layer * 4, (3, 3), padding='same')(res_x)
  res_x = Activation('relu')(res_x)
  x_quarter = Add()([x_quarter, res_x])
  return(x_quarter)

# Function to create generator
def generator(nodes_per_layer=32):
    input_x = Input(shape=(256, 256, 3))

    x = Conv2D(nodes_per_layer, (3, 3), padding='same')(input_x)
    x = Activation('relu')(x)

    # 3 residual blocks for full
    x = full_residual(nodes_per_layer, x)
    x = full_residual(nodes_per_layer, x)
    x = full_residual(nodes_per_layer, x)

    # 6 residual blocks for half
    x_half = Conv2D(nodes_per_layer * 2, (2, 2), strides=(2, 2))(input_x)
    x_half = Activation('relu')(x_half)

    x_half = half_residual(nodes_per_layer, x_half)
    x_half = half_residual(nodes_per_layer, x_half)
    x_half = half_residual(nodes_per_layer, x_half)
    x_half = half_residual(nodes_per_layer, x_half)
    x_half = half_residual(nodes_per_layer, x_half)
    x_half = half_residual(nodes_per_layer, x_half)

    x_quarter = Conv2D(nodes_per_layer * 4, (2, 2), strides=(2, 2))(x_half)
    x_quarter = Activation('relu')(x_quarter)

    # 6 residual blocks for quarter
    x_quarter = quarter_residual(nodes_per_layer, x_quarter)
    x_quarter = quarter_residual(nodes_per_layer, x_quarter)
    x_quarter = quarter_residual(nodes_per_layer, x_quarter)
    x_quarter = quarter_residual(nodes_per_layer, x_quarter)
    x_quarter = quarter_residual(nodes_per_layer, x_quarter)
    x_quarter = quarter_residual(nodes_per_layer, x_quarter)

    x_quarter = Conv2DTranspose(nodes_per_layer * 2, (2, 2), strides=(2, 2))(x_quarter)
    x_quarter = Activation('relu')(x_quarter)

    x_half = Add()([x_quarter, x_half])

    x_half = Conv2DTranspose(nodes_per_layer, (2, 2), strides=(2, 2))(x_half)
    x_half = Activation('relu')(x_half)

    x = Add()([x, x_half])
    x = Conv2D(3, (3, 3), padding='same')(x)
    output_x = Activation('relu')(x)
    return Model(input_x, output_x)

#===========================Compile CycleGAN===================================
# Create generators for 2 way style transfer
gen_a2b = generator()
gen_b2a = generator()

# Create discriminators to validate styles
disc_a = discriminator()
disc_b = discriminator()

optimizer1 = Adam(learning_rate=0.0001)

disc_a.compile(optimizer=optimizer1, loss='binary_crossentropy', metrics=['acc'])
disc_b.compile(optimizer=optimizer1, loss='binary_crossentropy', metrics=['acc'])

disc_a.trainable = False
disc_b.trainable = False

# Create the style A to B GAN architecture
input_a = Input(shape=(256, 256, 3))
output_image_b = gen_a2b(input_a)
output_disc_b = disc_b(output_image_b)
output_gen_a = gen_b2a(output_image_b)
GAN_a2b = Model(input_a, [output_disc_b, output_gen_a, output_image_b])

# Create the style B to A GAN architecture
input_b = Input(shape=(256, 256, 3))
output_image_a = gen_b2a(input_b)
output_disc_a = disc_a(output_image_a)
output_gen_b = gen_a2b(output_image_a)
GAN_b2a = Model(input_b, [output_disc_a, output_gen_b, output_image_a])

optimizer2 = Adam(learning_rate=0.0005)

GAN_a2b.compile(optimizer=optimizer2, loss=['binary_crossentropy', 'mae', 'mae'], loss_weights=[0.1, 1, 0.01])
GAN_b2a.compile(optimizer=optimizer2, loss=['binary_crossentropy', 'mae', 'mae'], loss_weights=[0.1, 1, 0.01])

plot_model(gen_a2b, to_file='gen_a2b.png', show_shapes=True, show_layer_names=True)
plot_model(gen_b2a, to_file='gen_b2a.png', show_shapes=True, show_layer_names=True)
plot_model(disc_a, to_file='disc_a.png', show_shapes=True, show_layer_names=True)
plot_model(disc_b, to_file='disc_b.png', show_shapes=True, show_layer_names=True)
plot_model(GAN_a2b, to_file='GAN_a2b.png', show_shapes=True, show_layer_names=True)
plot_model(GAN_b2a, to_file='GAN_b2a.png', show_shapes=True, show_layer_names=True)

#==============================Plot image function=============================
def plot_output(step):
    filename = "GANmodel_%d" % step
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        if i % 4 == 0:
          image = cv2.imread('monet2photo/testA/' + random.choice(os.listdir('monet2photo/testA')))
          plt.imshow(image)
          plt.axis('off')
        elif i % 4 == 1:
          image = cv2.resize(image, (256, 256))
          image = np.expand_dims(image, axis = 0)
          image_translated = gen_a2b.predict(image)
          plt.imshow(image.reshape(256,256,3))
          plt.axis('off')
        elif i % 4 == 2:
          image = cv2.imread('monet2photo/testB/' + random.choice(os.listdir('monet2photo/testB')))
          plt.imshow(image)
          plt.axis('off')
        elif i % 4 == 3:
          image = cv2.resize(image, (256, 256))
          image = np.expand_dims(image, axis = 0)
          image_translated = gen_b2a.predict(image)
          plt.imshow(image.reshape(256,256,3))
          plt.axis('off')
          plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')

#=============================Train CycleGAN===================================
def train_gan(batch_size=128, epoch=100, save_interval=1):
    
    # simple data augmentation implicitly including resizing to 256,256
    feature_datagen_a = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
    feature_datagen_b = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
    
    # pipeline to flow images from a directory in batches
    feature_image_generator_a = feature_datagen_a.flow_from_directory('images_a', seed=1, class_mode=None, batch_size = batch_size)
    feature_image_generator_b = feature_datagen_b.flow_from_directory('images_b', seed=1, class_mode=None, batch_size = batch_size)
    
    # core training loop, loads batch of images, generators and discriminators on batch
    for i in tqdm(range(0, epoch)):    
      # load a batch of images of each class into memory
      images_a_batch = next(feature_image_generator_a)
      images_b_batch = next(feature_image_generator_b)
      target_a_batch = np.ones([len(images_a_batch),1])
      target_b_batch = np.ones([len(images_b_batch),1])
    
      # fit each generator
      GAN_a2b.train_on_batch(images_a_batch, [target_a_batch, images_a_batch, images_a_batch])
      GAN_b2a.train_on_batch(images_b_batch, [target_b_batch, images_b_batch, images_b_batch])
    
      # create a new set of false images to train discriminator
      images_b_batch_fake = gen_a2b.predict(images_a_batch, batch_size=1)
      images_a_batch_fake = gen_b2a.predict(images_b_batch, batch_size=1)
      target_a_batch_fake = np.zeros([len(images_a_batch_fake),1])
      target_b_batch_fake = np.zeros([len(images_b_batch_fake),1])
    
      # combine fake and real images by class
      images_a_batch_discriminator = np.concatenate((images_a_batch, images_a_batch_fake), axis=0)
      images_b_batch_discriminator = np.concatenate((images_b_batch, images_b_batch_fake), axis=0)
      target_a_batch_discriminator = np.concatenate((target_a_batch, target_a_batch_fake), axis=0)
      target_b_batch_discriminator = np.concatenate((target_b_batch, target_b_batch_fake), axis=0)
    
      # fit discriminator to determine real vs fake images in a class
      disc_a.train_on_batch(images_a_batch_discriminator, target_a_batch_discriminator)
      disc_b.train_on_batch(images_b_batch_discriminator, target_b_batch_discriminator)
    
      # create a second training set for the discriminators of all real images mixing the classes
      images_a_batch_discriminator = np.concatenate((images_a_batch, images_b_batch), axis=0)
      images_b_batch_discriminator = np.concatenate((images_b_batch, images_a_batch), axis=0)
      target_a_batch_discriminator = np.concatenate((target_a_batch, target_a_batch_fake), axis=0)
      target_b_batch_discriminator = np.concatenate((target_b_batch, target_b_batch_fake), axis=0)
    
      # train discriminators to determine real images of class a from real images of class b
      disc_a.train_on_batch(images_a_batch_discriminator, target_a_batch_discriminator)
      disc_b.train_on_batch(images_b_batch_discriminator, target_b_batch_discriminator)
      
      if save_interval>0 and (i+1)%save_interval==0:
          plot_output(i)