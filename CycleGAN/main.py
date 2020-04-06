import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ZeroPadding2D
from keras.models import Sequential, load_model, Input
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
from keras import backend
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from sklearn.utils import shuffle
from PIL import Image, ImageOps
import tensorflow as tf

#============================Get images========================================
# Grab Monet art images from folder
images_a = []
for filename in tqdm(os.listdir('Image/monet2photo/trainA')):
    temp = np.array(img_to_array(load_img('Image/monet2photo/trainA/'+filename)), dtype=float)
    hor = 256 - temp.shape[0]
    ver = 256 - temp.shape[1]
    if hor%2 != 0:
        temp = np.pad(temp, ((hor//2 + 1, hor//2), (ver//2, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    elif ver%2 != 0:
        temp = np.pad(temp, ((hor//2, hor//2), (ver//2 + 1, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    else:
    # Pad resized images with zeros such that they are all 256x256x3
        temp = np.pad(temp, ((hor//2, hor//2), (ver//2, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    # Store images into a list
    images_a.append(np.array(temp, dtype=float))
    
# Grab real images from folder
images_b = []
for filename in tqdm(os.listdir('Image/monet2photo/trainB')):
    temp = np.array(img_to_array(load_img('Image/monet2photo/trainB/'+filename)), dtype=float)
    hor = 256 - temp.shape[0]
    ver = 256 - temp.shape[1]
    if hor%2 != 0:
        temp = np.pad(temp, ((hor//2 + 1, hor//2), (ver//2, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    elif ver%2 != 0:
        temp = np.pad(temp, ((hor//2, hor//2), (ver//2 + 1, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    else:
    # Pad resized images with zeros such that they are all 256x256x3
        temp = np.pad(temp, ((hor//2, hor//2), (ver//2, ver//2), (0, 0)),
              mode='constant', constant_values=0)
    # Store images into a list
    images_b.append(np.array(temp, dtype=float))
    
# Normalise RGB intensities, reshape and forced into array
X_a = [1.0/255*x for x in images_a]
X_a = [x.reshape(256, 256, 3) for x in X_a]
X_a = np.array(X_a)
del(images_a)

# Normalise RGB intensities, reshape and forced into array
X_b = [1.0/255*x for x in images_b]
X_b = [x.reshape(256, 256, 3) for x in X_b]
X_b = np.array(X_b)
del(images_b)

#======================Creat GAN unit model function===========================
def create_GUnit(X, depth = 64, dropout = 0.5):
    # Define architecture of the discriminator (police AI)
    D = Sequential()
    # First layer
    D.add(Conv2D(depth*1, 5, strides=2, input_shape=X.shape[1:],padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    # Second layer
    D.add(Conv2D(depth*2, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    # Third layer
    D.add(Conv2D(depth*4, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    # Forth layer
    D.add(Conv2D(depth*8, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    # Output layer
    D.add(Conv2D(1, 5, padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    
    D.add(Conv2DTranspose(depth*8, 5, strides=1, padding='same', kernel_initializer='glorot_normal'))
    D.add(BatchNormalization(momentum=0.8))
    D.add(LeakyReLU(alpha=0.2))
    
    D.add(Conv2DTranspose(depth*4, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(BatchNormalization(momentum=0.8))
    D.add(LeakyReLU(alpha=0.2))
    
    D.add(Conv2DTranspose(depth*2, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(BatchNormalization(momentum=0.8))
    D.add(LeakyReLU(alpha=0.2))
    
    D.add(Conv2DTranspose(depth*1, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(BatchNormalization(momentum=0.8))
    D.add(LeakyReLU(alpha=0.3))
    
    D.add(Conv2DTranspose(3, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(Activation('sigmoid'))
    
    return(D)
    
#======================Create discriminator model function=====================
def create_discriminator(X, depth = 64, dropout = 0.5):
    # Define architecture of the discriminator (police AI)
    D = Sequential()
    # First layer
    D.add(Conv2D(depth*1, 5, strides=2, input_shape=X.shape[1:],padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    # Second layer
    D.add(Conv2D(depth*2, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    # Third layer
    D.add(Conv2D(depth*4, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    # Forth layer
    D.add(Conv2D(depth*8, 5, strides=2, padding='same', kernel_initializer='glorot_normal'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))
    D.add(Flatten())
    # Output layer
    D.add(Dense(1, kernel_initializer='glorot_normal'))
    D.add(Activation('sigmoid'))
    
    return(D)

#=====================Create the GAN component models==========================
G_AB = create_GUnit(X_a)
D = create_discriminator(X_a)

optimizer1 = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
optimizer2 = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)

D.compile(loss='binary_crossentropy', optimizer=optimizer1,metrics=['accuracy'])
G_AB.compile(loss='binary_crossentropy', optimizer=optimizer1,metrics=['accuracy'])

# Define architecture of GAN
GAN = Sequential()
GAN.add(G_AB)  # Adding the generator
GAN.add(D)  # Adding the discriminator 
GAN.compile(loss='binary_crossentropy', optimizer=optimizer2, metrics=['accuracy'])



#temp = G_AB.predict(X_a[1].reshape(1, 256, 256, 3))
epoch = 10
batch_size = 32
for i in range(epoch):
    #=====================Train discriminator==============================
    # Randomly select n (batch_size) number of images from X
    images_real = X_a[np.random.randint(0,X_a.shape[0], size=batch_size), :, :, :]
    # Produce n number of fake images with generator
    images_fake = G_AB.predict(images_real)
    # Concat real and fake images
    x = np.concatenate((images_real, images_fake))
    # Create labels
    y = np.ones([2*batch_size, 1]) 
    y[batch_size:, :] = 0
    # Shuffle the real and fake images
    x,y = shuffle(x,y)
    # Make discriminator trainable
    D.trainable = True
    # Train discriminator on the sampled data
    d_loss = D.train_on_batch(x, y)
    
    #=========================Train GAN====================================
    # Create labels
    y = np.ones([batch_size, 1])
    # Generate n number of 100D noise vectors
    images_real = X_a[np.random.randint(0,X_a.shape[0], size=batch_size), :, :, :]
    # Produce n number of fake images with generator
    images_fake = G_AB.predict(images_real)
    # Freeze weights in discriminator
    D.trainable = False
    # Train GAN on the generated data
    gan_loss = GAN.train_on_batch(images_fake, y)
    # Print loss and accuracy values 
    log_msg = "epoch %d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
    log_msg = "%s  [GAN loss: %f, acc: %f]" % (log_msg, gan_loss[0], gan_loss[1])
    print(log_msg)