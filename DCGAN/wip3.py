import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ZeroPadding2D
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
import tensorflow as tf

#============================Get images========================================
images = []
# Grab images from folder
for filename in tqdm(os.listdir('Image/Train/Resized')):
    if np.random.normal(0, 1, 1) > 0.89:
        temp = np.array(img_to_array(load_img('Image/Train/Resized/'+filename)), dtype=float)
        hor = 64 - temp.shape[0]
        ver = 64 - temp.shape[1]
        if hor%2 != 0:
            temp = np.pad(temp, ((hor//2 + 1, hor//2), (ver//2, ver//2), (0, 0)),
                  mode='constant', constant_values=0)
        elif ver%2 != 0:
            temp = np.pad(temp, ((hor//2, hor//2), (ver//2 + 1, ver//2), (0, 0)),
                  mode='constant', constant_values=0)
        else:
        # Pad resized images with zeros such that they are all 32x32x3
            temp = np.pad(temp, ((hor//2, hor//2), (ver//2, ver//2), (0, 0)),
                  mode='constant', constant_values=0)
        # Store images into a list
        images.append(np.array(temp, dtype=float))

# Normalise RGB intensities, reshape and forced into array
X = [1.0/255*x for x in images]
X = [x.reshape(64, 64, 3) for x in X]
X = np.array(X)

del(images)
X = X[:10000]

#=========================Discriminator model==================================
noise = 32
def create_discriminator():
    disc_input = Input(shape=(X.shape[1], X.shape[2], X.shape[3]))
    
    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)
    
    optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    return discriminator

def create_generator():
    gen_input = Input(shape=(noise, ))
    
    x = Dense(128 * 8 * 8)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((8, 8, 128))(x)
    
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(3, 7, activation='tanh', padding='same')(x)
    
    generator = Model(gen_input, x)
    return generator

generator = create_generator()
discriminator = create_discriminator()
discriminator.trainable = False

gan_input = Input(shape=(noise, ))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')

#==========================Plot image function=================================
def plot_output(noise, step):
    filename = "GANmodel_%d.png" % step
    
    images = generator.predict(noise)

    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = image.reshape(images.shape[1], images.shape[2], images.shape[3])
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')
    
#==========================Plot loss function==================================
def plot_loss(d_performance, gan_performance, jump=100):
    plt.figure(figsize=(10, 10))
    plt.plot(d_performance[0::jump], label='discriminator')
    plt.plot(gan_performance[0::jump], label='GAN')
    plt.xlabel('epoch ({})'.format(jump))
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_over_epoch.png')
    plt.close('all')
    
#=========================Train GAN function===================================
def train_gan(X, model, batch_size, epoch, save_interval, noise_len=100):    
    d_losses = []
    gan_losses = []
    
    batch_per_epoch = int(round(X.shape[0]/batch_size))
    
    for i in range(epoch):
        start = 0
        for j in tqdm(range(batch_per_epoch)):
            latent_vectors = np.random.normal(size=(batch_size, noise))
            generated = generator.predict(latent_vectors)

            real = X[start:start + batch_size]
            combined_images = np.concatenate([generated, real])

            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            labels += .05 * np.random.random(labels.shape)

            d_loss = discriminator.train_on_batch(combined_images, labels)
            d_losses.append(d_loss)

            latent_vectors = np.random.normal(size=(batch_size, noise))
            misleading_targets = np.zeros((batch_size, 1))

            gan_loss = gan.train_on_batch(latent_vectors, misleading_targets)
            gan_losses.append(gan_loss)

            start += batch_size
            if start > X.shape[0] - batch_size:
                start = 0
        # Print loss and accuracy values 
        log_msg = "epoch %d: [D loss: %f]" % (i, d_loss)
        log_msg = "%s  [GAN loss: %f]" % (log_msg, gan_loss)
        print(log_msg)
        
        # Save ouputs
        if save_interval>0 and (i+1)%save_interval==0:
            noise_input = np.random.normal(0.0, 1.0, size=[16, noise_len])
            plot_output(noise=noise_input, step=(i+1))
    
    d_losses = [float(x) for x in d_losses]
    gan_losses = [float(x) for x in gan_losses]
    return(d_losses, gan_losses)

d_loss_ls, gan_loss_ls = train_gan(X=X, model=gan, batch_size=16, epoch=50, 
                                                         save_interval=1,
                                                         noise_len=32)