"""
Copyright (c) 2020, Heung Kit Leslie Chung
All rights reserved.
Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

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

#===========================Resize images======================================
img_path = "../Image/Train/tenk_celebs/100k"
out_path = "../Image/Train/Resized2"
for filename in tqdm(os.listdir(img_path)):
    temp = Image.open(img_path + '/' + filename)
    size = 64, 64
    temp.thumbnail(size, Image.ANTIALIAS)
    temp.save(out_path + '/' + filename, "JPEG")

#============================Get images========================================
images = []
# Grab images from folder
for filename in tqdm(os.listdir(out_path)):
    if np.random.normal(0, 1, 1) > 0.89:
        temp = np.array(img_to_array(load_img(out_path + '/' + filename)), dtype=float)
        hor = 64 - temp.shape[0]
        ver = 64 - temp.shape[1]
        if hor%2 != 0:
            temp = np.pad(temp, ((hor//2 + 1, hor//2), (ver//2, ver//2), (0, 0)),
                  mode='constant', constant_values=0)
        elif ver%2 != 0:
            temp = np.pad(temp, ((hor//2, hor//2), (ver//2 + 1, ver//2), (0, 0)),
                  mode='constant', constant_values=0)
        else:
        # Pad resized images with zeros such that they are all 64x64x3
            temp = np.pad(temp, ((hor//2, hor//2), (ver//2, ver//2), (0, 0)),
                  mode='constant', constant_values=0)
        # Store images into a list
        images.append(np.array(temp, dtype=float))

# Normalise RGB intensities, reshape and forced into array
X = [1.0/255*x for x in images]
X = [x.reshape(64, 64, 3) for x in X]
X = np.array(X)

# We need only 10k images to train
del(images)
X = X[:10000]

#=========================Discriminator model==================================
# Define architecture of the discriminator (police AI)
noise = 32
depth = 256

D = Sequential()
# Input + First layer
D.add(Conv2D(depth, 3, strides=1, input_shape=X.shape[1:]))
D.add(LeakyReLU())
# Second layer
D.add(Conv2D(depth, 4, strides=2))
D.add(LeakyReLU())
# Third layer
D.add(Conv2D(depth, 4, strides=2))
D.add(LeakyReLU())
# Forth layer
D.add(Conv2D(depth, 4, strides=2))
D.add(LeakyReLU())
# Fifth layer
D.add(Conv2D(depth, 4, strides=2))
D.add(LeakyReLU())
# Output
D.add(Flatten())
D.add(Dropout(0.4))
D.add(Dense(1))
D.add(Activation('sigmoid'))

# Print out architecture of the discriminator
D.summary()
# Save model architecture as .PNG
plot_model(D, to_file='discriminator.png', show_shapes=True, show_layer_names=True)

#==========================Generator model=====================================
# Define architecture of the generator (fraudster AI)
depth = 128
dim = 8
noise_vec = 32

G = Sequential()
# Input + First layer
G.add(Dense(dim*dim*depth, input_dim=noise_vec))
G.add(LeakyReLU())
G.add(Reshape((dim, dim, depth)))
# Second layer
G.add(Conv2D(depth*2, 5, padding = 'same'))
G.add(LeakyReLU())
# Third layer
G.add(Conv2DTranspose(depth*2, 4, strides=2, padding = 'same'))
G.add(LeakyReLU())
# Third layer
G.add(Conv2DTranspose(depth*2, 4, strides=2, padding = 'same'))
G.add(LeakyReLU())
# Third layer
G.add(Conv2DTranspose(depth*2, 4, strides=2, padding = 'same'))
G.add(LeakyReLU())
# Third layer
G.add(Conv2DTranspose(depth*4, 5, strides=1, padding = 'same'))
G.add(LeakyReLU())
# Third layer
G.add(Conv2DTranspose(depth*4, 5, strides=1, padding = 'same'))
G.add(LeakyReLU())
# Output
G.add(Conv2DTranspose(3, 7, strides=1, activation = 'tanh', padding = 'same'))

# Print out architecture of the generator
G.summary()
# Save model architecture as .PNG
plot_model(G, to_file='generator.png', show_shapes=True, show_layer_names=True)

#================================DCGAN=========================================
# Define optimisers
optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
D.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
D.trainable = False

# Define architecture of GAN
GAN = Sequential()
GAN.add(G)  # Adding the generator
GAN.add(D)  # Adding the discriminator 

# Compile DCGAN
GAN.compile(loss='binary_crossentropy', optimizer=optimizer)

# Print out architecture of GAN
GAN.summary()
# Save model architecture as .PNG
plot_model(GAN, to_file='DCGAN.png', show_shapes=True, show_layer_names=True)
plot_model(GAN, to_file='DCGAN_expand.png', expand_nested=True, show_shapes=True, show_layer_names=True)

#==========================Plot image function=================================
def plot_output(noise, step):
    filename = "GANmodel_%d.png" % step
    
    images = G.predict(noise)

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
    plt.xlabel('iteration (Skipping every {}its)'.format(jump))
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_over_epoch.png')
    plt.close('all')
    
#=========================Train GAN function===================================
def train_gan(X, model, batch_size, epoch, save_interval, noise_len=32):    
    d_losses = []
    gan_losses = []
    
    batch_per_epoch = int(round(X.shape[0]/batch_size))
    
    for i in range(epoch):
        start = 0
        for j in tqdm(range(batch_per_epoch)):
            #=====================Train discriminator==========================
            noise_vec = np.random.normal(size=(batch_size, noise))
            images_fake = G.predict(noise_vec)

            images_real = X[start:start + batch_size]
            x = np.concatenate([images_fake, images_real])

            y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            y += .05 * np.random.random(y.shape)

            d_loss = D.train_on_batch(x, y)
            d_losses.append(d_loss[0])
            
            #=========================Train GAN================================
            noise_vec = np.random.normal(size=(batch_size, noise))
            y = np.zeros((batch_size, 1))

            gan_loss = GAN.train_on_batch(noise_vec, y)
            gan_losses.append(gan_loss)

            start += batch_size
            if start > X.shape[0] - batch_size:
                start = 0
        # Print loss and accuracy values 
        log_msg = "epoch %d: [D loss: %f]" % (i, d_loss[0])
        log_msg = "%s  [GAN loss: %f]" % (log_msg, gan_loss)
        print(log_msg)
        
        # Save ouputs
        if save_interval>0 and (i+1)%save_interval==0:
            noise_input = np.random.normal(0.0, 1.0, size=[16, noise_len])
            plot_output(noise=noise_input, step=(i+1))
    
    d_losses = [float(x) for x in d_losses]
    gan_losses = [float(x) for x in gan_losses]
    return(d_losses, gan_losses)

#=================================Train GAN====================================
d_loss_ls, gan_loss_ls = train_gan(X=X, model=GAN, batch_size=16, epoch=50, 
                                                         save_interval=1,
                                                         noise_len=32)

plot_loss(d_loss_ls, gan_loss_ls)

#================================Save model====================================
model_json = GAN.to_json()
with open("GAN_model.json", "w") as json_file:
    json_file.write(model_json)
GAN.save_weights("GAN_model.h5", overwrite=True)

#================================Result GIF====================================
import imageio
result_pwd = 'Result/Model11'
output_pwd = os.path.abspath(os.getcwd())
images = []
for filename in tqdm(os.listdir(result_pwd)):
    images.append(imageio.imread(result_pwd + '/' + filename))
imageio.mimsave(output_pwd + '/' + result_pwd + '/' + 'GAN.gif', images)

#=======================Manipulating input vector==============================
features = 32
preset = [-2, -1, -.5, -.2, -.1, 0, .1, .2, .5, 1, 2]

plt.figure(1, figsize=(2 * len(preset), features * 2))
i = 0
for feature in range(features):
    for value in preset:
        plt.subplot(features, len(preset), i+1)
        latent_vector = np.zeros((1, 32))
        latent_vector[0, feature] = value
        img = generator.predict(latent_vector)[0]
        plt.imshow(img)
        plt.savefig('controlled_shifts.png')
        plt.axis('off')
        i += 1
plt.show()