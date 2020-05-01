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
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, LeakyReLU, AveragePooling2D, Embedding
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ZeroPadding2D, Concatenate
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

#===========================Get resized images=================================
folder_path = '../Image/Train/CelebA/img_align_celeba/img_align_celeba/'
len(os.listdir(folder_path))

total = 10000

width = 178
height = 208
diff = (height - width) // 2

new_width = 64
new_height = 64

crop_rect = (0, diff, width, height - diff)

X = []
for filename in tqdm(sorted(os.listdir(folder_path))[:total]):
    pic = Image.open(folder_path + filename).crop(crop_rect)
    pic.thumbnail((new_width, new_height), Image.ANTIALIAS)
    X.append(np.uint8(pic))

X = np.array(X) / 255
X.shape

#===========================Get attribute labels===============================
Y = pd.read_csv('../Image/Train/CelebA/list_attr_celeba.csv')
Y = Y[['Black_Hair','Blond_Hair','Eyeglasses','Male','Smiling']]
Y = np.array(Y[:total])
Y = np.array([(x+1)/2.0 for x in Y])

#=========================Discriminator model==================================
# Define architecture of the discriminator (police AI)
class_input = Input(shape=Y.shape[1:])
embedding1 = Embedding(5, 25)(class_input)
dense1 = Dense(64*64)(embedding1)
reshape1 = Reshape((64, 64, 5))(dense1)
dense2 = Dense(3)(reshape1)

image_input = Input(shape=X.shape[1:])
concatenate1 = Concatenate()([image_input, dense2])

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

disc_out = D(concatenate1)

disc_model = Model([class_input, image_input], disc_out)

# Save model architecture as .PNG
plot_model(disc_model, to_file='discriminator.png', expand_nested=True, show_shapes=True, show_layer_names=True)

#==========================Generator model=====================================
# Define architecture of the generator (fraudster AI)
# Note: with each layer, the image gets larger but with reduced depth
depth = 128
dim = 8
noise_vec = 32 + 5

gen_model = Sequential()
# Input + First layer
gen_model.add(Dense(dim*dim*depth, input_dim=noise_vec))
gen_model.add(LeakyReLU())
gen_model.add(Reshape((dim, dim, depth)))
# Second layer
gen_model.add(Conv2D(depth*2, 5, padding = 'same'))
gen_model.add(LeakyReLU())
# Third layer
gen_model.add(Conv2DTranspose(depth*2, 4, strides=2, padding = 'same'))
gen_model.add(LeakyReLU())
# Forth layer
gen_model.add(Conv2DTranspose(depth*2, 4, strides=2, padding = 'same'))
gen_model.add(LeakyReLU())
# Fifth layer
gen_model.add(Conv2DTranspose(depth*2, 4, strides=2, padding = 'same'))
gen_model.add(LeakyReLU())
# Sixth layer
gen_model.add(Conv2DTranspose(depth*4, 5, strides=1, padding = 'same'))
gen_model.add(LeakyReLU())
# Seventh layer
gen_model.add(Conv2DTranspose(depth*4, 5, strides=1, padding = 'same'))
gen_model.add(LeakyReLU())
# Output
gen_model.add(Conv2DTranspose(3, 7, strides=1, activation = 'tanh', padding = 'same'))

# Save model architecture as .PNG
plot_model(gen_model, to_file='generator.png', expand_nested=True, show_shapes=True, show_layer_names=True)

#==================================CGAN========================================
# Define optimisers
optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
disc_model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
disc_model.trainable = False

# Define architecture of CGAN
inputs = Input(shape = (37,))
gen_img = gen_model(inputs)
disc_class_inputs = Input(shape = (5,))
disc_outs = disc_model([disc_class_inputs, gen_img])

# Define CGAN input and output
GAN = Model([inputs, disc_class_inputs], disc_outs)

# Compile CGAN
GAN.compile(loss='binary_crossentropy', optimizer=optimizer)

# Save model architecture as .PNG
plot_model(GAN, show_shapes=True, show_layer_names=True)

#==========================Plot image function=================================
def plot_output(input_37, step):
    filename = "GANmodel_%d" % step
    
    images = gen_model.predict(input_37)

    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = image.reshape(X.shape[1], X.shape[2], X.shape[3])
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
    
#=========================Random label generator===============================
def random_flip():
    return [0,1][random.randrange(2)]

#================================Train CGAN====================================
batch_size = 8
latent_dim = 32 + 5
epoch = 10
save_interval = 1
def train_gan(X, batch_size, epoch, save_interval):
    batch_per_epoch = int(round(X.shape[0]/batch_size))
    d_losses = []
    gan_losses = []
    for i in range(epoch):
        start = 0
        for j in tqdm(range(batch_per_epoch)):
            #=============Train discriminator and auxiliary models=============            
            images_real = X[start:start + batch_size]
            images_label = Y[start:start + batch_size]
            
            random_label = np.array([np.array([random_flip() for i in range(5)]) for i in range(batch_size)])
            noise = np.random.normal(0, 1, size=(batch_size, 32))
            images_fake = gen_model.predict(np.hstack((noise, random_label)))
            
            x = np.concatenate([images_real, images_fake])
            label = np.concatenate([images_label, random_label])
            y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            y += .05 * np.random.random(y.shape)
            
            d_loss = disc_model.train_on_batch([label, x], y)
            d_losses.append(d_loss[0])
            
            #========================Train CGAN================================
            random_label = np.array([np.array([random_flip() for i in range(5)]) for i in range(batch_size)])
            noise = np.random.normal(0, 1, size=(batch_size, 32))
            gen_input = np.concatenate((noise, random_label), axis=1)
            y = np.ones((batch_size, 1))
            
            gan_loss = GAN.train_on_batch([gen_input, random_label], y)
            gan_losses.append(gan_loss)
            
            start += batch_size
            if start > X.shape[0] - batch_size:
                start = 0
        
        log_msg = "epoch %d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_msg = "%s  [CGAN loss: %f]" % (log_msg, gan_loss)
        print(log_msg)
        
        d_losses.append(np.array(d_loss[0], dtype=float))
        gan_losses.append(np.array(gan_loss, dtype=float))
        
        # Save ouputs
        if save_interval>0 and (i+1)%save_interval==0:
            fixed = np.array([random_flip() for i in range(5)])
            test_label = np.stack([fixed for _ in range(16)], axis=0)
            noise = np.random.normal(0, 1, size=(16, 32))
            test_input = np.hstack((noise, test_label))
            plot_output(input_37=test_input, step=(i+1))
    
    d_losses = [float(x) for x in d_losses]
    gan_losses = [float(x) for x in gan_losses]
    
    return(d_losses, gan_losses)

#================================Train CGAN====================================
d_loss_hist, gan_loss_hist = train_gan(X=X, batch_size=16, epoch=20, save_interval=1)

plot_loss(d_loss_ls, gan_loss_ls, 1)