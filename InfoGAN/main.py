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
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, LeakyReLU, ReLU, AveragePooling2D
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ZeroPadding2D
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras import backend
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from skimage.transform import resize
from sklearn.utils import shuffle
from PIL import Image, ImageOps
import tensorflow as tf
from imageio import imread
from keras.datasets import mnist

#=============================Load MNIST dataset===============================
(X, Y), (_, _) = mnist.load_data()
X = 1.0/255*X
X = np.array([x.reshape(28, 28, 1) for x in X])
Y = to_categorical(Y, 10)

latent_dim = 110
#====================Discriminator and Auxiliary model=========================
optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
input_disc = Input(shape = (28, 28, 1))
 
conv_1 = Conv2D(16, 3, padding = 'same', activation = LeakyReLU(alpha=0.2), kernel_initializer='random_normal')(input_disc)
batch_norm1 = BatchNormalization()(conv_1)
pool_1 = AveragePooling2D(strides = (2,2))(batch_norm1)
conv_2 = Conv2D(32, 3, padding = 'same', activation = LeakyReLU(alpha=0.2), kernel_initializer='random_normal')(pool_1)
batch_norm2 = BatchNormalization()(conv_2)
pool_2 = AveragePooling2D(strides = (2,2))(batch_norm2)
conv_3 = Conv2D(64, 3, padding = 'same', activation = LeakyReLU(alpha=0.2), kernel_initializer='random_normal')(pool_2)
batch_norm3 = BatchNormalization()(conv_3)
pool_3 = AveragePooling2D(strides = (2,2))(conv_3)
flatten_1 = Flatten()(pool_3)
output = Dense(1, activation = 'sigmoid', kernel_initializer='random_normal')(flatten_1)
q_output_catgorical = Dense(10, activation = 'softmax')(flatten_1)
    
disc_model = Model(input_disc, output)
disc_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
q_model = Model(input_disc, q_output_catgorical)
q_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
disc_model.summary()
q_model.summary()
 
plot_model(disc_model, show_shapes=True, show_layer_names=True)
plot_model(q_model, show_shapes=True, show_layer_names=True)

#==========================Generator model=====================================
optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
input_gen = Input(shape = (latent_dim,))
dense1 = Reshape((7,7,16))(Dense(7*7*16)(input_gen))
 
batch_norm_1 = BatchNormalization()(dense1)
trans_1 = Conv2DTranspose(128, 3, padding='same', activation=LeakyReLU(alpha=0.2), strides=(2, 2), kernel_initializer='random_normal')(batch_norm_1)
batch_norm_2 = BatchNormalization()(trans_1)
trans_2 = Conv2DTranspose(128, 3, padding='same', activation=LeakyReLU(alpha=0.2), strides=(2, 2), kernel_initializer='random_normal')(batch_norm_2)
output = Conv2D(1, (28,28), activation='sigmoid', padding='same', kernel_initializer='random_normal')(trans_2)
gen_model = Model(input_gen, output)
gen_model.compile(loss='binary_crossentropy', optimizer=optimizer)
gen_model.summary()

plot_model(gen_model, show_shapes=True, show_layer_names=True)

#===============Combine Discriminator and Auxiliary models=====================
inputs = Input(shape = (latent_dim,)) 
gen_img = gen_model(inputs)
    
disc_model.trainable = False

disc_outs = disc_model(gen_img)
q_outs = q_model(gen_img)
    
comb_model = Model(inputs, [disc_outs, q_outs])
comb_model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=optimizer, metrics=['accuracy'])
comb_model.summary()

plot_model(comb_model, show_shapes=True, show_layer_names=True)
    
#==============================Train InfoGAN===================================
batch_size = 16
latent_dim = 100 + 10
iterations = 60000

for i in range(iterations):
    #===============Train discriminator and auxiliary models===================
    images_index = np.random.randint(0, X.shape[0], size = (batch_size))
    images_real = X[images_index]
    images_label = Y[images_index]
    
    random_label = to_categorical(np.random.randint(0,10,batch_size), 10)
    noise = np.random.normal(0, 1, size=(batch_size, 100))
    images_fake = gen_model.predict(np.hstack((noise, random_label)))
    
    x = np.concatenate((images_real, images_fake))
    # Create labels
    disc_y = np.ones([2*batch_size, 1]) 
    disc_y[batch_size:, :] = 0
    q_y = np.concatenate((images_label, random_label))
    # Shuffle the real and fake images
    x,disc_y, q_y = shuffle(x,disc_y, q_y)
    
    disc_trainable = True
 
    d_loss = disc_model.train_on_batch(x, disc_y)
    q_loss = q_model.train_on_batch(x, q_y)
 
    #==========================Train InfoGAN===================================
    random_label = to_categorical(np.random.randint(0,10,batch_size), 10)
    noise = np.random.normal(0, 1, size=(batch_size, 100)) 
    images_fake = gen_model.predict(np.hstack((noise, random_label)))
    
    disc_trainable = False
 
    x = np.hstack((noise, random_label))
 
    gan_loss = comb_model.train_on_batch(x, [np.ones((batch_size,1)), random_label])