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
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.datasets import mnist

#=============================Load MNIST dataset===============================
# Load MNIST dataset
(X, Y), (_, _) = mnist.load_data()
# Normalise values to range 0 ~ 1
X = 1.0/255*X
X = np.array([x.reshape(28, 28, 1) for x in X])
# One-hot encode labels
Y = to_categorical(Y, 10)

# Our input vector for generator = 100D noise + 10D category info
latent_dim = 110
#====================Discriminator and Auxiliary model=========================
# Define input shape of our MNIST images
input_disc = Input(shape = (28, 28, 1))

# 1st Conv layer
# In: 28 x 28 x 1
# Out: 28 x 28 x 16
conv_1 = Conv2D(16, 3, padding = 'same', activation = LeakyReLU(alpha=0.2), kernel_initializer='random_normal')(input_disc)
batch_norm1 = BatchNormalization()(conv_1)
# 1st Pooling layer
# In: 28 x 28 x 16
# Out: 14 x 14 x 16
pool_1 = AveragePooling2D(strides = (2,2))(batch_norm1)
# 2nd Conv layer
# In: 14 x 14 x 16
# Out: 14 x 14 x 32
conv_2 = Conv2D(32, 3, padding = 'same', activation = LeakyReLU(alpha=0.2), kernel_initializer='random_normal')(pool_1)
batch_norm2 = BatchNormalization()(conv_2)
# 2nd Pooling layer
# In: 14 x 14 x 32
# Out: 7 x 7 x 32
pool_2 = AveragePooling2D(strides = (2,2))(batch_norm2)
# 3rd Conv layer
# In: 7 x 7 x 32
# Out: 7 x 7 x 64
conv_3 = Conv2D(64, 3, padding = 'same', activation = LeakyReLU(alpha=0.2), kernel_initializer='random_normal')(pool_2)
batch_norm3 = BatchNormalization()(conv_3)
# 3rd Pooling layer
# In: 7 x 7 x 64
# Out: 3 x 3 x 64
pool_3 = AveragePooling2D(strides = (2,2))(conv_3)
# Flatten layer
# In: 3 x 3 x 64
# Out: 1 x 576
flatten_1 = Flatten()(pool_3)
# Discriminator output
output = Dense(1, activation = 'sigmoid', kernel_initializer='random_normal')(flatten_1)
# Define discriminator input and output
disc_model = Model(input_disc, output)

# Auxiliary model output
q_output_catgorical = Dense(10, activation = 'softmax')(flatten_1)
# Define auxiliary model input and output
q_model = Model(input_disc, q_output_catgorical)

# Note that these two models are one and the same with different final layers
# Print out architecture of the discriminator    
disc_model.summary()
# Print out architecture of the auxiliary model
q_model.summary()

# Save model architecture as .PNG 
plot_model(disc_model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)
plot_model(q_model, to_file='auxiliary.png', show_shapes=True, show_layer_names=True)

#==========================Generator model=====================================
# Define input shape of our input vector (100D noise + 10D category info)
input_gen = Input(shape = (latent_dim,))
# 1st dense layer
# In: 1 x 110
# Out: 7 x 7 x 16
dense1 = Reshape((7,7,16))(Dense(7*7*16)(input_gen)) 
batch_norm_1 = BatchNormalization()(dense1)
# 1st Conv Transpose layer
# In: 7 x 7 x 16
# Out: 14 x 14 x 128
trans_1 = Conv2DTranspose(128, 3, padding='same', activation=LeakyReLU(alpha=0.2), strides=(2, 2), kernel_initializer='random_normal')(batch_norm_1)
batch_norm_2 = BatchNormalization()(trans_1)
# 2nd Conv Transpose layer
# In: 14 x 14 x 128
# Out: 28 x 28 x 128
trans_2 = Conv2DTranspose(128, 3, padding='same', activation=LeakyReLU(alpha=0.2), strides=(2, 2), kernel_initializer='random_normal')(batch_norm_2)
# 1st Conv layer
# In: 28 x 28 x 128
# Out: 28 x 28 x 1
output = Conv2D(1, (28,28), activation='sigmoid', padding='same', kernel_initializer='random_normal')(trans_2)

# Define generator input and output
gen_model = Model(input_gen, output)
# Print out architecture of the generator
gen_model.summary()

# Save model architecture as .PNG 
plot_model(gen_model, to_file='generator.png', show_shapes=True, show_layer_names=True)

#===============Combine Discriminator and Auxiliary models=====================
# Define optimisers - optimisr1 will be used for all component models
optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)

disc_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
q_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
gen_model.compile(loss='binary_crossentropy', optimizer=optimizer)

# Define architecture of InfoGAN
# Starts with generator input = 110D vector
inputs = Input(shape = (latent_dim,)) 
# Generator output
gen_img = gen_model(inputs)

disc_model.trainable = False

# Generator output is the input for both discriminator and auxiliary models
disc_outs = disc_model(gen_img)
q_outs = q_model(gen_img)

# Define InfoGAN inout and output
comb_model = Model(inputs, [disc_outs, q_outs])
comb_model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

# Print out architecture of GAN
comb_model.summary()
# Save model architecture as .PNG 
plot_model(comb_model, to_file='InfoGAN.png', show_shapes=True, show_layer_names=True)

#==========================Plot image function=================================
def plot_output(input_110, step):
    filename = "GANmodel_%d" % step
    filename += "_testnum_%d" % test_num
    
    images = gen_model.predict(input_110)

    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = image.reshape(X.shape[1], X.shape[2])
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')
    
#==============================Train InfoGAN===================================
batch_size = 16
latent_dim = 100 + 10
iterations = 60000
save_interval = 3000

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
    
    log_msg = "epoch %d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
    log_msg = "%s  [Q loss: %f, acc: %f]" % (log_msg, q_loss[0], q_loss[1])
    log_msg = "%s  [GAN loss: %f, acc: %f]" % (log_msg, gan_loss[0], gan_loss[1])
    print(log_msg)
    
    # Save ouputs
    if save_interval>0 and (i+1)%save_interval==0:
        test_num = np.random.randint(0,10)
        test_label = to_categorical(np.full((batch_size, 1), test_num), 10)
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        test_input = np.hstack((noise, test_label))
        plot_output(input_110=test_input, step=(i+1))