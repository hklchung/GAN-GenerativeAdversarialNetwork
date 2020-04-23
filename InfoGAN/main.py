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
img_shape = (28,28,1)
img = Input(shape=img_shape)

# Shared layers between discriminator and auxiliary models
model = Sequential()
model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
model.add(ZeroPadding2D(padding=((0,1),(0,1))))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))
model.add(BatchNormalization(momentum=0.8))
model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))
model.add(BatchNormalization(momentum=0.8))
model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))
model.add(BatchNormalization(momentum=0.8))
model.add(Flatten())

img_embedding = model(img)

# Discriminator
disc = Dense(1, activation='sigmoid')(img_embedding)

# Auxiliary model
aux = Dense(128, activation='relu')(img_embedding)
label = Dense(10, activation='softmax')(aux)

disc_model = Model(img, disc)
q_model = Model(img, label)

# Note that these two models are one and the same with different final layers
# Print out architecture of the discriminator    
disc_model.summary()
# Print out architecture of the auxiliary model
q_model.summary()

# Save model architecture as .PNG 
plot_model(disc_model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)
plot_model(q_model, to_file='auxiliary.png', show_shapes=True, show_layer_names=True)

del(model)
#==========================Generator model=====================================
model = Sequential()

model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
model.add(Reshape((7, 7, 128)))
model.add(BatchNormalization(momentum=0.8))
model.add(UpSampling2D())
model.add(Conv2D(128, kernel_size=3, padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(momentum=0.8))
model.add(UpSampling2D())
model.add(Conv2D(64, kernel_size=3, padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(momentum=0.8))
model.add(Conv2D(1, kernel_size=3, padding='same'))
model.add(Activation("sigmoid"))

gen_input = Input(shape=(latent_dim,))
fake_image = model(gen_input)
# Define generator input and output
gen_model = Model(gen_input, fake_image)
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
    
    images = gen_model.predict(input_110)

    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(10, 10, i+1)
        image = images[i, :, :, :]
        image = image.reshape(X.shape[1], X.shape[2])
        plt.imshow(image, cmap='gray')
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
    
#==============================Train InfoGAN===================================
batch_size = 16
latent_dim = 100 + 10
epoch = 60000
save_interval = 3000
def train_gan(X, batch_size, epoch, save_interval):
    batch_per_epoch = int(round(X.shape[0]/batch_size))
    d_loss_hist = []
    gan_loss_hist = []
    for i in range(epoch):
        for j in tqdm(range(batch_per_epoch)):
            #=============Train discriminator and auxiliary models=============
            disc_model.trainable = True
            
            images_index = np.random.randint(0, X.shape[0], size = (batch_size))
            images_real = X[images_index]
            images_label = Y[images_index]
            
            random_label = to_categorical(np.random.randint(0,10,batch_size), 10)
            noise = np.random.normal(0, 1, size=(batch_size, 100))
            images_fake = gen_model.predict(np.hstack((noise, random_label)))
            
            d_loss_real = disc_model.train_on_batch(images_real, np.ones([batch_size, 1]))
            d_loss_fake = disc_model.train_on_batch(images_fake, np.zeros([batch_size, 1]))
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            #========================Train InfoGAN=============================
            disc_model.trainable = False
            
            gen_input = np.concatenate((noise, random_label), axis=1)
            gan_loss = comb_model.train_on_batch(gen_input, [np.ones([batch_size, 1]), random_label])
        
        log_msg = "epoch %d: [D loss (real): %f, acc: %f]" % (i, d_loss_real[0], d_loss_real[1])
        log_msg = "%s  [D loss (fake): %f, acc: %f]" % (log_msg, d_loss_fake[0], d_loss_fake[1])
        print(log_msg)
        
        d_loss_hist.append(np.array(d_loss[0], dtype=float))
        gan_loss_hist.append(np.array(gan_loss[0], dtype=float))
        
        # Save ouputs
        if save_interval>0 and (i+1)%save_interval==0:
            test_label = np.concatenate(([np.concatenate(np.full((10, 1), x)) for x in range(0, 10)]))
            test_label = to_categorical(test_label)
            noise = np.random.normal(0, 1, size=(100, 100))
            test_input = np.hstack((noise, test_label))
            plot_output(input_110=test_input, step=(i+1))
            
    d_loss_hist = [float(x) for x in d_loss_hist]
    gan_loss_hist = [float(x) for x in gan_loss_hist]
    
    return(d_loss_hist, gan_loss_hist)
            
#===============================Train InfoGAN==================================
d_loss_hist, gan_loss_hist = train_gan(X=X, batch_size=16, epoch=5, save_interval=1)

plot_loss(d_loss_hist, gan_loss_hist, jump = 1)