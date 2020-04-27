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
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, LeakyReLU, ReLU, AveragePooling2D, Embedding
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ZeroPadding2D, Concatenate
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.datasets import mnist, fashion_mnist

#=============================Load MNIST dataset===============================
# Load our image dataset
(X, Y), (_, _) = mnist.load_data()
#(X, Y), (_, _) = fashion_mnist.load_data()

X = 1.0/255*X
X = np.array([x.reshape(28, 28, 1) for x in X])
Y = to_categorical(Y, 10)

# Our input vector for generator = 256D noise + 10D category info
latent_dim = 266

#====================Discriminator and Auxiliary model=========================
# Define input shape of our MNIST images
img_shape = (28,28,1)
img = Input(shape=img_shape)

# Shared network
depth = 64
dropout = 0.25
D = Sequential()
D.add(Conv2D(depth, kernel_size=3, strides=2, input_shape=X.shape[1:], padding="same"))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Conv2D(depth*2, kernel_size=3, strides=2, padding="same"))
D.add(ZeroPadding2D(padding=((0,1),(0,1))))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(BatchNormalization(momentum=0.8))
D.add(Conv2D(depth*4, kernel_size=3, strides=2, padding="same"))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(BatchNormalization(momentum=0.8))
D.add(Flatten())

# Discriminator output
img_embedding = D(img)
disc = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(img_embedding)
disc_model = Model(img, disc)

# Auxiliary output
aux = Dense(128, activation='relu')(img_embedding)
label = Dense(10, activation='softmax')(aux)
q_model = Model(img, label)

# Note that these two models are one and the same with different output layers
# Print out architecture of the discriminator    
disc_model.summary()
# Print out architecture of the auxiliary model
q_model.summary()

# Save model architecture as .PNG 
plot_model(disc_model, to_file='disciminator.png', expand_nested=True, show_shapes=True, show_layer_names=True)
plot_model(q_model, to_file='auxiliary.png', expand_nested=True, show_shapes=True, show_layer_names=True)

#==========================Generator model=====================================
# Define architecture of the generator (fraudster AI)
# Note: with each layer, the image gets larger but with reduced depth
depth = 64*2
dim = 7
noise_vec = 266

gen_model = Sequential()

gen_model.add(Dense(depth*dim*dim, activation="relu", input_dim=noise_vec))
gen_model.add(Reshape((7, 7, 128)))
gen_model.add(BatchNormalization(momentum=0.8))
gen_model.add(UpSampling2D())
gen_model.add(Conv2D(128, kernel_size=3, padding="same"))
gen_model.add(Activation("relu"))
gen_model.add(BatchNormalization(momentum=0.8))
gen_model.add(UpSampling2D())
gen_model.add(Conv2D(64, kernel_size=3, padding="same"))
gen_model.add(Activation("relu"))
gen_model.add(BatchNormalization(momentum=0.8))
gen_model.add(Conv2D(1, kernel_size=3, padding='same'))
gen_model.add(Activation("sigmoid"))

# Print out architecture of the generator
gen_model.summary()
# Save model architecture as .PNG
plot_model(gen_model, to_file='generator.png', expand_nested=True, show_shapes=True, show_layer_names=True)

#===============Combine Discriminator and Generator models=====================
# Define optimisers - optimisr1 will be used for all component networks
optimizer1 = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
optimizer2 = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)

disc_model.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
q_model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
gen_model.compile(loss='binary_crossentropy', optimizer=optimizer1)

# Define architecture of InfoGAN
# Starts with generator input = 266D vector
inputs = Input(shape = (latent_dim,)) 
# Generator output
gen_img = gen_model(inputs)

disc_model.trainable = False

# Generator output is the input for both discriminator and auxiliary models
disc_outs = disc_model(gen_img)
q_outs = q_model(gen_img)

# Define InfoGAN inout and output
comb_model = Model(inputs, [disc_outs, q_outs])
comb_model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=optimizer2, metrics=['accuracy'])

# Print out architecture of GAN
comb_model.summary()
# Save model architecture as .PNG 
plot_model(comb_model, to_file='infogan.png', show_shapes=True, show_layer_names=True)
plot_model(comb_model, to_file='infogan_expand.png', expand_nested=True, show_shapes=True, show_layer_names=True)

#==========================Plot image function=================================
def plot_output(input_266, step):
    filename = "GANmodel_%d" % step
    
    images = gen_model.predict(input_266)

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
def plot_loss(d_performance, gan_performance, q_performance, jump=100):
    plt.figure(figsize=(10, 10))
    plt.plot(d_performance[0::jump], label='discriminator')
    plt.plot(q_performance[0::jump], label='q')
    plt.plot(gan_performance[0::jump], label='GAN')
    plt.xlabel('epoch ({})'.format(jump))
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_over_epoch.png')
    plt.close('all')
    
#==============================Train InfoGAN===================================
def train_gan(X, batch_size, epoch, save_interval):
    batch_per_epoch = int(round(X.shape[0]/batch_size))
    d_loss_hist = []
    q_loss_hist = []
    gan_loss_hist = []
    for i in range(epoch):
        for j in tqdm(range(batch_per_epoch)):
            #=============Train discriminator and auxiliary models=============
            disc_model.trainable = True
            half_batch = int(batch_size/2)
            
            images_index = np.random.randint(0, X.shape[0], size = (half_batch))
            images_real = X[images_index]
            images_label = Y[images_index]
            
            random_label = to_categorical(np.random.randint(0,10,half_batch), 10)
            noise = np.random.normal(0, 1, size=(half_batch, 256))
            images_fake = gen_model.predict(np.hstack((noise, random_label)))
            
            d_loss_real = disc_model.train_on_batch(images_real, np.ones([half_batch, 1]))
            d_loss_fake = disc_model.train_on_batch(images_fake, np.zeros([half_batch, 1]))
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            q_loss_real = q_model.train_on_batch(images_real, images_label)
            q_loss_fake = q_model.train_on_batch(images_fake, random_label)
            
            q_loss = 0.5 * np.add(q_loss_real, q_loss_fake)
            
            #========================Train InfoGAN=============================
            disc_model.trainable = False
            
            gen_input1 = np.concatenate((noise, random_label), axis=1)
            
            random_label2 = to_categorical(np.random.randint(0,10,half_batch), 10)
            noise2 = np.random.normal(0, 1, size=(half_batch, 256))
            gen_input2 = np.concatenate((noise2, random_label2), axis=1)
            
            gen_input = np.concatenate((gen_input1, gen_input2), axis=0)
            gan_loss = comb_model.train_on_batch(gen_input, [np.ones([batch_size, 1]), np.concatenate((random_label, random_label2), axis=0)])
        
        log_msg = "epoch %d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_msg = "%s  [Q loss: %f]" % (log_msg, q_loss[0])
        log_msg = "%s  [GAN loss: %f]" % (log_msg, gan_loss[0])
        print(log_msg)
        
        d_loss_hist.append(np.array(d_loss[0], dtype=float))
        q_loss_hist.append(np.array(q_loss[0], dtype=float))
        gan_loss_hist.append(np.array(gan_loss[0], dtype=float))
        
        # Save ouputs
        if save_interval>0 and (i+1)%save_interval==0:
            test_label = np.concatenate(([np.concatenate(np.full((10, 1), x)) for x in range(0, 10)]))
            test_label = to_categorical(test_label)
            noise = np.random.normal(0, 1, size=(100, 256))
            test_input = np.hstack((noise, test_label))
            plot_output(input_266=test_input, step=(i+1))
            
    d_loss_hist = [float(x) for x in d_loss_hist]
    gan_loss_hist = [float(x) for x in gan_loss_hist]
    
    return(d_loss_hist, q_loss_hist, gan_loss_hist)

#===============================Train InfoGAN==================================
d_loss_hist, q_loss_hist, gan_loss_hist = train_gan(X=X, batch_size=32, epoch=10, save_interval=1)

plot_loss(d_loss_hist, gan_loss_hist, q_loss_hist, jump = 1)