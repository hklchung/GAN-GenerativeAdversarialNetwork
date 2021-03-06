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
(X, Y), (_, _) = mnist.load_data()
#(X, Y), (_, _) = fashion_mnist.load_data()

X = 1.0/255*X
X = np.array([x.reshape(28, 28, 1) for x in X])
Y = to_categorical(Y, 10)

#=========================Discriminator model==================================
# Define architecture of the discriminator (police AI)
class_input = Input(shape=Y.shape[1:])
embedding1 = Embedding(10, 50)(class_input)
dense1 = Dense(28*28)(embedding1)
reshape1 = Reshape((28, 28, 10))(dense1)
dense2 = Dense(1)(reshape1)

image_input = Input(shape=X.shape[1:])
concatenate1 = Concatenate()([image_input, dense2])

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
# Output layer
D.add(Dense(1, kernel_initializer='glorot_normal'))
D.add(Activation('sigmoid'))

disc_out = D(concatenate1)

disc_model = Model([class_input, image_input], disc_out)

# Print out architecture of the discriminator
disc_model.summary()
# Save model architecture as .PNG
plot_model(disc_model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)

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
plot_model(gen_model, to_file='generator.png', show_shapes=True, show_layer_names=True)

#===============Combine Discriminator and Generator models=====================
# Define optimisers - optimisr1 will be used for the discriminator and generator
optimizer1 = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
optimizer2 = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)

disc_model.compile(loss='binary_crossentropy', optimizer=optimizer1,metrics=['accuracy'])
gen_model.compile(loss='binary_crossentropy', optimizer=optimizer1,metrics=['accuracy'])

inputs = Input(shape = (266,))
gen_img = gen_model(inputs)
disc_class_inputs = Input(shape = (10,))
disc_outs = disc_model([disc_class_inputs, gen_img])

# Define CGAN inout and output
comb_model = Model([inputs, disc_class_inputs], disc_outs)
comb_model.compile(loss=['binary_crossentropy'], optimizer=optimizer2, metrics=['accuracy'])

# Print out architecture of GAN
comb_model.summary()
# Save model architecture as .PNG 
plot_model(comb_model, to_file='CGAN.png', show_shapes=True, show_layer_names=True)
plot_model(comb_model, to_file='CGAN_expand.png', expand_nested=True, show_shapes=True, show_layer_names=True)

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
def plot_loss(d_performance, gan_performance, jump=100):
    plt.figure(figsize=(10, 10))
    plt.plot(d_performance[0::jump], label='discriminator')
    plt.plot(gan_performance[0::jump], label='GAN')
    plt.xlabel('epoch ({})'.format(jump))
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_over_epoch.png')
    plt.close('all')
    
#================================Train CGAN====================================
batch_size = 16
latent_dim = 256 + 10
epoch = 1000
save_interval = 100
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
            noise = np.random.normal(0, 1, size=(batch_size, 256))
            images_fake = gen_model.predict(np.hstack((noise, random_label)))
            
            d_loss_real = disc_model.train_on_batch([images_label, images_real], np.ones([batch_size, 1]))
            d_loss_fake = disc_model.train_on_batch([random_label, images_fake], np.zeros([batch_size, 1]))
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            #========================Train CGAN================================
            disc_model.trainable = False
            
            gen_input = np.concatenate((noise, random_label), axis=1)
            gan_loss = comb_model.train_on_batch([gen_input, random_label], np.ones([batch_size, 1]))
        
        log_msg = "epoch %d: [D loss (real): %f, acc: %f]" % (i, d_loss_real[0], d_loss_real[1])
        log_msg = "%s  [D loss (fake): %f, acc: %f]" % (log_msg, d_loss_fake[0], d_loss_fake[1])
        log_msg = "%s  [CGAN loss: %f, acc: %f]" % (log_msg, gan_loss[0], gan_loss[1])
        print(log_msg)
        
        d_loss_hist.append(np.array(d_loss[0], dtype=float))
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
    
    return(d_loss_hist, gan_loss_hist)

#===============================Train CGAN=====================================
d_loss_hist, gan_loss_hist = train_gan(X=X, batch_size=16, epoch=5, save_interval=1)

plot_loss(d_loss_hist, gan_loss_hist, jump = 1)