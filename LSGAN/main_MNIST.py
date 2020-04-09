import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, LeakyReLU, ReLU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ZeroPadding2D
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
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
(X, _), (_, _) = mnist.load_data()
# KEY CHANGE: RGB values normalised to range -1 to 1 as we have tanh activation in generator
X = (X-127.5)/127.5
X = np.array([x.reshape(28, 28, 1) for x in X])

#=========================Discriminator model==================================
# Define architecture of the discriminator (police AI)
depth = 64
dropout = 0.5
D = Sequential()
# Input layer
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
D.add(Conv2D(depth*8, 5, strides=1, padding='same', kernel_initializer='glorot_normal'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Flatten())
# Output layer
D.add(Dense(1, kernel_initializer='glorot_normal'))
# KEY CHANGE: linear activation (rather than sigmoid like in DCGAN)
D.add(Activation('linear'))

# Print out architecture of the discriminator
D.summary()
# Save model architecture as .PNG
plot_model(D, to_file='discriminator.png', show_shapes=True, show_layer_names=True)

#==========================Generator model=====================================
# Define architecture of the generator (fraudster AI)
# Note: with each layer, the image gets larger but with reduced depth
dropout = 0.5
depth = 64*4
dim = 7
noise_vec = 256
G = Sequential()
# Input layer
# In: 100
# Out: dim x dim x depth
G.add(Dense(dim*dim*depth, input_dim=noise_vec, kernel_initializer='glorot_normal'))
G.add(BatchNormalization(momentum=0.9))
G.add(LeakyReLU(alpha=0.3))
G.add(Reshape((dim, dim, depth)))
G.add(Dropout(dropout))
# Second layer
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/2), 5, padding='same', kernel_initializer='glorot_normal'))
G.add(BatchNormalization(momentum=0.8))
G.add(LeakyReLU(alpha=0.3))
# Third layer
# In: 2*dim x 2*dim x depth/2
# Out: 4*dim x 4*dim x depth/4
G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/4), 5, padding='same', kernel_initializer='glorot_normal'))
G.add(BatchNormalization(momentum=0.8))
G.add(LeakyReLU(alpha=0.3))
# Output layer
# In: 4*dim x 4*dim x depth/4
# Out: 28 x 28 x 1 RGB scale image [0.0,1.0] per pixel
G.add(Conv2DTranspose(1, 5, padding='same', kernel_initializer='glorot_normal'))
# KEY CHANGE: tanh activation (rather than sigmoid like in DCGAN)
G.add(Activation('tanh'))

# Print out architecture of the generator
G.summary()
# Save model architecture as .PNG
plot_model(G, to_file='generator.png', show_shapes=True, show_layer_names=True)

#============================Optimiser=========================================
# Define optimisers - optimisr1 will be used for the discriminator and generator
optimizer1 = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
optimizer2 = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)

# KEY CHANGE: mse loss (rather than binary_crossentropy like in DCGAN)
D.compile(loss='mse', optimizer=optimizer1,metrics=['accuracy'])
G.compile(loss='mse', optimizer=optimizer1,metrics=['accuracy'])

# Define architecture of GAN
GAN = Sequential()
GAN.add(G)  # Adding the generator
GAN.add(D)  # Adding the discriminator 
GAN.compile(loss='mse', optimizer=optimizer2, metrics=['accuracy'])

# Print out architecture of GAN
GAN.summary()

#==========================Plot image function=================================
def plot_output(noise, step):
    filename = "GANmodel_%d.png" % step
    
    images = G.predict(noise)

    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
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
    
#========================Plot accuracy function================================
def plot_accuracy(d_performance, gan_performance, jump=100):
    plt.figure(figsize=(10, 10))
    plt.plot(d_performance[0::jump], label='discriminator')
    plt.plot(gan_performance[0::jump], label='GAN')
    plt.xlabel('epoch ({})'.format(jump))
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('accuracy_over_epoch.png')
    plt.close('all')

#=========================Train GAN function===================================
def train_gan(X, model, batch_size, epoch, save_interval, pretrain=False, pretrain_num=100, noise_len=100):
    if pretrain == True:
        if pretrain_num > X.shape[0]:
            pretrain_num = X.shape[0]
        else:
            # Randomly select n (pretrain_num) number of images from X
            images_real = X[np.random.randint(0,X.shape[0], size=pretrain_num), :, :, :]
            # Generate n number of 100D noise vectors
            noise = np.random.normal(0.0, 1.0, size=[pretrain_num, noise_len])
            # Produce n number of fake images with generator
            images_fake = G.predict(noise)
            # Concat real and fake images
            x = np.concatenate((images_real, images_fake))
            # Create labels
            y = np.ones([2*pretrain_num, 1])
            y[pretrain_num:, :] = 0
            # Shuffle the real and fake images
            x,y = shuffle(x,y)
            # Make discriminator trainable
            D.trainable = True
            # Train discriminator on the sampled data
            D.fit(x, y, batch_size=batch_size, epochs=1, validation_split=0.15)
    else:
        None
    
    d_performance = []
    gan_performance = []
    d_accuracy = []
    gan_accuracy = []
    for i in range(epoch):
        #=====================Train discriminator==============================
        # Randomly select n (batch_size) number of images from X
        images_real = X[np.random.randint(0,X.shape[0], size=batch_size), :, :, :]
        # Generate n number of 100D noise vectors
        noise = np.random.normal(0.0, 1.0, size=[batch_size, noise_len])
        # Produce n number of fake images with generator
        images_fake = G.predict(noise)
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
        noise = np.random.normal(0.0, 1.0, size=[batch_size, noise_len])
        # Freeze weights in discriminator
        D.trainable = False
        # Train GAN on the generated data
        gan_loss = GAN.train_on_batch(noise, y)
        # Print loss and accuracy values 
        log_msg = "epoch %d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_msg = "%s  [GAN loss: %f, acc: %f]" % (log_msg, gan_loss[0], gan_loss[1])
        print(log_msg)
        
        d_performance.append(np.array(d_loss[0], dtype=float))
        gan_performance.append(np.array(gan_loss[0], dtype=float))
        
        d_accuracy.append(np.array(d_loss[1], dtype=float))
        gan_accuracy.append(np.array(gan_loss[1], dtype=float))
        
        # Save ouputs
        if save_interval>0 and (i+1)%save_interval==0:
            noise_input = np.random.normal(0.0, 1.0, size=[16, noise_len])
            plot_output(noise=noise_input, step=(i+1))
    
    d_loss_ls = [float(x) for x in d_performance]
    gan_loss_ls = [float(x) for x in gan_performance]
    d_acc_ls = [float(x) for x in d_accuracy]
    gan_acc_ls = [float(x) for x in gan_accuracy]
    return(d_loss_ls, gan_loss_ls, d_acc_ls, gan_acc_ls)

#=================================Train GAN====================================
# Hints:
# Input noise - longer noise vector produce better results (100, 128, 256)
# --- to change noise_len, make sure to also change generator input size
# Batch size - pick smaller batch size like 8, 16, 32, 64
# Pre-training may hurt performance
d_loss_ls, gan_loss_ls, d_acc_ls, gan_acc_ls = train_gan(X=X, model=GAN, batch_size=16, epoch=2000, 
                                                         save_interval=100, pretrain=False, pretrain_num=20000,
                                                         noise_len=256)
plot_loss(d_loss_ls, gan_loss_ls)
plot_accuracy(d_acc_ls, gan_acc_ls)

#================================Save model====================================
model_json = GAN.to_json()
with open("LSGAN_MNIST_model.json", "w") as json_file:
    json_file.write(model_json)
GAN.save_weights("LSGAN_MNIST_model.h5")

#================================Result GIF====================================
import imageio
result_pwd = 'Result/MNIST/'
output_pwd = os.path.abspath(os.getcwd())
images = []
for filename in tqdm(os.listdir(result_pwd)):
    images.append(imageio.imread(result_pwd + '/' + filename))
imageio.mimsave(output_pwd + '/' + result_pwd + '/' + 'LSGAN_MNIST.gif', images)