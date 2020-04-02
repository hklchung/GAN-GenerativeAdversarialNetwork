import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, ZeroPadding2D
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from sklearn.utils import shuffle
from PIL import Image, ImageOps
import tensorflow as tf

#===========================Resize images======================================
for filename in tqdm(os.listdir('Image/Train')):
    temp = Image.open('Image/Train/' + filename)
    size = 32, 32
    temp.thumbnail(size, Image.ANTIALIAS)
    temp.save('Image/Train/Resized/' + filename, "JPEG")

#============================Get images========================================
images = []
# Grab images from folder
for filename in tqdm(os.listdir('Image/Train/Resized')):
    temp = np.array(img_to_array(load_img('Image/Train/Resized/'+filename)), dtype=float)
    hor = 32 - temp.shape[0]
    ver = 32 - temp.shape[1]
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
X = [x.reshape(32, 32, 3) for x in X]
X = np.array(X)

#=========================Discriminator model==================================
# Define architecture of the discriminator (police AI)
depth = 64
dropout = 0.5
D = Sequential()
# Input layer
D.add(Conv2D(depth*1, 5, strides=2, input_shape=X.shape[1:],padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
# Second layer
D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
# Third layer
D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
# Forth layer
D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Flatten())
# Output layer
D.add(Dense(1))
D.add(Activation('sigmoid'))

# Print out architecture of the discriminator
D.summary()
# Save model architecture as .PNG
plot_model(D, to_file='discriminator.png', show_shapes=True, show_layer_names=True)

#==========================Generator model=====================================
# Define architecture of the generator (fraudster AI)
# Note: with each layer, the image gets larger but with reduced depth
dropout = 0.5
depth = 64*4
dim = 8
G = Sequential()
# Input layer
# In: 100
# Out: dim x dim x depth
G.add(Dense(dim*dim*depth, input_dim=100))
G.add(BatchNormalization(momentum=0.9))
G.add(LeakyReLU(alpha=0.3))
G.add(Reshape((dim, dim, depth)))
G.add(Dropout(dropout))
# Second layer
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
G.add(BatchNormalization(momentum=0.8))
G.add(LeakyReLU(alpha=0.3))
# Third layer
# In: 2*dim x 2*dim x depth/2
# Out: 4*dim x 4*dim x depth/4
G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
G.add(BatchNormalization(momentum=0.8))
G.add(LeakyReLU(alpha=0.3))
# Forth layer
# In: 4*dim x 4*dim x depth/4
# Out: 8*dim x 8*dim x depth/8
G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
G.add(BatchNormalization(momentum=0.8))
G.add(LeakyReLU(alpha=0.3))
# Output layer
# In: 8*dim x 8*dim x depth/8
# Out: 32 x 32 x 3 RGB scale image [0.0,1.0] per pixel
G.add(Conv2DTranspose(3, 5, padding='same'))
G.add(Activation('sigmoid'))

# Print out architecture of the generator
G.summary()
# Save model architecture as .PNG
plot_model(G, to_file='generator.png', show_shapes=True, show_layer_names=True)

#============================Optimiser=========================================
# Define optimisers - optimisr1 will be used for the discriminator and generator
optimizer1 = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
optimizer2 = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)

D.compile(loss='binary_crossentropy', optimizer=optimizer1,metrics=['accuracy'])
G.compile(loss='binary_crossentropy', optimizer=optimizer1,metrics=['accuracy'])

# Define architecture of GAN
GAN = Sequential()
GAN.add(G)  # Adding the generator
GAN.add(D)  # Adding the discriminator 
GAN.compile(loss='binary_crossentropy', optimizer=optimizer2, metrics=['accuracy'])

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
        
        # Save ouputs
        if save_interval>0 and (i+1)%save_interval==0:
            noise_input = np.random.normal(0.0, 1.0, size=[16, noise_len])
            plot_output(noise=noise_input, step=(i+1))
    
    d_performance = [float(x) for x in d_performance]
    gan_performance = [float(x) for x in gan_performance]
    return(d_performance, gan_performance)

#=================================Train GAN====================================
# Hints:
# Input noise - longer noise vector produce better results (100, 128, 256)
# --- to change noise_len, make sure to also change generator input size
# Batch size - pick smaller batch size like 8, 16, 32, 64
# Pre-training may hurt performance
d_performance, gan_performance = train_gan(X=X, model=GAN, batch_size=32, epoch=2000, 
                                           save_interval=100, pretrain=False, pretrain_num=20000,
                                           noise_len=100)
plot_loss(d_performance, gan_performance)

#================================Result GIF====================================
import imageio
result_pwd = 'Result/Model4'
output_pwd = os.path.abspath(os.getcwd())
images = []
for filename in tqdm(os.listdir(result_pwd)):
    images.append(imageio.imread(result_pwd + '/' + filename))
imageio.mimsave(output_pwd + '/' + result_pwd + '/' + 'GAN.gif', images)