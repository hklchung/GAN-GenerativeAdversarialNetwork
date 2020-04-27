[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![Keras 2.3.1](https://img.shields.io/badge/keras-2.3.1-green.svg?style=plastic)
![TensorFlow-GPU 2.1.0](https://img.shields.io/badge/tensorflow_gpu-2.1.0-green.svg?style=plastic)
![Scikit Image 0.15.0](https://img.shields.io/badge/scikit_image-0.15.0-green.svg?style=plastic)
![Scikit Learn 0.21.3](https://img.shields.io/badge/scikit_learn-0.21.3-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

<br />
<p align="center">
  <a href="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork">
    <img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/DCGAN/Result/Model4/GAN.gif?raw=true" height="150">
  </a>

  <h3 align="center">Generative Adversarial Network</h3>

  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
* [Usage](#usage)
  * [DCGAN](#dcgan)
  * [LSGAN](#lsgan)
  * [InfoGAN](#infogan)
  * [CGAN](#cgan)
* [Contributing](#contributing)
* [Contact](#contact)
* [Known Issues](#known-issues)

<!-- ABOUT THE PROJECT -->
## About the Project
Generative Adversarial Networks, or GAN, is a class of machine learning systems where two neural networks contest with each other. Generally, a model called the Generator is trained on real images which then generate new images that look at least superficially authentic to human observers while another model called the Discriminator distinguishes images produced by the Generator from real images. In this project, I aim to build a various types of GAN models using the 100k Celebrity dataset from kaggle to generate fake human faces and the MNIST dataset to generate some numbers. For more on GAN, please visit: <a href="https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf"><strong>Ian Goodfellow's GAN paper</strong></a>.

All GAN implementations will be done using Keras with Tensorflow backend. This project aims to help beginners to get started with hands-on GAN implementation with hints and tips on how to improve performance with various GAN architectures.

<!-- GETTING STARTED -->
## Getting Started
Hope you are now excited to start building GAN on your machine. To get started, please ensure you have the below packages installed.

<!-- PREREQUISITES -->
### Prerequisites
* Keras==2.3.1
* Scikit-Image==0.15.0
* Scikit-Learn==0.21.3
* PIL==6.2.0
* Tensorflow-gpu==2.1.0
* Numpy==1.18.2

<!-- USAGE -->
## Usage
This projects is divided into 2 parts. With the foundational level GANs, namely DCGAN and LSGAN codes, we will be running through the below listed steps.
1. Download the <a href="https://www.kaggle.com/greg115/celebrities-100k"><strong>100k Celebrities Images Dataset</strong></a>
2. Run resize images to scale down image size to 32 x 32 (default)
3. Load images into session
4. Build the GAN
5. Train the GAN
6. Export a result .gif file

We will also explore extensions of the foundational GANs with variants such as CGAN, InfoGAN and CycleGAN where due to resource limitations, we will run with the MNIST dataset to produce results that are less demanding on the hardware.

<!-- DCGAN -->
### DCGAN
<details><summary>Click to expand</summary>
<p>
DCGAN is also known as Deep Convolutional Generative Adversarial Network, where two models are trained simultaneously by an adversarial process. A generator learns to create images that look real, while a discriminator learns to tell the real and fake images apart. During training, the generator progressively becomes better at creating images that look real, while the discriminator becomes better at telling them apart. The process reaches equilibrium when the discriminator can no longer distinguish real images from fakes, i.e. accuracy maintains at 50%.

Results from DCGAN training with below listed configurations.
<table>
  <tbody>
    <tr>
      <th>Results</th>
      <th>Configuration</th>
    </tr>
    <tr>
      <td><img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/DCGAN/Result/Model7/GANmodel_1700.png?raw=true" height="250"></td>
      <td width="50%">
        <ul>
          <li>no pre-training</li>
          <li>batch_size = 32</li>
          <li>epoch = 1</li>
          <li>noise_len = 100</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/DCGAN/Result/Model9/GANmodel_2500.png?raw=true" height="250"></td>
      <td width="50%">
        <ul>
          <li>no pre-training</li>
          <li>batch_size = 16</li>
          <li>epoch = 1</li>
          <li>noise_len = 256</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/DCGAN/Result/Model10/GANmodel_40.png?raw=true" height="250"></td>
      <td width="50%">
        <ul>
          <li>no pre-training</li>
          <li>batch_size = 128</li>
          <li>epoch = 40</li>
          <li>noise_len = 256</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

Below is a summary of what we have done in our DCGAN code file <a href="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/DCGAN/main.py"><strong>main.py</strong></a>.
1. Resized celebrity images to 32x32x3
2. Load images into session and normalised RGB intensities
3. Created the discriminator and generator models
4. Stacked the two models into GAN
5. Train the GAN by repeating the following
  * (Optional) First pre-train our discriminator to understand what it is looking for
  * Create 100D noise vectors and feed into the generator to create n number of fake images
  * Randomly select n number of real images and concatenate with the fake images from generator
  * Train the discriminator with this batch of images
  * Then freeze the weights on the discriminator
  * Create a new set of 100D noise vectors and again feed into the generator to create n number of fake images
  * Force all labels to be 1 (for "real images")
  * Train the GAN with this batch of images

Training DCGAN successfully is difficult as we are trying to train two models that compete with each other at the same time, and optimisation can oscillate between solutions so much that the generator can collapse. Below are some tips on how to train a DCGAN succesfully.
1. Increase length of input noise vectors - Start with 100 and try 128 and 256
2. Decrease batch size - Start with 64 and try 32, 16 and 8. Smaller batch size generally leads to rapid learning but a volatile learning process with higher variance in the classification accuracy. Whereas larger batch sizes slow down the learning process but the final stages result in a convergence to a more stable model exemplified by lower variance in classification accuracy.
3. No pre-training of discriminator
4. Training longer does not necessarily lead to better results - So don't set the epoch parameter too high
5. The discriminator model needs to be really good at distinguishing the fake from real images but it cannot overpower the generator, therefore both of these models should be as good as possible through maximising the depth of the network that can be supported by your machine

You can also try to configure the below settings.
1. GAN network architecture
2. Values of dropout, LeakyReLU alpha, BatchNormalization momentum
3. Change activation of generator to 'tanh'
4. Change optimiser from RMSProp to Adam
5. Change optimisation metric from 'binary_crossentropy' to Wasserstein loss function
6. Try various kinds of noise sampling, e.g. uniform sampling
7. Soft labelling
8. Separate batches of real and fake images when training discriminator

One of the key limitations of DCGAN is that it occupies a lot of memory during training and typically only works well with small, thumbnail sized images.
</p>
</details>

<!-- LSGAN -->
### LSGAN
<details><summary>Click to expand</summary>
<p>
LSGAN is also known as Least Squares Generative Adversarial Network. This architecture was developed and described by Mao et al., 2016 in the paper <a href="https://arxiv.org/abs/1611.04076"><strong>Least Squares Generative Adversarial Networks</strong></a>, where the author described LSGAN as <i>"...able to generate higher quality images than regular GANs ... LSGANs perform more stable during the learning process."</i>

LSGAN is heuristically identical with DCGAN with below changes in code:
* 'linear' for activation in the discriminator
* 'tanh' for activation in the generator
* 'mse' for loss metric rather than binary corssentropy  

Results from LSGAN training with below listed configurations.
<table>
  <tbody>
    <tr>
      <th>Results</th>
      <th>Configuration</th>
    </tr>
    <tr>
      <td><img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/LSGAN/Result/100kCelebs/GANmodel_1900.png?raw=true" height="250"></td>
      <td width="50%">
        <ul>
          <li>no pre-training</li>
          <li>batch_size = 16</li>
          <li>epoch = 1</li>
          <li>noise_len = 256</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

Below is a summary of what we have done in our LSGAN code file <a href="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/LSGAN/main_100kCeleb.py"><strong>main.py</strong></a>.
1. Resized celebrity images to 32x32x3
2. Load images into session and normalised RGB intensities into range -1 to 1
3. Created the discriminator and generator models
4. Stacked the two models into GAN
5. Train the GAN (process as per DCGAN, see above)

You can also try to configure the below settings.
1. GAN network architecture
2. Values of dropout, LeakyReLU alpha, BatchNormalization momentum
3. Change optimiser from RMSProp to Adam
4. Try various kinds of noise sampling, e.g. uniform sampling
5. Soft labelling
6. Separate batches of real and fake images when training discriminator
</p>
</details>

<!-- INFOGAN -->
### InfoGAN
<details><summary>Click to expand</summary>
<p>
InfoGAN is an information-theoretic extention to the Generative Adversarial Network. This architecture was developed and described by Chen et al., 2016 in the paper <a href="https://arxiv.org/abs/1606.03657"><strong>InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets</strong></a>, where the author described InfoGAN as <i>"... a generative adversarial network that also maximizes the mutual information between a small subset of the latent variables and the observation."</i>

In a well-trained vanilla GAN, the generator model randomly generate images that cannot be distinguished by the discriminator from the rest of the learning set. There is no control over what type of images would be generated. With InfoGAN, this becomes possible through manipulation of the input vector for the generator.

So how do we control the output in InfoGAN?
<img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/InfoGAN/InfoGAN_idea.png?raw=true" height="550">

The above diagram outlines the structure of the network in InfoGAN. We can see that InfoGAN is an extention of DCGAN with new components such as the latent codes c (also known as control vector/variables) and the auxiliary distribution Q(c|X) output which comes from a modified discriminator model. Here the discriminator box denotes a single network of shared weights for 
* A discriminator model that validates the input images
* An auxiliary model that predicts the control variables

At each step of training, we would first train the discriminator to learn to separate real and fake images. Then we freeze the weights on the discriminator and train the generator to produce fake images, given a set of control variables. The discriminator will then tell us how bad the fake images were and we update the weights in the generator to improve the quality of fake images. 

Results from InfoGAN training with below listed configurations. Please note that each row of images denotes one configuration of the control vector.
<table>
  <tbody>
    <tr>
      <th>Results</th>
      <th>Configuration</th>
    </tr>
    <tr>
      <td><img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/InfoGAN/Result/Model2/GANmodel_10.png?raw=true" height="250"></td>
      <td width="50%">
        <ul>
          <li>no pre-training</li>
          <li>batch_size = 32</li>
          <li>epoch = 10</li>
          <li>noise_len = 256 + 10</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

Below is a summary of what we have done in our InfoGAN code file <a href="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/InfoGAN/main.py"><strong>main.py</strong></a>.
1. Load MNIST dataset (default shape 28 x 28 x 1)
2. Normalised intensities into range 0 to 1
3. Created the discriminator, auxiliary and generator models
4. Stacked the three models into InfoGAN
5. Train the GAN by repeating the following
  * Create and stack 100D noise vectors and 10D one-hot encoding vectors (representing random value between 0 and 9)
  * Feed the stacked vectors (variable: gen_input) into the generator to create n number of fake images
  * Train the discriminator with this batch of fake images
  * Randomly select n number of real images
  * Train the discriminator with this batch of real images
  * Then freeze the weights on the discriminator
  * Using the same gen_input variable and force all labels to be 1 (for "real images")
  * Train the GAN with this batch of images
</p>
</details>

<!-- CGAN -->
### CGAN
<details><summary>Click to expand</summary>
<p>

CGAN or Conditional GAN is just like the InfoGAN where the generator is above to take upon a control vector to produce image of a particular desired type. This architecture was developed and described by Mirza and Osindero, 2014 in the paper <a href="https://arxiv.org/abs/1411.1784"><strong>Conditional Generative Adversarial Nets</strong></a>, where the author described CGAN as <i>"... conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator."</i>

So how do we control the output in CGAN?
<img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/CGAN/CGAN_idea.png?raw=true" height="550">

The above diagram outlines the structure of the network in CGAN. We can see that CGAN is similar to InfoGAN in that it is an extention of DCGAN with new components such as the control vector y which is fed into both the generator and the discriminator.

At each step of training, we would first train the discriminator to learn to separate real and fake images. Then we freeze the weights on the discriminator and train the generator to produce fake images, given a set of control variables. The same set of control variables and the images are then both feed into the discriminator which will then tell us how bad the fake images were and we update the weights in the generator to improve the quality of fake images. 

Results from CGAN training with below listed configurations. Please note that each row of images denotes one configuration of the control vector.
<table>
  <tbody>
    <tr>
      <th>Results</th>
      <th>Configuration</th>
    </tr>
    <tr>
      <td><img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/CGAN/Result/MNIST/GANmodel_5.png?raw=true" height="250"></td>
      <td width="50%">
        <ul>
          <li>no pre-training</li>
          <li>batch_size = 16</li>
          <li>epoch = 5</li>
          <li>noise_len = 256 + 10</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><img src="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/CGAN/Result/Fashion_MNIST/GANmodel_5.png?raw=true" height="250"></td>
      <td width="50%">
        <ul>
          <li>no pre-training</li>
          <li>batch_size = 16</li>
          <li>epoch = 5</li>
          <li>noise_len = 256 + 10</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

Below is a summary of what we have done in our CGAN code file <a href="https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/blob/master/CGAN/main.py"><strong>main.py</strong></a>.
1. Load MNIST dataset (default shape 28 x 28 x 1)
2. Normalised intensities into range 0 to 1
3. Created the discriminator and generator models
4. Stacked the two models into CGAN
5. Train the GAN by repeating the following
  * Create and stack 256D noise vectors and 10D one-hot encoding vectors (representing random value between 0 and 9)
  * Feed the stacked vectors into the generator to create n number of fake images
  * Train the discriminator with this batch of fake images and the same 10D one-hot encoding vectors from before
  * Randomly select n number of real images and their corresponding 10D one-hot encoding vectors
  * Train the discriminator with this batch of real images and their 10D vectors
  * Then freeze the weights on the discriminator
  * Using the same noise vector, the 10D one-hot encoding vectors and force all labels to be 1 (for "real images")
  * Train the GAN with this batch of images and 10D one-hot encoding vectors
</p>
</details>

<!-- CONTRIBUTING -->
## Contributing
I welcome anyone to contribute to this project so if you are interested, feel free to add your code.
Alternatively, if you are not a programmer but would still like to contribute to this project, please click on the request feature button at the top of the page and provide your valuable feedback.

<!-- CONTACT -->
## Contact
* [Leslie Chung](https://github.com/hklchung)

<!-- KNOWN ISSUES -->
## Known issues
* Incompatibility with Tensorflow V2 or later versions
* Training may take a very long time if you do not have a GPU available
* If you have previously installed tensorflow-gpu with pip, tensorflow may be unable to detect your GPU. To overcome this issue, first uninstall tensorflow-gpu, then reinstall with conda.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/hklchung/GAN-GenerativeAdversarialNetwork.svg?style=flat-square
[contributors-url]: https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hklchung/GAN-GenerativeAdversarialNetwork.svg?style=flat-square
[forks-url]: https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/network/members
[stars-shield]: https://img.shields.io/github/stars/hklchung/GAN-GenerativeAdversarialNetwork.svg?style=flat-square
[stars-url]: https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/stargazers
[issues-shield]: https://img.shields.io/github/issues/hklchung/GAN-GenerativeAdversarialNetwork.svg?style=flat-square
[issues-url]: https://github.com/hklchung/GAN-GenerativeAdversarialNetwork/issues
