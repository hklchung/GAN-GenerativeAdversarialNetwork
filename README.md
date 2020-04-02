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
* [Contributing](#contributing)
* [Contact](#contact)
* [Known Issues](#known-issues)

<!-- ABOUT THE PROJECT -->
## About the Project
Generative Adversarial Networks, or GAN, is a class of machine learning systems where two neural networks contest with each other. Generally, a model called the Generator is trained on real images which then generate new images that look at least superficially authentic to human observers while another model called the Discriminator distinguishes images produced by the Generator from real images. In this project, I aim to build a various types of GAN models using the 100k Celebrity dataset from kaggle to generate fake human faces. For more on GAN, please visit: <a href="https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf"><strong>Ian Goodfellow's GAN paper</strong></a>.

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
1. Download the <a href="https://www.kaggle.com/greg115/celebrities-100k"><strong>100k Celebrities Images Dataset</strong></a>
2. Run resize images to scale down image size to 32 x 32 (default)
3. Load images into session
4. Build the GAN
5. Train the GAN
6. Export a result .gif file

<!-- DCGAN -->
### DCGAN
DCGAN is also known as Deep Convolutional Generative Adversarial Network, where two models are trained simultaneously by an adversarial process. A generator learns to create images that look real, while a discriminator learns to tell the real and fake images apart. During training, the generator progressively becomes better at creating images that look real, while the discriminator becomes better at telling them apart. The process reaches equilibrium when the discriminator can no longer distinguish real images from fakes, i.e. accuracy maintains at 50%.

Results from DCGAN training at with below listed configurations.
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
          <li>epoch = 2000</li>
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
          <li>epoch = 5000</li>
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

You can also try to configure the below settings.
1. GAN network architecture
2. Values of dropout, LeakyReLU alpha, BatchNormalization momentum
3. Change activation of generator to 'tanh'
4. Change optimiser from RMSProp to Adam
5. Change optimisation metric from 'binary_crossentropy' to Wasserstein loss function
6. Try various kinds of noise sampling, e.g. uniform sampling
7. Soft labelling

One of the key limitations of DCGAN is that it occupies a lot of memory during training and typically only works well with small, thumbnail sized images.


<!-- CONTRIBUTING -->
## Contributing
I welcome anyone to contribute to this project so if you are interested, feel free to add your code.
Alternatively, if you are not a programmer but would still like to contribute to this project, please click on the request feature button at the top of the page and provide your valuable feedback.

<!-- CONTACT -->
## Contact
* [Leslie Chung](https://github.com/hklchung)

<!-- KNOWN ISSUES -->
## Known issues
Incompatibility with Tensorflow V2 or later versions.

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
