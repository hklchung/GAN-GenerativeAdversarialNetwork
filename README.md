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
  * [DCGANS](#dcgans)
* [Contributing](#contributing)
* [Contact](#contact)
* [Known Issues](#known-issues)

<!-- ABOUT THE PROJECT -->
## About the Project
Generative Adversarial Networks, or GAN, is a class of machine learning systems where two neural networks contest with each other. A model called the Generator is trained on real images which then generate new images that look at least superficially authentic to human observers while another model called the Discriminator distinguishes images produced by the Generator from real images. In this project, I aim to build a GAN using the 100k Celebrity dataset from kaggle to generate fake human faces. For more on GAN, please visit: <a href="https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf"><strong>Ian Goodfellow's GAN paper</strong></a>. 

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
---

<!-- DCGANS -->
### DCGANS
---

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
