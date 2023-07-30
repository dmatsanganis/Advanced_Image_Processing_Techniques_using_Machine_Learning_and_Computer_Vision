# Advanced Image Processing Techniques

This repository contains Python scripts for various advanced image processing techniques, focusing on image quantization using K-means clustering and feature extraction using Gabor filters.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Introduction

The project provides a comprehensive toolkit for image processing tasks. It focuses on two main aspects:

- **Image Quantization using K-Means Clustering:** An unsupervised machine learning algorithm used to partition an input image into different clusters based on pixel intensities in the color space.
  
- **Feature Extraction using Gabor Filters:** A linear filter used for texture analysis, which essentially analyzes whether there are any specific frequency content in the image in specific directions in a localized region around the point or region of analysis.

## Installation

### Dependencies

This project is implemented in Python and requires the following libraries:

- scikit-learn
- scikit-image
- NumPy
- OpenCV

To install these dependencies, run the following command:

```bash
pip install -r requirements.txt

Usage
To use the functions in this script, import them into your Python environment:

```bash
from functions import image_quantization, gabor_filtering
You can then call the functions on your image data.
