# Advanced Image Processing Techniques using Machine Learning and Computer Vision

This repository hosts a set of advanced image processing techniques developed in Python, specifically focusing on image quantization and feature extraction through K-means clustering and Gabor filters respectively.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation and Dependencies](#installation-and-dependencies)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Overview

The primary objective of this project is to provide a comprehensive suite of image processing functionalities. The main areas of focus include:

- **Image Quantization using K-Means Clustering:** This unsupervised machine learning technique is utilized to partition an image into distinct clusters based on pixel intensities within the LAB color space. The objective is to reduce the color palette of the image while preserving the overall visual perception and structure.

- **Feature Extraction using Gabor Filters:** Gabor filters, a group of linear filters used extensively in texture analysis, are applied in this project. These filters examine the presence of specific frequency content in the image across particular directions within a localized area around the point or region of analysis.

## Installation and Dependencies

This project is implemented using Python and depends on several packages within the machine learning and computer vision domains:

- scikit-learn
- scikit-image
- NumPy
- OpenCV

To install these dependencies, execute the following command:

```bash
pip install -r Application's Requirements.txt
