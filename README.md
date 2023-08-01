# Advanced Image Processing Techniques using Machine Learning and Computer Vision

This repository contains a suite of Python-based techniques for advanced image processing, with a particular emphasis on image quantization via K-means clustering and feature extraction using Gabor filters.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Methodology](#methodology)
3. [Installation and Dependencies](#installation-and-dependencies)
4. [Usage Instructions](#usage-instructions)
5. [Contributors](#contributors)

## Project Overview
 
The primary objective of this project is to offer a robust set of image processing functions. The key areas of focus include:

- **Image Quantization using K-Means Clustering:** This technique uses an unsupervised machine learning algorithm to divide an image into distinct clusters based on pixel intensities in the LAB color space. The goal is to minimize the color palette of the image while preserving the overall visual structure and perception.

- **Feature Extraction using Gabor Filters:** Gabor filters, a family of linear filters used widely in texture analysis, are employed in this project. These filters probe the presence of specific frequency content in the image in certain directions within a localized region around the point or region of analysis.

## Methodology

The project's methodology revolves around the following key stages:

- **Image Preprocessing:** This stage involves preparing the images for further processing.

- **Quantization and Feature Extraction:** This stage uses K-Means Clustering and Gabor Filters to extract meaningful features from the images.

- **Post-processing and Visualization:** The final stage involves processing the output from the previous stage and visualizing the results.

## Installation and Dependencies

This project has been implemented using Python and requires several libraries related to machine learning and computer vision:

- scikit-learn
- scikit-image
- NumPy
- OpenCV

## Usage Instructions

After successfully installing all the dependencies, you can import the functions in this script into your Python environment. After importing, you can call these functions on your image data as needed.


## Contributors

- [x] [Dimitris Matsanganis](https://github.com/dmatsanganis)



[![PowerPoint Presentation](https://img.shields.io/badge/PowerPoint-Presentation-brightgreen)](LINK_TO_YOUR_PPTX)


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
