# CUDA Blur Implementation in Google Colab

## Overview
This project demonstrates a GPU-accelerated blur Image implementation using CUDA in Google Colab. The code includes a sequential (CPU) implementation for comparison and showcases the performance benefits of GPU acceleration.

## Prerequisites
Before running the code, ensure that you have the following set up:
- Google Colab account
- Upload your image file (e.g., "SampleImage.jpg") to your Colab environment

## Getting Started
1. Open the provided Jupyter notebook file: `CUDA_Gaussian_Blur.ipynb`.
2. Upload your image file to the Colab environment.
3. Execute the cells in the notebook sequentially to run the CPU and GPU implementations.
4. View the generated blurred images and check the execution times.

## File Descriptions
- `Cuda.ipynb`: Jupyter notebook containing the CUDA C++ code and instructions.
- `SampleImage.jpg`: Example image for testing the implementation.

## Results
- Blurred images will be saved as:
  - `cpu_blurred_image.jpg`: Result of the CPU Gaussian blur.
  - `gpu_blurred_image.jpg`: Result of the GPU-accelerated Gaussian blur.

## Notes
- Make sure to adjust the image file path in the code if using a different file name or location.
- Experiment with different kernel sizes and sigma values for varying blur effects.


