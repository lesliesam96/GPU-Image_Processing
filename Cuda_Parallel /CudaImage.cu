#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to calculate Gaussian weights
__device__ float gaussianWeight(float x, float y, float sigma)
{
    float sigma2 = 2.0f * sigma * sigma;
    float t = (x * x + y * y) / sigma2;
    return exp(-t) / (M_PI * sigma2);
}

// CUDA kernel for Gaussian blur
__global__ void gaussianBlurCUDA(const unsigned char *input, unsigned char *output,
                                 int width, int height, float sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float sum = 0.0f;
        float totalWeight = 0.0f;

        // Sample 3x3 neighborhood for simplicity... you can increase the kernel size
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int pixelX = x + dx;
                int pixelY = y + dy;

                if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height)
                {
                    float weight = gaussianWeight(dx, dy, sigma);
                    sum += input[pixelY * width + pixelX] * weight;
                    totalWeight += weight;
                }
            }
        }
        // Normalize and cast to unsigned char before assigning to output
        output[y * width + x] = (unsigned char)((sum / totalWeight) + 0.5f); // Add 0.5 for rounding
    }
}

int main()
{
    // Image Path
    std::string imagePath = "/content/SampleImage.jpg"; // Replace with your uploaded image path
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cerr << "OpenCV version: " << CV_VERSION << std::endl;
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    cv::Mat blurredImageCPU(image.size(), image.type());
    cv::Mat blurredImageGPU(image.size(), image.type());

    // CPU Gaussian Blur (for timing comparison)
    auto startCPU = std::chrono::high_resolution_clock::now();
    cv::GaussianBlur(image, blurredImageCPU, cv::Size(3, 3), 3.0);
    auto endCPU = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, image.total());
    cudaMalloc(&d_output, image.total());

    // Copy input image to device
    cudaMemcpy(d_input, image.data, image.total(), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockSize(16, 16); // 2D block
    dim3 gridSize((image.cols + blockSize.x - 1) / blockSize.x,
                  (image.rows + blockSize.y - 1) / blockSize.y); // 2D grid

    // CUDA Gaussian Blur
    auto startGPU = std::chrono::high_resolution_clock::now();
    gaussianBlurCUDA<<<gridSize, blockSize>>>(d_input, d_output, image.cols, image.rows, 3.0);
    cudaDeviceSynchronize();
    auto endGPU = std::chrono::high_resolution_clock::now();

    // Copy result back to host for CPU
    cv::Mat blurredImageHost(image.size(), image.type());
    cudaMemcpy(blurredImageHost.data, d_output, image.total(), cudaMemcpyDeviceToHost);

    // Calculate execution times
    auto cpuDuration = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();
    auto gpuDuration = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();

    std::cout << "CPU Time: " << cpuDuration << " ms" << std::endl;
    std::cout << "GPU Time: " << gpuDuration << " ms" << std::endl;

    // Save ONLY blurred images
    cv::imwrite("/content/cpu_blurred_image.jpg", blurredImageHost);
    cv::imwrite("/content/gpu_blurred_image.jpg", blurredImageGPU);

    // Display confirmation message
    std::cout << "Blurred images saved as: \n"
              << " - cpu_blurred_image.jpg\n"
              << " - gpu_blurred_image.jpg\n";

    // Release memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
