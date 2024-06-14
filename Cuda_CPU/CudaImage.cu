#include </usr/include/opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main()
{
    // Image Path
    std::string imagePath = "/content/SampleImage.jpeg"; // Replace with your uploaded image path
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cerr << "OpenCV version: " << CV_VERSION << std::endl;
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    cv::Mat blurredImageCPU(image.size(), image.type());

    // CPU Gaussian Blur
    auto startCPU = std::chrono::high_resolution_clock::now();
    cv::GaussianBlur(image, blurredImageCPU, cv::Size(3, 3), 3.0);
    auto endCPU = std::chrono::high_resolution_clock::now();

    // Calculate execution times
    auto cpuDuration = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();
    std::cout << "CPU Time: " << cpuDuration << " ms" << std::endl;

    // Save ONLY blurred images
    cv::imwrite("/content/cpu_blurred_image.jpeg", blurredImageCPU);

    // Display confirmation message
    std::cout << "Blurred image saved as: \n"
              << " - cpu_blurred_image.jpeg\n";

    return 0;
}
