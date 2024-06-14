cmake_minimum_required(VERSION 3.5)
project(CudaOpenCVProject LANGUAGES CXX CUDA)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set CUDA architecture (change according to your GPU architecture)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -arch=sm_75)

# Specify include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add CUDA executable with explicitly specifying source file
add_executable(CudaImage CudaImage.cu)

# Set CUDA architectures property for the target (replace with your GPU's compute capability)
set_target_properties(CudaImage PROPERTIES CUDA_ARCHITECTURES 75)

# Link OpenCV libraries
target_link_libraries(CudaImage ${OpenCV_LIBS})
