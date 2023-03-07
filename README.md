CUDA C Command Line Pixel Sorting Image Processing Project

This is a project developed by Justin Newman, Andrew Fleming, and TJ Davies for processing images using pixel sorting techniques on NVIDIA GPUs with CUDA C.
Requirements

    NVIDIA GPU with CUDA capability
    CUDA Toolkit
    OpenCV
    CMake

Installation

    Clone the repository:

    bash

git clone https://github.com/JustinMatthewNewman/cs470_CUDA.git

Create a build directory:

bash

mkdir build
cd build

Configure the project using CMake:

cmake ..


Build the project:

    make

Usage

The project takes two command line arguments: the path to the input image and the path to the output image. Example usage:

lua

./pixel_sorting input.jpg output.jpg

References

    CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/index.html
    OpenCV Documentation: https://docs.opencv.org/
    CMake Documentation: https://cmake.org/docu