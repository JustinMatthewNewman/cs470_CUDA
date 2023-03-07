CUDA C Command Line Pixel Sorting Image Processing Project

This is a project developed by Justin Newman, Andrew Fleming, and TJ Davies for processing images using pixel sorting techniques on NVIDIA GPUs with CUDA C.
Requirements

    NVIDIA GPU with CUDA capability
    CUDA Toolkit
    OpenCV
    CMake

Installation

Clone the repository:

    git clone https://github.com/JustinMatthewNewman/cs470_CUDA.git

Change to build directory:

    cd cs470_CUDA

Build the project:

    make serial

Usage: Parallel Image Processing <option(s)> image-file :
  Options are:
	-d 	desaturate <threshold>
	-g	gaussian blur <threshold>
	-r	rotate
	-b	background removal <threshold>
	-s	sorting <threshold>

Example usage:

./serial input.jpg -s 100

References

    CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/index.html
    OpenCV Documentation: https://docs.opencv.org/
    CMake Documentation: https://cmake.org/docu
