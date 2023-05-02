
# CUDA PNG Image Processing for PixelSorting Effect

![Alt text](/encode_video/frames/frame_150.png "Optional title")

This project handles transparent PNG images using [libpng](http://www.libpng.org/pub/png/libpng.html), the official PNG reference library. It supports almost all PNG features, is extensible, and has been extensively tested for over 23 years.

## Example Run

```bash
make clean
make
./serial -s 100 input.png output.png
./serial -b 75 input.png output.png
./serial -g 1 input.png output.png
./serial -d input.png output.png
```

Now try exporting a video.

```bash
bash export_video.sh input.png
```


=======================================================================

Setup Troubleshooting Instructions

To compile the project, you will need to complete a few setup steps.
1. Install libpng

First, configure the build:

```bash
#tar -xvf libpng-1.6.39.tar.xz
cd libpng-1.6.39
./configure --prefix=$HOME/local
```
The --prefix option specifies the installation directory. In this case, we use a directory named local in the user's home directory. You can change this to another location if you prefer.

Next, compile and install the library:

```bash
make
make install
```
Now you have libpng installed in the specified directory (e.g., $HOME/local).
2. Update Compiler and Linker Flags

To use the locally installed libpng library, you will need to update the compiler and linker flags in your build process. Add -I$HOME/local/include to the compiler flags and -L$HOME/local/lib -lpng to the linker flags.

For example, if you were compiling the serial.c file, you would use the following command:

```bash
gcc serial.c -I$HOME/local/include -L$HOME/local/lib -lpng -o serial
```
3. Update the LD_LIBRARY_PATH Environment Variable

Finally, you'll need to update the LD_LIBRARY_PATH environment variable to include the path to the locally installed libpng library, so that the runtime linker can find it:

```bash
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
```
You can add this line to your .bashrc or .bash_profile file to make it permanent. Now you should be able to use the libpng library without sudo access.
Sharing Compiled Code

If you want to share your compiled code with others, they won't need to perform these steps as long as the shared binary is built with the required libraries (statically linked). When you compile your code with static linking, the necessary library files are included directly in the compiled binary, so other developers won't need to install the library separately.

To compile your code with static linking, you need to compile the libpng library with the --enable-static option:

```bash
./configure --prefix=$HOME/local --enable-static
```
Then, compile your code using the static version of the library:

```bash
gcc serial.c -I$HOME/local/include -L$HOME/local/lib -Wl,-Bstatic -lpng -Wl,-Bdynamic -lz -lm -o serial
```
In this command, we use -Wl,-Bstatic and -Wl,-Bdynamic to specify which libraries should be linked statically and which should be linked dynamically. In this case, we are linking libpng statically and other required libraries (like zlib and libm) dynamically.

Note that statically linked binaries will be larger in size, as they include the required libraries. If you prefer a smaller binary size, you can share the dynamically linked binary, but other developers will need to install the libpng library (either using sudo or locally) and set the appropriate environment variables (e.g., LD_LIBRARY_PATH).

After images have been processed, you can determine the percentage difference in the images using ImageMagicK
```bash
gcc compare_image.c -o compare
```
Then the comparison can occur with...
```bash
./compare -c image1.png image2.png
```
