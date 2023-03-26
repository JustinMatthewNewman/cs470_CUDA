[For Users]


Example run:

./serial -b 10 input.png


[For Developers]

configure the build:

cd libpng-x.x.x
./configure --prefix=$HOME/local

<!-- The --prefix option specifies the installation directory. In this case, we use a directory 
named local in the user's home directory. You can change this to another location if you prefer. -->

    Compile and install the library:

make
make install

<!-- Now you have libpng installed in the specified directory (e.g., $HOME/local).

    To use the locally installed libpng library, you will need to update the compiler and linker flags in your build process. Add -I$HOME/local/include to the compiler flags and -L$HOME/local/lib -lpng to the linker flags.

For example, if you were compiling the serial.c file mentioned in the previous answer, you would use the following command: -->


gcc serial.c -I$HOME/local/include -L$HOME/local/lib -lpng -o serial

    Finally, you'll need to update the LD_LIBRARY_PATH environment variable to include the path to the locally installed libpng library, so that the runtime linker can find it:


export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH

You can add this line to your .bashrc or .bash_profile file to make it permanent. Now you should be able to use the libpng library without sudo access.



If you want to share your compiled code with others, they won't need to perform these steps as long as the shared binary is built with the required libraries (statically linked). When you compile your code with static linking, the necessary library files are included directly in the compiled binary, so other developers won't need to install the library separately.

To compile your code with static linking, you need to compile the libpng library with the --enable-static option:

bash

./configure --prefix=$HOME/local --enable-static

Then, compile your code using the static version of the library:

bash

gcc serial.c -I$HOME/local/include -L$HOME/local/lib -Wl,-Bstatic -lpng -Wl,-Bdynamic -lz -lm -o serial

In this command, we use -Wl,-Bstatic and -Wl,-Bdynamic to specify which libraries should be linked statically and which should be linked dynamically. In this case, we are linking libpng statically and other required libraries (like zlib and libm) dynamically.

Note that statically linked binaries will be larger in size, as they include the required libraries. If you prefer a smaller binary size, you can share the dynamically linked binary, but other developers will need to install the libpng library (either using sudo or locally) and set the appropriate environment variables (e.g., LD_LIBRARY_PATH).