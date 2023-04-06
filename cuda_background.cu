#include "png_io.h"
#include "timer.h"
#include <getopt.h>
#include <math.h>
#include <png.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// To compile:
// nvcc -ccbin gcc -g -O3 cuda_background.cu -lpng -lm -o cuda_background

// =========================== KERNEL ============================

__global__ void
background_removal (png_bytepp in_row_pointers, png_bytepp out_row_pointers,
                    int width, int height, int threshold)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < width * height; i += stride)
    {
      int x = i % width;
      int y = i / width;

      // get the pixel
      png_bytep in_pixel = &in_row_pointers[y][x * 4];
      png_bytep out_pixel = &out_row_pointers[y][x * 4];

      int grey = (in_pixel[0] + in_pixel[1] + in_pixel[2]) / 3;
      // Saturate the input_copy by thresholding
      int grey_threshold = (grey < threshold) ? 0 : grey;

      if (abs (in_pixel[0] - grey_threshold) < threshold)
        { 
          out_pixel[0] = 150;
          out_pixel[1] = 150;
          out_pixel[2] = 150;
          out_pixel[3] = 75;
        }
      else
        {
          out_pixel[0] = in_pixel[0];
          out_pixel[1] = in_pixel[1];
          out_pixel[2] = in_pixel[2];
          out_pixel[3] = in_pixel[3];
        }
    }
}

int
main (int argc, char *argv[])
{

  // parse command-line parameters
  if (argc != 4)
    {
      printf ("Usage: %s <threshold> <input_image> <output_image>\n", argv[0]);
      return EXIT_FAILURE;
    }
  int threshold = strtol (argv[1], NULL, 10);
  char *input_filename = argv[2];
  char *output_filename = argv[3];

  png_uint_32 width, height;
  int bit_depth, color_type;
  png_bytepp in_row_pointers;
  png_bytepp out_row_pointers;
  png_bytepp cuda_in_row_pointers;

  // =========================== READ ============================
  START_TIMER (read)
  if (read_png (input_filename, &width, &height, &bit_depth, &color_type,
                &in_row_pointers)
      != 0)
    {
      return 1;
    }
  STOP_TIMER (read);
  // =============================================================

  // a cudaMallocManaged call to allocate memory for the input and output
  // images
     cudaMallocManaged (&cuda_in_row_pointers, height * sizeof (png_bytep));
  for (png_uint_32 i = 0; i < height; i++)
    {
      cudaMallocManaged (&cuda_in_row_pointers[i], width * 4 * sizeof (png_byte));
    }

  // copy the input image to the cudaMallocManaged memory
  for (png_uint_32 i = 0; i < height; i++)
    {
      for (png_uint_32 j = 0; j < width * 4; j++)
        {
          cuda_in_row_pointers[i][j] = in_row_pointers[i][j];
        }
    }

  // a cudaMallocManaged call to allocate memory for the output image

    cudaMallocManaged (&out_row_pointers, height * sizeof (png_bytep));
  for (png_uint_32 i = 0; i < height; i++)
    {
      cudaMallocManaged (&out_row_pointers[i], width * 4 * sizeof (png_byte));
    }

  // draw a red 25x25 block in the top left corner of the image. for testing

  for (int y = 0; y < 25; y++)
    {
      for (int x = 0; x < 25; x++)
        {
          png_bytep pixel = &out_row_pointers[y][x * 4];
          pixel[0] = 255;
          pixel[1] = 0;
          pixel[2] = 0;
          pixel[3] = 255;
        }
    }

  int blockSize = 256;

  // int devId;
  // cudaGetDevice (&devId);
  // int numSMs;
  // cudaDeviceGetAttribute (&numSMs, cudaDevAttrMultiProcessorCount, devId);

  int numBlocks = (width * height + blockSize - 1) / blockSize;

  START_TIMER (background)
  background_removal<<<numBlocks, blockSize>>> (cuda_in_row_pointers, out_row_pointers, width,
                                  height, threshold);
  // // get the error
  // cudaError_t err = cudaGetLastError ();
  // if (err != cudaSuccess)
  //   printf ("Error: %s \n", cudaGetErrorString (err));

  cudaDeviceSynchronize ();
  STOP_TIMER (background)

  // =========================== WRITE ===========================
  START_TIMER (save)
  if (write_png (output_filename, width, height, bit_depth, color_type,
                 out_row_pointers)
      != 0)
    {
      printf ("Failed to write to PNG\n");
      return 1;
    }
  STOP_TIMER (save)

  printf ("READ: %.6f  BACKGROUND: %.6f  SAVE: %.6f\n", GET_TIMER (read),
          GET_TIMER (background), GET_TIMER (save));

  // Free the memory
  free (in_row_pointers);

  for (png_uint_32 i = 1; i < height; i++)
  {
    cudaFree (cuda_in_row_pointers[i]);
    cudaFree (out_row_pointers[i]);
  }

  cudaFree (out_row_pointers);
  cudaFree (cuda_in_row_pointers);

  return EXIT_SUCCESS;
}