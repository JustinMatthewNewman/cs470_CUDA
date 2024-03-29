#include "par_image_processing.h"
#include "png_io.h"
#include "timer.h"
#include <getopt.h>
#include <math.h>
#include <png.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Constants for CUDA
int blockSize = 256;


int
main (int argc, char *argv[])
{
  int opt;
  int threshold;
  bool d_flag = false;
  bool r_flag = false;
  bool b_flag = false;
  bool f_flag = false;
  bool s_flag = false;
  bool m_flag = false;
  char *input_filename = NULL;
  char *output_filename = NULL;
  int target_x;
  int target_y;


  // Parse the command line options
  while ((opt = getopt (argc, argv, "d:rb:f:m:s:")) != -1)
    {
      switch (opt)
        {
        case 'd':
          d_flag = true;
          break;
        case 'r':
          r_flag = true;
          break;          
        case 'b':
          b_flag = true;
          target_x = atoi (optarg);
          target_y = atoi (argv[optind++]);
          threshold = atoi (argv[optind++]);
                    //threshold = atoi (optarg);

          break;
        case 'f':
          f_flag = true;
          target_x = atoi (optarg);
          target_y = atoi (argv[optind++]);
          threshold = atoi (argv[optind++]);
                    //threshold = atoi (optarg);

          break;
        case 'm':
          m_flag = true;
          threshold = atoi (optarg);
          break;
        case 's':
          s_flag = true;
          threshold = atoi (optarg);
          break;
        default:
          usage ();
          return 1;
        }
    }

  // Get the input and output filenames from the command line arguments
  if (optind < argc)
    {
      if (d_flag) {
        optind--;
      }
      input_filename = argv[optind++];
    }
  else
    {
      printf ("Error: no input image file specified\n");
      usage ();
      return 1;
    }

  if (optind < argc)
    {
      output_filename = argv[optind];
    }
  else
    {
      printf ("Error: no output image file specified\n");
      return 1;
    }

  png_uint_32 width, height;
  int bit_depth, color_type;
  png_bytepp in_row_pointers;
  png_bytepp out_row_pointers;
  png_bytepp mid_row_pointers;
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

  cudaMallocManaged(&cuda_in_row_pointers, height * sizeof (png_bytep));
  for (png_uint_32 i = 0; i < height; i++)
    {
      cudaMallocManaged(&cuda_in_row_pointers[i], width * 4 * sizeof(png_byte));
    }
  
  for (png_uint_32 i = 0; i < height; i++)
    {
      for (png_uint_32 j = 0; j < width * 4; j++)
        {
          cuda_in_row_pointers[i][j] = in_row_pointers[i][j];
        }
    }
    

  int new_height = height;
  int new_width = width;

  if (r_flag) {
    new_height = width;
    new_width = height;
  }

  cudaMallocManaged(&out_row_pointers, new_height * sizeof (png_bytep));
  cudaMallocManaged(&mid_row_pointers, new_height * sizeof (png_bytep));
  for (png_uint_32 i = 0; i < new_height; i++)
    {
      cudaMallocManaged(&out_row_pointers[i], new_width * 4 * sizeof(png_byte));
      cudaMallocManaged(&mid_row_pointers[i], new_width * 4 * sizeof(png_byte));
    }

  int numBlocks = (width * height + blockSize - 1) / blockSize;
  // dim3 dimGrid((height-1)/16 + 1, (width-1)/16+1, 1);
  // dim3 dimBlock(16,16,1);

  // // =========================== Grey ========================
  START_TIMER (grey)
  if (d_flag)
    {
      greyscale<<<numBlocks, blockSize>>>
        (cuda_in_row_pointers, out_row_pointers, width, height);
      cudaDeviceSynchronize();
    }
  STOP_TIMER (grey)
  // // =========================================================

  // // =========================== Blur ========================
  START_TIMER (blur)
  // if (g_flag)
  //   {
  //     gaussian_blur (in_row_pointers, out_row_pointers, width, height,
  //                    threshold, 100.0);
  //   }
  STOP_TIMER (blur)
  // // =========================================================

  // // =========================== Rotate ======================
  START_TIMER (rotate)
  if (r_flag)
    {
      rotate_90<<<numBlocks, blockSize>>>
        (cuda_in_row_pointers, out_row_pointers, width, height);
        cudaDeviceSynchronize();
    }
  STOP_TIMER (rotate)
  // // =========================================================

  // ============================== Background remove ===========
  START_TIMER (removal)
  if (b_flag)
    {
      // Usage: ./par -b 15 15 75 input.png output.png
      //                 x  y  threshold
      removal<<<numBlocks, blockSize>>> (cuda_in_row_pointers, out_row_pointers, 
      mid_row_pointers, width, height, target_x, target_y, threshold, 'b');
      cudaDeviceSynchronize();
    }
  // ======================== Foreground remove =================
  if (f_flag)
    {
      // Usage: ./par -f 15 15 75 input.png output.png
      //                 x  y  threshold
      removal<<<numBlocks, blockSize>>> (cuda_in_row_pointers, out_row_pointers, 
      mid_row_pointers, width, height, target_x, target_y, threshold, 'f');
      cudaDeviceSynchronize();
    }
  STOP_TIMER (removal)
  // ============================================================

  // ======================== Median Filter =====================
  START_TIMER (median)
  if (m_flag)
    {
      // threshold is meaningless
      // Usage: ./par -m 1 input7.png output7.png 
      median<<<numBlocks, blockSize>>> (cuda_in_row_pointers, out_row_pointers, 
                                        width, height);
      cudaDeviceSynchronize();
    }
  STOP_TIMER (median)
  // ============================================================


  // // =========================== Sort ========================
    START_TIMER (sort)
    if (s_flag)
    {
        pixel_sort_kernel<<<numBlocks, blockSize>>> (cuda_in_row_pointers, out_row_pointers, width, height, threshold);
        cudaDeviceSynchronize();
    }
    STOP_TIMER (sort)
  // =========================================================

  cudaDeviceSynchronize();
  START_TIMER (save)
  if (write_png (output_filename, new_width, new_height, bit_depth, color_type,
                 out_row_pointers)
      != 0)
    {
      printf("Failed to write to PNG\n");
      return 1;
    }
  STOP_TIMER (save)

  // Display timing results
  printf ("READ: %.6f  BACKGROUND: %.6f  GREY: %.6f  BLUR: %.6f  SORT: %.6f  "
          "ROTATE: %.6f  SAVE: %.6f\n",
          GET_TIMER (read), GET_TIMER (removal), GET_TIMER (grey),
          GET_TIMER (blur), GET_TIMER (sort), GET_TIMER (rotate), GET_TIMER (save));
    //double totalTime = GET_TIMER(read) + GET_TIMER(removal) + GET_TIMER(grey) + 
                   //GET_TIMER(blur) + GET_TIMER(sort) + GET_TIMER(rotate) + GET_TIMER(save);
  //printf ("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %d, ",
          //GET_TIMER (read), GET_TIMER (removal), GET_TIMER (grey),
          //GET_TIMER (blur), GET_TIMER (sort), GET_TIMER (rotate), GET_TIMER (save), totalTime, 4);


  for (png_uint_32 i = 0; i < height; i++)
    {
      cudaFree (mid_row_pointers[i]);
      cudaFree (cuda_in_row_pointers[i]);
      if (!r_flag) {
        cudaFree (out_row_pointers[i]);
      }
    }

  cudaFree (mid_row_pointers);
  cudaFree (cuda_in_row_pointers);
  cudaFree (out_row_pointers);
  free(in_row_pointers);

  return 0;
}
