#include "image_processing.h"
#include "png_io.h"
#include "timer.h"
#include <getopt.h>
#include <math.h>
#include <png.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int
main (int argc, char *argv[])
{
  int opt;
  int threshold;
  bool d_flag = false;
  bool g_flag = false;
  bool r_flag = false;
  bool b_flag = false;
  bool s_flag = false;
  char *input_filename = NULL;
  char *output_filename = NULL;

  // Parse the command line options
  while ((opt = getopt (argc, argv, "dg:rb:s:")) != -1)
    {
      switch (opt)
        {
        case 'd':
          d_flag = true;
          break;
        case 'g':
          g_flag = true;
          threshold = atoi (optarg);
          break;
        case 'r':
          r_flag = true;
          break;
        case 'b':
          b_flag = true;
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
      usage ();
      return 1;
    }

  png_uint_32 width, height;
  int bit_depth, color_type;
  png_bytepp in_row_pointers;
  png_bytepp out_row_pointers;

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

  int new_height = height;
  int new_width = width;
  if (r_flag) {
    new_height = width;
    new_width = height;
  }
  out_row_pointers = (png_bytep *)malloc (new_height * sizeof (png_bytep));
  for (png_uint_32 i = 0; i < new_height; i++)
    {
      out_row_pointers[i] = (png_byte *)malloc (new_width * 4 * sizeof (png_byte));
    }

  // // =========================== Grey ========================
  START_TIMER (grey)
  if (d_flag)
    {
      greyscale (in_row_pointers, out_row_pointers, width, height);
    }
  STOP_TIMER (grey)
  // // =========================================================

  // // =========================== Blur ========================
  START_TIMER (blur)
  if (g_flag)
    {
      gaussian_blur (in_row_pointers, out_row_pointers, width, height,
                     threshold, 100.0);
    }
  STOP_TIMER (blur)
  // // =========================================================

  // // =========================== Rotate ======================
  START_TIMER (rotate)
  if (r_flag)
    {
      rotate_90 (in_row_pointers, out_row_pointers, width, height);
    }
  STOP_TIMER (rotate)
  // // =========================================================

  // ============================== BG remove ===================
  START_TIMER (background)
  if (b_flag)
    {
      background_removal (in_row_pointers, out_row_pointers, width, height,
                          threshold);
    }
  STOP_TIMER (background)
  // ============================================================

  // // =========================== Sort ========================
  START_TIMER (sort)
  if (s_flag)
    {
      pixel_sort (in_row_pointers, out_row_pointers, width, height, threshold);
    }
  STOP_TIMER (sort)
  // =========================================================

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
          GET_TIMER (read), GET_TIMER (background), GET_TIMER (grey),
          GET_TIMER (blur), GET_TIMER (sort), GET_TIMER (rotate), GET_TIMER (save));
  for (png_uint_32 i = 0; i < height; i++)
    {
      free (in_row_pointers[i]);
      if (!r_flag) {
        free (out_row_pointers[i]);
      }
    }
  free (in_row_pointers);
  if (!r_flag) {
    free (out_row_pointers);
  }
  return 0;
}
