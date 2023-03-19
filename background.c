
#include "netpbm.h"
#include "timer.h"
#include <math.h>
#include <stdio.h>

/**
 * Converts a given PPM pixel array into greyscale.
 * Uses the following formula for the conversion:
 *      V = (R * 0.21) + (G * 0.72) + (B * 0.07)
 * Where V is the grey value and RGB are the red, green, and blue values,
 * respectively.
 */
void
color_to_grey (pixel_t *in, pixel_t *out, int width, int height)
{
  for (int i = 0; i < width * height; i++)
    {
      rgb_t v = (rgb_t)round (in[i].red * 0.21 + in[i].green * 0.72
                              + in[i].blue * 0.07);
      out[i].red = out[i].green = out[i].blue = v;
    }
}

void
background_removal (ppm_t *input_copy, pixel_t *in, pixel_t *out, int width,
                    int height, int threshold)
{

  // Convert the input_copy to greyscale
  color_to_grey (in, input_copy->pixels, width, height);

  // Saturate the input_copy by thresholding
  for (int i = 0; i < width * height; i++)
    {
      if (input_copy->pixels[i].red < threshold)
        {
          input_copy->pixels[i].red = 0;
          input_copy->pixels[i].green = 0;
          input_copy->pixels[i].blue = 0;
        }
    }
  // Compare absolute difference between input and input_copy.
  // If the difference is less than the threshold, set the output to white.
  for (int i = 0; i < width * height; i++)
    {
      out[i].red = in[i].red;
      out[i].green = in[i].green;
      out[i].blue = in[i].blue;

      if (abs (in[i].red - input_copy->pixels[i].red) > threshold)
        {
          out[i].red = 255;
          out[i].green = 255;
          out[i].blue = 255;
        }
    }
}

void
background_removal_averaging (pixel_t *in, pixel_t *out, int width, int height,
                              int threshold)
{
  // Get the most common color in the image 
  // of all the pixels in a 75x75 window
  // in top left corner of the image.
  int r = 0;
  int g = 0;
  int b = 0;
  for (int i = 0; i < 75; i++)
    {
      for (int j = 0; j < 75; j++)
        {
          r += in[i * width + j].red;
          g += in[i * width + j].green;
          b += in[i * width + j].blue;
        }
    }
  r /= 5625;
  g /= 5625;
  b /= 5625;

  // Compare absolute difference between input and the most common color.
  // If the difference is less than the threshold, set the output to white.
  // Otherwise, set the output to the input.
  for (int i = 0; i < width * height; i++)
    {
      if (abs (in[i].red - r) < threshold && abs (in[i].green - g) < threshold
          && abs (in[i].blue - b) < threshold)
        {
          out[i].red = 255;
          out[i].green = 255;
          out[i].blue = 255;
        }
      else
        {
          out[i].red = in[i].red;
          out[i].green = in[i].green;
          out[i].blue = in[i].blue;
        }
    }
}

int
main (int argc, char *argv[])
{
  if (argc != 6)
    {
      printf ("Usage: %s <infile> <outfile> <width> <height> <threshold>\n",
              argv[0]);
      exit (EXIT_FAILURE);
    }

  char *in = argv[1];
  char *out = argv[2];
  int width = strtol (argv[3], NULL, 10), height = strtol (argv[4], NULL, 10);
  long total_pixels = width * height;
  int threshold = strtol (argv[5], NULL, 10);

  // Allocate memory for images
  ppm_t *input
      = (ppm_t *)malloc (sizeof (ppm_t) + (total_pixels * sizeof (pixel_t)));
  ppm_t *output
      = (ppm_t *)malloc (sizeof (ppm_t) + (total_pixels * sizeof (pixel_t)));
  ppm_t *copy
      = (ppm_t *)malloc (sizeof (ppm_t) + (total_pixels * sizeof (pixel_t)));

  // Read image
  START_TIMER (read)
  read_in_ppm (in, input);
  read_in_ppm (in, copy);
  STOP_TIMER (read)

  // Verify dimensions
  if (width != input->width || height != input->height)
    {
      printf ("ERROR: given dimensions do not match file\n");
      exit (EXIT_FAILURE);
    }

  // Copy header to output image
  copy_header_ppm (input, output);

  // Convert to greyscale
  START_TIMER (background_removal)
  background_removal(copy, input->pixels, output->pixels, width, height, threshold);
  // background_removal_averaging (input->pixels, output->pixels, width, height, threshold);
  STOP_TIMER (background_removal)

  // Swap buffers in preparation for blurring
  memcpy (input->pixels, output->pixels, total_pixels * sizeof (pixel_t));

  // Save output image
  START_TIMER (save)
  write_out_ppm (out, output);
  STOP_TIMER (save)

  // Display timing results
  printf ("READ: %.6f  BACKGROUND: %.6f  SAVE: %.6f\n", GET_TIMER (read),
          GET_TIMER (background_removal), GET_TIMER (save));

  free (input);
  free (output);
  free (copy);

  return 0;
}
