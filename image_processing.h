#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <png.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>


// Function prototypes

void rotate_90(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height);

void usage();

int get_next_non_white_x(png_bytep *row_pixels, int x, int width, int white_threshold);

int get_next_white_x(png_bytep *row_pixels, int x, int width, int white_threshold);

void sort_pixels_by_brightness(png_bytep *unsorted, png_bytep *sorted, int sorting_length);

int get_brightness(png_bytep pixel);

void pixel_sort(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height, int threshold);

double *create_gaussian_kernel(int radius, double sigma);

void gaussian_blur(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height, int radius, double sigma);

void greyscale(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height);

void background_removal(png_bytep *in_row_pointers, png_bytep *out_row_pointers, 
                        int width, int height, int threshold);

void background_removal_averaging(png_bytep *in_row_pointers, png_bytep *out_row_pointers, 
                                  int width, int height, int threshold);

void foreground_removal(png_bytep *in_row_pointers, png_bytep *out_row_pointers, 
                        int width, int height, int threshold);

void target_removal(png_bytep *in_row_pointers, png_bytep *out_row_pointers, 
                int width, int height, int threshold, int target_x, int target_y);


// ============================= Rotate ==================================

/**
 * Rotates the provided image 90 degrees to the right
 */
void
rotate_90(png_bytep * in_row_pointers, png_bytep * out_row_pointers, int width,
  int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      png_bytep in_pixel = & in_row_pointers[y][x * 4];
      png_bytep out_pixel = & out_row_pointers[x][(height - y - 1) * 4];
      out_pixel[0] = in_pixel[0];
      out_pixel[1] = in_pixel[1];
      out_pixel[2] = in_pixel[2];
      out_pixel[3] = in_pixel[3];
    }
  }
  
}

// =======================================================================

// =======================================================================



// ========================= Utilities ===================================


void
usage() {
  printf("Usage: ./serial <option(s)> image-file\n");
  printf("Options are:\n");
  printf("\t-d\tdesaturate\n");
  printf("\t-r\trotate\n");
  printf("\t-b\tbackground removal <threshold>\n");
  #printf("\t-t\ttarget removal <threshold> <target-x> <target-y>\n");
  printf("\t-s\tsorting <threshold>\n");
}

// Helper function to get the first non-white x value in the row
int
get_next_non_white_x(png_bytep * row_pixels, int x, int width,
  int white_threshold) {
  while (true) {
    png_bytep pixel = row_pixels[x];
    int brightness = get_brightness(pixel);
    if (brightness < white_threshold) {
      return x;
    }
    x++;
    if (x >= width) {
      return -1;
    }
  }
}

// Helper function to get the next white x value in the row
int get_next_white_x(png_bytep * row_pixels, int x, int width, int white_threshold) {
  x++;
  while (true) {
    if (x >= width) {
      return width - 1;
    }

    png_bytep pixel = row_pixels[x];
    int brightness = get_brightness(pixel);

    if (brightness >= white_threshold) {
      return x - 1;
    }
    x++;
  }
}


/**
 * Sort Pixels by brightness
 */
void
sort_pixels_by_brightness(png_bytep * unsorted, png_bytep * sorted,
  int sorting_length) {
  for (int i = 0; i < sorting_length - 1; i++) {
    for (int j = 0; j < sorting_length - i - 1; j++) {
      int brightness1 = get_brightness(unsorted[j]);
      int brightness2 = get_brightness(unsorted[j + 1]);
      if (brightness1 > brightness2) {
        png_bytep temp = unsorted[j];
        unsorted[j] = unsorted[j + 1];
        unsorted[j + 1] = temp;
      }
    }
  }

  for (int i = 0; i < sorting_length; i++) {
    sorted[i] = unsorted[i];
  }
}

// Helper function to get the brightness of a pixel
int
get_brightness(png_bytep pixel) {
  int r = pixel[0];
  int g = pixel[1];
  int b = pixel[2];
  return (r + g + b) / 3;
}
// ===============================================================




// ======================== Pixel Sort  ===================================
void
pixel_sort(png_bytep * in_row_pointers, png_bytep * out_row_pointers, int width,
  int height, int threshold) {
  for (int y = 0; y < height; y++) {
    png_bytep * row_pixels = (png_bytep * ) malloc(sizeof(png_bytep) * width);
    for (int x = 0; x < width; x++) {
      row_pixels[x] = & in_row_pointers[y][x * 4];
    }
    int x_start = 0;
    int x_end = 0;
    while (x_end < width - 1) {
      x_start = get_next_non_white_x(row_pixels, x_start, width, threshold);
      x_end = get_next_white_x(row_pixels, x_start, width, threshold);
           // printf("segfault is here\n");
      if (x_start < 0)
        break;
      int sorting_length = x_end - x_start;
      png_bytep * unsorted = (png_bytep * ) malloc(sizeof(png_bytep) * sorting_length);
      png_bytep * sorted = (png_bytep * ) malloc(sizeof(png_bytep) * sorting_length);
      for (int i = 0; i < sorting_length; i++) {
        unsorted[i] = row_pixels[x_start + i];
      }
      sort_pixels_by_brightness(unsorted, sorted, sorting_length);
      for (int i = 0; i < sorting_length; i++) {
        row_pixels[x_start + i] = sorted[i];
      }
      x_start = x_end + 1;
      // free(unsorted);
      // free(sorted);
    }
    for (int x = 0; x < width; x++) {
      png_bytep in_pixel = row_pixels[x];
      png_bytep out_pixel = & out_row_pointers[y][x * 4];
      out_pixel[0] = in_pixel[0];
      out_pixel[1] = in_pixel[1];
      out_pixel[2] = in_pixel[2];
      out_pixel[3] = in_pixel[3];
    }
    //free(row_pixels);
  }
}
// ================================================================



// ============================== Blur ====================================
double *
  create_gaussian_kernel(int radius, double sigma) {
    int size = 2 * radius + 1;
    double * kernel = (double * ) malloc(size * size * sizeof(double));
    double sum = 0;
    double two_sigma_sq = 2.0 * sigma * sigma;
    for (int y = -radius; y <= radius; y++) {
      for (int x = -radius; x <= radius; x++) {
        int index = (y + radius) * size + (x + radius);
        kernel[index] = exp(-(x * x + y * y) / two_sigma_sq) / (M_PI * two_sigma_sq);
        sum += kernel[index];
      }
    }
    for (int i = 0; i < size * size; i++) {
      kernel[i] /= sum;
    }
    return kernel;
  }

void
gaussian_blur(png_bytep * in_row_pointers, png_bytep * out_row_pointers,
  int width, int height, int radius, double sigma) {
  double * kernel = create_gaussian_kernel(radius, sigma);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
      for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
          int src_x = x + dx;
          int src_y = y + dy;
          if (src_x >= 0 && src_x < width && src_y >= 0 &&
            src_y < height) {
            int kernel_index = (dy + radius) * (2 * radius + 1) + (dx + radius);
            double kernel_value = kernel[kernel_index];
            png_bytep in_pixel = & in_row_pointers[src_y][src_x * 4];
            sum_r += in_pixel[0] * kernel_value;
            sum_g += in_pixel[1] * kernel_value;
            sum_b += in_pixel[2] * kernel_value;
            sum_a += in_pixel[3] * kernel_value;
          }
        }
      }
      png_bytep out_pixel = & out_row_pointers[y][x * 4];
      out_pixel[0] = (png_byte) round(sum_r);
      out_pixel[1] = (png_byte) round(sum_g);
      out_pixel[2] = (png_byte) round(sum_b);
      out_pixel[3] = (png_byte) round(sum_a);
    }
  }
  //free(kernel);
}
// ==========================================================================


// ================================== Desaturation =============================

void
greyscale(png_bytep * in_row_pointers, png_bytep * out_row_pointers, int width,
  int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      png_bytep in_pixel = & in_row_pointers[y][x * 4];
      png_bytep out_pixel = & out_row_pointers[y][x * 4];
      int grey = (in_pixel[0] + in_pixel[1] + in_pixel[2]) / 3;
      out_pixel[0] = grey;
      out_pixel[1] = grey;
      out_pixel[2] = grey;
      // Preserve the alpha channel
      out_pixel[3] = in_pixel[3];
    }
  }
}
// ==========================================================================


// ============================== Background Removal ========================

void
background_removal(png_bytep * in_row_pointers, png_bytep * out_row_pointers,
  int width, int height, int threshold) {
  // Convert the input_copy to greyscale

  // get the pixel at coordinate 10, 10
  png_bytep back_pixel = &in_row_pointers[15][15 * 4];


  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      png_bytep in_pixel = & in_row_pointers[y][x * 4];
      png_bytep out_pixel = & out_row_pointers[y][x * 4];
  
      // Background removal technique inspired by Ian Chu Te
      // https://www.kaggle.com/code/ianchute/background-removal-cieluv-color-thresholding/notebook

      // create grey scale pixel colors
      float back_r = back_pixel[0];
      float back_g = back_pixel[1];
      float back_b = back_pixel[2];

      // get mean of the red pixel
      float red_mean = (back_r + in_pixel[0]) / 2;

      float diff_r = abs(in_pixel[0] - back_r);
      float diff_g = abs(in_pixel[1] - back_g);
      float diff_b = abs(in_pixel[2] - back_b);

      // https://www.compuphase.com/cmetric.htm
      float distance = sqrt((2 + (red_mean/256)) * pow(diff_r, 2) + 4 * pow(diff_g, 2) + 
                              (2 + ((255 - red_mean)/256)) * pow(diff_b, 2));
                           
      if (distance < (float) threshold) {
        out_pixel[0] = 0;
        out_pixel[1] = 0;
        out_pixel[2] = 0;
        out_pixel[3] = 0; // Set alpha channel to 0 for transparency
      } else {
        out_pixel[0] = in_pixel[0];
        out_pixel[1] = in_pixel[1];
        out_pixel[2] = in_pixel[2];
        out_pixel[3] = in_pixel[3]; // Preserve the alpha channel
      }
    }
  }
  
  // int passes = 0;

  // while (passes < 2)
  // {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {

      // get the median of the pixels using the local pixel values 
      // get the surrounding pixels
      short local_red[25];
      short local_green[25];
      short local_blue[25];

      for (int j = 0; j < 25; j++)
      {
          int x_offset = j % 5;
          int y_offset = j / 5;

          int src_x = x + x_offset - 2;
          int src_y = y + y_offset - 2;

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height)
        {
          png_bytep in_pixel = &out_row_pointers[src_y][src_x * 4];
          local_red[j] = in_pixel[0];
          local_green[j] = in_pixel[1];
          local_blue[j] = in_pixel[2];
        }
      }

      // sort the local pixels. shift all of the colors at once based on their index
      for (int j = 0; j < 25; j++)
      {
        for (int k = j + 1; k < 25; k++)
        {
          float j_avg = (local_red[j] + local_green[j] + local_blue[j]) / 3;
          float k_avg = (local_red[k] + local_green[k] + local_blue[k]) / 3;
          if (j_avg > k_avg)
          {
            short temp_red = local_red[j];
            local_red[j] = local_red[k];
            local_red[k] = temp_red;

            short temp_green = local_green[j];
            local_green[j] = local_green[k];
            local_green[k] = temp_green;

            short temp_blue = local_blue[j];
            local_blue[j] = local_blue[k];
            local_blue[k] = temp_blue;
          }
        }
      }
      // get the pixel
      png_bytep out_pixel = &out_row_pointers[y][x * 4];
      // set the median of the pixels
      out_pixel[0] = local_red[12];
      out_pixel[1] = local_green[12];
      out_pixel[2] = local_blue[12];
      
      // if the pixel used to be a background pixel but is now a foreground pixel, 
      // set the alpha channel to 255
      if (out_pixel[0] != 0 && out_pixel[1] != 0 && out_pixel[2] != 0)
      {
        out_pixel[3] = 255;
      }
      else
      {
        out_pixel[3] = 0;
      }
    }
  }
  //   passes += 1;
  // }
  
}

// ==========================================================================


// ==================== Background Removal Averaging =========================

void
background_removal_averaging(png_bytep * in_row_pointers, png_bytep * out_row_pointers,
  int width, int height, int threshold) {
  // Average a 75x75 pixel grid to get the background color
  int background_r = 0, background_g = 0, background_b = 0;
  int grid_size = 75;
  // First, get the average color of the top left corner
  for (int y = 0; y < grid_size; y++) {
    for (int x = 0; x < grid_size; x++) {
      png_bytep in_pixel = & in_row_pointers[y][x * 4];
      background_r += in_pixel[0];
      background_g += in_pixel[1];
      background_b += in_pixel[2];
    }
  }
  background_r /= grid_size * grid_size;
  background_g /= grid_size * grid_size;
  background_b /= grid_size * grid_size;

  // Compare the average color of the top left corner to the rest of the image
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      png_bytep in_pixel = & in_row_pointers[y][x * 4];
      png_bytep out_pixel = & out_row_pointers[y][x * 4];
      // Compare absolute difference between input and input_copy.
      if (abs(in_pixel[0] - background_r) < threshold &&
        abs(in_pixel[1] - background_g) < threshold &&
        abs(in_pixel[2] - background_b) < threshold) {
        out_pixel[0] = 0;
        out_pixel[1] = 0;
        out_pixel[2] = 0;
        out_pixel[3] = 0; // Set alpha channel to 0 for transparency
      } else {
        out_pixel[0] = in_pixel[0];
        out_pixel[1] = in_pixel[1];
        out_pixel[2] = in_pixel[2];
        out_pixel[3] = in_pixel[3]; // Preserve the alpha channel
      }
    }
  }
}

// ==========================================================================

// =========================== Foreground Removal ===========================

void
foreground_removal(png_bytep * in_row_pointers, png_bytep * out_row_pointers,
  int width, int height, int threshold) {
  // Convert the input_copy to greyscale
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      png_bytep in_pixel = & in_row_pointers[y][x * 4];
      png_bytep out_pixel = & out_row_pointers[y][x * 4];
      int grey = (in_pixel[0] + in_pixel[1] + in_pixel[2]) / 3;
      // Saturate the input_copy by thresholding
      int grey_threshold = (grey < threshold) ? 0 : grey;

      if (abs(in_pixel[0] - grey_threshold) < threshold)  {
        out_pixel[0] = in_pixel[0];
        out_pixel[1] = in_pixel[1];
        out_pixel[2] = in_pixel[2];
        out_pixel[3] = in_pixel[3]; // Preserve the alpha channel
      } else {
        out_pixel[0] = 0;
        out_pixel[1] = 0;
        out_pixel[2] = 0;
        out_pixel[3] = 0; // Set alpha channel to 0 for transparency
      }
    }
  }
}

// ==========================================================================

// ========================== Targeting =====================================

void
target_removal(png_bytep * in_row_pointers, png_bytep * out_row_pointers,
  int width, int height, int threshold, int target_x, int target_y) {

    // Average a 50x50 pixel grid around the target to get the background color
    int background_r = 0, background_g = 0, background_b = 0;
    int grid_size = 25;

    // First, get the average color of the target area
    for (int y = target_y - grid_size / 2; y < target_y + grid_size / 2; y++) {
      for (int x = target_x - grid_size / 2; x < target_x + grid_size / 2; x++) {
        png_bytep in_pixel = & in_row_pointers[y][x * 4];
        background_r += in_pixel[0];
        background_g += in_pixel[1];
        background_b += in_pixel[2];
      }
    }

    background_r /= grid_size * grid_size;
    background_g /= grid_size * grid_size;
    background_b /= grid_size * grid_size;

    // Compare the average color of the target area to the rest of the image
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        png_bytep in_pixel = & in_row_pointers[y][x * 4];
        png_bytep out_pixel = & out_row_pointers[y][x * 4];

        if (abs(in_pixel[0] - background_r) < threshold &&
          abs(in_pixel[1] - background_g) < threshold &&
          abs(in_pixel[2] - background_b) < threshold) {
          out_pixel[0] = 0;
          out_pixel[1] = 0;
          out_pixel[2] = 0;
          out_pixel[3] = 0; // Set alpha channel to 0 for transparency
        } else {
          out_pixel[0] = in_pixel[0];
          out_pixel[1] = in_pixel[1];
          out_pixel[2] = in_pixel[2];
          out_pixel[3] = in_pixel[3]; // Preserve the alpha channel
        }
      }
    }
  }

#endif // IMAGE_PROCESSING_H
