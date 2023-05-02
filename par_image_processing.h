#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <png.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>


// Function prototypes

__global__
void rotate_90(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height);

void usage();

int get_next_non_white_x(png_bytep *row_pixels, int x, int width, int white_threshold);

int get_next_white_x(png_bytep *row_pixels, int x, int width, int white_threshold);

void sort_pixels_by_brightness(png_bytep *unsorted, png_bytep *sorted, int sorting_length);

int get_brightness(png_bytep pixel);

void pixel_sort(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height, int threshold);

double *create_gaussian_kernel(int radius, double sigma);

void gaussian_blur(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height, int radius, double sigma);

__global__
void greyscale(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height);

__global__
void removal (png_bytepp in_row_pointers, png_bytepp out_row_pointers, png_bytepp mid_row_pointers,
          int width, int height, int target_x, int target_y, int threshold, char removal_type);

__global__ void
median (png_bytepp in_row_pointers, png_bytepp out_row_pointers, int width, int height);


// ============================= Rotate ==================================

/**
 * Rotates the provided image 90 degrees to the right
 */
__global__
void
rotate_90(png_bytep * in_row_pointers, png_bytep * out_row_pointers, int width,
  int height) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int y = index; y < height; y+=stride) {
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
  printf("\t-d\tdesaturate <threshold>\n");
  printf("\t-g\tgaussian blur <threshold>\n");
  printf("\t-r\trotate\n");
  printf("\t-b\tbackground removal <threshold>\n");
  printf("\t-a\tbackground removal averaging <threshold>\n");
  printf("\t-f\tforeground removal <threshold>\n");
  printf("\t-t\ttarget removal <threshold> <target-x> <target-y>\n");
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

__global__
void
greyscale(png_bytep * in_row_pointers, png_bytep * out_row_pointers, int width,
  int height) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int y = index; y < height; y += stride) {
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


// ============================== Removal ===================================

__global__ void
removal (png_bytepp in_row_pointers, png_bytepp out_row_pointers, png_bytepp mid_row_pointers,
  int width, int height, int target_x, int target_y, int threshold, char removal_type)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  int stride = blockDim.x * gridDim.x;
  
  if (index_x >= width * height)
  {
    return;
  }

  // get the pixel coordinate
  png_bytep back_pixel = &in_row_pointers[target_x][target_y * 4];

  for (int i = index_y * blockDim.x * gridDim.x + index_x; 
          i < width * height; i += blockDim.y * gridDim.y * stride)
    {
      int x = i % width;
      int y = i / width;

      // Background removal technique inspired by Ian Chu Te
      // https://www.kaggle.com/code/ianchute/background-removal-cieluv-color-thresholding/notebook

      png_bytep in_pixel = &in_row_pointers[y][x * 4];
      png_bytep mid_pixel = &mid_row_pointers[y][x * 4];
      
      // create grey scale pixel colors
      float back_r = back_pixel[0];
      float back_g = back_pixel[1];
      float back_b = back_pixel[2];

      // get mean of the red pixel
      float red_mean = (back_r + in_pixel[0]) / 2;

      // diff the colors of current pixel and the reference pixel
      float diff_r = abs(in_pixel[0] - back_r);
      float diff_g = abs(in_pixel[1] - back_g);
      float diff_b = abs(in_pixel[2] - back_b);

      // Compare the colors with a distance calculation
      // https://www.compuphase.com/cmetric.htm
      float distance = sqrt((2 + (red_mean/256)) * pow(diff_r, 2) + 4 * pow(diff_g, 2) + 
                              (2 + ((255 - red_mean)/256)) * pow(diff_b, 2));

      if (in_pixel[3] != 0)
      {
        if (distance < (float) threshold)
        {
          if (removal_type == 'b')
          { 
            mid_pixel[0] = 0;
            mid_pixel[1] = 0;
            mid_pixel[2] = 0;
            mid_pixel[3] = 0;
          }
          else
          {
            mid_pixel[0] = in_pixel[0];
            mid_pixel[1] = in_pixel[1];
            mid_pixel[2] = in_pixel[2];
            mid_pixel[3] = in_pixel[3];
          }
        }
        else
        {
          if (removal_type == 'b')
          {
            mid_pixel[0] = in_pixel[0];
            mid_pixel[1] = in_pixel[1];
            mid_pixel[2] = in_pixel[2];
            mid_pixel[3] = in_pixel[3];
          }
          else
          {
            mid_pixel[0] = 0;
            mid_pixel[1] = 0;
            mid_pixel[2] = 0;
            mid_pixel[3] = 0;
          }
        }
      }
    }

    // Perform median filtering using the same local pixel values defined above
    // first, synchronize threads
    __syncthreads();

    int passes = 0;
    // double pass for CUDA implementation cleans up banding noise
    while (passes < 2)
    {
        for (int i = index_y * blockDim.x * gridDim.x + index_x; 
            i < width * height; i += blockDim.y * gridDim.y * stride)
      {
        int x = i % width;
        int y = i / width;

        // storage of surrounding pixel colors
        // 5x5 median grid
        short local_colors[25][3];
      
      // get the surrounding pixels colors and store them in the local pixel arrays
        for (int z = 0; z < 25; z++)
        {
          int x_offset = z % 5;
          int y_offset = z / 5;

          int src_x = x + x_offset - 2;
          int src_y = y + y_offset - 2;

          if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height)
          {
            png_bytep mid_pixel = &mid_row_pointers[src_y][src_x * 4];
            local_colors[z][0] = mid_pixel[0];
            local_colors[z][1] = mid_pixel[1];
            local_colors[z][2] = mid_pixel[2];
          }
        }
      
      // sort the local pixels. shift all of the colors at once based on their index
      for (int j = 0; j < 25; j++)
      {
        for (int k = j + 1; k < 25; k++)
        {
          float j_luma = (local_colors[j][0] + local_colors[j][1] + local_colors[j][2]) / 3;
          float k_luma = (local_colors[k][0] + local_colors[k][1] + local_colors[k][2]) / 3;

          if (j_luma > k_luma)
          {
            short temp_red = local_colors[j][0];
            short temp_green = local_colors[j][1];
            short temp_blue = local_colors[j][2];

            local_colors[j][0] = local_colors[k][0];
            local_colors[j][1] = local_colors[k][1];
            local_colors[j][2] = local_colors[k][2];

            local_colors[k][0] = temp_red;
            local_colors[k][1] = temp_green;
            local_colors[k][2] = temp_blue;
          }
        }
      }
      png_bytep out_pixel = &out_row_pointers[y][x * 4];
      // set the median of the pixels
      out_pixel[0] = local_colors[12][0];
      out_pixel[1] = local_colors[12][1];
      out_pixel[2] = local_colors[12][2];

            
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
      passes += 1;
      __syncthreads();
    }
}

// ==========================================================================


// ============================== Median Filter =============================

__global__ void
median (png_bytepp in_row_pointers, png_bytepp out_row_pointers, int width, int height)
{

  // Isolated median filter for troubleshooting purposes
  // https://en.wikipedia.org/wiki/Median_filter

  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  int stride = blockDim.x * gridDim.x;
  
  if (index_x >= width * height)
  {
    return;
  }

  for (int i = index_y * blockDim.x * gridDim.x + index_x; 
      i < width * height; i += blockDim.y * gridDim.y * stride)
  {
    int x = i % width;
    int y = i / width;

    // storage of surrounding pixel colors
    // 5x5 median grid
    short local_colors[25][3];
  
    // get the surrounding pixels colors and store them in the local pixel arrays
    for (int z = 0; z < 25; z++)
    {
      int x_offset = z % 5;
      int y_offset = z / 5;

      int src_x = x + x_offset - 2;
      int src_y = y + y_offset - 2;

      if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height)
      {
        png_bytep in_pixel = &in_row_pointers[src_y][src_x * 4];
        local_colors[z][0] = in_pixel[0];
        local_colors[z][1] = in_pixel[1];
        local_colors[z][2] = in_pixel[2];
      }
    }
    // Sort the local pixels by color average
    for (int j = 0; j < 25; j++)
    {
      for (int k = j + 1; k < 25; k++)
      {
        float j_luma = (local_colors[j][0] + local_colors[j][1] + local_colors[j][2]) / 3;
        float k_luma = (local_colors[k][0] + local_colors[k][1] + local_colors[k][2]) / 3;

        if (j_luma > k_luma)
        {
          short temp_red = local_colors[j][0];
          short temp_green = local_colors[j][1];
          short temp_blue = local_colors[j][2];

          local_colors[j][0] = local_colors[k][0];
          local_colors[j][1] = local_colors[k][1];
          local_colors[j][2] = local_colors[k][2];

          local_colors[k][0] = temp_red;
          local_colors[k][1] = temp_green;
          local_colors[k][2] = temp_blue;
        }
      }
    }
    png_bytep out_pixel = &out_row_pointers[y][x * 4];
    // set the median of the pixels
    out_pixel[0] = local_colors[12][0];
    out_pixel[1] = local_colors[12][1];
    out_pixel[2] = local_colors[12][2];
    out_pixel[3] = 255;
  }
}

#endif // IMAGE_PROCESSING_H
