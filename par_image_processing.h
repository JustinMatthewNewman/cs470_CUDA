#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <png.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Function prototypes

__global__
void rotate_90(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height);

void usage();

__device__
int get_next_non_white_x(png_bytep *row_pixels, int x, int width, int white_threshold);

__device__
int get_next_white_x(png_bytep *row_pixels, int x, int width, int white_threshold);

void sort_pixels_by_brightness(png_bytep *unsorted, png_bytep *sorted, int sorting_length);

int get_brightness(png_bytep pixel);

__global__
void pixel_sort_kernel(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height, int threshold);

__device__
void swap_pixels(png_bytep px1, png_bytep px2);

__device__
float compute_brightness(png_bytep px);

__global__
void greyscale(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height);

__global__
void removal (png_bytepp in_row_pointers, png_bytepp out_row_pointers, png_bytepp mid_row_pointers,
          int width, int height, int target_x, int target_y, int threshold, char removal_type);

__global__ void
median (png_bytepp in_row_pointers, png_bytepp out_row_pointers, int width, int height);


// ================================================================




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
  
  for (int y = index; y < height; y += stride) {
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
  printf("Usage: ./par <option(s)> image-file\n");
  printf("Options are:\n");
  printf("\t-d\tdesaturate <threshold>\n");
  printf("\t-g\tgaussian blur <threshold>\n");
  printf("\t-r\trotate\n");
  printf("\t-b\tback-removal <x> <y> <threshold>\n");
  printf("\t-f\tfore-removal <x> <y> <threshold>\n");
  printf("\t-s\tsorting <threshold>\n");
  printf("\t-m\tmedian <threshold>\n");
}


// cuda utils

__device__
float compute_brightness(png_bytep pixel) {
      int r = pixel[0];
      int g = pixel[1];
      int b = pixel[2];
      return (r + g + b) / 3;
}

__device__
void swap_pixels(png_bytep px1, png_bytep px2) {
    for (int i = 0; i < 4; i++) {
        png_byte temp = px1[i];
        px1[i] = px2[i];
        px2[i] = temp;
    }
}

__device__
int get_next_non_white_x(png_bytep row, int x, int width, int white_threshold) {
    while (true) {
        int brightness = compute_brightness(row + x * 4);
        if (brightness < white_threshold) {
            return x;
        }
        x++;
        if (x >= width) {
            return -1;
        }
    }
}

__device__
int get_next_white_x(png_bytep row, int x, int width, int white_threshold) {
    x++;
    while (true) {
        if (x >= width) {
            return width - 1;
        }

        int brightness = compute_brightness(row + x * 4);

        if (brightness >= white_threshold) {
            return x - 1;
        }
        x++;
    }
}

__global__
void pixel_sort_kernel(png_bytep *in_row_pointers, png_bytep *out_row_pointers, int width, int height, int threshold) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < height) {
        png_bytep in_row = in_row_pointers[row_idx];
        png_bytep out_row = out_row_pointers[row_idx];

        // Copy input row to output row
        for (int x = 0; x < width * 4; x++) {
            out_row[x] = in_row[x];
        }

        int x_start = 0;
        int x_end = 0;
        while (x_end < width - 1) {
            x_start = get_next_non_white_x(out_row, x_start, width, threshold);
            x_end = get_next_white_x(out_row, x_start, width, threshold);

            if (x_start < 0)
                break;

            int sorting_length = x_end - x_start + 1;
            for (int i = 0; i < sorting_length - 1; i++) {
                int max_idx = i;
                float max_brightness = compute_brightness(out_row + (x_start + i) * 4);
                for (int j = i + 1; j < sorting_length; j++) {
                    float brightness = compute_brightness(out_row + (x_start + j) * 4);
                    if (brightness < max_brightness) {
                        max_brightness = brightness;
                        max_idx = j;
                    }
                }
                swap_pixels(out_row + (x_start + i) * 4, out_row + (x_start + max_idx) * 4);
            }
            x_start = x_end + 1;
        }
    }
}

// =========================================================================

// ================================== Desaturation =========================

__global__
void
greyscale(png_bytep * in_row_pointers, png_bytep * out_row_pointers, int width,
  int height) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < width * height; i += stride)
   {
      int x = i % width;
      int y = i / width;
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
