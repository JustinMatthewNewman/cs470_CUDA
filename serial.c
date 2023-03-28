#include "timer.h"
#include <getopt.h>
#include <math.h>
#include <png.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Function prototypes
png_bytep * read_png_file(const char * filename, int * width, int * height,
  size_t * row_size);
void write_png_file(const char * filename, png_bytep * row_pointers, int width,
  int height);
void background_removal(png_bytep * in_row_pointers,
  png_bytep * out_row_pointers, int width, int height,
  int threshold, size_t row_size);
int get_brightness(png_bytep pixel);
void usage();


// ============================= Rotate ==================================

/**
 *
 * Currently broken and unimplemented
 *
 */
void
rotate_90(png_bytep * in_row_pointers, png_bytep * out_row_pointers, int width,
  int height, size_t row_size) {
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



// ========================= Utilities ===================================


void
usage() {
  printf("Usage: ./serial <option(s)> image-file\n");
  printf("Options are:\n");
  printf("\t-d\tdesaturate <threshold>\n");
  printf("\t-g\tgaussian blur <threshold>\n");
  printf("\t-r\trotate\n");
  printf("\t-b\tbackground removal <threshold>\n");
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
  int height, size_t row_size, int threshold) {
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
      free(unsorted);
      free(sorted);

    }
    for (int x = 0; x < width; x++) {
      png_bytep in_pixel = row_pixels[x];
      png_bytep out_pixel = & out_row_pointers[y][x * 4];
      out_pixel[0] = in_pixel[0];
      out_pixel[1] = in_pixel[1];
      out_pixel[2] = in_pixel[2];
      out_pixel[3] = in_pixel[3];
    }
    free(row_pixels);
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
  int width, int height, int radius, double sigma,
  size_t row_size) {
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
  free(kernel);
}
// ==========================================================================


// ================================== Desaturation =============================

void
greyscale(png_bytep * in_row_pointers, png_bytep * out_row_pointers, int width,
  int height, size_t row_size) {
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
  int width, int height, int threshold, size_t row_size) {
  // Convert the input_copy to greyscale
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      png_bytep in_pixel = & in_row_pointers[y][x * 4];
      png_bytep out_pixel = & out_row_pointers[y][x * 4];
      int grey = (in_pixel[0] + in_pixel[1] + in_pixel[2]) / 3;
      // Saturate the input_copy by thresholding
      int grey_threshold = (grey < threshold) ? 0 : grey;
      // Compare absolute difference between input and input_copy.
      // If the difference is less than the threshold, set the output to
      // transparent.
      if (abs(in_pixel[0] - grey_threshold) > threshold) {
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

// ==============================================================================

// ============================== Input / Output ================================


// ============================    READ      =====================================
png_bytep *
  read_png_file(const char * filename, int * width, int * height, size_t * row_size) {
    FILE * fp = fopen(filename, "rb");
    if (!fp) {
      fprintf(stderr, "Error: Unable to open the file %s for reading.\n",
        filename);
      exit(1);
    }
    png_structp png
      = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
      fclose(fp);
      fprintf(stderr, "Error: Unable to create png read structure.\n");
      exit(1);
    }
    png_infop info = png_create_info_struct(png);
    if (!info) {
      fclose(fp);
      png_destroy_read_struct( & png, NULL, NULL);
      fprintf(stderr, "Error: Unable to create png info structure.\n");
      exit(1);
    }
    if (setjmp(png_jmpbuf(png))) {
      fclose(fp);
      png_destroy_read_struct( & png, & info, NULL);
      fprintf(stderr, "Error: Error encountered while reading png file.\n");
      exit(1);
    }
    png_init_io(png, fp);
    png_read_info(png, info);
    * width = png_get_image_width(png, info);
    * height = png_get_image_height(png, info);
    png_read_update_info(png, info);
    * row_size = png_get_rowbytes(png, info);
    png_bytep * row_pointers = (png_bytep * ) malloc(sizeof(png_bytep) * ( * height));
    for (int y = 0; y < * height; y++) {
      row_pointers[y] = (png_byte * ) malloc( * row_size);
    }
    png_read_image(png, row_pointers);
    fclose(fp);
    return row_pointers;
  }
// ====================================================================

// =============================  Write  ==============================
void
write_png_file(const char * filename, png_bytep * row_pointers, int width,
  int height) {
  FILE * fp = fopen(filename, "wb");
  if (!fp) {
    printf("Error: Failed to open file %s\n", filename);
    return;
  }

  png_structp png
    = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) {
    printf("Error: Failed to create PNG write structure\n");
    fclose(fp);
    return;
  }

  png_infop info = png_create_info_struct(png);
  if (!info) {
    printf("Error: Failed to create PNG info structure\n");
    png_destroy_write_struct( & png, NULL);
    fclose(fp);
    return;
  }

  if (setjmp(png_jmpbuf(png))) {
    printf("Error: PNG error during writing image\n");
    png_destroy_write_struct( & png, & info);
    fclose(fp);
    return;
  }

  png_init_io(png, fp);
  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
    PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);
  fclose(fp);
  png_destroy_write_struct( & png, & info);
}
// ===================================================================================


// ===================================================================================

// ================================ Main Function ====================================

// ===================================================================================

int
main(int argc, char * argv[]) {
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
  while ((opt = getopt(argc, argv, "dg:rb:s:")) != -1) {
    switch (opt) {
    case 'd':
      d_flag = true;
      break;
    case 'g':
      g_flag = true;
      threshold = atoi(optarg);
      break;
    case 'r':
      r_flag = true;
      break;
    case 'b':
      b_flag = true;
      threshold = atoi(optarg);
      break;
    case 's':
      s_flag = true;
      threshold = atoi(optarg);
      break;
    default:
      usage();
      return 1;
    }
  }

  // Get the input and output filenames from the command line arguments
  if (optind < argc) {
    input_filename = argv[optind++];
  } else {
    printf("Error: no input image file specified\n");
    return 1;
  }

  if (optind < argc) {
    output_filename = argv[optind];
  } else {
    printf("Error: no output image file specified\n");
    return 1;
  }

  int width, height;
  png_bytep * in_row_pointers;
  size_t row_size;
  START_TIMER(read)
  in_row_pointers = read_png_file(input_filename, & width, & height, & row_size);
  STOP_TIMER(read)
  png_bytep * out_row_pointers = (png_bytep * ) malloc(sizeof(png_bytep) * height);
  for (int y = 0; y < height; y++) {
    out_row_pointers[y] = (png_byte * ) malloc(row_size);
  }
  // Call the selected functions
  // =========================== Grey ========================
  START_TIMER(grey)
  if (d_flag) {
    greyscale(in_row_pointers, out_row_pointers, width, height, row_size);
  }
  STOP_TIMER(grey)
  // =========================================================

  // =========================== Blur ========================
  START_TIMER(blur)
  if (g_flag) {
    gaussian_blur(in_row_pointers, out_row_pointers, width, height,
      threshold, 100.0, row_size);
  }
  STOP_TIMER(blur)
  // =========================================================

  // =========================== Rotate ======================
  START_TIMER(rotate)
  if (r_flag) {
    rotate_90(in_row_pointers, out_row_pointers, width, height, row_size);
  }
  STOP_TIMER(rotate)

  // =========================================================

  START_TIMER(background)
  if (b_flag) {
    background_removal(in_row_pointers, out_row_pointers, width, height,
      threshold, row_size);
  }
  STOP_TIMER(background)
  // =========================== Sort ========================
  START_TIMER(sort)
  if (s_flag) {
    pixel_sort(in_row_pointers, out_row_pointers, width, height, row_size,
      threshold);
  }
  STOP_TIMER(sort)
  // =========================================================

  // Save output image
  START_TIMER(save)
  write_png_file(output_filename, out_row_pointers, width, height);
  STOP_TIMER(save)

  // Display timing results
  printf("READ: %.6f  BACKGROUND: %.6f  GREY: %.6f  BLUR: %.6f  SORT: %.6f  "
    "SAVE: %.6f\n",
    GET_TIMER(read), GET_TIMER(background), GET_TIMER(grey),
    GET_TIMER(blur), GET_TIMER(sort), GET_TIMER(save));

  // Free the memory allocated for row pointers

  for (int y = 0; y < height; y++) {
    free(in_row_pointers[y]);
    if (! r_flag)
      free(out_row_pointers[y]);
  }

  free(in_row_pointers);
  free(out_row_pointers);
  return 0;
}
