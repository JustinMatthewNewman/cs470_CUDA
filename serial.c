#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

#include <math.h>
#include "netpbm.h"
#include "timer.h"

// Function declarations
void rotate(const char* filename);
void remove_background(const char* filename, int threshold);
void pixel_sorting(const char* filename, int threshold);


// The size of the blur box
#define BLUR_SIZE 6

/**
 * Converts a given PPM pixel array into greyscale.
 * Uses the following formula for the conversion:
 *      V = (R * 0.21) + (G * 0.72) + (B * 0.07)
 * Where V is the grey value and RGB are the red, green, and blue values, respectively.
 */
void color_to_grey(pixel_t *in, pixel_t *out, int width, int height)
{
    for (int i = 0; i < width * height; i++) {
        rgb_t v = (rgb_t) round(
              in[i].red * 0.21
            + in[i].green * 0.72
            + in[i].blue * 0.07);
        out[i].red = out[i].green = out[i].blue = v;
    }
}

/**
 * Blurs a given PPM pixel array through box blurring.
 * Strength of blur can be adjusted through the BLUR_SIZE value.
 */
void blur(pixel_t *in, pixel_t *out, int width, int height)
{
    for (int i = 0; i < width * height; i++) {
        int row = i / width;
        int col = i % width;
        float avg_red = 0, avg_green = 0, avg_blue = 0;
        int pixel_count = 0;

        for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; blur_row++) {
            for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; blur_col++) {
                int curr_row = row + blur_row;
                int curr_col = col + blur_col;

                if (curr_row > -1 && curr_row < height && curr_col > -1 && curr_col < width) {
                    int curr_index = curr_row * width + curr_col;
                    avg_red += in[curr_index].red;
                    avg_green += in[curr_index].green;
                    avg_blue += in[curr_index].blue;
                    pixel_count++;
                }
            }
        }

        pixel_t result = {.red   = (rgb_t) lroundf(avg_red   / pixel_count),
                          .green = (rgb_t) lroundf(avg_green / pixel_count),
                          .blue  = (rgb_t) lroundf(avg_blue  / pixel_count)
                         };
        out[i] = result;
    }
}



// Main function
int main(int argc, char* argv[]) {

    char *in = argv[1];
    char *out = argv[2];
    int width = strtol(argv[3], NULL, 10),
        height = strtol(argv[4], NULL, 10);
    long total_pixels = width * height;

    // Allocate memory for images
    ppm_t *input  = (ppm_t*)malloc(sizeof(ppm_t) + (total_pixels * sizeof(pixel_t)));
    ppm_t *output = (ppm_t*)malloc(sizeof(ppm_t) + (total_pixels * sizeof(pixel_t)));

    // Read image
    START_TIMER(read)
    read_in_ppm(in, input);
    STOP_TIMER(read)

    // Verify dimensions
    if(width != input->width || height != input->height) {
        printf("ERROR: given dimensions do not match file\n");
        exit(EXIT_FAILURE);
    }

    // Copy header to output image
    copy_header_ppm(input, output);


    int opt;
    int threshold;
    bool d_flag = false;
    bool g_flag = false;
    bool r_flag = false;
    bool b_flag = false;
    bool s_flag = false;
    const char* filename = NULL;

    

    // Parse the command line options
    while ((opt = getopt(argc, argv, "d:g:rb:s:")) != -1) {
        switch (opt) {
        case 'd':
            d_flag = true;
            threshold = atoi(optarg);
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
            printf("Usage: %s <option(s)> image-file\n", argv[0]);
            printf("Options are:\n");
            printf("\t-d\tdesaturate <threshold>\n");
            printf("\t-g\tgaussian blur <threshold>\n");
            printf("\t-r\trotate\n");
            printf("\t-b\tbackground removal <threshold>\n");
            printf("\t-s\tsorting <threshold>\n");
            return 1;
        }
    }

    // Get the filename from the command line arguments
    if (optind < argc) {
        filename = argv[optind];
    }
    else {
        printf("Error: no image file specified\n");
        return 1;
    }

    // Call the selected functions
    // =========================== Grey ========================
    START_TIMER(grey)
    if (d_flag) {
            color_to_grey(input->pixels, output->pixels, width, height);
            if (g_flag) {
                memcpy(input->pixels, output->pixels, total_pixels * sizeof(pixel_t));
            }
            
    }
    STOP_TIMER(grey)
    // =========================================================


    // =========================== Blur ========================
    START_TIMER(blur)
    if (g_flag) {
        blur(input->pixels, output->pixels, width, height);
            if (d_flag) {
                memcpy(input->pixels, output->pixels, total_pixels * sizeof(pixel_t));
            }
    }
    STOP_TIMER(blur)
    // =========================================================

    if (r_flag) {
        rotate(filename);
    }
    if (b_flag) {
        remove_background(filename, threshold);
    }
    if (s_flag) {
        pixel_sorting(filename, threshold);
    }

    // Save output image
    START_TIMER(save)
    write_out_ppm(out, output);
    STOP_TIMER(save)

    // Display timing results
    printf("READ: %.6f  GREY: %.6f  BLUR: %.6f  SAVE: %.6f\n",
           GET_TIMER(read), GET_TIMER(grey), GET_TIMER(blur), GET_TIMER(save));

    free(input);
    free(output);

    return 0;

}

// Function definitions

void rotate(const char* filename) {
    // Implement the rotate function here
    printf("Rotate %s\n", filename);
}

void remove_background(const char* filename, int threshold) {
    // Implement the remove_background function here
    printf("Remove background from %s with threshold %d\n", filename, threshold);
}

void pixel_sorting(const char* filename, int threshold) {
    // Implement the pixel_sorting function here
    printf("Pixel sorting %s with threshold %d\n", filename, threshold);
}
