/**
 * CS 470 CUDA Lab
 *
 * Originally written by William Lovo in Spring 2019 as a research project.
 */

#ifndef CUDA_LAB_NETPBM_H
#define CUDA_LAB_NETPBM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Maximum number of pixels written to a line
#define MAX_PIXELS_PER_LINE 60

// RGB value
typedef unsigned short rgb_t;

// Color pixel
typedef struct pixel {
    rgb_t red;
    rgb_t green;
    rgb_t blue;
} pixel_t;

// Portable PixMap structure
typedef struct ppm {
    char magic[4];
    int width;
    int height;
    short max_val;
    pixel_t pixels[];
} ppm_t;

/**
 * Reads image data from a PPM file.
 */
void read_in_ppm(char *filename, ppm_t *image)
{
    FILE *fin = fopen(filename, "r");
    if (!fin) {
        printf("ERROR: could not open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    /* The standard PPM fields */
    char magic[4];
    int width, height;
    short max_val;

    /* Reads the file, assuming file structure */
    if (fscanf(fin, "%s\n", magic) != 1) {
        printf("ERROR: reading magic number\n");
        exit(EXIT_FAILURE);
    }
    /* ImageMagick may produce an extra line when converting the image to a PPM.
     * We first check to see if we find the two integer pattern. If it is not found, then we
     * assume the line exists and ignore it. */
    if (fscanf(fin, "%d %d\n", &width, &height) != 2) {
        if (fscanf(fin, "%*s\n%d %d\n", &width, &height) != 2) {
            printf("ERROR: reading width/height\n");
            exit(EXIT_FAILURE);
        }
    }
    if (fscanf(fin, "%hd\n", &max_val) != 1) {
        printf("ERROR: reading maximum value\n");
        exit(EXIT_FAILURE);
    }

    /* Writes discovered values into the given PPM structure */
    snprintf(image->magic, 4, "%s", magic);
    image->width = width;
    image->height = height;
    image->max_val = max_val;

    /* Local temporary variables */
    long pix_count = 0;
    rgb_t r, g, b;

    /* Begin reading pixel RGB values */
    while (fscanf(fin, "%hu %hu %hu", &r, &g, &b) == 3) {
        pixel_t current_pix = {.red = r, .green = g, .blue = b};
        image->pixels[pix_count] = current_pix;
        pix_count++;
    }

    fclose(fin);
}

/**
 * Copies PPM header information into another PPM header
 */
void copy_header_ppm(ppm_t *img, ppm_t *other)
{
    snprintf(other->magic, 4, "P3");
    other->width   = img->width;
    other->height  = img->height;
    other->max_val = img->max_val;
}

/**
 * Writes out a PPM structure to disk.
 */
void write_out_ppm(char *filename, ppm_t *image)
{
    FILE *fout = fopen(filename, "w");
    if (!fout) {
        printf("ERROR: could not open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    if (fprintf(fout, "%s\n", image->magic) < 0) {
        printf("ERROR: writing magic number\n");
        exit(EXIT_FAILURE);
    }
    if (fprintf(fout, "%d %d\n", image->width, image->height) < 0) {
        printf("ERROR: writing width/height\n");
        exit(EXIT_FAILURE);
    }
    if (fprintf(fout, "%hd\n", image->max_val) < 0) {
        printf("ERROR: writing maximum value\n");
        exit(EXIT_FAILURE);
    }

    long pix_count = 0;
    long total_num_pix = image->width * image->height;
    pixel_t curr_pix;

    while (pix_count < total_num_pix) {
        curr_pix = image->pixels[pix_count];

        if (pix_count % MAX_PIXELS_PER_LINE == 0) {
            if (fprintf(fout, "\n") < 0) {
                printf("ERROR: writing maximum value\n");
                exit(EXIT_FAILURE);
            }
        }

        if (fprintf(fout, "%hu %hu %hu ",
                    curr_pix.red, curr_pix.green, curr_pix.blue) < 0) {
            printf("ERROR: writing pixel_t value\n");
            exit(EXIT_FAILURE);
        }

        pix_count++;
    }

    fclose(fout);
}

/**
 * Prints PPM header information (for debugging purposes).
 */
void print_header_ppm(ppm_t *img)
{
    printf("Magic= %s, Width= %d, Height= %d, Max Value= %hd\n",
           img->magic, img->width, img->height, img->max_val);
}

#endif //CUDA_LAB_NETPBM_H
