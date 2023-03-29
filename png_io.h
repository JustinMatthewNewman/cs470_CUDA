// png_utils.h

#ifndef PNG_IO_H
#define PNG_IO_H

#include <png.h>
#include <stdlib.h>


int read_png(const char *filename, png_uint_32 *width, png_uint_32 *height, int *bit_depth, int *color_type, png_bytepp *row_pointers);
int write_png(const char *filename, png_uint_32 width, png_uint_32 height, int bit_depth, int color_type, png_bytepp row_pointers);


int read_png(const char *filename, png_uint_32 *width, png_uint_32 *height, int *bit_depth, int *color_type, png_bytepp *row_pointers) {
    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        fprintf(stderr, "Failed to open file %s for reading.\n", filename);
        return 1;
    }
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(infile);
        fprintf(stderr, "Failed to create png read struct.\n");
        return 1;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(infile);
        fprintf(stderr, "Failed to create png info struct.\n");
        return 1;
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(infile);
        fprintf(stderr, "Error during png_read_info.\n");
        return 1;
    }
    png_init_io(png_ptr, infile);
    png_read_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, NULL, NULL, NULL);
    png_set_expand(png_ptr);
    png_set_strip_16(png_ptr);
    png_set_gray_to_rgb(png_ptr);
    png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);
    png_read_update_info(png_ptr, info_ptr);
    *row_pointers = (png_bytepp)malloc(*height * sizeof(png_bytep));
    for (png_uint_32 i = 0; i < *height; i++) {
        (*row_pointers)[i] = (png_bytep)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }
    png_read_image(png_ptr, *row_pointers);
    fclose(infile);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

int write_png(const char *filename, png_uint_32 width, png_uint_32 height, int bit_depth, int color_type, png_bytepp row_pointers) {
    FILE *outfile = fopen(filename, "wb");
    if (!outfile) {
        fprintf(stderr, "Failed to open file %s for writing.\n", filename);
        return 1;
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(outfile);
        fprintf(stderr, "Failed to create png write struct.\n");
        return 1;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(outfile);
        fprintf(stderr, "Failed to create png info struct.\n");
        return 1;
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(outfile);
        fprintf(stderr, "Error during png_write_info.\n");
        return 1;
    }
    png_init_io(png_ptr, outfile);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    fclose(outfile);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return 0;
}


#endif // PNG_UTILS_H
