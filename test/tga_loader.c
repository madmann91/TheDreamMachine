#include "tga_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char raw_tga[12] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static unsigned char rle_tga[12] = {0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0};

int load_raw_tga(rgb** texture, unsigned int* tex_w, unsigned int* tex_h, FILE* fp)
{
    unsigned char info[6];
    if (fread(info, sizeof(char), 6, fp) != 6)
        return 0;

    int width = info[1] * 256 + info[0];
    int height = info[3] * 256 + info[2];
    int bpp = info[4];

    if (width <= 0 || height <= 0 || (bpp != 24 && bpp != 32))
        return 0;

    *tex_w = width;
    *tex_h = height;
    *texture = malloc(width * height * sizeof(rgb));

    int line_size = bpp * width / 8;
    unsigned char* line = malloc(line_size);    

    for (int y = 0; y < height; y++) {
        if (fread(line, sizeof(unsigned char), line_size, fp) != line_size)
            return 0;

        if (bpp == 32) {
            /* Load RGBA data */
            rgb* ptr = *texture + y * width;
            for (int x = 0, i = 0; x < width; x++, i += 4) {
                (ptr + x)->q.b = line[i + 0];
                (ptr + x)->q.g = line[i + 1];
                (ptr + x)->q.r = line[i + 2];
                (ptr + x)->q.a = line[i + 3];
            }
        } else {
            /* Load RGB data */
            rgb* ptr = *texture + y * width;
            for (int x = 0, i = 0; x < width; x++, i += 3) {
                (ptr + x)->q.b = line[i + 0];
                (ptr + x)->q.g = line[i + 1];
                (ptr + x)->q.r = line[i + 2];
                (ptr + x)->q.a = 255;
            }
        }
    }

    free(line);
    fclose(fp);
    return 1;
}

int load_rle_tga(rgb** texture, unsigned int* tex_w, unsigned int* tex_h, FILE* fp)
{
    unsigned char info[6];

    if (fread(info, sizeof(char), 6, fp) != 6)
        return 0;

    int width = info[1] * 256 + info[0];
    int height = info[3] * 256 + info[2];
    int bpp = info[4];

    if (width <= 0 || height <= 0 || (bpp != 24 && bpp != 32))
        return 0;

    *tex_w = width;
    *tex_h = height;
    *texture = malloc(width * height * sizeof(rgb));

    int cur_pixel = 0;
    int pixel_count = width * height;

    while (cur_pixel < pixel_count) {
        unsigned char chunk_header;

        if (fread(&chunk_header, sizeof(char), 1, fp) != 1)
            return 0;

        if (chunk_header < 128) {
            /* RAW chunk, read it and copy it directly */
            chunk_header++;
            unsigned char line[128 * 4];
            if (fread(line, sizeof(unsigned char) * bpp / 8, chunk_header, fp) != chunk_header)
                return 0;

            if (bpp == 32) {
                rgb* ptr = *texture + cur_pixel;
                for (int x = 0, i = 0; x < chunk_header; x++, i += 4) {
                    (ptr + x)->q.b = line[i + 0];
                    (ptr + x)->q.g = line[i + 1];
                    (ptr + x)->q.r = line[i + 2];
                    (ptr + x)->q.a = line[i + 3];
                }
                cur_pixel += chunk_header;
            } else {
                rgb* ptr = *texture + cur_pixel;
                for (int x = 0, i = 0; x < chunk_header; x++, i += 3) {
                    (ptr + x)->q.b = line[i + 0];
                    (ptr + x)->q.g = line[i + 1];
                    (ptr + x)->q.r = line[i + 2];
                    (ptr + x)->q.a = 255;
                }
                cur_pixel += chunk_header;
            }
        } else {
            chunk_header -= 127;
            unsigned char pixel[4];

            /* Read one pixel and duplicate it chunk_header times */
            if (fread(pixel, sizeof(unsigned char) * bpp / 8, 1, fp) != 1)
                return 0;
            
            if (bpp == 32) {
                rgb* ptr = *texture + cur_pixel;
                for (int x = 0, i = 0; x < chunk_header; x++, i += 4) {
                    (ptr + x)->q.b = pixel[0];
                    (ptr + x)->q.g = pixel[1];
                    (ptr + x)->q.r = pixel[2];
                    (ptr + x)->q.a = pixel[3];
                }
                cur_pixel += chunk_header;
            } else {
                rgb* ptr = *texture + cur_pixel;
                for (int x = 0, i = 0; x < chunk_header; x++, i += 3) {
                    (ptr + x)->q.b = pixel[0];
                    (ptr + x)->q.g = pixel[1];
                    (ptr + x)->q.r = pixel[2];
                    (ptr + x)->q.a = 255;
                }
                cur_pixel += chunk_header;
            }
        }
    }

    fclose(fp);
    return 1;
}

int load_tga(rgb** texture, unsigned int* tex_w, unsigned int* tex_h, const char* file_name)
{
    FILE* fp = fopen(file_name, "rb");
    if (!fp)
        return 0;

    unsigned char header[12];
    if (fread(header, sizeof(unsigned char), 12, fp) != 12)
        return 0;

    /* Detect the type of texture */
    if (!memcmp(header, raw_tga, sizeof(unsigned char) * 12)) {
        return load_raw_tga(texture, tex_w, tex_h, fp);
    } else if (!memcmp(header, rle_tga, sizeof(unsigned char) * 12)) {
        return load_rle_tga(texture, tex_w, tex_h, fp);
    }

    return 0;
}

void destroy_tga(rgb* texture)
{
    free(texture);
}
