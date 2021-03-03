#include "intr.h"
#include "mem.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_PACKETS 1000000

static inline float dir_inv(float dir)
{
    return (dir == 0) ? FLT_MAX : 1.0f / dir; 
}

int main(int argc, char** argv)
{
    const float SSE_ALIGN(vertices[16]) = {1, 0, 0, 1,
                                           0, 1, 0, 1,
                                           0, 0, 1, 1};

    ray_packet4* packets = aligned_malloc(NUM_PACKETS * sizeof(ray_packet4), 16);

    srand(time(NULL));
    for (int i = 0; i < NUM_PACKETS; i++) {
        for (int j = 0; j < 4; j++) {
            float theta = (float)rand() / (float)RAND_MAX * 2. * M_PI;
            float phi = (float)rand() / (float)RAND_MAX * M_PI;

            packets[i].org[0 + j] = 3 * cos(theta) * sin(phi);
            packets[i].org[4 + j] = 3 * sin(theta) * sin(phi);
            packets[i].org[8 + j] = 3 * cos(phi);

            packets[i].dir[0 + j] = -packets[i].org[0 + j];
            packets[i].dir[4 + j] = -packets[i].org[4 + j];
            packets[i].dir[8 + j] = -packets[i].org[8 + j];

            packets[i].inv_dir[0 + j] = dir_inv(packets[i].dir[0 + j]);
            packets[i].inv_dir[4 + j] = dir_inv(packets[i].dir[4 + j]);
            packets[i].inv_dir[8 + j] = dir_inv(packets[i].dir[8 + j]);
        }
    }

    packet4_hit SSE_ALIGN(hit);

    clock_t ck = clock();
    for (int i = 0; i < NUM_PACKETS; i++)
        intr_packet4_tri(packets + i, vertices + 0, vertices + 4, vertices + 8, &hit, 1);
    
    printf("%d rays\n", 4 * NUM_PACKETS);
    printf("%ld ms\n", (clock() - ck) * 1000 / CLOCKS_PER_SEC);
    
    aligned_free(packets);    

    return 0;
}
