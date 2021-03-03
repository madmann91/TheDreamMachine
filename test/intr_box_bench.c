#include "intr.h"
#include "mem.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define COUNT 10000000

int main(int argc, char** argv)
{
    srand((int)clock() + (int)time(NULL));

    ray_packet4* packets = aligned_malloc(sizeof(ray_packet4) * COUNT, 16);
    for (int i = 0; i < COUNT; i++) {
        for (int j = 0; j < 3 * 4; j++) {
            packets[i].dir[j] = (float)rand() / (float)RAND_MAX;
            packets[i].org[j] = 3 * (float)rand() / (float)RAND_MAX - 1;
            packets[i].inv_dir[j] = 1.0f / packets[i].dir[j];
        }
    }

    unsigned long num = 0;
    const float SSE_ALIGN(box_min[4]) = {0, 0, 0, 0};
    const float SSE_ALIGN(box_max[4]) = {1, 1, 1, 1};
    const float SSE_ALIGN(tmin[4]) = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

    clock_t ck = clock();
    for (int i = 0; i < COUNT; i++) {
        int hit = intr_packet4_box(packets + i, box_min, box_max, tmin);

        num = (hit        & 1) ? num + 1 : num;
        num = ((hit >> 1) & 1) ? num + 1 : num;
        num = ((hit >> 2) & 1) ? num + 1 : num;
        num = ((hit >> 3) & 1) ? num + 1 : num;
    }
    
    printf("%lu hits\n", num);
    printf("%ld ms\n", (clock() - ck) * 1000 / CLOCKS_PER_SEC);

    aligned_free(packets);

    return 0;
}
