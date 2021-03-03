#include "intr.h"
#include "mem.h"
#include <stdio.h>

#define SHIFT 80

int main(int argc, char** argv)
{
    float SSE_ALIGN(box_min[4]) = {0, 0, 0, 1};
    float SSE_ALIGN(box_max[4]) = {1, 1, 1, 1}; 

    /* Box : min(0, 0, 2) max(1, 1, 3) */
    float SSE_ALIGN(frustum[4 * 6]) = {0, 0, -1, 2,
                            0, 0, 1, -3,
                            0, -1, 0, 0,
                            0, 1, 0, -1,
                            -1, 0, 0, 0,
                            1, 0, 0, -1};

    int result;
    float center = 0;
    for (int i = 0; i < SHIFT; i++) {
        frustum[3] -= 0.05;
        frustum[4 + 3] += 0.05;
        center = (frustum[3] - frustum[4 + 3]) / 2;
        result = intr_frustum_box(frustum, box_min, box_max);
        printf("center : %f, result : %d\n", center, result);
    }

    return 0;
}
