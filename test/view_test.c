#include "view.h"
#include "intr.h"
#include "mem.h"
#include <stdio.h>

int main(int argc, char** argv)
{
    view_info view;
    float SSE_ALIGN(eye[4]) = {0.0f, 0.0f, -100.0f, 1.0f};
    float SSE_ALIGN(pos[4]) = {0.0f, 0.0f, 0.0f, 1.0f};
    float SSE_ALIGN(up[4]) = {0.0f, 1.0f, 0.0f, 0.0f};

    setup_view_persp(&view,
                     eye, pos, up,
                     60.0f, 1.0f,
                     0.1f, 1000.0f);

    float SSE_ALIGN(frustum[6 * 4]);
    ray_packet4 SSE_ALIGN(packets[16 * 16]);
    build_packet4_persp(&view, packets, 0 , 0,
                        16, 16,
                        512, 512);

    build_packet4_frustum_persp(packets, 16, 16, view.near, view.far, frustum);

    for (int i = 0; i < 6; i++) {
        float* plane = frustum + 4 * i;
        printf("plane %d : %f %f %f %f\n", i, plane[0], plane[1], plane[2], plane[3]);
    }

    return 0;
}
