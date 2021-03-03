#include "intr.h"
#include "mem.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

int main(int argc, char** argv)
{
    const float SSE_ALIGN(vertices[16]) = {0.5, 0.5, 0, 1,
                                           1.5, 2.5, 1.0, 1,
                                           0.5, 1.5, 0, 1};
    const float inf = -logf(0.0f);
    
    ray_packet4 SSE_ALIGN(packet) = 
    { 
        /* origins */
        {  -30,   0.2f,   0.4f,   0.6f,
          0.5f,   0.2f,   0.4f,   0.6f,
          -100,     -3,     -3,     -3},
         
        /* directions */
        {0, 0, 0, 0,
         0, 0, 0, 0,
         1, 1, 1, 1},
         
        /* inverse direction */
        {inf, inf, inf, inf,
         inf, inf, inf, inf,
           1,   1,   1,   1}
    };

    packet4_hit SSE_ALIGN(hit);

    hit.t[0] = FLT_MAX;
    hit.t[1] = FLT_MAX;
    hit.t[2] = FLT_MAX;
    hit.t[3] = FLT_MAX;
    
    hit.intr[0] = 0;
    hit.intr[1] = 0;
    hit.intr[2] = 0;
    hit.intr[3] = 0;

    intr_packet4_tri(&packet, vertices + 0, vertices + 4, vertices + 8, &hit, 1);
    for (int i = 0; i < 4; i++) {
        if (hit.intr[i] == 1) {
            printf("%d    t = %f, u = %f, v = %f\n", i, hit.t[i], hit.u[i], hit.v[i]);
        }
    }

    return 0;
}
