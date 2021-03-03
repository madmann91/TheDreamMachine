#include "intr.h"
#include "mem.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define SHIFT 80

int main(int argc, char** agv)
{
    float SSE_ALIGN(box_min[4]) = {0, 0, 0, 1};
    float SSE_ALIGN(box_max[4]) = {1, 1, 1, 1};
    float SSE_ALIGN(tmin[4]) = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    const float inf = -logf(0.0f);

    /* TEST 1 : one packet against the unit cube */
    /* 1st : hit x */
    /* 2nd : hit y */
    /* 3rd : miss */
    /* 4th : miss */
    ray_packet4 SSE_ALIGN(packet) = 
    { 
        /* origins */
        {   -1,   0.5f,    1,  -1,
         0.99f,     -1,    2,   2,
         0.99f,   0.5f,    2,   2},
         
        /* directions */
        {1, 0, 0, 1,
         0, 1, 0, 0,
         0, 0, 1, 0},
         
        /* inverse direction */
        {  1, inf, inf,   1,
         inf,   1, inf, inf,
         inf, inf,   1, inf}
    };

    int hit = intr_packet4_box(&packet, box_min, box_max, tmin);
    printf("%d %d %d %d\n", hit & 1, (hit >> 1) & 1 , (hit >> 2) & 1, (hit >> 3) & 1);

    /* TEST 2 : moving packet against unit cube again */
    ray_packet4 SSE_ALIGN(packet2) = 
    { 
        /* origins */
        {   -1,     -1,     -1,     -1,
          0.5f,   0.5f,   0.5f,   0.5f,
         -1.50, -1.55f, -1.60f, -1.65f},
         
		/* directions */
		{1, 1, 1, 1,
		 0, 0, 0, 0,
         0, 0, 0, 0},
         
		/* inverse direction */
		{  1,   1,   1,   1,
		 inf, inf, inf, inf,
		 inf, inf, inf, inf}
	};

	for (int i = 0; i < SHIFT; i++) {
		packet2.org[ 8] += .05;
		packet2.org[ 9] += .05;
		packet2.org[10] += .05;
		packet2.org[11] += .05;
	
		int hit = intr_packet4_box(&packet2, box_min, box_max, tmin);		
		printf("%d %d %d %d    %f\n", hit & 1, (hit >> 1) & 1 , (hit >> 2) & 1, (hit >> 3) & 1, packet2.org[8]);
	}    
    
    return 0;
}
