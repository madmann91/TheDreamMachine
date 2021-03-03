#include "../intr.h"
#include "../linalg.inl"

int intr_frustum_box(const float* frustum, const float* box_min, const float* box_max)
{
    float n[3];

    for (int i = 0; i < 6; i++) {
        const float* plane = frustum + 4 * i;

        n[0] = (plane[0] < 0) ? box_max[0] : box_min[0];
        n[1] = (plane[1] < 0) ? box_max[1] : box_min[1];
        n[2] = (plane[2] < 0) ? box_max[2] : box_min[2];

        if (vector3_dot(n, plane) > -plane[3])
            return 0;
    }

    return 1;
}
