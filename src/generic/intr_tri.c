#include "../intr.h"
#include "../linalg.inl"
#include <math.h>

#define EPSILON 0.00001f

static inline int intr_ray_tri(const float* ray_org,
                               const float* ray_dir,
                               const float* v0,
                               const float* v1,
                               const float* v2,
                               float* t, float* u, float* v)
{
    /* e0 = v1 - v0 and e1 = v2 - v0 */
    float e0[3], e1[3];
    vector3_subtract(v1, v0, e0);
    vector3_subtract(v2, v0, e1);

    /* pvec = cross(dir, e1) */
    const float pvec[3] = {
        ray_dir[4] * e1[2] - ray_dir[8] * e1[1],
        ray_dir[8] * e1[0] - ray_dir[0] * e1[2],
        ray_dir[0] * e1[1] - ray_dir[4] * e1[0]
    };
    float det = vector3_dot(e0, pvec);
#ifdef CULLING
    if (det < EPSILON) return 0;
#else
    if (det > -EPSILON && det < EPSILON) return 0;
#endif

    float inv_det = 1.0f / det;

    /* tvec = org - v0 */
    const float tvec[3] = {
        ray_org[0] - v0[0],
        ray_org[4] - v0[1],
        ray_org[8] - v0[2]
    };

    /* u = dot(pvec, tvec) * inv_det */
    float next_u = vector3_dot(tvec, pvec) * inv_det;
    if (next_u < 0.0f || next_u > 1.0f) return 0;

    /* qvec = cross(tvec, e0) */
    float qvec[3];
    vector3_cross(tvec, e0, qvec);

    /* v = dot(dir, qvec) * inv_det */
    float next_v = (ray_dir[0] * qvec[0] + ray_dir[4] * qvec[1] + ray_dir[8] * qvec[2]) * inv_det;
    if (next_v < 0.0f || next_u + next_v  > 1.0f) return 0;
 
    float next_t = vector3_dot(e1, qvec) * inv_det;

    if (next_t > 0 && next_t < *t) {
        *t = next_t;
        *u = next_u;
        *v = next_v;
        return 1;
    }

    return 0;
}

void intr_packet4_tri(const ray_packet4* packet,
                      const float* v0,
                      const float* v1,
                      const float* v2,
                      packet4_hit* hit,
                      int tri_id)
{
    for (int i = 0; i < 4; i++) {
        if (intr_ray_tri(packet->org + i, packet->dir + i,
                         v0, v1, v2,
                         hit->t + i,
                         hit->u + i,
                         hit->v + i)) {
            hit->intr[i] = tri_id;
        }
    }
}
