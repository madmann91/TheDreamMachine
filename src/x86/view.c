#include "../view.h"
#include "../linalg.inl"
#include "../mem.h"
#include "x86.inl"
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void rotate_view(float angle,
                 const float* axis,
                 float* pos)
{
    quat temp, quat_view;

    temp[0] = axis[0] * sinf(angle / 2);
    temp[1] = axis[1] * sinf(angle / 2);
    temp[2] = axis[2] * sinf(angle / 2);
    temp[3] = cosf(angle / 2);

    vector3_copy(pos, quat_view);
    quat_view[3] = 0;

    float result[4];
    quat_mult(temp, quat_view, result);
    quat_conj(temp, temp);
    quat_mult(result, temp, pos);
}

void update_view(float tx, float ty,
                 float rx, float ry,
                 float* up,
                 float* eye,
                 float* pos)
{
    float dir[3], right[3];
    vector3_subtract(pos, eye, dir);
    vector3_scale(dir, 1.0f / vector3_length(dir), dir);
    vector3_cross(dir, up, right);

    /* Rotate the view */
    if (rx != 0 || ry != 0) {
        rotate_view(ry, right, dir);
        rotate_view(rx, up, dir);
        pos[0] = eye[0] + dir[0];
        pos[1] = eye[1] + dir[1];
        pos[2] = eye[2] + dir[2];

        vector3_cross(right, dir, up);
        vector3_scale(up, 1.0f / vector3_length(up), up);
    }

    /* Translate the eye orthogonally to its dir */
    if (tx != 0 || ty != 0) {
        /* First, translate along x */
        if (tx != 0) {
            vector3_scale(right, -tx, right);
            vector3_add(eye, right, eye);
            vector3_add(pos, right, pos);
        }

        /* Then y */
        if (ty != 0) {
            vector3_scale(dir, ty, dir);
            vector3_add(eye, dir, eye);
            vector3_add(pos, dir, pos);
        }
    }
}

void setup_view_persp(view_info* view,
                      const float* eye,
                      const float* pos,
                      const float* up,
                      float fov, float ratio,
                      float near, float far)
{
    vector3_subtract(pos, eye, view->dir);
    vector3_scale(view->dir, 1.0f / vector3_length(view->dir), view->dir);

    vector3_cross(view->dir, up, view->right);
    vector3_scale(view->right, 1.0f / vector3_length(view->right), view->right);

    vector3_cross(view->right, view->dir, view->up);

    float offset = vector3_dot(view->dir, eye);

    /* Near plane */
    vector3_negate(view->dir, view->near);
    view->near[3] = offset + near;

    /* Far plane */
    vector3_copy(view->dir, view->far);
    view->far[3] = -offset - far;

    view->wnear = near * tanf(fov * M_PI / 180.0f * 0.5f);
    view->hnear = view->wnear / ratio;

    vector3_copy(eye, view->eye);
    view->dnear = near;
}

void build_packet4_persp(const view_info* view,
                         ray_packet4* packets,
                         unsigned int i, unsigned int j,
                         unsigned int packet_w, unsigned int packet_h,
                         unsigned int view_w, unsigned int view_h)
{
    const __m128 eye = _mm_loadu_ps(view->eye);
    const __m128 up = _mm_loadu_ps(view->up);
    const __m128 right = _mm_loadu_ps(view->right);

    __m128 top_left;
    {
        const __m128 dir = _mm_loadu_ps(view->dir);
        const __m128 up_point    = _mm_mul_ps(up,    _mm_set1_ps(view->hnear * (2 * (float)j / (float)view_h - 1)));
        const __m128 right_point = _mm_mul_ps(right, _mm_set1_ps(view->wnear * (2 * (float)i / (float)view_w - 1)));
        top_left = _mm_add_ps(sse_madd(dir, _mm_set1_ps(view->dnear), eye),
                              _mm_add_ps(up_point, right_point));
    }

    const __m128 up_offset    = _mm_mul_ps(up,    _mm_set1_ps((view->hnear * 2) / (float)view_h));
    const __m128 right_offset = _mm_mul_ps(right, _mm_set1_ps((view->wnear * 2) / (float)view_w));

    /* Build initial position */
    __m128 vert_x;
    {
        const __m128 inc_right = _mm_shuffle_ps(right_offset, right_offset, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 inc_up    = _mm_shuffle_ps(up_offset,    up_offset,    _MM_SHUFFLE(0, 0, 0, 0));
        vert_x = _mm_shuffle_ps(top_left, top_left, _MM_SHUFFLE(0, 0, 0, 0));
        vert_x = _mm_add_ss(vert_x, inc_right);
        vert_x = _mm_shuffle_ps(vert_x, vert_x, _MM_SHUFFLE(0, 1, 0, 1));
        vert_x = _mm_add_ps(vert_x, _mm_shuffle_ps(_mm_setzero_ps(), inc_up, _MM_SHUFFLE(3, 2, 1, 0)));
    }

    __m128 vert_y;
    {
        const __m128 inc_right = _mm_shuffle_ps(right_offset, right_offset, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 inc_up    = _mm_shuffle_ps(up_offset,    up_offset,    _MM_SHUFFLE(1, 1, 1, 1));
        vert_y = _mm_shuffle_ps(top_left, top_left, _MM_SHUFFLE(1, 1, 1, 1));
        vert_y = _mm_add_ss(vert_y, inc_right);
        vert_y = _mm_shuffle_ps(vert_y, vert_y, _MM_SHUFFLE(0, 1, 0, 1));
        vert_y = _mm_add_ps(vert_y, _mm_shuffle_ps(_mm_setzero_ps(), inc_up, _MM_SHUFFLE(3, 2, 1, 0)));
    }

    __m128 vert_z;
    {
        const __m128 inc_right = _mm_shuffle_ps(right_offset, right_offset, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 inc_up    = _mm_shuffle_ps(up_offset,    up_offset,    _MM_SHUFFLE(2, 2, 2, 2));
        vert_z = _mm_shuffle_ps(top_left, top_left, _MM_SHUFFLE(2, 2, 2, 2));
        vert_z = _mm_add_ss(vert_z, inc_right);
        vert_z = _mm_shuffle_ps(vert_z, vert_z, _MM_SHUFFLE(0, 1, 0, 1));
        vert_z = _mm_add_ps(vert_z, _mm_shuffle_ps(_mm_setzero_ps(), inc_up, _MM_SHUFFLE(3, 2, 1, 0)));
    }
    
    const __m128 eye_x = _mm_shuffle_ps(eye, eye, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 eye_y = _mm_shuffle_ps(eye, eye, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 eye_z = _mm_shuffle_ps(eye, eye, _MM_SHUFFLE(2, 2, 2, 2));
    
    __m128 right_x, right_y, right_z;
    __m128 up_x, up_y, up_z;
    {
        const __m128 inc_up    = _mm_add_ps(up_offset, up_offset);
        const __m128 inc_right = _mm_add_ps(right_offset, right_offset);
        
        right_x = _mm_shuffle_ps(inc_right, inc_right, _MM_SHUFFLE(0, 0, 0, 0));
        right_y = _mm_shuffle_ps(inc_right, inc_right, _MM_SHUFFLE(1, 1, 1, 1));
        right_z = _mm_shuffle_ps(inc_right, inc_right, _MM_SHUFFLE(2, 2, 2, 2));
        
        up_x = _mm_shuffle_ps(inc_up, inc_up, _MM_SHUFFLE(0, 0, 0, 0));
        up_y = _mm_shuffle_ps(inc_up, inc_up, _MM_SHUFFLE(1, 1, 1, 1));
        up_z = _mm_shuffle_ps(inc_up, inc_up, _MM_SHUFFLE(2, 2, 2, 2));
    }

    /* Build packets */
    ray_packet4* packet = packets;
    for (unsigned int y = 0; y < packet_h; y += 2) {
        __m128 horz_x = vert_x;
        __m128 horz_y = vert_y;
        __m128 horz_z = vert_z;

        for (unsigned int x = 0; x < packet_w; x += 2) {                    
            _mm_store_ps(packet->org + 0, horz_x);
            _mm_store_ps(packet->org + 4, horz_y);
            _mm_store_ps(packet->org + 8, horz_z);

            const __m128 dir_x = _mm_sub_ps(horz_x, eye_x);
            const __m128 dir_y = _mm_sub_ps(horz_y, eye_y);
            const __m128 dir_z = _mm_sub_ps(horz_z, eye_z);
            _mm_store_ps(packet->dir + 0, dir_x);
            _mm_store_ps(packet->dir + 4, dir_y);
            _mm_store_ps(packet->dir + 8, dir_z);

            /* We do not use sse_rcp because we don't need that much precision */
            const __m128 inv_x = _mm_rcp_ps(dir_x);
            const __m128 inv_y = _mm_rcp_ps(dir_y);
            const __m128 inv_z = _mm_rcp_ps(dir_z);
            _mm_store_ps(packet->inv_dir + 0, inv_x);
            _mm_store_ps(packet->inv_dir + 4, inv_y);
            _mm_store_ps(packet->inv_dir + 8, inv_z);
            
            horz_x = _mm_add_ps(horz_x, right_x);
            horz_y = _mm_add_ps(horz_y, right_y);
            horz_z = _mm_add_ps(horz_z, right_z);
            packet++;
        }

        vert_x = _mm_add_ps(vert_x, up_x);
        vert_y = _mm_add_ps(vert_y, up_y);
        vert_z = _mm_add_ps(vert_z, up_z);
    }
}

void build_packet4_frustum_persp(const ray_packet4* packets,
                                 unsigned int packet_w, unsigned int packet_h,
                                 const float* near_plane, const float* far_plane,
                                 float* packet_frustum)
{
    const unsigned int top_left = 0;
    const unsigned int top_right = packet_w / 2 - 1;
    const unsigned int bottom_left = (packet_h - 2) * packet_w / 4;
    const unsigned int bottom_right = packet_h * packet_w / 4 - 1;

#define COMPUTE_PLANE(a, b, p, r) \
    (r)[0] = (b)[4] * (a)[8] - (b)[8] * (a)[4]; \
    (r)[1] = (b)[8] * (a)[0] - (b)[0] * (a)[8]; \
    (r)[2] = (b)[0] * (a)[4] - (b)[4] * (a)[0]; \
    (r)[3] = -(r)[0] * (p)[0] - (r)[1] * (p)[4] - (r)[2] * (p)[8];
    
    /* Build frustum : top */
    COMPUTE_PLANE(packets[top_right].dir,
                  packets[top_left].dir,
                  packets[top_left].org,
                  packet_frustum);
    /* Left */
    COMPUTE_PLANE(packets[top_left].dir,
                  packets[bottom_left].dir,
                  packets[top_left].org,
                  packet_frustum + 4);
    /* Bottom */
    COMPUTE_PLANE(packets[bottom_left].dir + 3,
                  packets[bottom_right].dir + 3,
                  packets[bottom_left].org + 3,
                  packet_frustum + 8);
    /* Right */
    COMPUTE_PLANE(packets[bottom_right].dir + 3,
                  packets[top_right].dir + 3,
                  packets[top_right].org + 3,
                  packet_frustum + 12);

#undef COMPUTE_PLANE
    
    /* Near & far */
    vector4_copy(near_plane, packet_frustum + 16);
    vector4_copy(far_plane, packet_frustum + 20);
}

void build_packet4_shadow(ray_packet4* packets,
                          unsigned int num_packets,
                          const ray_packet4* prev_packets,
                          const packet4_hit* hits,
                          const float* light_pos)
{
    __m128 light_x, light_y, light_z;
    {
        const __m128 light = _mm_load_ps(light_pos);
        light_x = _mm_shuffle_ps(light, light, _MM_SHUFFLE(0, 0, 0, 0));
        light_y = _mm_shuffle_ps(light, light, _MM_SHUFFLE(1, 1, 1, 1));
        light_z = _mm_shuffle_ps(light, light, _MM_SHUFFLE(2, 2, 2, 2));
    }

    for (unsigned int i = 0; i < num_packets; i++) {
        __m128i mask = _mm_cmplt_epi32(_mm_load_si128((__m128i*)hits[i].intr), _mm_setzero_si128());

        _mm_store_ps(packets[i].org + 0, light_x);
        _mm_store_ps(packets[i].org + 4, light_y);
        _mm_store_ps(packets[i].org + 8, light_z);

        __m128 dir_x, dir_y, dir_z;
        if (_mm_movemask_ps(_mm_castsi128_ps(mask)) != 0x0F) {
            const __m128 hit_t = _mm_load_ps(hits[i].t);

            {
                const __m128 prev_org = _mm_load_ps(prev_packets[i].org);
                const __m128 prev_dir = _mm_load_ps(prev_packets[i].dir);
                dir_x = _mm_sub_ps(sse_madd(prev_dir, hit_t, prev_org), light_x);
            }

            {
                const __m128 prev_org = _mm_load_ps(prev_packets[i].org + 4);
                const __m128 prev_dir = _mm_load_ps(prev_packets[i].dir + 4);
                dir_y = _mm_sub_ps(sse_madd(prev_dir, hit_t, prev_org), light_y);
            }

            {
                const __m128 prev_org = _mm_load_ps(prev_packets[i].org + 8);
                const __m128 prev_dir = _mm_load_ps(prev_packets[i].dir + 8);
                dir_z = _mm_sub_ps(sse_madd(prev_dir, hit_t, prev_org), light_z);
            }

            dir_x = _mm_andnot_ps(_mm_castsi128_ps(mask), dir_x);
            dir_y = _mm_andnot_ps(_mm_castsi128_ps(mask), dir_y);
            dir_z = _mm_andnot_ps(_mm_castsi128_ps(mask), dir_z);
        } else {
            dir_x = _mm_setzero_ps();
            dir_y = _mm_setzero_ps();
            dir_z = _mm_setzero_ps();
        }

        _mm_store_ps(packets[i].dir + 0, dir_x);
        _mm_store_ps(packets[i].dir + 4, dir_y);
        _mm_store_ps(packets[i].dir + 8, dir_z);

        /* We do not use sse_rcp because we don't need that much precision */
        const __m128 inv_x = _mm_rcp_ps(dir_x);
        const __m128 inv_y = _mm_rcp_ps(dir_y);
        const __m128 inv_z = _mm_rcp_ps(dir_z);
        _mm_store_ps(packets[i].inv_dir + 0, inv_x);
        _mm_store_ps(packets[i].inv_dir + 4, inv_y);
        _mm_store_ps(packets[i].inv_dir + 8, inv_z);
    }
}

void build_packet4_frustum_shadow(const ray_packet4* packets,
                                  unsigned int num_packets,
                                  const float* light_pos,
                                  float* packet_frustum)
{
    /* Builds a bounding box for a shadow packet and store
     * the resulting 6 planes in the packet frustum
     */
    const __m128 light = _mm_load_ps(light_pos);
    const __m128 light_x = _mm_shuffle_ps(light, light, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 light_y = _mm_shuffle_ps(light, light, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 light_z = _mm_shuffle_ps(light, light, _MM_SHUFFLE(2, 2, 2, 2));

    __m128 min_x = light_x, min_y = light_y, min_z = light_z;
    __m128 max_x = light_x, max_y = light_y, max_z = light_z;

    for (unsigned int i = 0; i < num_packets; i++) {
        const __m128 pos_x = _mm_add_ps(light_x, _mm_load_ps(packets[i].dir + 0));
        const __m128 pos_y = _mm_add_ps(light_y, _mm_load_ps(packets[i].dir + 4));
        const __m128 pos_z = _mm_add_ps(light_z, _mm_load_ps(packets[i].dir + 8));

        min_x = _mm_min_ps(min_x, pos_x);
        min_y = _mm_min_ps(min_y, pos_y);
        min_z = _mm_min_ps(min_z, pos_z);

        max_x = _mm_max_ps(max_x, pos_x);
        max_y = _mm_max_ps(max_y, pos_y);
        max_z = _mm_max_ps(max_z, pos_z);
    }

    min_x = _mm_min_ps(min_x, _mm_shuffle_ps(min_x, min_x, _MM_SHUFFLE(3, 2, 3, 2)));
    min_y = _mm_min_ps(min_y, _mm_shuffle_ps(min_y, min_y, _MM_SHUFFLE(3, 2, 3, 2)));
    min_z = _mm_min_ps(min_z, _mm_shuffle_ps(min_z, min_z, _MM_SHUFFLE(3, 2, 3, 2)));

    max_x = _mm_max_ps(max_x, _mm_shuffle_ps(max_x, max_x, _MM_SHUFFLE(3, 2, 3, 2)));
    max_y = _mm_max_ps(max_y, _mm_shuffle_ps(max_y, max_y, _MM_SHUFFLE(3, 2, 3, 2)));
    max_z = _mm_max_ps(max_z, _mm_shuffle_ps(max_z, max_z, _MM_SHUFFLE(3, 2, 3, 2)));

    min_x = _mm_min_ss(min_x, _mm_shuffle_ps(min_x, min_x, _MM_SHUFFLE(1, 1, 1, 1)));
    min_y = _mm_min_ss(min_y, _mm_shuffle_ps(min_y, min_y, _MM_SHUFFLE(1, 1, 1, 1)));
    min_z = _mm_min_ss(min_z, _mm_shuffle_ps(min_z, min_z, _MM_SHUFFLE(1, 1, 1, 1)));

    max_x = _mm_max_ss(max_x, _mm_shuffle_ps(max_x, max_x, _MM_SHUFFLE(1, 1, 1, 1)));
    max_y = _mm_max_ss(max_y, _mm_shuffle_ps(max_y, max_y, _MM_SHUFFLE(1, 1, 1, 1)));
    max_z = _mm_max_ss(max_z, _mm_shuffle_ps(max_z, max_z, _MM_SHUFFLE(1, 1, 1, 1)));

    max_x = sse_negate(max_x);
    max_y = sse_negate(max_y);
    max_z = sse_negate(max_z);

    packet_frustum[0 * 4 + 0] = -1.0f;
    packet_frustum[0 * 4 + 1] = 0.0f;
    packet_frustum[0 * 4 + 2] = 0.0f;
    _mm_store_ss(packet_frustum + 0 * 4 + 3, min_x);
    packet_frustum[1 * 4 + 0] = 1.0f;
    packet_frustum[1 * 4 + 1] = 0.0f;
    packet_frustum[1 * 4 + 2] = 0.0f;
    _mm_store_ss(packet_frustum + 1 * 4 + 3, max_x);

    packet_frustum[2 * 4 + 0] = 0.0f;
    packet_frustum[2 * 4 + 1] = -1.0f;
    packet_frustum[2 * 4 + 2] = 0.0f;
    _mm_store_ss(packet_frustum + 2 * 4 + 3, min_y);
    packet_frustum[3 * 4 + 0] = 0.0f;
    packet_frustum[3 * 4 + 1] = 1.0f;
    packet_frustum[3 * 4 + 2] = 0.0f;
    _mm_store_ss(packet_frustum + 3 * 4 + 3, max_y);

    packet_frustum[4 * 4 + 0] = 0.0f;
    packet_frustum[4 * 4 + 1] = 0.0f;
    packet_frustum[4 * 4 + 2] = -1.0f;
    _mm_store_ss(packet_frustum + 4 * 4 + 3, min_z);
    packet_frustum[5 * 4 + 0] = 0.0f;
    packet_frustum[5 * 4 + 1] = 0.0f;
    packet_frustum[5 * 4 + 2] = 1.0f;
    _mm_store_ss(packet_frustum + 5 * 4 + 3, max_z);
}
