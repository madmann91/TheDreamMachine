#include "../view.h"
#include "../linalg.inl"
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
    float point_y[3], right_point[3], up_point[3];
    vector3_scale(view->dir, view->dnear, point_y);
    vector3_add(view->eye, point_y, point_y);

    vector3_scale(view->up, view->hnear * (2 * (float)j / (float)view_h - 1), up_point);
    vector3_scale(view->right, view->wnear * (2 * (float)i / (float)view_w - 1), right_point);
    vector3_add(point_y, up_point, point_y);
    vector3_add(point_y, right_point, point_y);

    float right_offset[3], up_offset[3];
    vector3_scale(view->up, (view->hnear * 2) / (float)view_h, up_offset);
    vector3_scale(view->right, (view->wnear * 2) / (float)view_w, right_offset);

    /* Build packets */
    ray_packet4* packet = packets;
    for (unsigned int y = 0; y < packet_h; y += 2) {
        float point_x[3];
        vector3_copy(point_y, point_x);
       
        for (unsigned int x = 0; x < packet_w; x += 2) {
            for (int k = 0; k < 2; k++) {
                packet->org[0 + k] = point_x[0];
                packet->org[4 + k] = point_x[1];
                packet->org[8 + k] = point_x[2];
                
                packet->org[0 + 2 + k] = point_x[0] + up_offset[0];
                packet->org[4 + 2 + k] = point_x[1] + up_offset[1];
                packet->org[8 + 2 + k] = point_x[2] + up_offset[2];

                packet->dir[0 + k] = packet->org[0 + k] - view->eye[0];
                packet->dir[4 + k] = packet->org[4 + k] - view->eye[1];
                packet->dir[8 + k] = packet->org[8 + k] - view->eye[2];
                
                packet->dir[0 + 2 + k] = packet->org[0 + 2 + k] - view->eye[0];
                packet->dir[4 + 2 + k] = packet->org[4 + 2 + k] - view->eye[1];
                packet->dir[8 + 2 + k] = packet->org[8 + 2 + k] - view->eye[2];

                packet->inv_dir[0 + k] = 1.0f / packet->dir[0 + k];
                packet->inv_dir[4 + k] = 1.0f / packet->dir[4 + k];
                packet->inv_dir[8 + k] = 1.0f / packet->dir[8 + k];
                
                packet->inv_dir[0 + 2 + k] = 1.0f / packet->dir[0 + 2 + k];
                packet->inv_dir[4 + 2 + k] = 1.0f / packet->dir[4 + 2 + k];
                packet->inv_dir[8 + 2 + k] = 1.0f / packet->dir[8 + 2 + k];
                
                vector3_add(point_x, right_offset, point_x);
            }
            packet++;
        }
        
        point_y[0] += 2 * up_offset[0];
        point_y[1] += 2 * up_offset[1];
        point_y[2] += 2 * up_offset[2];
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
    for (unsigned int i = 0; i < num_packets; i++) {
        for (int j = 0; j < 4; j++) {
            packets[i].org[0 + j] = light_pos[0];
            packets[i].org[4 + j] = light_pos[1];
            packets[i].org[8 + j] = light_pos[2];

            if (hits[i].intr[j] >= 0) {
                /* If the ray has hit, we can build a shadow ray */
                packets[i].dir[0 + j] = prev_packets[i].dir[0 + j] * hits[i].t[j] + prev_packets[i].org[0 + j] - light_pos[0];
                packets[i].dir[4 + j] = prev_packets[i].dir[4 + j] * hits[i].t[j] + prev_packets[i].org[4 + j] - light_pos[1];
                packets[i].dir[8 + j] = prev_packets[i].dir[8 + j] * hits[i].t[j] + prev_packets[i].org[8 + j] - light_pos[2];

                packets[i].inv_dir[0 + j] = 1.0f / packets[i].dir[0 + j];
                packets[i].inv_dir[4 + j] = 1.0f / packets[i].dir[4 + j];
                packets[i].inv_dir[8 + j] = 1.0f / packets[i].dir[8 + j];
            } else {
                /* Otherwise we build an empty ray */               
                packets[i].dir[0 + j] = 0.0f;
                packets[i].dir[4 + j] = 0.0f;
                packets[i].dir[8 + j] = 0.0f;
                
                packets[i].inv_dir[0 + j] = FLT_MAX;
                packets[i].inv_dir[4 + j] = FLT_MAX;
                packets[i].inv_dir[8 + j] = FLT_MAX;
            }
        }
    }
}

void build_packet4_frustum_shadow(const ray_packet4* packets,
                                  unsigned int num_packets,
                                  const float* light_pos,
                                  float* packet_frustum)
{
    /* Builds a bounding box for a shadow packet and store
     * the resulting 6 planes in the packet frustum */
    float min[3], max[3];
    vector3_copy(light_pos, min);
    vector3_copy(light_pos, max);

    for (unsigned int i = 0; i < num_packets; i++) {
        for (int j = 0; j < 4; j++) {
            const float pos[3] = {
                packets[i].dir[0 + j] + light_pos[0],
                packets[i].dir[4 + j] + light_pos[1],
                packets[i].dir[8 + j] + light_pos[2]
            };
            
            vector3_min(min, pos, min);
            vector3_max(max, pos, max);
        }
    }

    for (int i = 0; i < 6; i += 2) {
        int j = i / 2;
        packet_frustum[i * 4 + 0] = 0.0f;
        packet_frustum[i * 4 + 1] = 0.0f;
        packet_frustum[i * 4 + 2] = 0.0f;
        packet_frustum[i * 4 + 3] = min[j];
        packet_frustum[i * 4 + j] = -1.0f;

        packet_frustum[(i + 1) * 4 + 0] = 0.0f;
        packet_frustum[(i + 1) * 4 + 1] = 0.0f;
        packet_frustum[(i + 1) * 4 + 2] = 0.0f;
        packet_frustum[(i + 1) * 4 + 3] = -max[j];
        packet_frustum[(i + 1) * 4 + j] = 1.0f;
    }
}
