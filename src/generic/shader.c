#include "../shader.h"
#include "../linalg.inl"
#include <math.h>
#include <stdio.h>

#define NUM_CELLS 8

static inline void rgb_to_vector4(const rgb* from, float* to)
{
    to[0] = (float)from->q.r / 255.0f;
    to[1] = (float)from->q.g / 255.0f;
    to[2] = (float)from->q.b / 255.0f;
    to[3] = (float)from->q.a / 255.0f;
}

static inline void vector4_to_rgb(const float* from, rgb* to)
{
    to->q.r = round(from[0] * 255);
    to->q.g = round(from[1] * 255);
    to->q.b = round(from[2] * 255);
    to->q.a = round(from[3] * 255);
}

static inline void interpolate_normals(const float* bary,
                                       const face* triangle,
                                       const shader_info* shader,
                                       float* result)
{
    const float* n0 = shader->normals + 4 * triangle->normals[0];
    const float* n1 = shader->normals + 4 * triangle->normals[1];
    const float* n2 = shader->normals + 4 * triangle->normals[2];

    result[0] = n0[0] * bary[0] + n1[0] * bary[1] + n2[0] * bary[2];
    result[1] = n0[1] * bary[0] + n1[1] * bary[1] + n2[1] * bary[2];
    result[2] = n0[2] * bary[0] + n1[2] * bary[1] + n2[2] * bary[2];

    /* Renormalize result */
    vector3_scale(result, 1.0f / vector3_length(result), result);
}

static inline void interpolate_texcoords(const float* bary,
                                         const face* triangle,
                                         const shader_info* shader,
                                         float* result)
{
    const float* tex0 = shader->texcoords + 2 * triangle->texcoords[0];
    const float* tex1 = shader->texcoords + 2 * triangle->texcoords[1];
    const float* tex2 = shader->texcoords + 2 * triangle->texcoords[2];
    result[0] = tex0[0] * bary[0] + tex1[0] * bary[1] + tex2[0] * bary[2];
    result[1] = tex0[1] * bary[0] + tex1[1] * bary[1] + tex2[1] * bary[2];
}

static inline void bilinear_filtering(const float* texcoords,
                                      const shader_info* shader,
                                      const material* mat,
                                      float* result)
{
    float tex_x, tex_y;
    if (mat->tex_mode & TEX_REPEAT) {
        float tmp;
        tex_x = modff(texcoords[0], &tmp);
        tex_y = modff(texcoords[1], &tmp);
        if (tex_x < 0) tex_x = 1 + tex_x;
        if (tex_y < 0) tex_y = 1 + tex_y;
    } else {
        tex_x = (texcoords[0] < 0) ? 0 : ((texcoords[0] > 1) ? 1 : texcoords[0]);
        tex_y = (texcoords[1] < 0) ? 0 : ((texcoords[1] > 1) ? 1 : texcoords[1]);
    }

    tex_x *= mat->tex_w - 1;
    tex_y *= mat->tex_h - 1;

    const unsigned int xmin = floor(tex_x);
    const unsigned int ymin = floor(tex_y);

    float u = tex_x - xmin;
    float v = tex_y - ymin;
    float u_opp = 1 - u;
    float v_opp = 1 - v;

    float near_colors[4 * 4];

    rgb_to_vector4(mat->texture + ymin * mat->tex_w + xmin, near_colors + 0);
    rgb_to_vector4(mat->texture + ymin * mat->tex_w + xmin + 1, near_colors + 4);
    rgb_to_vector4(mat->texture + (ymin + 1) * mat->tex_w + xmin, near_colors + 8);
    rgb_to_vector4(mat->texture + (ymin + 1) * mat->tex_w + xmin + 1, near_colors + 12);

    for (int j = 0; j < 4; j++) {
        result[j] = (near_colors[j] * u_opp + near_colors[j + 4] * u) * v_opp +
                    (near_colors[j + 8] * u_opp + near_colors[j + 12] * u) * v;
    }
}

void packet4_phong_shader(const packet4_hit* hits,
                          const ray_packet4* packets,
                          const packet4_hit* shadow_hits,
                          const ray_packet4* shadow_packets,
                          unsigned int num_packets,
                          const shader_info* shader,
                          rgb* result)
{
    for (int i = 0; i < num_packets; i++) {
        for (int j = 0; j < 4; j++) {
            /* Skip the ray if it doesn't hit anything */
            if (hits[i].intr[j] < 0) {
                result[i * 4 + j] = shader->bg_color;
                continue;
            }

            const face* triangle = shader->faces + hits[i].intr[j];
            const material* mat = shader->materials + triangle->material;

            /* Barycentric coordinates */
            float bary[3];
            bary[0] = 1 - hits[i].u[j] - hits[i].v[j];
            bary[1] = hits[i].u[j];
            bary[2] = hits[i].v[j];

            /* Interpolate the normal */
            float avg_normal[3];
            interpolate_normals(bary, triangle, shader, avg_normal);

            /* Get the texture color */
            float tex_color[4] = {1, 1, 1, 1};
            if (mat->texture) {
                /* Interpolate the texture coordinates */
                float avg_texture[4];
                interpolate_texcoords(bary, triangle, shader, avg_texture);
                bilinear_filtering(avg_texture, shader, mat, tex_color);
            }

            float partial[4];
            vector4_scale4(shader->ambient, mat->ambient, partial);
            vector4_scale4(partial, tex_color, partial);

            /* Initialise view vector */
            float dir_to_eye[3];
            {
                const float* v = packets[i].dir + j;
                const float inv_len = -1.0f / sqrtf(v[0] * v[0] + v[4] * v[4] + v[8] * v[8]);
                dir_to_eye[0] = v[0] * inv_len;
                dir_to_eye[1] = v[4] * inv_len;
                dir_to_eye[2] = v[8] * inv_len;
            }

            /* Sum each light contribution */
            for (int l = 0; l < shader->num_lights; l++) {
                if (shadow_hits[l * num_packets + i].intr[j] >= 0) continue;

                float dir_to_light[3];
                {
                    const float* v = shadow_packets[l * num_packets + i].dir + j;
                    const float inv_len = -1.0f / sqrtf(v[0] * v[0] + v[4] * v[4] + v[8] * v[8]);
                    dir_to_light[0] = v[0] * inv_len;
                    dir_to_light[1] = v[4] * inv_len;
                    dir_to_light[2] = v[8] * inv_len;
                }

                float dot_d = vector3_dot(dir_to_light, avg_normal);
                if (dot_d < 0) dot_d = 0;
                const float diff[4] =
                {
                    (mat->diffuse[0] * tex_color[0]) * dot_d,
                    (mat->diffuse[1] * tex_color[1]) * dot_d,
                    (mat->diffuse[2] * tex_color[2]) * dot_d,
                    0.0f
                };

                float rm[3] =
                {
                    avg_normal[0] * 2 * dot_d - dir_to_light[0],
                    avg_normal[1] * 2 * dot_d - dir_to_light[1],
                    avg_normal[2] * 2 * dot_d - dir_to_light[2],
                };
                
                float dot_s = vector3_dot(rm, dir_to_eye);
                dot_s = (dot_s > 0) ? powf(dot_s, mat->alpha) : 0;
                const float spec[4] =
                {
                    mat->specular[0] * dot_s,
                    mat->specular[1] * dot_s,
                    mat->specular[2] * dot_s,
                    0.0f
                };
              
                partial[0] += diff[0] * shader->lights[l].diffuse[0] +
                              spec[0] * shader->lights[l].specular[0];
                partial[1] += diff[1] * shader->lights[l].diffuse[1] +
                              spec[1] * shader->lights[l].specular[1];
                partial[2] += diff[2] * shader->lights[l].diffuse[2] +
                              spec[2] * shader->lights[l].specular[2];
                partial[3] += diff[3] * shader->lights[l].diffuse[3] +
                              spec[3] * shader->lights[l].specular[3];
            }

            partial[0] = (partial[0] < 1) ? partial[0] : 1;
            partial[1] = (partial[1] < 1) ? partial[1] : 1;
            partial[2] = (partial[2] < 1) ? partial[2] : 1;
            partial[3] = (partial[3] < 1) ? partial[3] : 1;

            vector4_to_rgb(partial, result + i * 4 + j);
        }
    }
}

void packet4_cel_shader(const packet4_hit* hits,
                        const ray_packet4* packets,
                        const packet4_hit* shadow_hits,
                        const ray_packet4* shadow_packets,
                        unsigned int num_packets,
                        const shader_info* shader,
                        rgb* result)
{
    static const float values[NUM_CELLS] =
    {
        0.0f, 0.1f, 0.3f, 0.4f,
        0.6f, 0.8f, 1.0f, 1.0f
    };

    for (int i = 0; i < num_packets; i++) {
        for (int j = 0; j < 4; j++) {
            /* Skip the ray if it doesn't hit anything */
            if (hits[i].intr[j] < 0) {
                result[i * 4 + j] = shader->bg_color;
                continue;
            }

            const face* triangle = shader->faces + hits[i].intr[j];
            const material* mat = shader->materials + triangle->material;

            /* Barycentric coordinates */
            float bary[3];
            bary[0] = 1 - hits[i].u[j] - hits[i].v[j];
            bary[1] = hits[i].u[j];
            bary[2] = hits[i].v[j];

            /* Interpolate the normal */
            float avg_normal[3];
            interpolate_normals(bary, triangle, shader, avg_normal);

            /* Get the texture color */
            float tex_color[4] = {0, 0, 0, 0};
            if (mat->texture) {
                /* Interpolate the texture coordinates */
                float avg_texture[4];
                interpolate_texcoords(bary, triangle, shader, avg_texture);
                bilinear_filtering(avg_texture, shader, mat, tex_color);
            }

            float partial[4];
            vector4_scale4(shader->ambient, mat->ambient, partial);
            vector4_scale4(partial, tex_color, partial);

            /* Sum each light contribution */
            for (int l = 0; l < shader->num_lights; l++) {
                if (shadow_hits[l * num_packets + i].intr[j] >= 0) continue;

                float dir_to_light[3];
                {
                    const float* v = shadow_packets[l * num_packets + i].dir + j;
                    const float inv_len = -1.0f / sqrtf(v[0] * v[0] + v[4] * v[4] + v[8] * v[8]);
                    dir_to_light[0] = v[0] * inv_len;
                    dir_to_light[1] = v[4] * inv_len;
                    dir_to_light[2] = v[8] * inv_len;
                }

                float dot_d = vector3_dot(dir_to_light, avg_normal);
                if (dot_d < 0) dot_d = 0;
                dot_d = values[(int)(dot_d * (NUM_CELLS - 1))];
                const float diff[4] =
                {
                    (mat->diffuse[0] + tex_color[0]) * dot_d,
                    (mat->diffuse[1] + tex_color[1]) * dot_d,
                    (mat->diffuse[2] + tex_color[2]) * dot_d,
                    (mat->diffuse[3] + tex_color[3]) * dot_d
                };
              
                partial[0] += diff[0] * shader->lights[l].diffuse[0];
                partial[1] += diff[1] * shader->lights[l].diffuse[1];
                partial[2] += diff[2] * shader->lights[l].diffuse[2];
                partial[3] += diff[3] * shader->lights[l].diffuse[3];
            }

            partial[0] = (partial[0] < 1) ? partial[0] : 1;
            partial[1] = (partial[1] < 1) ? partial[1] : 1;
            partial[2] = (partial[2] < 1) ? partial[2] : 1;
            partial[3] = (partial[3] < 1) ? partial[3] : 1;

            vector4_to_rgb(partial, result + i * 4 + j);
        }
    }
}
