#include "../shader.h"
#include "../mem.h"
#include "shader_utils.inl"
#include "x86.inl"
#include <math.h>
#include <immintrin.h>

#define NUM_CELLS 8

void packet4_phong_shader(const packet4_hit* hits,
                          const ray_packet4* packets,
                          const packet4_hit* shadow_hits,
                          const ray_packet4* shadow_packets,
                          unsigned int num_packets,
                          const shader_info* shader,
                          rgb* result)
{
    const material def_mat = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, 0, 0, 0, 0, NULL};
    
    for (int i = 0; i < num_packets; i++) {
        /* Skip this packet if no intersection is found */
        if (hits[i].intr[0] < 0 && hits[i].intr[1] < 0 &&
            hits[i].intr[2] < 0 && hits[i].intr[3] < 0) {
            result[4 * i + 0] = shader->bg_color;
            result[4 * i + 1] = shader->bg_color;
            result[4 * i + 2] = shader->bg_color;
            result[4 * i + 3] = shader->bg_color;
            continue;
        }
        
        /* Fill in the packet info */
        const face* f0 = (hits[i].intr[0] >= 0) ? shader->faces + hits[i].intr[0] : NULL;
        const face* f1 = (hits[i].intr[1] >= 0) ? shader->faces + hits[i].intr[1] : NULL;
        const face* f2 = (hits[i].intr[2] >= 0) ? shader->faces + hits[i].intr[2] : NULL;
        const face* f3 = (hits[i].intr[3] >= 0) ? shader->faces + hits[i].intr[3] : NULL;
        
        const material* mat0 = (hits[i].intr[0] >= 0) ? shader->materials + f0->material : &def_mat;
        const material* mat1 = (hits[i].intr[1] >= 0) ? shader->materials + f1->material : &def_mat;
        const material* mat2 = (hits[i].intr[2] >= 0) ? shader->materials + f2->material : &def_mat;
        const material* mat3 = (hits[i].intr[3] >= 0) ? shader->materials + f3->material : &def_mat;
        
        const packet4_info SSE_ALIGN(info) =
        {
            {f0, f1, f2, f3},
            {mat0, mat1, mat2, mat3},
            {(f0 != NULL) && (mat0->texture != NULL),
            (f1 != NULL) && (mat1->texture != NULL),
            (f2 != NULL) && (mat2->texture != NULL),
            (f3 != NULL) && (mat3->texture != NULL)}
        };
        
        /* Compute the ambient color */
        __m128 col_r = _mm_mul_ps(_mm_set1_ps(shader->ambient[0]),
                                  _mm_set_ps(mat3->ambient[0],
                                             mat2->ambient[0],
                                             mat1->ambient[0],
                                             mat0->ambient[0]));
        
        __m128 col_g = _mm_mul_ps(_mm_set1_ps(shader->ambient[1]),
                                  _mm_set_ps(mat3->ambient[1],
                                             mat2->ambient[1],
                                             mat1->ambient[1],
                                             mat0->ambient[1]));
        
        __m128 col_b = _mm_mul_ps(_mm_set1_ps(shader->ambient[2]),
                                  _mm_set_ps(mat3->ambient[2],
                                             mat2->ambient[2],
                                             mat1->ambient[2],
                                             mat0->ambient[2]));
        
        /* Interpolation coefficients */
        const __m128 v = _mm_load_ps(hits[i].u);
        const __m128 t = _mm_load_ps(hits[i].v);
        const __m128 u = _mm_sub_ps(_mm_set1_ps(1), _mm_add_ps(t, v));
        
        /* Find texture color */
        __m128 tex_r, tex_g, tex_b;
        if (info.tex[0] || info.tex[1] || info.tex[2] || info.tex[3]) {
            __m128 t_x, t_y;
            get_packet4_texcoords(&info, shader, u, v, t, &t_x, &t_y);
            packet4_bilinear_filtering(&info, shader, t_x, t_y, &tex_r, &tex_g, &tex_b);
        } else {
            tex_r = _mm_set1_ps(1.0f);
            tex_g = _mm_set1_ps(1.0f);
            tex_b = _mm_set1_ps(1.0f);
        }
        
        /* Compute the interpolated normal */
        __m128 n_x, n_y, n_z;
        get_packet4_normals(&info, shader, u, v, t, &n_x, &n_y, &n_z);
        
        /* Compute the eye direction */
        __m128 deye_x, deye_y, deye_z;
        {
            deye_x = _mm_load_ps(packets[i].dir + 0);
            deye_y = _mm_load_ps(packets[i].dir + 4);
            deye_z = _mm_load_ps(packets[i].dir + 8);
            
            const __m128 len = sse_dot3(deye_x, deye_x,
                                        deye_y, deye_y,
                                        deye_z, deye_z);
            const __m128 factor = sse_negate(sse_rsqrt(len));
            
            deye_x = _mm_mul_ps(deye_x, factor);
            deye_y = _mm_mul_ps(deye_y, factor);
            deye_z = _mm_mul_ps(deye_z, factor);
        }
        
        /* Sum each light contribution */
        for (int l = 0; l < shader->num_lights; l++) {
            const packet4_hit* shadow = shadow_hits + l * num_packets + i;
            
            /* Early exit if no ray is lit in this packet */
            if (shadow->intr[0] >= 0 && shadow->intr[1] >= 0 &&
                shadow->intr[2] >= 0 && shadow->intr[3] >= 0) {
                continue;
            }
            
            /* Compute the light direction */
            __m128 dlight_x, dlight_y, dlight_z;
            {
                dlight_x = _mm_load_ps(shadow_packets[l * num_packets + i].dir + 0);
                dlight_y = _mm_load_ps(shadow_packets[l * num_packets + i].dir + 4);
                dlight_z = _mm_load_ps(shadow_packets[l * num_packets + i].dir + 8);
                
                const __m128 len = sse_dot3(dlight_x, dlight_x,
                                            dlight_y, dlight_y,
                                            dlight_z, dlight_z);
                const __m128 factor = sse_negate(sse_rsqrt(len));
                
                dlight_x = _mm_mul_ps(dlight_x, factor);
                dlight_y = _mm_mul_ps(dlight_y, factor);
                dlight_z = _mm_mul_ps(dlight_z, factor);
            }
            
            /* Diffuse part */
            __m128 light_r, light_g, light_b;
            const __m128 dot_d = _mm_max_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(dlight_x, n_x),
                                                                  _mm_mul_ps(dlight_y, n_y)),
                                                       _mm_mul_ps(dlight_z, n_z)),
                                            _mm_setzero_ps());
            {           
                const __m128 mat_r = _mm_set_ps(mat3->diffuse[0], mat2->diffuse[0], mat1->diffuse[0], mat0->diffuse[0]);
                const __m128 mat_g = _mm_set_ps(mat3->diffuse[1], mat2->diffuse[1], mat1->diffuse[1], mat0->diffuse[1]);
                const __m128 mat_b = _mm_set_ps(mat3->diffuse[2], mat2->diffuse[2], mat1->diffuse[2], mat0->diffuse[2]);
                
                const __m128 diff_r = _mm_mul_ps(dot_d, _mm_mul_ps(mat_r, tex_r));
                const __m128 diff_g = _mm_mul_ps(dot_d, _mm_mul_ps(mat_g, tex_g));
                const __m128 diff_b = _mm_mul_ps(dot_d, _mm_mul_ps(mat_b, tex_b));
                
                light_r = _mm_mul_ps(diff_r, _mm_set1_ps(shader->lights[l].diffuse[0]));
                light_g = _mm_mul_ps(diff_g, _mm_set1_ps(shader->lights[l].diffuse[1]));
                light_b = _mm_mul_ps(diff_b, _mm_set1_ps(shader->lights[l].diffuse[2]));
            }
            
            /* Specular part */
            {
                __m128 rm_x = _mm_mul_ps(dot_d, n_x);
                __m128 rm_y = _mm_mul_ps(dot_d, n_y);
                __m128 rm_z = _mm_mul_ps(dot_d, n_z);
                
                rm_x = _mm_sub_ps(_mm_add_ps(rm_x, rm_x), dlight_x);
                rm_y = _mm_sub_ps(_mm_add_ps(rm_y, rm_y), dlight_y);
                rm_z = _mm_sub_ps(_mm_add_ps(rm_z, rm_z), dlight_z);
                
                const __m128 dot_s = _mm_max_ps(sse_dot3(deye_x, rm_x,
                                                         deye_y, rm_y,
                                                         deye_z, rm_z),
                                                _mm_set1_ps(0.0001f));
                
                const __m128 alpha = _mm_set_ps(mat3->alpha, mat2->alpha, mat1->alpha, mat0->alpha);
                const __m128 factor = sse_pow(dot_s, alpha);
                
                const __m128 mat_r = _mm_set_ps(mat3->specular[0], mat2->specular[0], mat1->specular[0], mat0->specular[0]);
                const __m128 mat_g = _mm_set_ps(mat3->specular[1], mat2->specular[1], mat1->specular[1], mat0->specular[1]);
                const __m128 mat_b = _mm_set_ps(mat3->specular[2], mat2->specular[2], mat1->specular[2], mat0->specular[2]);
                
                light_r = sse_madd(factor, _mm_mul_ps(mat_r, _mm_set1_ps(shader->lights[l].specular[0])), light_r);
                light_g = sse_madd(factor, _mm_mul_ps(mat_g, _mm_set1_ps(shader->lights[l].specular[1])), light_g);
                light_b = sse_madd(factor, _mm_mul_ps(mat_b, _mm_set1_ps(shader->lights[l].specular[2])), light_b);
            }
            
            const __m128 shadow_mask = _mm_castsi128_ps(_mm_cmplt_epi32(_mm_set_epi32(shadow->intr[3],
                                                                                      shadow->intr[2],
                                                                                      shadow->intr[1],
                                                                                      shadow->intr[0]),
                                                                        _mm_setzero_si128()));
            
            col_r = _mm_add_ps(col_r, _mm_and_ps(shadow_mask, light_r));
            col_g = _mm_add_ps(col_g, _mm_and_ps(shadow_mask, light_g));
            col_b = _mm_add_ps(col_b, _mm_and_ps(shadow_mask, light_b));
        }
        
        col_r = _mm_min_ps(col_r, _mm_set1_ps(1.0f));
        col_g = _mm_min_ps(col_g, _mm_set1_ps(1.0f));
        col_b = _mm_min_ps(col_b, _mm_set1_ps(1.0f));
        
        const __m128i icol_r = _mm_cvtps_epi32(_mm_mul_ps(col_r, _mm_set1_ps(255.0f)));
        const __m128i icol_g = _mm_cvtps_epi32(_mm_mul_ps(col_g, _mm_set1_ps(255.0f)));
        const __m128i icol_b = _mm_cvtps_epi32(_mm_mul_ps(col_b, _mm_set1_ps(255.0f)));
        
        const __m128i col = _mm_or_si128(icol_r, _mm_or_si128(_mm_slli_epi32(icol_g, 8),
                                                              _mm_slli_epi32(icol_b, 16)));
        
        const __m128i intr_mask = _mm_cmplt_epi32(_mm_load_si128((__m128i*)hits[i].intr), _mm_set1_epi32(0));
        
        const __m128i final_col = _mm_or_si128(_mm_and_si128(intr_mask, _mm_set1_epi32(shader->bg_color.p)),
                                               _mm_andnot_si128(intr_mask, col));
        _mm_store_si128((__m128i*)result + i, final_col);
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

    const __m128 num_cells = _mm_set1_ps((float)(NUM_CELLS - 1));
    
    const material def_mat = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, 0, 0, 0, 0, NULL};
    
    for (int i = 0; i < num_packets; i++) {
        /* Skip this packet if no intersection is found */
        if (hits[i].intr[0] < 0 && hits[i].intr[1] < 0 &&
            hits[i].intr[2] < 0 && hits[i].intr[3] < 0) {
            result[4 * i + 0] = shader->bg_color;
            result[4 * i + 1] = shader->bg_color;
            result[4 * i + 2] = shader->bg_color;
            result[4 * i + 3] = shader->bg_color;
            continue;
        }
        
        /* Fill in the packet info */
        const face* f0 = (hits[i].intr[0] >= 0) ? shader->faces + hits[i].intr[0] : NULL;
        const face* f1 = (hits[i].intr[1] >= 0) ? shader->faces + hits[i].intr[1] : NULL;
        const face* f2 = (hits[i].intr[2] >= 0) ? shader->faces + hits[i].intr[2] : NULL;
        const face* f3 = (hits[i].intr[3] >= 0) ? shader->faces + hits[i].intr[3] : NULL;
        
        const material* mat0 = (hits[i].intr[0] >= 0) ? shader->materials + f0->material : &def_mat;
        const material* mat1 = (hits[i].intr[1] >= 0) ? shader->materials + f1->material : &def_mat;
        const material* mat2 = (hits[i].intr[2] >= 0) ? shader->materials + f2->material : &def_mat;
        const material* mat3 = (hits[i].intr[3] >= 0) ? shader->materials + f3->material : &def_mat;
        
        const packet4_info SSE_ALIGN(info) =
        {
            {f0, f1, f2, f3},
            {mat0, mat1, mat2, mat3},            
            {(f0 != NULL) && (mat0->texture != NULL),
            (f1 != NULL) && (mat1->texture != NULL),
            (f2 != NULL) && (mat2->texture != NULL),
            (f3 != NULL) && (mat3->texture != NULL)}
        };
        
        /* Compute the ambient color */
        __m128 col_r = _mm_mul_ps(_mm_set1_ps(shader->ambient[0]),
                                  _mm_set_ps(mat3->ambient[0],
                                             mat2->ambient[0],
                                             mat1->ambient[0],
                                             mat0->ambient[0]));
        
        __m128 col_g = _mm_mul_ps(_mm_set1_ps(shader->ambient[1]),
                                  _mm_set_ps(mat3->ambient[1],
                                             mat2->ambient[1],
                                             mat1->ambient[1],
                                             mat0->ambient[1]));
        
        __m128 col_b = _mm_mul_ps(_mm_set1_ps(shader->ambient[2]),
                                  _mm_set_ps(mat3->ambient[2],
                                             mat2->ambient[2],
                                             mat1->ambient[2],
                                             mat0->ambient[2]));
        
        /* Interpolation coefficients */
        const __m128 v = _mm_load_ps(hits[i].u);
        const __m128 t = _mm_load_ps(hits[i].v);
        const __m128 u = _mm_sub_ps(_mm_set1_ps(1), _mm_add_ps(t, v));
        
        /* Find texture color */
        __m128 tex_r, tex_g, tex_b;
        if (info.tex[0] || info.tex[1] || info.tex[2] || info.tex[3]) {
            __m128 t_x, t_y;
            get_packet4_texcoords(&info, shader, u, v, t, &t_x, &t_y);
            packet4_bilinear_filtering(&info, shader, t_x, t_y, &tex_r, &tex_g, &tex_b);
        } else {
            tex_r = _mm_set1_ps(1.0f);
            tex_g = _mm_set1_ps(1.0f);
            tex_b = _mm_set1_ps(1.0f);
        }
        
        /* Compute the interpolated normal */
        __m128 n_x, n_y, n_z;
        get_packet4_normals(&info, shader, u, v, t, &n_x, &n_y, &n_z);
        
        /* Compute the eye direction */
        __m128 deye_x, deye_y, deye_z;
        {
            deye_x = _mm_load_ps(packets[i].dir + 0);
            deye_y = _mm_load_ps(packets[i].dir + 4);
            deye_z = _mm_load_ps(packets[i].dir + 8);
            
            const __m128 len = sse_dot3(deye_x, deye_x,
                                        deye_y, deye_y,
                                        deye_z, deye_z);
            const __m128 factor = sse_negate(sse_rsqrt(len));
            
            deye_x = _mm_mul_ps(deye_x, factor);
            deye_y = _mm_mul_ps(deye_y, factor);
            deye_z = _mm_mul_ps(deye_z, factor);
        }
        
        /* Sum each light contribution */
        for (int l = 0; l < shader->num_lights; l++) {
            const packet4_hit* shadow = shadow_hits + l * num_packets + i;
            
            /* Early exit if no ray is lit in this packet */
            if (shadow->intr[0] >= 0 && shadow->intr[1] >= 0 &&
                shadow->intr[2] >= 0 && shadow->intr[3] >= 0) {
                continue;
            }
            
            /* Compute the light direction */
            __m128 dlight_x, dlight_y, dlight_z;
            {
                dlight_x = _mm_load_ps(shadow_packets[l * num_packets + i].dir + 0);
                dlight_y = _mm_load_ps(shadow_packets[l * num_packets + i].dir + 4);
                dlight_z = _mm_load_ps(shadow_packets[l * num_packets + i].dir + 8);
                
                const __m128 len = sse_dot3(dlight_x, dlight_x,
                                            dlight_y, dlight_y,
                                            dlight_z, dlight_z);
                const __m128 factor = sse_negate(sse_rsqrt(len));
                
                dlight_x = _mm_mul_ps(dlight_x, factor);
                dlight_y = _mm_mul_ps(dlight_y, factor);
                dlight_z = _mm_mul_ps(dlight_z, factor);
            }
            
            /* Diffuse part */
            __m128 light_r, light_g, light_b;
            __m128 dot_d = _mm_max_ps(sse_dot3(dlight_x, n_x,
                                               dlight_y, n_y,
                                               dlight_z, n_z),
                                      _mm_setzero_ps());
            __m128i cell_id = _mm_cvtps_epi32(_mm_mul_ps(dot_d, num_cells));
            int SSE_ALIGN(ids[4]);
            _mm_store_si128((__m128i*)ids, cell_id);
            dot_d = _mm_set_ps(values[ids[3]],
                               values[ids[2]],
                               values[ids[1]],
                               values[ids[0]]);
            
            {           
                const __m128 mat_r = _mm_set_ps(mat3->diffuse[0], mat2->diffuse[0], mat1->diffuse[0], mat0->diffuse[0]);
                const __m128 mat_g = _mm_set_ps(mat3->diffuse[1], mat2->diffuse[1], mat1->diffuse[1], mat0->diffuse[1]);
                const __m128 mat_b = _mm_set_ps(mat3->diffuse[2], mat2->diffuse[2], mat1->diffuse[2], mat0->diffuse[2]);
                
                const __m128 diff_r = _mm_mul_ps(dot_d, _mm_mul_ps(mat_r, tex_r));
                const __m128 diff_g = _mm_mul_ps(dot_d, _mm_mul_ps(mat_g, tex_g));
                const __m128 diff_b = _mm_mul_ps(dot_d, _mm_mul_ps(mat_b, tex_b));
                
                light_r = _mm_mul_ps(diff_r, _mm_set1_ps(shader->lights[l].diffuse[0]));
                light_g = _mm_mul_ps(diff_g, _mm_set1_ps(shader->lights[l].diffuse[1]));
                light_b = _mm_mul_ps(diff_b, _mm_set1_ps(shader->lights[l].diffuse[2]));
            }
            
            const __m128 shadow_mask = _mm_castsi128_ps(_mm_cmplt_epi32(_mm_set_epi32(shadow->intr[3],
                                                                                      shadow->intr[2],
                                                                                      shadow->intr[1],
                                                                                      shadow->intr[0]),
                                                                        _mm_setzero_si128()));
            
            col_r = _mm_add_ps(col_r, _mm_and_ps(shadow_mask, light_r));
            col_g = _mm_add_ps(col_g, _mm_and_ps(shadow_mask, light_g));
            col_b = _mm_add_ps(col_b, _mm_and_ps(shadow_mask, light_b));
        }
        
        col_r = _mm_min_ps(col_r, _mm_set1_ps(1.0f));
        col_g = _mm_min_ps(col_g, _mm_set1_ps(1.0f));
        col_b = _mm_min_ps(col_b, _mm_set1_ps(1.0f));
        
        const __m128i icol_r = _mm_cvtps_epi32(_mm_mul_ps(col_r, _mm_set1_ps(255.0f)));
        const __m128i icol_g = _mm_cvtps_epi32(_mm_mul_ps(col_g, _mm_set1_ps(255.0f)));
        const __m128i icol_b = _mm_cvtps_epi32(_mm_mul_ps(col_b, _mm_set1_ps(255.0f)));
        
        const __m128i col = _mm_or_si128(icol_r, _mm_or_si128(_mm_slli_epi32(icol_g, 8),
                                                              _mm_slli_epi32(icol_b, 16)));
        
        const __m128i intr_mask = _mm_cmplt_epi32(_mm_load_si128((__m128i*)hits[i].intr), _mm_set1_epi32(0));
        
        const __m128i final_col = _mm_or_si128(_mm_and_si128(intr_mask, _mm_set1_epi32(shader->bg_color.p)),
                                               _mm_andnot_si128(intr_mask, col));
        _mm_store_si128((__m128i*)result + i, final_col);
    }
}
