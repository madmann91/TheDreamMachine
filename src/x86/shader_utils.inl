/** (C) 2013-2014 MadMann's Company
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */
 
#ifndef DREAM_SHADER_UTILS_INL
#define DREAM_SHADER_UTILS_INL

#include "../shader.h"
#include "x86.inl"

/** Contains the faces, materials and texture flags for a single ray packet */
typedef struct
{
    const face* faces[4];
    const material* mats[4];
    int tex[4];
} packet4_info;

/** Interpolates texture coordinates */
static inline void get_packet4_texcoords(const packet4_info* info,
                                         const shader_info* shader,
                                         __m128 u, __m128 v, __m128 t,
                                         __m128* t_x, __m128* t_y)
{
    const face* f0 = info->faces[0];
    const face* f1 = info->faces[1];
    const face* f2 = info->faces[2];
    const face* f3 = info->faces[3];
    
    /* Interpolate the texture coordinates */
    {
        const __m128 t0 = (info->tex[0]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f0->texcoords[0]) : _mm_setzero_ps();        
        const __m128 t1 = (info->tex[1]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f1->texcoords[0]) : _mm_setzero_ps();
        const __m128 t2 = (info->tex[2]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f2->texcoords[0]) : _mm_setzero_ps();
        const __m128 t3 = (info->tex[3]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f3->texcoords[0]) : _mm_setzero_ps();
        
        const __m128 t01_xxyy = _mm_unpacklo_ps(t0, t1);
        const __m128 t23_xxyy = _mm_unpacklo_ps(t2, t3);
        
        *t_x = _mm_mul_ps(_mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(1, 0, 1, 0)), u);
        *t_y = _mm_mul_ps(_mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(3, 2, 3, 2)), u);
    }
    
    {
        const __m128 t0 = (info->tex[0]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f0->texcoords[1]) : _mm_setzero_ps();        
        const __m128 t1 = (info->tex[1]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f1->texcoords[1]) : _mm_setzero_ps();
        const __m128 t2 = (info->tex[2]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f2->texcoords[1]) : _mm_setzero_ps();
        const __m128 t3 = (info->tex[3]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f3->texcoords[1]) : _mm_setzero_ps();
        
        const __m128 t01_xxyy = _mm_unpacklo_ps(t0, t1);
        const __m128 t23_xxyy = _mm_unpacklo_ps(t2, t3);
        
        *t_x = sse_madd(_mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(1, 0, 1, 0)), v, *t_x);
        *t_y = sse_madd(_mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(3, 2, 3, 2)), v, *t_y);
    }
    
    {
        const __m128 t0 = (info->tex[0]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f0->texcoords[2]) : _mm_setzero_ps();        
        const __m128 t1 = (info->tex[1]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f1->texcoords[2]) : _mm_setzero_ps();
        const __m128 t2 = (info->tex[2]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f2->texcoords[2]) : _mm_setzero_ps();
        const __m128 t3 = (info->tex[3]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f3->texcoords[2]) : _mm_setzero_ps();
        
        const __m128 t01_xxyy = _mm_unpacklo_ps(t0, t1);
        const __m128 t23_xxyy = _mm_unpacklo_ps(t2, t3);
        
        *t_x = sse_madd(_mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(1, 0, 1, 0)), t, *t_x);
        *t_y = sse_madd(_mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(3, 2, 3, 2)), t, *t_y);
    }
}

/** Interpolates & normalizes triangle normals */
static inline void get_packet4_normals(const packet4_info* info,
                                       const shader_info* shader,
                                       __m128 u, __m128 v, __m128 t,
                                       __m128* n_x, __m128* n_y, __m128* n_z)
{
    const face* f0 = info->faces[0];
    const face* f1 = info->faces[1];
    const face* f2 = info->faces[2];
    const face* f3 = info->faces[3];
    
    /* Interpolate the normals */
    {
        const __m128 n0 = (f0 != NULL) ? _mm_load_ps(shader->normals + 4 * f0->normals[0]) : _mm_setzero_ps();        
        const __m128 n1 = (f1 != NULL) ? _mm_load_ps(shader->normals + 4 * f1->normals[0]) : _mm_setzero_ps();
        const __m128 n2 = (f2 != NULL) ? _mm_load_ps(shader->normals + 4 * f2->normals[0]) : _mm_setzero_ps();
        const __m128 n3 = (f3 != NULL) ? _mm_load_ps(shader->normals + 4 * f3->normals[0]) : _mm_setzero_ps();
        
        const __m128 n01_xxyy = _mm_unpacklo_ps(n0, n1);
        const __m128 n23_xxyy = _mm_unpacklo_ps(n2, n3);
        const __m128 n01_zzww = _mm_unpackhi_ps(n0, n1);
        const __m128 n23_zzww = _mm_unpackhi_ps(n2, n3);
        
        *n_x = _mm_mul_ps(_mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(1, 0, 1, 0)), u);
        *n_y = _mm_mul_ps(_mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(3, 2, 3, 2)), u);
        *n_z = _mm_mul_ps(_mm_shuffle_ps(n01_zzww, n23_zzww, _MM_SHUFFLE(1, 0, 1, 0)), u);
    }
    
    {
        const __m128 n0 = (f0 != NULL) ? _mm_load_ps(shader->normals + 4 * f0->normals[1]) : _mm_setzero_ps();        
        const __m128 n1 = (f1 != NULL) ? _mm_load_ps(shader->normals + 4 * f1->normals[1]) : _mm_setzero_ps();
        const __m128 n2 = (f2 != NULL) ? _mm_load_ps(shader->normals + 4 * f2->normals[1]) : _mm_setzero_ps();
        const __m128 n3 = (f3 != NULL) ? _mm_load_ps(shader->normals + 4 * f3->normals[1]) : _mm_setzero_ps();
        
        const __m128 n01_xxyy = _mm_unpacklo_ps(n0, n1);
        const __m128 n23_xxyy = _mm_unpacklo_ps(n2, n3);
        const __m128 n01_zzww = _mm_unpackhi_ps(n0, n1);
        const __m128 n23_zzww = _mm_unpackhi_ps(n2, n3);
        
        *n_x = sse_madd(_mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(1, 0, 1, 0)), v, *n_x);
        *n_y = sse_madd(_mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(3, 2, 3, 2)), v, *n_y);
        *n_z = sse_madd(_mm_shuffle_ps(n01_zzww, n23_zzww, _MM_SHUFFLE(1, 0, 1, 0)), v, *n_z);
    }

    {
        const __m128 n0 = (f0 != NULL) ? _mm_load_ps(shader->normals + 4 * f0->normals[2]) : _mm_setzero_ps();        
        const __m128 n1 = (f1 != NULL) ? _mm_load_ps(shader->normals + 4 * f1->normals[2]) : _mm_setzero_ps();
        const __m128 n2 = (f2 != NULL) ? _mm_load_ps(shader->normals + 4 * f2->normals[2]) : _mm_setzero_ps();
        const __m128 n3 = (f3 != NULL) ? _mm_load_ps(shader->normals + 4 * f3->normals[2]) : _mm_setzero_ps();
        
        const __m128 n01_xxyy = _mm_unpacklo_ps(n0, n1);
        const __m128 n23_xxyy = _mm_unpacklo_ps(n2, n3);
        const __m128 n01_zzww = _mm_unpackhi_ps(n0, n1);
        const __m128 n23_zzww = _mm_unpackhi_ps(n2, n3);
        
        *n_x = sse_madd(_mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(1, 0, 1, 0)), t, *n_x);
        *n_y = sse_madd(_mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(3, 2, 3, 2)), t, *n_y);
        *n_z = sse_madd(_mm_shuffle_ps(n01_zzww, n23_zzww, _MM_SHUFFLE(1, 0, 1, 0)), t, *n_z);
    }
    
    /* Normalize them */
    {
        const __m128 len_n = sse_dot3(*n_x, *n_x,
                                      *n_y, *n_y,
                                      *n_z, *n_z);
    
        const __m128 inv_len = sse_rsqrt(len_n);
        
        *n_x = _mm_mul_ps(*n_x, inv_len);
        *n_y = _mm_mul_ps(*n_y, inv_len);
        *n_z = _mm_mul_ps(*n_z, inv_len);
    }
}

/** Interpolates texture colors */
static inline void packet4_bilinear_filtering(const packet4_info* info,
                                              const shader_info* shader,
                                              __m128 t_x, __m128 t_y,
                                              __m128* col_r, __m128* col_g, __m128* col_b)
{   
    const material* mat0 = info->mats[0];
    const material* mat1 = info->mats[1];
    const material* mat2 = info->mats[2];
    const material* mat3 = info->mats[3];
    
    /* Texture repeat mode */
    __m128 tx_rep, ty_rep;
    {
        const __m128 trunc_x = _mm_cvtepi32_ps(_mm_cvttps_epi32(t_x));
        const __m128 trunc_y = _mm_cvtepi32_ps(_mm_cvttps_epi32(t_y));
        tx_rep = _mm_sub_ps(t_x, trunc_x);
        ty_rep = _mm_sub_ps(t_y, trunc_y);
        
        const __m128 tx_mask = _mm_cmplt_ps(tx_rep, _mm_setzero_ps());
        const __m128 ty_mask = _mm_cmplt_ps(ty_rep, _mm_setzero_ps());
        tx_rep = _mm_add_ps(tx_rep, _mm_and_ps(tx_mask, _mm_set1_ps(1.0f)));
        ty_rep = _mm_add_ps(ty_rep, _mm_and_ps(ty_mask, _mm_set1_ps(1.0f)));
    }
    
    /* Texture clamp mode (u, v) = (tx, ty) > 0 ? ((tx, ty) < 1 ? (tx, ty) : 1) : 0 */
    const __m128 tx_clamp = _mm_min_ps(_mm_max_ps(t_x, _mm_setzero_ps()), _mm_set1_ps(1.0f));
    const __m128 ty_clamp = _mm_min_ps(_mm_max_ps(t_y, _mm_setzero_ps()), _mm_set1_ps(1.0f));
    
    /* Compute right texture coordinates without conditionnals */
    __m128 final_tx, final_ty;
    {
        const __m128i tex_mode = _mm_and_si128(_mm_set_epi32(mat3->tex_mode,
                                                             mat2->tex_mode,
                                                             mat1->tex_mode,
                                                             mat0->tex_mode),
                                               _mm_set1_epi32(TEX_REPEAT));
        const __m128 mode_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(tex_mode, _mm_set1_epi32(0)));
        final_tx = sse_blend(tx_rep, tx_clamp, mode_mask);
        final_ty = sse_blend(ty_rep, ty_clamp, mode_mask);
        
        final_tx = _mm_mul_ps(final_tx, _mm_set_ps(mat3->tex_w - 1,
                                                   mat2->tex_w - 1,
                                                   mat1->tex_w - 1,
                                                   mat0->tex_w - 1));
        
        final_ty = _mm_mul_ps(final_ty, _mm_set_ps(mat3->tex_h - 1,
                                                   mat2->tex_h - 1,
                                                   mat1->tex_h - 1,
                                                   mat0->tex_h - 1));
    }
    
    int SSE_ALIGN(int_x[4]);
    int SSE_ALIGN(int_y[4]);
    const __m128i floor_tx = _mm_cvttps_epi32(final_tx);
    const __m128i floor_ty = _mm_cvttps_epi32(final_ty);
    _mm_store_si128((__m128i*)int_x, floor_tx);
    _mm_store_si128((__m128i*)int_y, floor_ty);

    /* Interpolate pixels horizontally */
    const __m128 u = _mm_sub_ps(final_tx, _mm_cvtepi32_ps(floor_tx));
    const __m128 one_u = _mm_sub_ps(_mm_set1_ps(1.0f), u);
    
    __m128 up_r, up_g, up_b;
    const rgb def_pix = {0xFFFFFFFF};
    {
        const rgb pix0 = (info->tex[0]) ? mat0->texture[int_y[0] * mat0->tex_w + int_x[0] + 0] : def_pix;
        const rgb pix1 = (info->tex[1]) ? mat1->texture[int_y[1] * mat1->tex_w + int_x[1] + 0] : def_pix;
        const rgb pix2 = (info->tex[2]) ? mat2->texture[int_y[2] * mat2->tex_w + int_x[2] + 0] : def_pix;
        const rgb pix3 = (info->tex[3]) ? mat3->texture[int_y[3] * mat3->tex_w + int_x[3] + 0] : def_pix;
        
        up_r = _mm_mul_ps(one_u, _mm_set_ps(pix3.q.r, pix2.q.r, pix1.q.r, pix0.q.r));
        up_g = _mm_mul_ps(one_u, _mm_set_ps(pix3.q.g, pix2.q.g, pix1.q.g, pix0.q.g));
        up_b = _mm_mul_ps(one_u, _mm_set_ps(pix3.q.b, pix2.q.b, pix1.q.b, pix0.q.b));
    }
    
    {
        const rgb pix0 = (info->tex[0]) ? mat0->texture[int_y[0] * mat0->tex_w + int_x[0] + 1] : def_pix;
        const rgb pix1 = (info->tex[1]) ? mat1->texture[int_y[1] * mat1->tex_w + int_x[1] + 1] : def_pix;
        const rgb pix2 = (info->tex[2]) ? mat2->texture[int_y[2] * mat2->tex_w + int_x[2] + 1] : def_pix;
        const rgb pix3 = (info->tex[3]) ? mat3->texture[int_y[3] * mat3->tex_w + int_x[3] + 1] : def_pix;
        
        up_r = sse_madd(u, _mm_set_ps(pix3.q.r, pix2.q.r, pix1.q.r, pix0.q.r), up_r);
        up_g = sse_madd(u, _mm_set_ps(pix3.q.g, pix2.q.g, pix1.q.g, pix0.q.g), up_g);
        up_b = sse_madd(u, _mm_set_ps(pix3.q.b, pix2.q.b, pix1.q.b, pix0.q.b), up_b);
    }
    
    /* Next line */
    int_y[0]++;
    int_y[1]++;
    int_y[2]++;
    int_y[3]++;
    
    __m128 down_r, down_g, down_b;
    {
        const rgb pix0 = (info->tex[0]) ? mat0->texture[int_y[0] * mat0->tex_w + int_x[0] + 0] : def_pix;
        const rgb pix1 = (info->tex[1]) ? mat1->texture[int_y[1] * mat1->tex_w + int_x[1] + 0] : def_pix;
        const rgb pix2 = (info->tex[2]) ? mat2->texture[int_y[2] * mat2->tex_w + int_x[2] + 0] : def_pix;
        const rgb pix3 = (info->tex[3]) ? mat3->texture[int_y[3] * mat3->tex_w + int_x[3] + 0] : def_pix;
        
        down_r = _mm_mul_ps(one_u, _mm_set_ps(pix3.q.r, pix2.q.r, pix1.q.r, pix0.q.r));
        down_g = _mm_mul_ps(one_u, _mm_set_ps(pix3.q.g, pix2.q.g, pix1.q.g, pix0.q.g));
        down_b = _mm_mul_ps(one_u, _mm_set_ps(pix3.q.b, pix2.q.b, pix1.q.b, pix0.q.b));
    }
    
    {
        const rgb pix0 = (info->tex[0]) ? mat0->texture[int_y[0] * mat0->tex_w + int_x[0] + 1] : def_pix;
        const rgb pix1 = (info->tex[1]) ? mat1->texture[int_y[1] * mat1->tex_w + int_x[1] + 1] : def_pix;
        const rgb pix2 = (info->tex[2]) ? mat2->texture[int_y[2] * mat2->tex_w + int_x[2] + 1] : def_pix;
        const rgb pix3 = (info->tex[3]) ? mat3->texture[int_y[3] * mat3->tex_w + int_x[3] + 1] : def_pix;
        
        down_r = sse_madd(u, _mm_set_ps(pix3.q.r, pix2.q.r, pix1.q.r, pix0.q.r), down_r);
        down_g = sse_madd(u, _mm_set_ps(pix3.q.g, pix2.q.g, pix1.q.g, pix0.q.g), down_g);
        down_b = sse_madd(u, _mm_set_ps(pix3.q.b, pix2.q.b, pix1.q.b, pix0.q.b), down_b);
    }
    
    /* Interpolate vertically and normalize */
    {
        const __m128 v = _mm_sub_ps(final_ty, _mm_cvtepi32_ps(floor_ty));
        const __m128 one_v = _mm_sub_ps(_mm_set1_ps(1.0f), v);
    
        *col_r = sse_madd(one_v, up_r, _mm_mul_ps(v, down_r));
        *col_g = sse_madd(one_v, up_g, _mm_mul_ps(v, down_g));
        *col_b = sse_madd(one_v, up_b, _mm_mul_ps(v, down_b));
        
        *col_r = _mm_mul_ps(*col_r, _mm_set1_ps(1.0f / 255.0f));
        *col_g = _mm_mul_ps(*col_g, _mm_set1_ps(1.0f / 255.0f));
        *col_b = _mm_mul_ps(*col_b, _mm_set1_ps(1.0f / 255.0f));
    }
}

#if defined(__AVX__)
/** Contains the faces, materials and texture flags for a single ray packet */
typedef struct
{
    const face* faces[8];
    const material* mats[8];
    int tex[8];
} packet8_info;

/** Interpolates texture coordinates */
static inline void get_packet8_texcoords(const packet8_info* info,
                                         const shader_info* shader,
                                         __m256 u, __m256 v, __m256 t,
                                         __m256* t_x, __m256* t_y)
{
    const face* f0 = info->faces[0];
    const face* f1 = info->faces[1];
    const face* f2 = info->faces[2];
    const face* f3 = info->faces[3];
    const face* f4 = info->faces[4];
    const face* f5 = info->faces[5];
    const face* f6 = info->faces[6];
    const face* f7 = info->faces[7];
    
    /* Interpolate the texture coordinates */
    {
        const __m128 t0 = (info->tex[0]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f0->texcoords[0]) : _mm_setzero_ps();        
        const __m128 t1 = (info->tex[1]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f1->texcoords[0]) : _mm_setzero_ps();
        const __m128 t2 = (info->tex[2]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f2->texcoords[0]) : _mm_setzero_ps();
        const __m128 t3 = (info->tex[3]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f3->texcoords[0]) : _mm_setzero_ps();
        const __m128 t4 = (info->tex[4]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f4->texcoords[0]) : _mm_setzero_ps();        
        const __m128 t5 = (info->tex[5]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f5->texcoords[0]) : _mm_setzero_ps();
        const __m128 t6 = (info->tex[6]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f6->texcoords[0]) : _mm_setzero_ps();
        const __m128 t7 = (info->tex[7]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f7->texcoords[0]) : _mm_setzero_ps();
        
        const __m128 t01_xxyy = _mm_unpacklo_ps(t0, t1);
        const __m128 t23_xxyy = _mm_unpacklo_ps(t2, t3);
        const __m128 t45_xxyy = _mm_unpacklo_ps(t4, t5);
        const __m128 t67_xxyy = _mm_unpacklo_ps(t6, t7);
        
        const __m128 t0123_x = _mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
        const __m128 t4567_x = _mm_shuffle_ps(t45_xxyy, t67_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
        const __m128 t0123_y = _mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
        const __m128 t4567_y = _mm_shuffle_ps(t45_xxyy, t67_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
        
        *t_x = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(t0123_x), t4567_x, 1), u);
        *t_y = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(t0123_y), t4567_y, 1), u);
    }
    
    {
        const __m128 t0 = (info->tex[0]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f0->texcoords[1]) : _mm_setzero_ps();        
        const __m128 t1 = (info->tex[1]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f1->texcoords[1]) : _mm_setzero_ps();
        const __m128 t2 = (info->tex[2]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f2->texcoords[1]) : _mm_setzero_ps();
        const __m128 t3 = (info->tex[3]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f3->texcoords[1]) : _mm_setzero_ps();
        const __m128 t4 = (info->tex[4]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f4->texcoords[1]) : _mm_setzero_ps();        
        const __m128 t5 = (info->tex[5]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f5->texcoords[1]) : _mm_setzero_ps();
        const __m128 t6 = (info->tex[6]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f6->texcoords[1]) : _mm_setzero_ps();
        const __m128 t7 = (info->tex[7]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f7->texcoords[1]) : _mm_setzero_ps();
        
        const __m128 t01_xxyy = _mm_unpacklo_ps(t0, t1);
        const __m128 t23_xxyy = _mm_unpacklo_ps(t2, t3);
        const __m128 t45_xxyy = _mm_unpacklo_ps(t4, t5);
        const __m128 t67_xxyy = _mm_unpacklo_ps(t6, t7);
        
        const __m128 t0123_x = _mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
        const __m128 t4567_x = _mm_shuffle_ps(t45_xxyy, t67_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
        const __m128 t0123_y = _mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
        const __m128 t4567_y = _mm_shuffle_ps(t45_xxyy, t67_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
        
        *t_x = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(t0123_x), t4567_x, 1), v, *t_x);
        *t_y = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(t0123_y), t4567_y, 1), v, *t_y);
    }
    
    {
        const __m128 t0 = (info->tex[0]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f0->texcoords[2]) : _mm_setzero_ps();        
        const __m128 t1 = (info->tex[1]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f1->texcoords[2]) : _mm_setzero_ps();
        const __m128 t2 = (info->tex[2]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f2->texcoords[2]) : _mm_setzero_ps();
        const __m128 t3 = (info->tex[3]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f3->texcoords[2]) : _mm_setzero_ps();
        const __m128 t4 = (info->tex[4]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f4->texcoords[2]) : _mm_setzero_ps();        
        const __m128 t5 = (info->tex[5]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f5->texcoords[2]) : _mm_setzero_ps();
        const __m128 t6 = (info->tex[6]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f6->texcoords[2]) : _mm_setzero_ps();
        const __m128 t7 = (info->tex[7]) ? _mm_loadl_pi(_mm_setzero_ps(), (__m64*)shader->texcoords + f7->texcoords[2]) : _mm_setzero_ps();
        
        const __m128 t01_xxyy = _mm_unpacklo_ps(t0, t1);
        const __m128 t23_xxyy = _mm_unpacklo_ps(t2, t3);
        const __m128 t45_xxyy = _mm_unpacklo_ps(t4, t5);
        const __m128 t67_xxyy = _mm_unpacklo_ps(t6, t7);
        
        const __m128 t0123_x = _mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
        const __m128 t4567_x = _mm_shuffle_ps(t45_xxyy, t67_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
        const __m128 t0123_y = _mm_shuffle_ps(t01_xxyy, t23_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
        const __m128 t4567_y = _mm_shuffle_ps(t45_xxyy, t67_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
        
        *t_x = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(t0123_x), t4567_x, 1), v, *t_x);
        *t_y = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(t0123_y), t4567_y, 1), v, *t_y);
    }
}

/** Interpolates & normalizes triangle normals */
static inline void get_packet8_normals(const packet8_info* info,
                                       const shader_info* shader,
                                       __m256 u, __m256 v, __m256 t,
                                       __m256* n_x, __m256* n_y, __m256* n_z)
{
    const face* f0 = info->faces[0];
    const face* f1 = info->faces[1];
    const face* f2 = info->faces[2];
    const face* f3 = info->faces[3];
    const face* f4 = info->faces[4];
    const face* f5 = info->faces[5];
    const face* f6 = info->faces[6];
    const face* f7 = info->faces[7];
    
    /* Interpolate the normals */
    {
        const __m128 n0 = (f0 != NULL) ? _mm_load_ps(shader->normals + 4 * f0->normals[0]) : _mm_setzero_ps();        
        const __m128 n1 = (f1 != NULL) ? _mm_load_ps(shader->normals + 4 * f1->normals[0]) : _mm_setzero_ps();
        const __m128 n2 = (f2 != NULL) ? _mm_load_ps(shader->normals + 4 * f2->normals[0]) : _mm_setzero_ps();
        const __m128 n3 = (f3 != NULL) ? _mm_load_ps(shader->normals + 4 * f3->normals[0]) : _mm_setzero_ps();
        const __m128 n4 = (f4 != NULL) ? _mm_load_ps(shader->normals + 4 * f4->normals[0]) : _mm_setzero_ps();        
        const __m128 n5 = (f5 != NULL) ? _mm_load_ps(shader->normals + 4 * f5->normals[0]) : _mm_setzero_ps();
        const __m128 n6 = (f6 != NULL) ? _mm_load_ps(shader->normals + 4 * f6->normals[0]) : _mm_setzero_ps();
        const __m128 n7 = (f7 != NULL) ? _mm_load_ps(shader->normals + 4 * f7->normals[0]) : _mm_setzero_ps();
        
        /* X & Y */
        {
            const __m128 n01_xxyy = _mm_unpacklo_ps(n0, n1);
            const __m128 n23_xxyy = _mm_unpacklo_ps(n2, n3);
            const __m128 n45_xxyy = _mm_unpacklo_ps(n4, n5);
            const __m128 n67_xxyy = _mm_unpacklo_ps(n6, n7);
            
            {
                const __m128 n0123_x = _mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
                const __m128 n4567_x = _mm_shuffle_ps(n45_xxyy, n67_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
                *n_x = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_x), n4567_x, 1), u);
            }
            
            {
                const __m128 n0123_y = _mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
                const __m128 n4567_y = _mm_shuffle_ps(n45_xxyy, n67_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
                *n_y = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_y), n4567_y, 1), u);
            }
        }
        
        /* Z */
        {
            const __m128 n01_zzww = _mm_unpackhi_ps(n0, n1);
            const __m128 n23_zzww = _mm_unpackhi_ps(n2, n3);
            const __m128 n45_zzww = _mm_unpackhi_ps(n4, n5);
            const __m128 n67_zzww = _mm_unpackhi_ps(n6, n7);
        
            const __m128 n0123_z = _mm_shuffle_ps(n01_zzww, n23_zzww, _MM_SHUFFLE(1, 0, 1, 0));
            const __m128 n4567_z = _mm_shuffle_ps(n45_zzww, n67_zzww, _MM_SHUFFLE(1, 0, 1, 0));
        
            *n_z = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_z), n4567_z, 1), u);
        }
    }
    
    {
        const __m128 n0 = (f0 != NULL) ? _mm_load_ps(shader->normals + 4 * f0->normals[1]) : _mm_setzero_ps();        
        const __m128 n1 = (f1 != NULL) ? _mm_load_ps(shader->normals + 4 * f1->normals[1]) : _mm_setzero_ps();
        const __m128 n2 = (f2 != NULL) ? _mm_load_ps(shader->normals + 4 * f2->normals[1]) : _mm_setzero_ps();
        const __m128 n3 = (f3 != NULL) ? _mm_load_ps(shader->normals + 4 * f3->normals[1]) : _mm_setzero_ps();
        const __m128 n4 = (f4 != NULL) ? _mm_load_ps(shader->normals + 4 * f4->normals[1]) : _mm_setzero_ps();        
        const __m128 n5 = (f5 != NULL) ? _mm_load_ps(shader->normals + 4 * f5->normals[1]) : _mm_setzero_ps();
        const __m128 n6 = (f6 != NULL) ? _mm_load_ps(shader->normals + 4 * f6->normals[1]) : _mm_setzero_ps();
        const __m128 n7 = (f7 != NULL) ? _mm_load_ps(shader->normals + 4 * f7->normals[1]) : _mm_setzero_ps();
        
        /* X & Y */
        {
            const __m128 n01_xxyy = _mm_unpacklo_ps(n0, n1);
            const __m128 n23_xxyy = _mm_unpacklo_ps(n2, n3);
            const __m128 n45_xxyy = _mm_unpacklo_ps(n4, n5);
            const __m128 n67_xxyy = _mm_unpacklo_ps(n6, n7);
            
            {
                const __m128 n0123_x = _mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
                const __m128 n4567_x = _mm_shuffle_ps(n45_xxyy, n67_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
                *n_x = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_x), n4567_x, 1), u, *n_x);
            }
            
            {
                const __m128 n0123_y = _mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
                const __m128 n4567_y = _mm_shuffle_ps(n45_xxyy, n67_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
                *n_y = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_y), n4567_y, 1), u, *n_y);
            }
        }
        
        /* Z */
        {
            const __m128 n01_zzww = _mm_unpackhi_ps(n0, n1);
            const __m128 n23_zzww = _mm_unpackhi_ps(n2, n3);
            const __m128 n45_zzww = _mm_unpackhi_ps(n4, n5);
            const __m128 n67_zzww = _mm_unpackhi_ps(n6, n7);
        
            const __m128 n0123_z = _mm_shuffle_ps(n01_zzww, n23_zzww, _MM_SHUFFLE(1, 0, 1, 0));
            const __m128 n4567_z = _mm_shuffle_ps(n45_zzww, n67_zzww, _MM_SHUFFLE(1, 0, 1, 0));
        
            *n_z = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_z), n4567_z, 1), u, *n_z);
        }
    }

    {
        const __m128 n0 = (f0 != NULL) ? _mm_load_ps(shader->normals + 4 * f0->normals[2]) : _mm_setzero_ps();        
        const __m128 n1 = (f1 != NULL) ? _mm_load_ps(shader->normals + 4 * f1->normals[2]) : _mm_setzero_ps();
        const __m128 n2 = (f2 != NULL) ? _mm_load_ps(shader->normals + 4 * f2->normals[2]) : _mm_setzero_ps();
        const __m128 n3 = (f3 != NULL) ? _mm_load_ps(shader->normals + 4 * f3->normals[2]) : _mm_setzero_ps();
        const __m128 n4 = (f4 != NULL) ? _mm_load_ps(shader->normals + 4 * f4->normals[2]) : _mm_setzero_ps();        
        const __m128 n5 = (f5 != NULL) ? _mm_load_ps(shader->normals + 4 * f5->normals[2]) : _mm_setzero_ps();
        const __m128 n6 = (f6 != NULL) ? _mm_load_ps(shader->normals + 4 * f6->normals[2]) : _mm_setzero_ps();
        const __m128 n7 = (f7 != NULL) ? _mm_load_ps(shader->normals + 4 * f7->normals[2]) : _mm_setzero_ps();
        
        /* X & Y */
        {
            const __m128 n01_xxyy = _mm_unpacklo_ps(n0, n1);
            const __m128 n23_xxyy = _mm_unpacklo_ps(n2, n3);
            const __m128 n45_xxyy = _mm_unpacklo_ps(n4, n5);
            const __m128 n67_xxyy = _mm_unpacklo_ps(n6, n7);
            
            {
                const __m128 n0123_x = _mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
                const __m128 n4567_x = _mm_shuffle_ps(n45_xxyy, n67_xxyy, _MM_SHUFFLE(1, 0, 1, 0));
                *n_x = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_x), n4567_x, 1), u, *n_x);
            }
            
            {
                const __m128 n0123_y = _mm_shuffle_ps(n01_xxyy, n23_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
                const __m128 n4567_y = _mm_shuffle_ps(n45_xxyy, n67_xxyy, _MM_SHUFFLE(3, 2, 3, 2));
                *n_y = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_y), n4567_y, 1), u, *n_y);
            }
        }
        
        /* Z */
        {
            const __m128 n01_zzww = _mm_unpackhi_ps(n0, n1);
            const __m128 n23_zzww = _mm_unpackhi_ps(n2, n3);
            const __m128 n45_zzww = _mm_unpackhi_ps(n4, n5);
            const __m128 n67_zzww = _mm_unpackhi_ps(n6, n7);
        
            const __m128 n0123_z = _mm_shuffle_ps(n01_zzww, n23_zzww, _MM_SHUFFLE(1, 0, 1, 0));
            const __m128 n4567_z = _mm_shuffle_ps(n45_zzww, n67_zzww, _MM_SHUFFLE(1, 0, 1, 0));
        
            *n_z = avx_madd(_mm256_insertf128_ps(_mm256_castps128_ps256(n0123_z), n4567_z, 1), u, *n_z);
        }
    }
    
    /* Normalize them */
    {
        const __m256 len_n = avx_dot3(*n_x, *n_x,
                                      *n_y, *n_y,
                                      *n_z, *n_z);
    
        const __m256 inv_len = avx_rsqrt(len_n);
        
        *n_x = _mm256_mul_ps(*n_x, inv_len);
        *n_y = _mm256_mul_ps(*n_y, inv_len);
        *n_z = _mm256_mul_ps(*n_z, inv_len);
    }
}

/** Interpolates texture colors */
static inline void packet8_bilinear_filtering(const packet8_info* info,
                                              const shader_info* shader,
                                              __m256 t_x, __m256 t_y,
                                              __m256* col_r, __m256* col_g, __m256* col_b)
{   
    const material* mat0 = info->mats[0];
    const material* mat1 = info->mats[1];
    const material* mat2 = info->mats[2];
    const material* mat3 = info->mats[3];
    const material* mat4 = info->mats[0];
    const material* mat5 = info->mats[1];
    const material* mat6 = info->mats[2];
    const material* mat7 = info->mats[3];
    
    /* Texture repeat mode */
    __m256 tx_rep, ty_rep;
    {
        const __m256 trunc_x = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(t_x));
        const __m256 trunc_y = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(t_y));
        tx_rep = _mm256_sub_ps(t_x, trunc_x);
        ty_rep = _mm256_sub_ps(t_y, trunc_y);
        
        const __m256 tx_mask = _mm256_cmp_ps(tx_rep, _mm256_setzero_ps(),_CMP_LT_OQ);
        const __m256 ty_mask = _mm256_cmp_ps(ty_rep, _mm256_setzero_ps(), _CMP_LT_OQ);
        tx_rep = _mm256_add_ps(tx_rep, _mm256_and_ps(tx_mask, _mm256_set1_ps(1.0f)));
        ty_rep = _mm256_add_ps(ty_rep, _mm256_and_ps(ty_mask, _mm256_set1_ps(1.0f)));
    }
    
    /* Texture clamp mode (u, v) = (tx, ty) > 0 ? ((tx, ty) < 1 ? (tx, ty) : 1) : 0 */
    const __m256 tx_clamp = _mm256_min_ps(_mm256_max_ps(t_x, _mm256_setzero_ps()), _mm256_set1_ps(1.0f));
    const __m256 ty_clamp = _mm256_min_ps(_mm256_max_ps(t_y, _mm256_setzero_ps()), _mm256_set1_ps(1.0f));
    
    /* Compute right texture coordinates without conditionnals */
    __m256 final_tx, final_ty;
    {
        const __m256i tex_mode = avx_andi(avx_set1i(TEX_REPEAT),
                                          avx_seti(mat7->tex_mode,
                                                   mat6->tex_mode,
                                                   mat5->tex_mode,
                                                   mat4->tex_mode,
                                                   mat3->tex_mode,
                                                   mat2->tex_mode,
                                                   mat1->tex_mode,
                                                   mat0->tex_mode));
        const __m256 mode_mask = _mm256_cmp_ps(_mm256_castsi256_ps(tex_mode), _mm256_setzero_ps(), _CMP_EQ_OQ);
        final_tx = _mm256_blendv_ps(tx_rep, tx_clamp, mode_mask);
        final_ty = _mm256_blendv_ps(ty_rep, ty_clamp, mode_mask);
        
        final_tx = _mm256_mul_ps(final_tx, _mm256_set_ps(mat7->tex_w - 1,
                                                         mat6->tex_w - 1,
                                                         mat5->tex_w - 1,
                                                         mat4->tex_w - 1,
                                                         mat3->tex_w - 1,
                                                         mat2->tex_w - 1,
                                                         mat1->tex_w - 1,
                                                         mat0->tex_w - 1));
        
        final_ty = _mm256_mul_ps(final_ty, _mm256_set_ps(mat7->tex_h - 1,
                                                         mat6->tex_h - 1,
                                                         mat5->tex_h - 1,
                                                         mat4->tex_h - 1,
                                                         mat3->tex_h - 1,
                                                         mat2->tex_h - 1,
                                                         mat1->tex_h - 1,
                                                         mat0->tex_h - 1));
    }
    
    int AVX_ALIGN(int_x[8]);
    int AVX_ALIGN(int_y[8]);
    const __m256i floor_tx = _mm256_cvttps_epi32(final_tx);
    const __m256i floor_ty = _mm256_cvttps_epi32(final_ty);
    _mm256_store_si256((__m256i*)int_x, floor_tx);
    _mm256_store_si256((__m256i*)int_y, floor_ty);

    /* Interpolate pixels horizontally */
    const __m256 u = _mm256_sub_ps(final_tx, _mm256_cvtepi32_ps(floor_tx));
    const __m256 one_u = _mm256_sub_ps(_mm256_set1_ps(1.0f), u);
    
    __m256 up_r, up_g, up_b;
    const rgb def_pix = {0xFFFFFFFF};
    {
        const rgb pix0 = (info->tex[0]) ? mat0->texture[int_y[0] * mat0->tex_w + int_x[0] + 0] : def_pix;
        const rgb pix1 = (info->tex[1]) ? mat1->texture[int_y[1] * mat1->tex_w + int_x[1] + 0] : def_pix;
        const rgb pix2 = (info->tex[2]) ? mat2->texture[int_y[2] * mat2->tex_w + int_x[2] + 0] : def_pix;
        const rgb pix3 = (info->tex[3]) ? mat3->texture[int_y[3] * mat3->tex_w + int_x[3] + 0] : def_pix;
        const rgb pix4 = (info->tex[4]) ? mat4->texture[int_y[4] * mat4->tex_w + int_x[4] + 0] : def_pix;
        const rgb pix5 = (info->tex[5]) ? mat5->texture[int_y[5] * mat5->tex_w + int_x[5] + 0] : def_pix;
        const rgb pix6 = (info->tex[6]) ? mat6->texture[int_y[6] * mat6->tex_w + int_x[6] + 0] : def_pix;
        const rgb pix7 = (info->tex[7]) ? mat7->texture[int_y[7] * mat7->tex_w + int_x[7] + 0] : def_pix;
        
        up_r = _mm256_mul_ps(one_u, _mm256_set_ps(pix7.q.r, pix6.q.r, pix5.q.r, pix4.q.r,
                                                  pix3.q.r, pix2.q.r, pix1.q.r, pix0.q.r));
        up_g = _mm256_mul_ps(one_u, _mm256_set_ps(pix7.q.g, pix6.q.g, pix5.q.g, pix4.q.g,
                                                  pix3.q.g, pix2.q.g, pix1.q.g, pix0.q.g));
        up_b = _mm256_mul_ps(one_u, _mm256_set_ps(pix7.q.b, pix6.q.b, pix5.q.b, pix4.q.b,
                                                  pix3.q.b, pix2.q.b, pix1.q.b, pix0.q.b));
    }
    
    {
        const rgb pix0 = (info->tex[0]) ? mat0->texture[int_y[0] * mat0->tex_w + int_x[0] + 1] : def_pix;
        const rgb pix1 = (info->tex[1]) ? mat1->texture[int_y[1] * mat1->tex_w + int_x[1] + 1] : def_pix;
        const rgb pix2 = (info->tex[2]) ? mat2->texture[int_y[2] * mat2->tex_w + int_x[2] + 1] : def_pix;
        const rgb pix3 = (info->tex[3]) ? mat3->texture[int_y[3] * mat3->tex_w + int_x[3] + 1] : def_pix;
        const rgb pix4 = (info->tex[4]) ? mat4->texture[int_y[4] * mat4->tex_w + int_x[4] + 1] : def_pix;
        const rgb pix5 = (info->tex[5]) ? mat5->texture[int_y[5] * mat5->tex_w + int_x[5] + 1] : def_pix;
        const rgb pix6 = (info->tex[6]) ? mat6->texture[int_y[6] * mat6->tex_w + int_x[6] + 1] : def_pix;
        const rgb pix7 = (info->tex[7]) ? mat7->texture[int_y[7] * mat7->tex_w + int_x[7] + 1] : def_pix;
        
        up_r = avx_madd(u, _mm256_set_ps(pix7.q.r, pix6.q.r, pix5.q.r, pix4.q.r,
                                         pix3.q.r, pix2.q.r, pix1.q.r, pix0.q.r), up_r);
        up_g = avx_madd(u, _mm256_set_ps(pix7.q.g, pix6.q.g, pix5.q.g, pix4.q.g,
                                         pix3.q.g, pix2.q.g, pix1.q.g, pix0.q.g), up_g);
        up_b = avx_madd(u, _mm256_set_ps(pix7.q.b, pix6.q.b, pix5.q.b, pix4.q.b,
                                         pix3.q.b, pix2.q.b, pix1.q.b, pix0.q.b), up_b);
    }
    
    /* Next line */
    int_y[0]++;
    int_y[1]++;
    int_y[2]++;
    int_y[3]++;
    int_y[4]++;
    int_y[5]++;
    int_y[6]++;
    int_y[7]++;
    
    __m256 down_r, down_g, down_b;
    {
        const rgb pix0 = (info->tex[0]) ? mat0->texture[int_y[0] * mat0->tex_w + int_x[0] + 0] : def_pix;
        const rgb pix1 = (info->tex[1]) ? mat1->texture[int_y[1] * mat1->tex_w + int_x[1] + 0] : def_pix;
        const rgb pix2 = (info->tex[2]) ? mat2->texture[int_y[2] * mat2->tex_w + int_x[2] + 0] : def_pix;
        const rgb pix3 = (info->tex[3]) ? mat3->texture[int_y[3] * mat3->tex_w + int_x[3] + 0] : def_pix;
        const rgb pix4 = (info->tex[4]) ? mat4->texture[int_y[4] * mat4->tex_w + int_x[4] + 0] : def_pix;
        const rgb pix5 = (info->tex[5]) ? mat5->texture[int_y[5] * mat5->tex_w + int_x[5] + 0] : def_pix;
        const rgb pix6 = (info->tex[6]) ? mat6->texture[int_y[6] * mat6->tex_w + int_x[6] + 0] : def_pix;
        const rgb pix7 = (info->tex[7]) ? mat7->texture[int_y[7] * mat7->tex_w + int_x[7] + 0] : def_pix;
        
        down_r = _mm256_mul_ps(one_u, _mm256_set_ps(pix7.q.r, pix6.q.r, pix5.q.r, pix4.q.r,
                                                    pix3.q.r, pix2.q.r, pix1.q.r, pix0.q.r));
        down_g = _mm256_mul_ps(one_u, _mm256_set_ps(pix7.q.g, pix6.q.g, pix5.q.g, pix4.q.g,
                                                    pix3.q.g, pix2.q.g, pix1.q.g, pix0.q.g));
        down_b = _mm256_mul_ps(one_u, _mm256_set_ps(pix7.q.b, pix6.q.b, pix5.q.b, pix4.q.b,
                                                    pix3.q.b, pix2.q.b, pix1.q.b, pix0.q.b));
    }
    
    {
        const rgb pix0 = (info->tex[0]) ? mat0->texture[int_y[0] * mat0->tex_w + int_x[0] + 1] : def_pix;
        const rgb pix1 = (info->tex[1]) ? mat1->texture[int_y[1] * mat1->tex_w + int_x[1] + 1] : def_pix;
        const rgb pix2 = (info->tex[2]) ? mat2->texture[int_y[2] * mat2->tex_w + int_x[2] + 1] : def_pix;
        const rgb pix3 = (info->tex[3]) ? mat3->texture[int_y[3] * mat3->tex_w + int_x[3] + 1] : def_pix;
        const rgb pix4 = (info->tex[4]) ? mat4->texture[int_y[4] * mat4->tex_w + int_x[4] + 1] : def_pix;
        const rgb pix5 = (info->tex[5]) ? mat5->texture[int_y[5] * mat5->tex_w + int_x[5] + 1] : def_pix;
        const rgb pix6 = (info->tex[6]) ? mat6->texture[int_y[6] * mat6->tex_w + int_x[6] + 1] : def_pix;
        const rgb pix7 = (info->tex[7]) ? mat7->texture[int_y[7] * mat7->tex_w + int_x[7] + 1] : def_pix;
        
        down_r = avx_madd(u, _mm256_set_ps(pix7.q.r, pix6.q.r, pix5.q.r, pix4.q.r,
                                           pix3.q.r, pix2.q.r, pix1.q.r, pix0.q.r), down_r);
        down_g = avx_madd(u, _mm256_set_ps(pix7.q.g, pix6.q.g, pix5.q.g, pix4.q.g,
                                           pix3.q.g, pix2.q.g, pix1.q.g, pix0.q.g), down_g);
        down_b = avx_madd(u, _mm256_set_ps(pix7.q.b, pix6.q.b, pix5.q.b, pix4.q.b,
                                           pix3.q.b, pix2.q.b, pix1.q.b, pix0.q.b), down_b);
    }
    
    /* Interpolate vertically and normalize */
    {
        const __m256 v = _mm256_sub_ps(final_ty, _mm256_cvtepi32_ps(floor_ty));
        const __m256 one_v = _mm256_sub_ps(_mm256_set1_ps(1.0f), v);
    
        *col_r = avx_madd(one_v, up_r, _mm256_mul_ps(v, down_r));
        *col_g = avx_madd(one_v, up_g, _mm256_mul_ps(v, down_g));
        *col_b = avx_madd(one_v, up_b, _mm256_mul_ps(v, down_b));
        
        *col_r = _mm256_mul_ps(*col_r, _mm256_set1_ps(1.0f / 255.0f));
        *col_g = _mm256_mul_ps(*col_g, _mm256_set1_ps(1.0f / 255.0f));
        *col_b = _mm256_mul_ps(*col_b, _mm256_set1_ps(1.0f / 255.0f));
    }
}

#endif // defined(__AVX__)

#endif // DREAM_SHADER_UTILS_INL