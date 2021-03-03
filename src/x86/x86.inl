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

#ifndef DREAM_X86_INL
#define DREAM_X86_INL

#include "../mem.h"
#include <x86intrin.h>

#define SSE_SIGNMASK    (_mm_set1_ps(-0.0f))

/** Multiply add : a * b + c */
static inline  __m128 sse_madd(__m128 a, __m128 b, __m128 c)
{
#if defined(__FMA4__)
    return _mm_macc_ps(a, b, c);
#elif defined (__FMA__)
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

/** Multiply subtract : a * b - c */
static inline  __m128 sse_msub(__m128 a, __m128 b, __m128 c)
{
#if defined(__FMA4__)
    return _mm_msub_ps(a, b, c);
#elif defined (__FMA__)
    return _mm_fmsub_ps(a, b, c);
#else
    return _mm_sub_ps(_mm_mul_ps(a, b), c);
#endif
}

/** Negated multiply add : -(a * b) + c */
static inline  __m128 sse_nmadd(__m128 a, __m128 b, __m128 c)
{
#if defined(__FMA4__)
    return _mm_nmacc_ps(a, b, c);
#elif defined (__FMA__)
    return _mm_fnmadd_ps(a, b, c);
#else
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
#endif
}

/** Blend : (mask) ? b : a */
static inline __m128 sse_blend(__m128 a, __m128 b, __m128 mask)
{
#if defined(__SSE4_1__)
    return _mm_blendv_ps(a, b, mask);
#else
    return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
#endif
}

/** Part of cross product */
static inline __m128 sse_cross(__m128 a, __m128 b,
                               __m128 c, __m128 d)
{
    return sse_msub(a, b, _mm_mul_ps(c, d));
}

/** Negate 4 floats */
static inline __m128 sse_negate(__m128 a)
{
    return _mm_xor_ps(a, SSE_SIGNMASK);
}

/** 4 dot3 products */
static inline __m128 sse_dot3(__m128 ax, __m128 bx,
                              __m128 ay, __m128 by,
                              __m128 az, __m128 bz)
{
    return sse_madd(ax, bx, sse_madd(ay, by, _mm_mul_ps(az, bz)));
}

/** 4 dot4 products */
static inline __m128 sse_dot4(__m128 ax, __m128 bx,
                              __m128 ay, __m128 by,
                              __m128 az, __m128 bz,
                              __m128 aw, __m128 bw)
{
    return sse_madd(ax, bx, sse_madd(ay, by, sse_madd(az, bz, _mm_mul_ps(aw, bw))));
}

/** 4 components reciprocal - precise */
static inline __m128 sse_rcp(__m128 a)
{
    const __m128 r = _mm_rcp_ps(a);
    const __m128 r2 = _mm_mul_ps(a, r);
    const __m128 r3 = _mm_add_ps(r, r);
    return sse_nmadd(r2, r, r3);
}

/** 4 components reciprocal of square root - precise */
static inline __m128 sse_rsqrt(__m128 a)
{
    __m128 r = _mm_rsqrt_ps(a);
    const __m128 r1 = sse_nmadd(_mm_mul_ps(r, r), a, _mm_set1_ps(3.0f)); 
    r = _mm_mul_ps(r, _mm_mul_ps(r1, _mm_set1_ps(0.5f)));
    const __m128 r2 = sse_nmadd(_mm_mul_ps(r, r), a, _mm_set1_ps(3.0f)); 
    return _mm_mul_ps(r, _mm_mul_ps(r2, _mm_set1_ps(0.5f)));
}

/** Vector absolute value */
static inline __m128 sse_abs(__m128 a)
{
    return _mm_andnot_ps(SSE_SIGNMASK, a);
}

/** 4 components floor function */
static inline __m128 sse_floor(__m128 x)
{
    const __m128 tmp  = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
    return _mm_sub_ps(tmp, _mm_and_ps(_mm_cmpgt_ps(tmp, x), _mm_set1_ps(1.0f)));
}

/** 4 components integer power of 2 - exact */
static inline __m128 sse_pow2i(__m128i n)
{
    n = _mm_add_epi32(n, _mm_set1_epi32(0x7f));
    n = _mm_slli_epi32(n, 23);
    return _mm_castsi128_ps(n);
}

/** 4 components exponential function */
static inline __m128 sse_exp(__m128 x)
{
    const __m128 exp_hi = _mm_set1_ps(88.3762626647949f);
    const __m128 exp_lo = _mm_set1_ps(-88.3762626647949f);
    x = _mm_min_ps(x, exp_hi);
    x = _mm_max_ps(x, exp_lo);

    /* Express e**x = e**g 2**n
     *              = e**g e**( n loge(2) )
     *              = e**( g + n loge(2) )
     */
    const __m128 cephes_log2f = _mm_set1_ps(1.44269504088896341f);
    const __m128 z = sse_floor(sse_madd(cephes_log2f, x, _mm_set1_ps(0.5f)));   
    const __m128i n = _mm_cvttps_epi32(z);
    
    const __m128 cephes_c1 = _mm_set1_ps(0.693359375f);
    const __m128 cephes_c2 =  _mm_set1_ps(-2.12194440e-4f);
    x = sse_nmadd(z, cephes_c1, x);
    x = sse_nmadd(z, cephes_c2, x);    
    const __m128 xx = _mm_mul_ps(x, x);
    
    /* Theoretical peak relative error in [-0.5, +0.5] is 4.2e-9. */
    __m128 e = _mm_set1_ps(1.9875691500E-4f);
    e = sse_madd(e, x, _mm_set1_ps(1.3981999507E-3f));
    e = sse_madd(e, x, _mm_set1_ps(8.3334519073E-3f));
    e = sse_madd(e, x, _mm_set1_ps(4.1665795894E-2f));
    e = sse_madd(e, x, _mm_set1_ps(1.6666665459E-1f));
    e = sse_madd(e, x, _mm_set1_ps(5.0000001201E-1f));
    e = sse_madd(e, xx, _mm_add_ps(x, _mm_set1_ps(1.0f)));

    /* Multiply by power of 2 */
    return _mm_mul_ps(e, sse_pow2i(n));
}

/** 4 components logarithm function */
static inline __m128 sse_log(__m128 x)
{
    /* Code taken & adapted from http://gruntthepeon.free.fr/ssemath/ 
     * Under the zlib license
     */
    
    const __m128 invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());
    const __m128 min_norm = _mm_castsi128_ps(_mm_set1_epi32(0x00800000));
    
    /* Cut off denormalized stuff */
    x = _mm_max_ps(x, min_norm);
    __m128i emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
    
    /* Keep only the fractional part */
    const __m128 inv_mant = _mm_castsi128_ps(_mm_set1_epi32(~0x7f800000));
    x = _mm_and_ps(x, inv_mant);
    x = _mm_or_ps(x, _mm_set1_ps(0.5f));

    emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
    __m128 e = _mm_cvtepi32_ps(emm0);
    e = _mm_add_ps(e, _mm_set1_ps(1.0f));

    const __m128 cephes_sqrthf = _mm_set1_ps(0.707106781186547524f);
    __m128 mask = _mm_cmplt_ps(x, cephes_sqrthf);
    const __m128 tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, _mm_set1_ps(1.0f));
    e = _mm_sub_ps(e, _mm_and_ps(_mm_set1_ps(1.0f), mask));
    x = _mm_add_ps(x, tmp);

    __m128 z = _mm_mul_ps(x,x);

    __m128 y = _mm_set1_ps(7.0376836292E-2f);
    y = sse_madd(y, x, _mm_set1_ps(-1.1514610310E-1f));
    y = sse_madd(y, x, _mm_set1_ps(1.1676998740E-1f));
    y = sse_madd(y, x, _mm_set1_ps(-1.2420140846E-1f));
    y = sse_madd(y, x, _mm_set1_ps(+1.4249322787E-1f));
    y = sse_madd(y, x, _mm_set1_ps(-1.6668057665E-1f));
    y = sse_madd(y, x, _mm_set1_ps(+2.0000714765E-1f));
    y = sse_madd(y, x, _mm_set1_ps(-2.4999993993E-1f));
    y = sse_madd(y, x, _mm_set1_ps(+3.3333331174E-1f));
    
    y = _mm_mul_ps(y, x);
    y = _mm_mul_ps(y, z);

    y = sse_madd(e, _mm_set1_ps(-2.12194440e-4f), y);
    y = sse_nmadd(z, _mm_set1_ps(0.5f), y);
    x = _mm_add_ps(x, y);
    x = sse_madd(e, _mm_set1_ps(0.693359375f), x);
    
    /* Negative arg will be NAN */
    x = _mm_or_ps(x, invalid_mask);
    return x;
}

static inline __m128 sse_pow(__m128 x, __m128 y)
{
    return sse_exp(_mm_mul_ps(y, sse_log(x)));
}

#ifdef __AVX__

#define AVX_SIGNMASK    (_mm256_set1_ps(-0.0f))

/* Multiply add : a * b + c */
static inline  __m256 avx_madd(__m256 a, __m256 b, __m256 c)
{
#if defined(__FMA4__)
    return _mm256_macc_ps(a, b, c);
#elif defined (__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

/* Multiply subtract : a * b - c */
static inline  __m256 avx_msub(__m256 a, __m256 b, __m256 c)
{
#if defined(__FMA4__)
    return _mm256_msub_ps(a, b, c);
#elif defined (__FMA__)
    return _mm256_fmsub_ps(a, b, c);
#else
    return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
#endif
}

/* Negated multiply add : -(a * b) + c */
static inline  __m256 avx_nmadd(__m256 a, __m256 b, __m256 c)
{
#if defined(__FMA4__)
    return _mm256_nmacc_ps(a, b, c);
#elif defined (__FMA__)
    return _mm256_fnmadd_ps(a, b, c);
#else
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif
}

/** Part of cross product */
static inline __m256 avx_cross(__m256 a, __m256 b,
                               __m256 c, __m256 d)
{
    return avx_msub(a, b, _mm256_mul_ps(c, d));
}

/** Negate 4 floats */
static inline __m256 avx_negate(__m256 a)
{
    return _mm256_xor_ps(a, AVX_SIGNMASK);
}

/** 4 dot3 products */
static inline __m256 avx_dot3(__m256 ax, __m256 bx,
                              __m256 ay, __m256 by,
                              __m256 az, __m256 bz)
{
    return avx_madd(ax, bx, avx_madd(ay, by, _mm256_mul_ps(az, bz)));
}

/** 4 dot4 products */
static inline __m256 avx_dot4(__m256 ax, __m256 bx,
                              __m256 ay, __m256 by,
                              __m256 az, __m256 bz,
                              __m256 aw, __m256 bw)
{
    return avx_madd(ax, bx, avx_madd(ay, by, avx_madd(az, bz, _mm256_mul_ps(aw, bw))));
}

/** 4 components reciprocal - precise */
static inline __m256 avx_rcp(__m256 a)
{
    const __m256 r = _mm256_rcp_ps(a);
    const __m256 r2 = _mm256_mul_ps(a, r);
    const __m256 r3 = _mm256_add_ps(r, r);
    return avx_nmadd(r2, r, r3);
}

/** 4 components reciprocal of square root - precise */
static inline __m256 avx_rsqrt(__m256 a)
{
    __m256 r = _mm256_rsqrt_ps(a);
    const __m256 r1 = avx_nmadd(_mm256_mul_ps(r, r), a, _mm256_set1_ps(3.0f)); 
    r = _mm256_mul_ps(r, _mm256_mul_ps(r1, _mm256_set1_ps(0.5f)));
    const __m256 r2 = avx_nmadd(_mm256_mul_ps(r, r), a, _mm256_set1_ps(3.0f)); 
    return _mm256_mul_ps(r, _mm256_mul_ps(r2, _mm256_set1_ps(0.5f)));
}

/** Vector absolute value */
static inline __m256 avx_abs(__m256 a)
{
    return _mm256_andnot_ps(AVX_SIGNMASK, a);
}

/** 8 components floor function */
static inline __m256 avx_floor(__m256 x)
{
    const __m128 tmp_low  = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm256_castps256_ps128(x)));
    const __m128 tmp_high = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm256_extractf128_ps(x, 1)));
    const __m256 tmp = _mm256_insertf128_ps(_mm256_castps128_ps256(tmp_low), tmp_high, 1);
    return _mm256_sub_ps(tmp, _mm256_and_ps(_mm256_cmp_ps(tmp, x, _CMP_GT_OQ), _mm256_set1_ps(1.0f)));
}

/** 8 components integer power of 2 - exact */
static inline __m256 avx_pow2i(__m256i n)
{
#if defined(__AVX2__)
    n = _mm256_add_epi32(n, _mm256_set1_epi32(0x7f));
    n = _mm256_slli_epi32(n, 23);
    return _mm256_castsi256_ps(n);
#else
    const __m256 n_ps = _mm256_castsi256_ps(n);
    const __m128i n_low  = _mm_castps_si128(_mm256_castps256_ps128(n_ps));
    const __m128i n_high = _mm_castps_si128(_mm256_extractf128_ps(n_ps, 1));
    const __m128i p_low  = _mm_slli_epi32(_mm_add_epi32(n_low,  _mm_set1_epi32(0x7f)), 23);
    const __m128i p_high = _mm_slli_epi32(_mm_add_epi32(n_high, _mm_set1_epi32(0x7f)), 23);

    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(p_low)),
                                _mm_castsi128_ps(p_high),
                                1);
#endif
}

/** 8 components exponential function */
static inline __m256 avx_exp(__m256 x)
{
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    /* Express e**x = e**g 2**n
     *              = e**g e**( n loge(2) )
     *              = e**( g + n loge(2) )
     */
    const __m256 cephes_log2f = _mm256_set1_ps(1.44269504088896341f);
    const __m256 z = avx_floor(avx_madd(cephes_log2f, x, _mm256_set1_ps(0.5f)));   
    const __m256i n = _mm256_cvttps_epi32(z);
   
    const __m256 cephes_c1 = _mm256_set1_ps(0.693359375f);
    const __m256 cephes_c2 =  _mm256_set1_ps(-2.12194440e-4f);
    x = avx_nmadd(z, cephes_c1, x);
    x = avx_nmadd(z, cephes_c2, x);    
    const __m256 xx = _mm256_mul_ps(x, x);
    
    /* Theoretical peak relative error in [-0.5, +0.5] is 4.2e-9. */
    __m256 e = _mm256_set1_ps(1.9875691500E-4f);
    e = avx_madd(e, x, _mm256_set1_ps(1.3981999507E-3f));
    e = avx_madd(e, x, _mm256_set1_ps(8.3334519073E-3f));
    e = avx_madd(e, x, _mm256_set1_ps(4.1665795894E-2f));
    e = avx_madd(e, x, _mm256_set1_ps(1.6666665459E-1f));
    e = avx_madd(e, x, _mm256_set1_ps(5.0000001201E-1f));
    e = avx_madd(e, xx, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));

    /* Multiply by power of 2 */
    return _mm256_mul_ps(e, avx_pow2i(n));
}

/** Shift left eight integers */
static inline __m256i avx_srli(__m256i x, int shift)
{
#if defined(__AVX2__)
    return _mm256_srli_epi32(x, shift);
#else
    const __m256 x_ps = _mm256_castsi256_ps(x);
    const __m128i x_low  = _mm_castps_si128(_mm256_castps256_ps128(x_ps));
    const __m128i x_high = _mm_castps_si128(_mm256_extractf128_ps(x_ps, 1));
    const __m256 res = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(_mm_srli_epi32(x_low, shift))),
                                            _mm_castsi128_ps(_mm_srli_epi32(x_high, shift)), 1);
    return _mm256_castps_si256(res);
#endif
}

/** Subtract eight integers */
static inline __m256i avx_subi(__m256i a, __m256i b)
{
#if defined(__AVX2__)
    return _mm256_sub_epi32(a, b);
#else
    const __m256 a_ps = _mm256_castsi256_ps(a);
    const __m256 b_ps = _mm256_castsi256_ps(b);
    const __m128i res_low  = _mm_sub_epi32(_mm_castps_si128(_mm256_castps256_ps128(a_ps)),
                                           _mm_castps_si128(_mm256_castps256_ps128(b_ps)));
    const __m128i res_high = _mm_sub_epi32(_mm_castps_si128(_mm256_extractf128_ps(a_ps, 1)),
                                           _mm_castps_si128(_mm256_extractf128_ps(b_ps, 1)));
    const __m256 res = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(res_low)),
                                            _mm_castsi128_ps(res_high), 1);
    return _mm256_castps_si256(res);
#endif
}

/** Performs a biwise and on 8 integers */
static inline __m256i avx_andi(__m256i a, __m256i b)
{
#if defined(__AVX2__)
    return _mm256_and_si256(a, b);
#else
    const __m256 a_ps = _mm256_castsi256_ps(a);
    const __m256 b_ps = _mm256_castsi256_ps(b);
    const __m128i res_low  = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(a_ps)),
                                           _mm_castps_si128(_mm256_castps256_ps128(b_ps)));
    const __m128i res_high = _mm_and_si128(_mm_castps_si128(_mm256_extractf128_ps(a_ps, 1)),
                                           _mm_castps_si128(_mm256_extractf128_ps(b_ps, 1)));
    const __m256 res = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(res_low)),
                                            _mm_castsi128_ps(res_high), 1);
    return _mm256_castps_si256(res);
#endif
}

/** Sets the 8 integers of a register */
static inline __m256i avx_set1i(int x)
{
#if defined(__AVX2__)
    return _mm256_set1_epi32(x);
#else
    union {
        int i;
        float f;
    } value;
    value.i = x;
    return _mm256_castps_si256(_mm256_set1_ps(value.f));
#endif
}

/** Sets the 8 integers of a register */
static inline __m256i avx_seti(int x7, int x6, int x5, int x4,
                               int x3, int x2, int x1, int x0)
{
#if defined(__AVX2__)
    return _mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0);
#else
    union {
        int i[8];
        float f[8];
    } AVX_ALIGN(values);
    values.i[0] = x0;
    values.i[1] = x1;
    values.i[2] = x2;
    values.i[3] = x3;
    values.i[4] = x4;
    values.i[5] = x5;
    values.i[6] = x6;
    values.i[7] = x7;
    return _mm256_castps_si256(_mm256_load_ps(values.f));
#endif
}

/** 8 components logarithm function */
static inline __m256 avx_log(__m256 x)
{
    /* Code taken & adapted from http://gruntthepeon.free.fr/ssemath/ 
     * Under the zlib license
     */
    
    const __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);
    const __m256 min_norm = _mm256_castsi256_ps(avx_set1i(0x00800000));
    
    /* Cut off denormalized stuff */
    x = _mm256_max_ps(x, min_norm);
    __m256i emm0 = avx_srli(_mm256_castps_si256(x), 23);
    
    /* Keep only the fractional part */
    const __m256 inv_mant = _mm256_castsi256_ps(avx_set1i(~0x7f800000));
    x = _mm256_and_ps(x, inv_mant);
    x = _mm256_or_ps(x, _mm256_set1_ps(0.5f));

    emm0 = avx_subi(emm0, avx_set1i(0x7f));
    __m256 e = _mm256_cvtepi32_ps(emm0);
    e = _mm256_add_ps(e, _mm256_set1_ps(1.0f));

    const __m256 cephes_sqrthf = _mm256_set1_ps(0.707106781186547524f);
    __m256 mask = _mm256_cmp_ps(x, cephes_sqrthf, _CMP_LT_OQ);
    const __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, _mm256_set1_ps(1.0f));
    e = _mm256_sub_ps(e, _mm256_and_ps(_mm256_set1_ps(1.0f), mask));
    x = _mm256_add_ps(x, tmp);

    __m256 z = _mm256_mul_ps(x,x);

    __m256 y = _mm256_set1_ps(7.0376836292E-2f);
    y = avx_madd(y, x, _mm256_set1_ps(-1.1514610310E-1f));
    y = avx_madd(y, x, _mm256_set1_ps(1.1676998740E-1f));
    y = avx_madd(y, x, _mm256_set1_ps(-1.2420140846E-1f));
    y = avx_madd(y, x, _mm256_set1_ps(+1.4249322787E-1f));
    y = avx_madd(y, x, _mm256_set1_ps(-1.6668057665E-1f));
    y = avx_madd(y, x, _mm256_set1_ps(+2.0000714765E-1f));
    y = avx_madd(y, x, _mm256_set1_ps(-2.4999993993E-1f));
    y = avx_madd(y, x, _mm256_set1_ps(+3.3333331174E-1f));
    
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);

    y = avx_madd(e, _mm256_set1_ps(-2.12194440e-4f), y);
    y = avx_nmadd(z, _mm256_set1_ps(0.5f), y);
    x = _mm256_add_ps(x, y);
    x = avx_madd(e, _mm256_set1_ps(0.693359375f), x);
    
    /* Negative arg will be NAN */
    x = _mm256_or_ps(x, invalid_mask);
    return x;
}

static inline __m256 avx_pow(__m256 x, __m256 y)
{
    return avx_exp(_mm256_mul_ps(y, avx_log(x)));
}
#endif

#endif // DREAM_X86_INL
