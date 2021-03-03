#include "../intr.h"
#include "x86.inl"

#define EPSILON 0.00001f

void intr_packet4_tri(const ray_packet4* packet,
                      const float* v0,
                      const float* v1,
                      const float* v2,
                      packet4_hit* hit,
                      int tri_id)
{
    /* e0 = v1 - v0 and e1 = v2 - v0 */
    const __m128 v0_xyz = _mm_load_ps(v0);
    
    __m128 e0_x, e0_y, e0_z;
    __m128 e1_x, e1_y, e1_z;
    {
        const __m128 e0 = _mm_sub_ps(_mm_load_ps(v1), v0_xyz);
        const __m128 e1 = _mm_sub_ps(_mm_load_ps(v2), v0_xyz);

        e1_x = _mm_shuffle_ps(e1, e1, _MM_SHUFFLE(0, 0, 0, 0));
        e1_y = _mm_shuffle_ps(e1, e1, _MM_SHUFFLE(1, 1, 1, 1));
        e1_z = _mm_shuffle_ps(e1, e1, _MM_SHUFFLE(2, 2, 2, 2));

        e0_x = _mm_shuffle_ps(e0, e0, _MM_SHUFFLE(0, 0, 0, 0));
        e0_y = _mm_shuffle_ps(e0, e0, _MM_SHUFFLE(1, 1, 1, 1));
        e0_z = _mm_shuffle_ps(e0, e0, _MM_SHUFFLE(2, 2, 2, 2));
    }

    const __m128 dir_x = _mm_load_ps(packet->dir + 0);
    const __m128 dir_y = _mm_load_ps(packet->dir + 4);
    const __m128 dir_z = _mm_load_ps(packet->dir + 8);

    /* pvec = cross(dir, e1) */
    const __m128 pvec_x = sse_cross(dir_y, e1_z, dir_z, e1_y);
    const __m128 pvec_y = sse_cross(dir_z, e1_x, dir_x, e1_z);
    const __m128 pvec_z = sse_cross(dir_x, e1_y, dir_y, e1_x);

    /* tvec = org - v0 */
    __m128 tvec_x, tvec_y, tvec_z;
    {
        const __m128 org_x = _mm_load_ps(packet->org + 0);
        const __m128 org_y = _mm_load_ps(packet->org + 4);
        const __m128 org_z = _mm_load_ps(packet->org + 8);
        const __m128 v0_x = _mm_shuffle_ps(v0_xyz, v0_xyz, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 v0_y = _mm_shuffle_ps(v0_xyz, v0_xyz, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 v0_z = _mm_shuffle_ps(v0_xyz, v0_xyz, _MM_SHUFFLE(2, 2, 2, 2));

        tvec_x = _mm_sub_ps(org_x, v0_x);
        tvec_y = _mm_sub_ps(org_y, v0_y);
        tvec_z = _mm_sub_ps(org_z, v0_z);
    }
    
    /* inv_det = 1 / dot(e0, pvec) */
    const __m128 det = sse_dot3(e0_x, pvec_x,
                                e0_y, pvec_y,
                                e0_z, pvec_z);
    
    __m128 mask;
    /* det >= EPSILON && det <= -EPSILON */
#ifdef CULLING
    mask = _mm_cmpge_ps(det, _mm_set1_ps(EPSILON));
#else
    mask = _mm_or_ps(_mm_cmpge_ps(det, _mm_set1_ps(EPSILON)),
                     _mm_cmple_ps(det, _mm_set1_ps(-EPSILON)));
#endif
    if (!_mm_movemask_ps(mask))
        return;
    
#ifndef CULLING
    const __m128 inv_det = sse_rcp(det);
#endif
    /* u = dot(pvec, tvec) * inv_det */
    __m128 u = sse_dot3(pvec_x, tvec_x,
                        pvec_y, tvec_y,
                        pvec_z, tvec_z);
#ifndef CULLING
    u = _mm_mul_ps(u, inv_det);
#endif

    /* qvec = cross(tvec, e0) */
    const __m128 qvec_x = sse_cross(tvec_y, e0_z, tvec_z, e0_y);
    const __m128 qvec_y = sse_cross(tvec_z, e0_x, tvec_x, e0_z);
    const __m128 qvec_z = sse_cross(tvec_x, e0_y, tvec_y, e0_x);

    /* v = dot(dir, qvec) * inv_det */
    __m128 v = sse_dot3(dir_x, qvec_x,
                        dir_y, qvec_y,
                        dir_z, qvec_z);
#ifndef CULLING
    v = _mm_mul_ps(v, inv_det);
#endif

#ifdef CULLING
    /* u + v <= det, u >= 0, v >= 0 */
    mask = _mm_and_ps(mask, _mm_cmple_ps(_mm_add_ps(u, v), det));
    mask = _mm_and_ps(mask, _mm_cmpge_ps(u, _mm_setzero_ps()));
    mask = _mm_and_ps(mask, _mm_cmpge_ps(v, _mm_setzero_ps()));
    
    if (!_mm_movemask_ps(mask))
        return;

    const __m128 inv_det = sse_rcp(det);
    v = _mm_mul_ps(v, inv_det);
    u = _mm_mul_ps(u, inv_det);
#else
    /* u + v <= 1.0f, u >= 0, v >= 0 */
    mask = _mm_and_ps(mask, _mm_cmple_ps(_mm_add_ps(u, v), _mm_set1_ps(1.0f)));
    mask = _mm_and_ps(mask, _mm_cmpge_ps(u, _mm_setzero_ps()));
    mask = _mm_and_ps(mask, _mm_cmpge_ps(v, _mm_setzero_ps()));
    
    if (!_mm_movemask_ps(mask))
        return;
#endif
   
    /* t = dot (e1, qvec) * inv_det */
    __m128 t = sse_dot3(e1_x, qvec_x,
                        e1_y, qvec_y,
                        e1_z, qvec_z);
    t = _mm_mul_ps(t, inv_det);

    /* t >= 0, t < prev_t */
    mask = _mm_and_ps(mask, _mm_cmpge_ps(t, _mm_setzero_ps()));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(t, _mm_load_ps(hit->t)));

    int result = _mm_movemask_ps(mask);
#ifndef __SSE4_1__
    if (result & 1) {
        _mm_store_ss(hit->t + 0, t);
        _mm_store_ss(hit->v + 0, v);
        _mm_store_ss(hit->u + 0, u);
        hit->intr[0] = tri_id;
    }

    if (result & 2) {
        _mm_store_ss(hit->t + 1, _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 1, 1, 1)));
        _mm_store_ss(hit->v + 1, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)));
        _mm_store_ss(hit->u + 1, _mm_shuffle_ps(u, u, _MM_SHUFFLE(1, 1, 1, 1)));
        hit->intr[1] = tri_id;
    }

    if (result & 4) {
        _mm_store_ss(hit->t + 2, _mm_shuffle_ps(t, t, _MM_SHUFFLE(2, 2, 2, 2)));
        _mm_store_ss(hit->v + 2, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)));
        _mm_store_ss(hit->u + 2, _mm_shuffle_ps(u, u, _MM_SHUFFLE(2, 2, 2, 2)));
        hit->intr[2] = tri_id;
    }

    if (result & 8) {
        _mm_store_ss(hit->t + 3, _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 3, 3, 3)));
        _mm_store_ss(hit->v + 3, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)));
        _mm_store_ss(hit->u + 3, _mm_shuffle_ps(u, u, _MM_SHUFFLE(3, 3, 3, 3)));
        hit->intr[3] = tri_id;
    }
#else
    if (result & 1) {
        _MM_EXTRACT_FLOAT(hit->t[0], t, 0);
        _MM_EXTRACT_FLOAT(hit->u[0], u, 0);
        _MM_EXTRACT_FLOAT(hit->v[0], v, 0);
        hit->intr[0] = tri_id;
    }

    if (result & 2) {
        _MM_EXTRACT_FLOAT(hit->t[1], t, 1);
        _MM_EXTRACT_FLOAT(hit->u[1], u, 1);
        _MM_EXTRACT_FLOAT(hit->v[1], v, 1);
        hit->intr[1] = tri_id;
    }

    if (result & 4) {
        _MM_EXTRACT_FLOAT(hit->t[2], t, 2);
        _MM_EXTRACT_FLOAT(hit->u[2], u, 2);
        _MM_EXTRACT_FLOAT(hit->v[2], v, 2);
        hit->intr[2] = tri_id;
    }

    if (result & 8) {
        _MM_EXTRACT_FLOAT(hit->t[3], t, 3);
        _MM_EXTRACT_FLOAT(hit->u[3], u, 3);
        _MM_EXTRACT_FLOAT(hit->v[3], v, 3);
        hit->intr[3] = tri_id;
    }
#endif
}

#ifdef __AVX__
void intr_packet8_tri(const ray_packet8* packet,
                      const float* v0,
                      const float* v1,
                      const float* v2,
                      packet8_hit* hit,
                      int tri_id)
{

    /* e0 = v1 - v0 and e1 = v2 - v0 */
    const __m256 v0_xyz = _mm256_broadcast_ps((__m128*)v0);
    
    __m256 e0_x, e0_y, e0_z;
    __m256 e1_x, e1_y, e1_z;
    {
        const __m256 e0 = _mm256_sub_ps(_mm256_broadcast_ps((__m128*)v1), v0_xyz);
        const __m256 e1 = _mm256_sub_ps(_mm256_broadcast_ps((__m128*)v2), v0_xyz);

        e1_x = _mm256_shuffle_ps(e1, e1, _MM_SHUFFLE(0, 0, 0, 0));
        e1_y = _mm256_shuffle_ps(e1, e1, _MM_SHUFFLE(1, 1, 1, 1));
        e1_z = _mm256_shuffle_ps(e1, e1, _MM_SHUFFLE(2, 2, 2, 2));

        e0_x = _mm256_shuffle_ps(e0, e0, _MM_SHUFFLE(0, 0, 0, 0));
        e0_y = _mm256_shuffle_ps(e0, e0, _MM_SHUFFLE(1, 1, 1, 1));
        e0_z = _mm256_shuffle_ps(e0, e0, _MM_SHUFFLE(2, 2, 2, 2));
    }

    const __m256 dir_x = _mm256_load_ps(packet->dir + 0);
    const __m256 dir_y = _mm256_load_ps(packet->dir + 8);
    const __m256 dir_z = _mm256_load_ps(packet->dir + 16);

    /* pvec = cross(dir, e1) */
    const __m256 pvec_x = avx_cross(dir_y, e1_z, dir_z, e1_y);
    const __m256 pvec_y = avx_cross(dir_z, e1_x, dir_x, e1_z);
    const __m256 pvec_z = avx_cross(dir_x, e1_y, dir_y, e1_x);

    /* tvec = org - v0 */
    __m256 tvec_x, tvec_y, tvec_z;
    {
        const __m256 org_x = _mm256_load_ps(packet->org + 0);
        const __m256 org_y = _mm256_load_ps(packet->org + 8);
        const __m256 org_z = _mm256_load_ps(packet->org + 16);
        const __m256 v0_x = _mm256_shuffle_ps(v0_xyz, v0_xyz, _MM_SHUFFLE(0, 0, 0, 0));
        const __m256 v0_y = _mm256_shuffle_ps(v0_xyz, v0_xyz, _MM_SHUFFLE(1, 1, 1, 1));
        const __m256 v0_z = _mm256_shuffle_ps(v0_xyz, v0_xyz, _MM_SHUFFLE(2, 2, 2, 2));

        tvec_x = _mm256_sub_ps(org_x, v0_x);
        tvec_y = _mm256_sub_ps(org_y, v0_y);
        tvec_z = _mm256_sub_ps(org_z, v0_z);
    }
    
    /* inv_det = 1 / dot(e0, pvec) */
    const __m256 det = avx_dot3(e0_x, pvec_x,
                                e0_y, pvec_y,
                                e0_z, pvec_z);
    
    __m256 mask;
    /* det >= EPSILON && det <= -EPSILON */
#ifdef CULLING
    mask = _mm256_cmp_ps(det, _mm256_set1_ps(EPSILON),_CMP_GE_OQ);
#else
    mask = _mm256_or_ps(_mm256_cmp_ps(det, _mm256_set1_ps(EPSILON),  _CMP_GE_OQ)),
                        _mm256_cmp_ps(det, _mm256_set1_ps(-EPSILON), _CMP_LE_OQ));
#endif
    if (!_mm256_movemask_ps(mask))
        return;
    
#ifndef CULLING
    const __m256 inv_det = avx_rcp(det);
#endif
    /* u = dot(pvec, tvec) * inv_det */
    __m256 u = avx_dot3(pvec_x, tvec_x,
                        pvec_y, tvec_y,
                        pvec_z, tvec_z);
#ifndef CULLING
    u = _mm256_mul_ps(u, inv_det);
#endif

    /* qvec = cross(tvec, e0) */
    const __m256 qvec_x = avx_cross(tvec_y, e0_z, tvec_z, e0_y);
    const __m256 qvec_y = avx_cross(tvec_z, e0_x, tvec_x, e0_z);
    const __m256 qvec_z = avx_cross(tvec_x, e0_y, tvec_y, e0_x);

    /* v = dot(dir, qvec) * inv_det */
    __m256 v = avx_dot3(dir_x, qvec_x,
                        dir_y, qvec_y,
                        dir_z, qvec_z);
#ifndef CULLING
    v = _mm256_mul_ps(v, inv_det);
#endif

#ifdef CULLING
    /* u + v <= det, u >= 0, v >= 0 */
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(_mm256_add_ps(u, v), det, _CMP_LE_OQ));
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_GE_OQ));
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OQ));
    
    if (!_mm256_movemask_ps(mask))
        return;

    const __m256 inv_det = avx_rcp(det);
    v = _mm256_mul_ps(v, inv_det);
    u = _mm256_mul_ps(u, inv_det);
#else
    /* u + v <= 1.0f, u >= 0, v >= 0 */
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.0f), _CMP_LE_OQ));
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_GE_OQ));
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OQ));
    
    if (!_mm256_movemask_ps(mask))
        return;
#endif
   
    /* t = dot (e1, qvec) * inv_det */
    __m256 t = avx_dot3(e1_x, qvec_x,
                        e1_y, qvec_y,
                        e1_z, qvec_z);
    t = _mm256_mul_ps(t, inv_det);

    /* t >= 0, t < prev_t */
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(t, _mm256_setzero_ps(), _CMP_GE_OQ));
    mask = _mm256_and_ps(mask, _mm256_cmp_ps(t, _mm256_load_ps(hit->t), _CMP_LT_OQ));

    int result = _mm256_movemask_ps(mask);
    
    if (result & 0x0F) {
        const __m128 t0 = _mm256_castps256_ps128(t);
        const __m128 u0 = _mm256_castps256_ps128(u);
        const __m128 v0 = _mm256_castps256_ps128(v);
        
        if (result & 1) {
            _MM_EXTRACT_FLOAT(hit->t[0], t0, 0);
            _MM_EXTRACT_FLOAT(hit->u[0], u0, 0);
            _MM_EXTRACT_FLOAT(hit->v[0], v0, 0);
            hit->intr[0] = tri_id;
        }

        if (result & 2) {
            _MM_EXTRACT_FLOAT(hit->t[1], t0, 1);
            _MM_EXTRACT_FLOAT(hit->u[1], u0, 1);
            _MM_EXTRACT_FLOAT(hit->v[1], v0, 1);
            hit->intr[1] = tri_id;
        }

        if (result & 4) {
            _MM_EXTRACT_FLOAT(hit->t[2], t0, 2);
            _MM_EXTRACT_FLOAT(hit->u[2], u0, 2);
            _MM_EXTRACT_FLOAT(hit->v[2], v0, 2);
            hit->intr[2] = tri_id;
        }

        if (result & 8) {
            _MM_EXTRACT_FLOAT(hit->t[3], t0, 3);
            _MM_EXTRACT_FLOAT(hit->u[3], u0, 3);
            _MM_EXTRACT_FLOAT(hit->v[3], v0, 3);
            hit->intr[3] = tri_id;
        }
    }
    
    if (result & 0xF0) {
        const __m128 t1 = _mm256_extractf128_ps(t, 1);
        const __m128 u1 = _mm256_extractf128_ps(u, 1);
        const __m128 v1 = _mm256_extractf128_ps(v, 1);
        
        if (result & 16) {
            _MM_EXTRACT_FLOAT(hit->t[4], t1, 0);
            _MM_EXTRACT_FLOAT(hit->u[4], u1, 0);
            _MM_EXTRACT_FLOAT(hit->v[4], v1, 0);
            hit->intr[4] = tri_id;
        }

        if (result & 32) {
            _MM_EXTRACT_FLOAT(hit->t[5], t1, 1);
            _MM_EXTRACT_FLOAT(hit->u[5], u1, 1);
            _MM_EXTRACT_FLOAT(hit->v[5], v1, 1);
            hit->intr[5] = tri_id;
        }

        if (result & 64) {
            _MM_EXTRACT_FLOAT(hit->t[6], t1, 2);
            _MM_EXTRACT_FLOAT(hit->u[6], u1, 2);
            _MM_EXTRACT_FLOAT(hit->v[6], v1, 2);
            hit->intr[6] = tri_id;
        }

        if (result & 128) {
            _MM_EXTRACT_FLOAT(hit->t[7], t1, 3);
            _MM_EXTRACT_FLOAT(hit->u[7], u1, 3);
            _MM_EXTRACT_FLOAT(hit->v[7], v1, 3);
            hit->intr[7] = tri_id;
        }
    }
}
#endif
