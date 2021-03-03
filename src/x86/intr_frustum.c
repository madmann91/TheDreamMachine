#include "../intr.h"
#include "../mem.h"
#include "x86.inl"
#include <immintrin.h>

int intr_frustum_box(const float* frustum, const float* box_min, const float* box_max)
{
    const __m128 bmin = _mm_load_ps(box_min);
    const __m128 bmax = _mm_load_ps(box_max);
    
    /* The fourth component is set to 1 */
#ifdef __SSE4_1__
    const __m128 min = _mm_insert_ps(bmin, _mm_set1_ps(1.0f), (3 << 6) | (3 << 4));
    const __m128 max = _mm_insert_ps(bmax, _mm_set1_ps(1.0f), (3 << 6) | (3 << 4));
#else
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 min_hi = _mm_unpackhi_ps(bmin, one);
    const __m128 max_hi = _mm_unpackhi_ps(bmax, one);    
    const __m128 min = _mm_shuffle_ps(bmin, min_hi, _MM_SHUFFLE(1, 0, 1, 0));
    const __m128 max = _mm_shuffle_ps(bmax, max_hi, _MM_SHUFFLE(1, 0, 1, 0));
#endif
   
    for (int i = 0; i < 6; i++) {
        const __m128 plane = _mm_load_ps(frustum + 4 * i);

        const __m128 mask = _mm_cmplt_ps(plane, _mm_setzero_ps());
        const __m128 n = sse_blend(min, max, mask);
#ifdef __SSE4_1__
        const __m128 dot = _mm_dp_ps(n, plane, 0xF1);
#else
        
        const __m128 d = _mm_mul_ps(n, plane);
        const __m128 d0 = _mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 1, 0, 3));
        const __m128 d1 = _mm_shuffle_ps(d, d, _MM_SHUFFLE(1, 0, 3, 2));
        const __m128 d2 = _mm_shuffle_ps(d, d, _MM_SHUFFLE(0, 3, 2, 1));
        const __m128 dot = _mm_add_ss(_mm_add_ss(d0, d), _mm_add_ss(d1, d2));
#endif
        
        const __m128 ret = _mm_cmpgt_ss(dot, _mm_setzero_ps());
        if (_mm_movemask_ps(ret) & 1)
            return 0;
    }
    
    return 1;
}
