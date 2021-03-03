#include "../intr.h"
#include "x86.inl"

int intr_packet4_box(const ray_packet4* packet,
                     const float* box_min,
                     const float* box_max,
                     const float* prev_tmin)
{
    const __m128 min = _mm_load_ps(box_min);
    const __m128 max = _mm_load_ps(box_max);

    __m128 tmin;
    __m128 tmax;

    {
        const __m128 org_x = _mm_load_ps(packet->org);
        const __m128 inv_x = _mm_load_ps(packet->inv_dir);
        const __m128 min_x = _mm_shuffle_ps(min, min, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 max_x = _mm_shuffle_ps(max, max, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 tmin_x = _mm_mul_ps(_mm_sub_ps(min_x, org_x), inv_x);
        const __m128 tmax_x = _mm_mul_ps(_mm_sub_ps(max_x, org_x), inv_x);
        tmin = _mm_min_ps(tmin_x, tmax_x);
        tmax = _mm_max_ps(tmin_x, tmax_x);
    }

    {
        const __m128 org_y = _mm_load_ps(packet->org + 4);
        const __m128 inv_y = _mm_load_ps(packet->inv_dir + 4);
        const __m128 min_y = _mm_shuffle_ps(min, min, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 max_y = _mm_shuffle_ps(max, max, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 tmin_y = _mm_mul_ps(_mm_sub_ps(min_y, org_y), inv_y);
        const __m128 tmax_y = _mm_mul_ps(_mm_sub_ps(max_y, org_y), inv_y);
        tmin = _mm_max_ps(tmin, _mm_min_ps(tmin_y, tmax_y));
        tmax = _mm_min_ps(tmax, _mm_max_ps(tmin_y, tmax_y));
    }

    {
        const __m128 org_z = _mm_load_ps(packet->org + 8);
        const __m128 inv_z = _mm_load_ps(packet->inv_dir + 8);
        const __m128 min_z = _mm_shuffle_ps(min, min, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 max_z = _mm_shuffle_ps(max, max, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 tmin_z = _mm_mul_ps(_mm_sub_ps(min_z, org_z), inv_z);
        const __m128 tmax_z = _mm_mul_ps(_mm_sub_ps(max_z, org_z), inv_z);
        tmin = _mm_max_ps(tmin, _mm_min_ps(tmin_z, tmax_z));
        tmax = _mm_min_ps(tmax, _mm_max_ps(tmin_z, tmax_z));
    }
   
    __m128 hit = _mm_and_ps(_mm_cmpge_ps(tmax, tmin),
                            _mm_cmpge_ps(tmax, _mm_setzero_ps()));
    hit = _mm_and_ps(hit, _mm_cmple_ps(tmin, _mm_load_ps(prev_tmin)));
    
    return _mm_movemask_ps(hit);
}

#if defined(__AVX__)
int intr_packet8_box(const ray_packet8* packet,
                     const float* box_min,
                     const float* box_max,
                     const float* prev_tmin)
{

    const __m256 min = _mm256_broadcast_ps((__m128*)box_min);
    const __m256 max = _mm256_broadcast_ps((__m128*)box_max);

    __m256 tmin;
    __m256 tmax;

    {
        const __m256 org_x = _mm256_load_ps(packet->org);
        const __m256 inv_x = _mm256_load_ps(packet->inv_dir);
        const __m256 min_x = _mm256_shuffle_ps(min, min, _MM_SHUFFLE(0, 0, 0, 0));
        const __m256 max_x = _mm256_shuffle_ps(max, max, _MM_SHUFFLE(0, 0, 0, 0));
        const __m256 tmin_x = _mm256_mul_ps(_mm256_sub_ps(min_x, org_x), inv_x);
        const __m256 tmax_x = _mm256_mul_ps(_mm256_sub_ps(max_x, org_x), inv_x);
        tmin = _mm256_min_ps(tmin_x, tmax_x);
        tmax = _mm256_max_ps(tmin_x, tmax_x);
    }

    {
        const __m256 org_y = _mm256_load_ps(packet->org + 8);
        const __m256 inv_y = _mm256_load_ps(packet->inv_dir + 8);
        const __m256 min_y = _mm256_shuffle_ps(min, min, _MM_SHUFFLE(1, 1, 1, 1));
        const __m256 max_y = _mm256_shuffle_ps(max, max, _MM_SHUFFLE(1, 1, 1, 1));
        const __m256 tmin_y = _mm256_mul_ps(_mm256_sub_ps(min_y, org_y), inv_y);
        const __m256 tmax_y = _mm256_mul_ps(_mm256_sub_ps(max_y, org_y), inv_y);
        tmin = _mm256_max_ps(tmin, _mm256_min_ps(tmin_y, tmax_y));
        tmax = _mm256_min_ps(tmax, _mm256_max_ps(tmin_y, tmax_y));
    }

    {
        const __m256 org_z = _mm256_load_ps(packet->org + 16);
        const __m256 inv_z = _mm256_load_ps(packet->inv_dir + 16);
        const __m256 min_z = _mm256_shuffle_ps(min, min, _MM_SHUFFLE(2, 2, 2, 2));
        const __m256 max_z = _mm256_shuffle_ps(max, max, _MM_SHUFFLE(2, 2, 2, 2));
        const __m256 tmin_z = _mm256_mul_ps(_mm256_sub_ps(min_z, org_z), inv_z);
        const __m256 tmax_z = _mm256_mul_ps(_mm256_sub_ps(max_z, org_z), inv_z);
        tmin = _mm256_max_ps(tmin, _mm256_min_ps(tmin_z, tmax_z));
        tmax = _mm256_min_ps(tmax, _mm256_max_ps(tmin_z, tmax_z));
    }
   
    __m256 hit = _mm256_and_ps(_mm256_cmp_ps(tmax, tmin, _CMP_GE_OQ),
                               _mm256_cmp_ps(tmax, _mm256_setzero_ps(), _CMP_GE_OQ));
    hit = _mm256_and_ps(hit, _mm256_cmp_ps(tmin, _mm256_load_ps(prev_tmin), _CMP_GE_OQ));
    
    return _mm256_movemask_ps(hit);
}
#endif // defined(__AVX__)
