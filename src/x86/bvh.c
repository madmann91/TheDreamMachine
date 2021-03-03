#include "../mem.h"
#include "../bvh.h"
#include "../linalg.inl"
#include <float.h>
#include <xmmintrin.h>
#include <stdint.h>
#include <math.h>

#define BVH_BIN_COUNT 16                /* Number of bins */
#define CUTOFF_THRESHOLD 4              /* Linear cutoff threshold */
#define SAH_EPSILON FLT_EPSILON         /* SAH adjustment factor */

/* 48 bytes aligned */
typedef struct
{
    float min[4];
    float max[4];
    float center[4];
} bbox;

/* 32 bytes aligned */
typedef struct
{
    float min[3];
    int num_tris;
    float max[4];
} sah_bin;

/* Returns half the surface area of a SAH bin */
float half_area(__m128 min, __m128 max)
{
    const __m128 extents = _mm_sub_ps(max, min);
    const __m128 extents2 = _mm_shuffle_ps(extents, extents, _MM_SHUFFLE(3, 0, 2, 1));
    const __m128 area = _mm_mul_ps(extents, extents2);
    const __m128 area0 = _mm_shuffle_ps(area, area, _MM_SHUFFLE(0, 3, 2, 1));
    const __m128 area1 = _mm_shuffle_ps(area, area, _MM_SHUFFLE(1, 0, 3, 2));
    const __m128 ret = _mm_add_ss(area, _mm_add_ss(area0, area1));
    float retf;
    _mm_store_ss(&retf, ret);
    return retf;
}

/* Evaluate the cost of all the possible partitions */
unsigned int evaluate_partitions(const sah_bin* bins,
                                 unsigned int num_bins)
{
    float left_area[num_bins];
    int left_tris[num_bins];
    
    /* Sweep from the left */
    __m128 min = _mm_load_ps(bins[0].min);
    __m128 max = _mm_load_ps(bins[0].max);
    left_area[0] = half_area(min, max);
    left_tris[0] = bins[0].num_tris;
    for (unsigned int i = 1; i < num_bins - 1; i++) {
        min = _mm_min_ps(min, _mm_load_ps(bins[i].min));
        max = _mm_max_ps(max, _mm_load_ps(bins[i].max));

        left_area[i] = half_area(min, max);
        left_tris[i] = left_tris[i - 1] + bins[i].num_tris;
    }

    /* Sweep from the right */
    min = _mm_load_ps(bins[num_bins - 1].min);
    max = _mm_load_ps(bins[num_bins - 1].max);

    float right_area = half_area(min, max);
    int right_tris = bins[num_bins - 1].num_tris;
    float min_cost = right_area * right_tris +
                     left_area[num_bins - 2] * left_tris[num_bins - 2];
    unsigned int min_idx = num_bins - 1;

    for (int i = num_bins - 2; i > 0; i--) {
        min = _mm_min_ps(min, _mm_load_ps(bins[i].min));
        max = _mm_max_ps(max, _mm_load_ps(bins[i].max));

        right_area = half_area(min, max);
        right_tris = right_tris + bins[i].num_tris;

        const float sah_cost = right_area * right_tris +
                               left_area[i - 1] * left_tris[i - 1];
        if (sah_cost < min_cost) {
            min_idx = i;
            min_cost = sah_cost;
        }
    }
    
    return min_idx;
}

/* Makes the given node a leaf */
void make_leaf(bvh_node* node, unsigned int first, unsigned int last)
{
    node->num_tris = last - first + 1;
    node->node_data.tri_id = first;
}

/* Builds a BVH using SAH heuristic & binning */
unsigned int build_bvh_node(bvh_node* root,
                            bvh_node* node,
                            unsigned int node_count,
                            unsigned int* indices,
                            int first, int last,
                            bbox* boxes, unsigned int depth)
{
    /* Compute the bounding box for this node */
    __m128 node_min = _mm_set1_ps(FLT_MAX);
    __m128 node_max = _mm_set1_ps(-FLT_MAX);
    __m128 sse_ctr_min = _mm_set1_ps(FLT_MAX);
    __m128 sse_ctr_max = _mm_set1_ps(-FLT_MAX);

    for (int i = first; i <= last; i++) {
        const __m128 center = _mm_load_ps(boxes[indices[i]].center);
        node_min = _mm_min_ps(node_min, _mm_load_ps(boxes[indices[i]].min));
        node_max = _mm_max_ps(node_max, _mm_load_ps(boxes[indices[i]].max));
        sse_ctr_min = _mm_min_ps(sse_ctr_min, center);
        sse_ctr_max = _mm_max_ps(sse_ctr_max, center);
    }

    _mm_store_ps(node->box_max, node_max);
    _mm_store_ps(node->box_min, node_min);

    if (!depth || last - first <= CUTOFF_THRESHOLD - 1) {
        make_leaf(node, first, last);
        return node_count;
    }

    const __m128 sse_extents = _mm_sub_ps(sse_ctr_max, sse_ctr_min);

    float SSE_ALIGN(center_min[4]);
    float SSE_ALIGN(center_max[4]);
    float SSE_ALIGN(extents[4]);

    _mm_store_ps(center_max, sse_ctr_max);
    _mm_store_ps(center_min, sse_ctr_min); 
    _mm_store_ps(extents, sse_extents);
    
    /* Find largest axis */
    unsigned int axis = 0;
    if (extents[0] < extents[1])
        axis = 1;
    if (extents[axis] < extents[2])
        axis = 2;

    /* Build bins of equal size */
    const unsigned int num_bins = BVH_BIN_COUNT;
    sah_bin SSE_ALIGN(bins[num_bins]);

    for (unsigned int i = 0; i < num_bins; i++) {
        vector3_set_all(FLT_MAX, bins[i].min);
        vector3_set_all(-FLT_MAX, bins[i].max);
        bins[i].num_tris = 0;
    }

    /* Update each bin with the triangles bounds */
    const float sah_factor = num_bins * (1 - SAH_EPSILON) /
        (center_max[axis] - center_min[axis] + SAH_EPSILON);
    const float sah_offset = center_min[axis];

    for (int i = first; i <= last; i++) {
        const unsigned int tri_id = indices[i];
        const int bin_id = sah_factor * (boxes[tri_id].center[axis] - sah_offset);
        
        /* Grow the bin's bounding box */
        vector3_min(bins[bin_id].min, boxes[tri_id].min, bins[bin_id].min);
        vector3_max(bins[bin_id].max, boxes[tri_id].max, bins[bin_id].max);
        bins[bin_id].num_tris++;
    }

    /* Evaluate partition costs */
    unsigned int min_cost = evaluate_partitions(bins, num_bins);

    /* Partition the list into two sets : [0..min_cost - 1] and [min_cost..num_bins - 1] */
    int i = first, j = last;
    while (i <= j) {
        const int bin_id = sah_factor * (boxes[indices[i]].center[axis] - sah_offset);
        
        if (bin_id >= min_cost) {
            unsigned int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
            j--;
        } else {
            i++;
        }
    }

    /* Test if the partition was useful */
    if (i >= last || i <= first) {
        make_leaf(node, first, last);
        return node_count;
    }
    
    /* Setup this node as an inner node */
    node->num_tris = 0;
    node->node_data.child = node_count;

    /* The resulting partition is [first..i], [i + 1..last] */
    bvh_node* left = root + node_count, *right = left + 1;
    node_count = build_bvh_node(root, left, node_count + 2, indices, first, i - 1, boxes, depth - 1);
    node_count = build_bvh_node(root, right, node_count, indices, i, last, boxes, depth - 1);

    float child_distance[3];
    child_distance[0] = right->box_min[0] + right->box_max[0] - left->box_min[0] - left->box_max[0];
    child_distance[1] = right->box_min[1] + right->box_max[1] - left->box_min[1] - left->box_max[1];
    child_distance[2] = right->box_min[2] + right->box_max[2] - left->box_min[2] - left->box_max[2];

    unsigned int order_axis = 0;
    if (fabsf(child_distance[1]) > fabsf(child_distance[0]))
        order_axis = 1;
    if (fabsf(child_distance[2]) > fabsf(child_distance[order_axis]))
        order_axis = 2;

    node->axis = order_axis;
    node->order = (child_distance[order_axis] > 0) ? 0 : 1;

    return node_count;
}

/* Compute the bounding boxes of a list of triangles */
void compute_trangles_aabb(const unsigned int* indices, unsigned int num_tris,
                           const float* vertices, bbox* boxes)
{
    for (unsigned int i = 0; i < num_tris; i++) {
        bbox* box = boxes + i;
        const unsigned int* idx = indices + 3 * i;

        const __m128 vtx0 = _mm_load_ps(vertices + 4 * idx[0]);
        const __m128 vtx1 = _mm_load_ps(vertices + 4 * idx[1]);
        const __m128 vtx2 = _mm_load_ps(vertices + 4 * idx[2]);

        __m128 min = _mm_min_ps(vtx0, vtx1);
        __m128 max = _mm_max_ps(vtx0, vtx1);
        min = _mm_min_ps(min, vtx2);
        max = _mm_max_ps(max, vtx2);
        
        const __m128 center = _mm_mul_ps(_mm_add_ps(min, max), _mm_set1_ps(0.5f));
        
        _mm_store_ps(box->min, min);
        _mm_store_ps(box->max, max);
        _mm_store_ps(box->center, center);
    }
}

unsigned int build_bvh(const unsigned int* indices,
                       unsigned int num_tris,
                       const float* vertices,
                       bvh_obj* result)
{
    bvh_node* nodes = aligned_malloc(sizeof(bvh_node) * (2 * num_tris - 1), 16);
    bbox* boxes = aligned_malloc(sizeof(bbox) * num_tris, 16);

    /* Build bounding boxes */
    compute_trangles_aabb(indices, num_tris, vertices, boxes);

    /* Build indices */
    unsigned int* tri_ids = malloc(sizeof(unsigned int) * num_tris);
    for (unsigned int i = 0; i < num_tris; i++) {
        tri_ids[i] = i;
    }

    /* Find maximum depth */
    unsigned int max_depth = 0, num_log2 = num_tris;
    while (num_log2 > CUTOFF_THRESHOLD) {
        num_log2 >>= 1;
        max_depth += 2;
    }

    const unsigned int num_nodes = build_bvh_node(nodes, nodes, 1, tri_ids, 0, num_tris - 1, boxes, max_depth);
    aligned_free(boxes);

    result->tri_ids = tri_ids;
    result->root = nodes;
    result->depth = max_depth;

    return num_nodes;
}

void destroy_bvh(bvh_obj* bvh)
{
    aligned_free(bvh->root);
    free(bvh->tri_ids);
}
