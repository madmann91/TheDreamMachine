#include "bvh.h"
#include "linalg.inl"
#include <float.h>
#include <stdlib.h>

#define CUTOFF_THRESHOLD 4              /* Linear cutoff threshold */

typedef struct
{
    float min[3];
    float max[3];
    float center[3];
} bbox;

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
    float center_min[3], center_max[3];
    vector3_copy(boxes[indices[first]].min, node->box_min);
    vector3_copy(boxes[indices[first]].max, node->box_max);
    vector3_copy(boxes[indices[first]].center, center_min);
    vector3_copy(boxes[indices[first]].center, center_max);

    for (int i = first + 1; i <= last; i++) {
        const bbox* box = boxes + indices[i];
        vector3_min(node->box_min, box->min, node->box_min);
        vector3_max(node->box_max, box->max, node->box_max);
        vector3_min(center_min, box->center, center_min);
        vector3_max(center_max, box->center, center_max);
    }

    if (!depth || last - first <= CUTOFF_THRESHOLD - 1) {
        make_leaf(node, first, last);
        return node_count;
    }
    
    /* Find largest axis */
    float extents[3];
    vector3_subtract(center_max, center_min, extents);
    unsigned int axis = 0;
    if (extents[0] < extents[1])
        axis = 1;
    if (extents[axis] < extents[2])
        axis = 2;
    
    float med = 0.5f * (center_max[axis] + center_min[axis]);

    /* Partition the list into two sets : [0..min_cost - 1] and [min_cost..num_bins - 1] */
    int i = first, j = last;
    while (i <= j) {
        const bbox* box = boxes + indices[i];
        
        if (box->center[axis] > med) {
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
    node->axis = axis;
    node->order = 0;

    /* The resulting partition is [first..i], [i + 1..last] */
    bvh_node* left = root + node_count, *right = left + 1;
    node_count = build_bvh_node(root, left, node_count + 2, indices, first, i - 1, boxes, depth - 1);
    node_count = build_bvh_node(root, right, node_count, indices, i, last, boxes, depth - 1);

    return node_count;
}

/* Compute the bounding boxes of a list of triangles */
void compute_trangles_aabb(const unsigned int* indices, unsigned int num_tris,
                           const float* vertices, bbox* boxes)
{
    for (unsigned int i = 0; i < num_tris; i++) {
        bbox* box = boxes + i;
        const unsigned int* idx = indices + 3 * i;

        vector3_copy(vertices + 4 * idx[0], box->min);
        vector3_copy(vertices + 4 * idx[0], box->max);
        vector3_min(box->min, vertices + 4 * idx[1], box->min);
        vector3_max(box->max, vertices + 4 * idx[1], box->max);
        vector3_min(box->min, vertices + 4 * idx[2], box->min);
        vector3_max(box->max, vertices + 4 * idx[2], box->max);

        vector3_add(box->max, box->min, box->center);
        vector3_scale(box->center, 0.5f, box->center);
    }
}

unsigned int build_bvh(const unsigned int* indices,
                       unsigned int num_tris,
                       const float* vertices,
                       bvh_obj* result) {
    bvh_node* nodes = malloc(sizeof(bvh_node) * (2 * num_tris - 1));
    bbox* boxes = malloc(sizeof(bbox) * num_tris);

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
        max_depth++;
    }

    const unsigned int num_nodes = build_bvh_node(nodes, nodes, 1, tri_ids, 0, num_tris - 1, boxes, max_depth);
    nodes = realloc(nodes, sizeof(bvh_node) * num_nodes);
    free(boxes);

    result->tri_ids = tri_ids;
    result->root = nodes;
    result->depth = max_depth;

    return num_nodes;
}

void destroy_bvh(bvh_obj* bvh)
{
    free(bvh->root);
    free(bvh->tri_ids);
}
