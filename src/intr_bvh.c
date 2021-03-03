#include "intr.h"
#include "mem.h"
#include "linalg.inl"
#include <float.h>
#include <math.h>

typedef struct
{
    const bvh_node* node;
    unsigned int active_ray;
} stack_node;

unsigned int get_first_packet4_hit(const ray_packet4* packets,
                                   const packet4_hit* hits,
                                   unsigned int num_packets,
                                   const float* frustum,
                                   unsigned int active,
                                   const bvh_node* node)
{
    /* First test with the active packet */
    if (intr_packet4_box(packets + active, node->box_min, node->box_max, hits[active].t) != 0)
        return active;

    /* Then test with the frustum */    
    if (!intr_frustum_box(frustum, node->box_min, node->box_max))
        return num_packets;

    /* Test all packets */
    for (unsigned int i = active + 1; i < num_packets; i++) {
        if (intr_packet4_box(packets + i, node->box_min, node->box_max, hits[i].t) != 0) {
            return i;
        }
    }

    return num_packets;
}

unsigned int get_last_packet4_hit(const ray_packet4* packets,
                                  const packet4_hit* hits,
                                  unsigned int num_packets,
                                  const float* frustum,
                                  unsigned int active,
                                  const bvh_node* node)
{
    /* Find last ray hit */
    for (unsigned int i = num_packets - 1; i > active; i--) {
        if (intr_packet4_box(packets + i, node->box_min, node->box_max, hits[i].t) != 0) {
            return i + 1;
        }
    }

    return active + 1;
}

static inline void closest_intr_packet4_node(const bvh_obj* bvh,
                                             const bvh_node* node,
                                             const unsigned int* indices,
                                             const float* vertices,
                                             const ray_packet4* packets,
                                             unsigned int first,
                                             unsigned int last,
                                             packet4_hit* hits)
{
    /* Loop over all triangles in the node */
    for (unsigned int i = first; i < last; i++) {
        const ray_packet4* packet = packets + i;

        for (unsigned int j = 0; j < node->num_tris; j++) {
            const unsigned int tri_id = bvh->tri_ids[node->node_data.tri_id + j];
            
            const float* v0 = vertices + indices[tri_id * 3 + 0] * 4;
            const float* v1 = vertices + indices[tri_id * 3 + 1] * 4;
            const float* v2 = vertices + indices[tri_id * 3 + 2] * 4;
            intr_packet4_tri(packet, v0, v1, v2, hits + i, tri_id);
        }
    }
}

void closest_intr_packet4_bvh(const ray_packet4* packets,
                              unsigned int num_packets,
                              const float* frustum,
                              const bvh_obj* bvh,
                              const unsigned int* indices,
                              const float* vertices,
                              packet4_hit* hits)
{
    stack_node stack[bvh->depth + 1];
    
    for (unsigned int i = 0; i < num_packets; i++) {
        hits[i].t[0] = FLT_MAX;
        hits[i].t[1] = FLT_MAX;
        hits[i].t[2] = FLT_MAX;
        hits[i].t[3] = FLT_MAX;

        hits[i].intr[0] = -1;
        hits[i].intr[1] = -1;
        hits[i].intr[2] = -1;
        hits[i].intr[3] = -1;
    }
    
    /* Push the first node onto the stack */
    unsigned int cur_depth = 1;
    stack[0].node = bvh->root;
    stack[0].active_ray = 0;

    while (cur_depth > 0) {
        /* Pop the next element */
        cur_depth--;
        const bvh_node* node = stack[cur_depth].node;

        unsigned int first = get_first_packet4_hit(packets, hits, num_packets, frustum, stack[cur_depth].active_ray, node);
        if (first < num_packets) {
            if (node->num_tris == 0) {
                /* At least one packet intersected the node, go through children */
                const int order = (packets[first].dir[node->axis] > 0) ? node->order : 1 - node->order;
                const bvh_node* child0 = bvh->root + node->node_data.child + order;
                const bvh_node* child1 = bvh->root + node->node_data.child + 1 - order;

                stack[cur_depth].active_ray = first;
                stack[cur_depth++].node = child1;

                stack[cur_depth].active_ray = first;
                stack[cur_depth++].node = child0;
            } else {
                unsigned int last = get_last_packet4_hit(packets, hits, num_packets, frustum, first, node);
                closest_intr_packet4_node(bvh, node, indices, vertices, packets, first, last, hits);
            }
        }
    }
}

static inline int dead_packet4(const packet4_hit* hit)
{
    /* A packet is dead if all the rays in it have touched a triangle */
    return !((hit->intr[0] & 0x80000000) | (hit->intr[1] & 0x80000000) |
             (hit->intr[2] & 0x80000000) | (hit->intr[3] & 0x80000000));
}

static inline void first_intr_packet4_node(const bvh_obj* bvh,
                                           const bvh_node* node,
                                           const unsigned int* indices,
                                           const float* vertices,
                                           const ray_packet4* packets,
                                           unsigned int first,
                                           unsigned int last,
                                           packet4_hit* hits)
{
    /* For each active packet */
    for (unsigned int i = first; i < last; i++) {
        const ray_packet4* packet = packets + i;
        
        /* Loop over all triangles in the node */
        for (unsigned int j = 0; j < node->num_tris; j++) {
            /* If the packet has already hit a triangle, skip it */
            if (dead_packet4(hits + i)) break;

            const unsigned int tri_id = bvh->tri_ids[node->node_data.tri_id + j];
            
            const float* v0 = vertices + indices[tri_id * 3 + 0] * 4;
            const float* v1 = vertices + indices[tri_id * 3 + 1] * 4;
            const float* v2 = vertices + indices[tri_id * 3 + 2] * 4;
            intr_packet4_tri(packet, v0, v1, v2, hits + i, tri_id);
        }
    }
}

void first_intr_packet4_bvh(const ray_packet4* packets,
                            unsigned int num_packets,
                            const float* frustum,
                            const bvh_obj* bvh,
                            const unsigned int* indices,
                            const float* vertices,
                            packet4_hit* hits)
{
    stack_node stack[bvh->depth + 1];

    const float max_dist = 0.9999f;
    for (unsigned int i = 0; i < num_packets; i++) {
        hits[i].t[0] = max_dist;
        hits[i].t[1] = max_dist;
        hits[i].t[2] = max_dist;
        hits[i].t[3] = max_dist;

        hits[i].intr[0] = -1;
        hits[i].intr[1] = -1;
        hits[i].intr[2] = -1;
        hits[i].intr[3] = -1;
    }

    /* Push the first node onto the stack */
    unsigned int cur_depth = 1;
    stack[0].node = bvh->root;
    stack[0].active_ray = 0;

    /* Store the first and last alive ray */
    unsigned int first_alive = 0;
    unsigned int last_alive = num_packets;

    while (cur_depth > 0 && last_alive > first_alive) {
        /* Pop the next element */
        cur_depth--;
        const bvh_node* node = stack[cur_depth].node;
        unsigned int first = (first_alive > stack[cur_depth].active_ray)
                             ? first_alive
                             : stack[cur_depth].active_ray;

        /* Get the first hit. Start at max(first_alive, first_active) */
        first = get_first_packet4_hit(packets, hits, last_alive, frustum, first, node);

        if (first < last_alive) {
            if (node->num_tris == 0) {
                /* At least one packet intersected the node, go through children */
                const int order = (packets[first].dir[node->axis] > 0) ? node->order : 1 - node->order;
                const bvh_node* child0 = bvh->root + node->node_data.child + order;
                const bvh_node* child1 = bvh->root + node->node_data.child + 1 - order;

                stack[cur_depth].active_ray = first;
                stack[cur_depth++].node = child1;

                stack[cur_depth].active_ray = first;
                stack[cur_depth++].node = child0;
            } else {
                unsigned int last = get_last_packet4_hit(packets, hits, last_alive, frustum, first, node);
                first_intr_packet4_node(bvh, node, indices, vertices, packets, first, last, hits);

                /* Change the first and last alive rays */
                while (first_alive < last_alive && dead_packet4(hits + first_alive)) {
                    first_alive++;
                }

                while (last_alive > first_alive && dead_packet4(hits + last_alive - 1)) {
                    last_alive--;
                }
            }
        }
    }
}
