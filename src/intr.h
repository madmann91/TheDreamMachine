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

#ifndef DREAM_INTR_H
#define DREAM_INTR_H

#include "bvh.h"
#include <stdint.h>

/** Precomputed triangle data */
typedef struct
{
    int empty;
} tri_data;

/** Packet of 4 rays (components stored as xxxx yyyy zzzz) */
typedef struct
{
    float org[4 * 3];          /**< Rays origins */
    float dir[4 * 3];          /**< Rays directions */
    float inv_dir[4 * 3];      /**< Precomputed inverse coordinates of dir */
} ray_packet4;

/** Packet of 4 rays intersection result */
typedef struct 
{
    float t[4];               /**< Intersection times */ 
    float u[4];               /**< Barycentric coordinates */ 
    float v[4];               /**< Barycentric coordinates */
    int32_t intr[4];          /**< Index of intersected triangle or -1 */
} packet4_hit;

/** Packet of 8 rays (components stored as xxxxxxxx yyyyyyyy zzzzzzzz) */
typedef struct
{
    float org[8 * 3];          /**< Rays origins */
    float dir[8 * 3];          /**< Rays directions */
    float inv_dir[8 * 3];      /**< Precomputed inverse coordinates of dir */
} ray_packet8;

/** Packet of 8 rays intersection result */
typedef struct 
{
    float t[8];               /**< Intersection times */ 
    float u[8];               /**< Barycentric coordinates */ 
    float v[8];               /**< Barycentric coordinates */
    int32_t intr[8];          /**< Index of intersected triangle or -1 */
} packet8_hit;

/** Intersects a triangle with a ray packet
 * \param[in]  packet    Packet of rays (16 bytes aligned)
 * \param[in]  v0        First triangle vertex (16 bytes aligned)
 * \param[in]  v1        Second triangle vertex (16 bytes aligned)
 * \param[in]  v2        Third triangle vertex (16 bytes aligned)
 * \param[in]  tri_id    Triangle index (to fill the intr field in the hit result)
 * \param[out] hit       Result of the intersection (16 bytes aligned)
 */
void intr_packet4_tri(const ray_packet4* packet,
                      const float* v0,
                      const float* v1,
                      const float* v2,
                      packet4_hit* hit,
                      int tri_id);

/** Intersects a triangle with a ray packet
 * \param[in]  packet    Packet of rays (32 bytes aligned)
 * \param[in]  v0        First triangle vertex (16 bytes aligned)
 * \param[in]  v1        Second triangle vertex (16 bytes aligned)
 * \param[in]  v2        Third triangle vertex (16 bytes aligned)
 * \param[in]  tri_id    Triangle index (to fill the intr field in the hit result)
 * \param[out] hit       Result of the intersection (32 bytes aligned)
 */
void intr_packet8_tri(const ray_packet8* packet,
                      const float* v0,
                      const float* v1,
                      const float* v2,
                      packet8_hit* hit,
                      int tri_id);

/** Intersects a triangle with a ray packet
 * \param[in]  packet    Packet of rays (16 bytes aligned)
 * \param[in]  tri_data  Precomputed triangle data (16 bytes aligned)
 * \param[in]  tri_id    Triangle index (to fill the intr field in the hit result)
 * \param[out] hit       Result of the intersection (16 bytes aligned)
 */
void intr_packet4_tri_fast(const ray_packet4* packet,
                           const tri_data* tri,
                           packet4_hit* hit,
                           int tri_id);

/** Intersects a triangle with a ray packet
 * \param[in]  packet    Packet of rays (32 bytes aligned)
 * \param[in]  tri_data  Precomputed triangle data (16 bytes aligned)
 * \param[in]  tri_id    Triangle index (to fill the intr field in the hit result)
 * \param[out] hit       Result of the intersection (32 bytes aligned)
 */
void intr_packet8_tri_fast(const ray_packet8* packet,
                           const tri_data* tri,
                           packet8_hit* hit,
                           int tri_id);

/** Intersects a box with a packet
 * \param[in]  ray_packet  Packet to intersect (16 bytes aligned)
 * \param[in]  box_min     Minimal point of the box (16 bytes aligned)
 * \param[in]  box_max     Maximal point of the box (16 bytes aligned)
 * \param[in]  prev_tmin   Previous intersection time (for culling, 16 bytes aligned)
 * \return 0 iff no intersection was found
 */
int intr_packet4_box(const ray_packet4* packet,
                     const float* box_min,
                     const float* box_max,
                     const float* prev_tmin);

/** Intersects a box with a packet
 * \param[in]  ray_packet  Packet to intersect (16 bytes aligned)
 * \param[in]  box_min     Minimal point of the box (16 bytes aligned)
 * \param[in]  box_max     Maximal point of the box (16 bytes aligned)
 * \param[in]  prev_tmin   Previous intersection time (for culling, 16 bytes aligned)
 * \return 0 iff no intersection was found
 */
int intr_packet8_box(const ray_packet8* packet,
                     const float* box_min,
                     const float* box_max,
                     const float* prev_tmin);

/** Intersects a frustum with a box
 * \param[in] frustum     Frustum planes (16 bytes aligned, normals pointing outside)
 * \param[in] box_min     Minimal point of the box (16 bytes aligned)
 * \param[in] box_max     Maximal point of the box (16 bytes aligned)
 * \return 0 iff no intersection was found
 */
int intr_frustum_box(const float* frustum,
                     const float* box_min,
                     const float* box_max);

/** Intersects a bvh with several packets, finds the closest intersection
 * \param[in]  packets       List of ray packets (16 bytes aligned)
 * \param[in]  num_packets   Number of packets
 * \param[in]  frustum       Bounding frustum of the packet (16 bytes aligned)
 * \param[in]  bvh           BVH object
 * \param[in]  indices       Vertex indices
 * \param[in]  vertices      Triangles vertices (16 bytes aligned)
 * \param[out] hits          Array of hits
 */
void closest_intr_packet4_bvh(const ray_packet4* packets,
                              unsigned int num_packets,
                              const float* frustum,
                              const bvh_obj* bvh,
                              const unsigned int* indices,
                              const float* vertices,
                              packet4_hit* hits);

/** Intersects a bvh with several packets, finds the closest intersection
 * \param[in]  packets       List of ray packets (16 bytes aligned)
 * \param[in]  num_packets   Number of packets
 * \param[in]  frustum       Bounding frustum of the packet (16 bytes aligned)
 * \param[in]  bvh           BVH object
 * \param[in]  tris          Precomputed triangles (16 bytes aligned)
 * \param[out] hits          Array of hits
 */
void closest_intr_packet4_bvh_fast(const ray_packet4* packets,
                                   unsigned int num_packets,
                                   const float* frustum,
                                   const bvh_obj* bvh,
                                   const tri_data* tris,
                                   packet4_hit* hits);

/** Intersects a bvh with several packets, finds the closest intersection
 * \param[in]  packets       List of ray packets (32 bytes aligned)
 * \param[in]  num_packets   Number of packets
 * \param[in]  frustum       Bounding frustum of the packet (16 bytes aligned)
 * \param[in]  bvh           BVH object
 * \param[in]  indices       Vertex indices
 * \param[in]  vertices      Triangles vertices (16 bytes aligned)
 * \param[out] hits          Array of hits
 */
void closest_intr_packet8_bvh(const ray_packet8* packets,
                              unsigned int num_packets,
                              const float* frustum,
                              const bvh_obj* bvh,
                              const unsigned int* indices,
                              const float* vertices,
                              packet8_hit* hits);

/** Intersects a bvh with several packets, finds the closest intersection
 * \param[in]  packets       List of ray packets (32 bytes aligned)
 * \param[in]  num_packets   Number of packets
 * \param[in]  frustum       Bounding frustum of the packet (16 bytes aligned)
 * \param[in]  bvh           BVH object
 * \param[in]  tris          Precomputed triangles (16 bytes aligned)
 * \param[out] hits          Array of hits
 */
void closest_intr_packet8_bvh_fast(const ray_packet8* packets,
                                   unsigned int num_packets,
                                   const float* frustum,
                                   const bvh_obj* bvh,
                                   const tri_data* tris,
                                   packet8_hit* hits);

/** Intersects a bvh with several packets, stops at the first intersection between (0..1)
 * \param[in]  packets       List of ray packets (16 bytes aligned)
 * \param[in]  num_packets   Number of packets
 * \param[in]  frustum       Bounding frustum of the packet (16 bytes aligned)
 * \param[in]  bvh           BVH object
 * \param[in]  indices       Vertex indices
 * \param[in]  vertices      Triangles vertices (16 bytes aligned)
 * \param[out] hits          Array of hits
 */
void first_intr_packet4_bvh(const ray_packet4* packets,
                            unsigned int num_packets,
                            const float* frustum,
                            const bvh_obj* bvh,
                            const unsigned int* indices,
                            const float* vertices,
                            packet4_hit* hits);

/** Intersects a bvh with several packets, stops at the first intersection between (0..1)
 * \param[in]  packets       List of ray packets (16 bytes aligned)
 * \param[in]  num_packets   Number of packets
 * \param[in]  frustum       Bounding frustum of the packet (16 bytes aligned)
 * \param[in]  bvh           BVH object
 * \param[in]  tris          Precomputed triangles (16 bytes aligned)
 * \param[out] hits          Array of hits
 */
void first_intr_packet4_bvh_fast(const ray_packet4* packets,
                                 unsigned int num_packets,
                                 const float* frustum,
                                 const bvh_obj* bvh,
                                 const tri_data* tris,
                                 packet4_hit* hits);

/** Intersects a bvh with several packets, stops at the first intersection between (0..1)
 * \param[in]  packets       List of ray packets (32 bytes aligned)
 * \param[in]  num_packets   Number of packets
 * \param[in]  frustum       Bounding frustum of the packet (16 bytes aligned)
 * \param[in]  bvh           BVH object
 * \param[in]  indices       Vertex indices
 * \param[in]  vertices      Triangles vertices (16 bytes aligned)
 * \param[out] hits          Array of hits
 */
void first_intr_packet8_bvh(const ray_packet8* packets,
                            unsigned int num_packets,
                            const float* frustum,
                            const bvh_obj* bvh,
                            const unsigned int* indices,
                            const float* vertices,
                            packet8_hit* hits);

/** Intersects a bvh with several packets, stops at the first intersection between (0..1)
 * \param[in]  packets       List of ray packets (32 bytes aligned)
 * \param[in]  num_packets   Number of packets
 * \param[in]  frustum       Bounding frustum of the packet (16 bytes aligned)
 * \param[in]  bvh           BVH object
 * \param[in]  tris          Precomputed triangles (16 bytes aligned)
 * \param[out] hits          Array of hits
 */
void first_intr_packet8_bvh_fast(const ray_packet8* packets,
                                 unsigned int num_packets,
                                 const float* frustum,
                                 const bvh_obj* bvh,
                                 const tri_data* tris,
                                 packet8_hit* hits);

#endif // DREAM_INTR_H
