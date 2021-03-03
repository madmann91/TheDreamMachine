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

/**
 * \file bvh.h
 * \brief BVH creation and destruction
 * \author Arsène Pérard-Gayot, Camille Brugel
 * \date 10 juin 2013
 */

#ifndef DREAM_BVH_H
#define DREAM_BVH_H

#include <stdint.h>

/**
 * \struct bvh_node
 * Represent a BVH node
 * 32 bytes aligned
 */
typedef struct
{
    float box_min[3];               /**< Minimum dimension of the bounding box */
    union {
        uint32_t child;             /**< Children index (for inner nodes) */
        uint32_t tri_id;            /**< First triangle id (for leaf nodes) */
    } node_data;
    float box_max[3];               /**< Maximum dimension of the bounding box */
    uint16_t num_tris;              /**< Number of triangles (for a leaf node) */
    uint8_t axis;                   /**< Dominant axis (0, 1 or 2) */
    uint8_t order;                  /**< Traversal order (0 or 1) */
} bvh_node;

/**
 * \struct bvh_obj
 * Represent a BVH object
 * 32 bytes aligned
 */
typedef struct
{
    unsigned int* tri_ids;          /**< Triangle indices in the mesh */
    bvh_node* root;
    unsigned int depth;
} bvh_obj;

/** Builds a BVH using a greedy SAH approximation and a binning algorithm as
 * described in "On fast Construction of SAH-based Bounding Volume Hierarchies"
 * (Ingo Wald - 2007) to speed up construction times.
 * \param[in]  indices     Vertex indices of the triangle mesh
 * \param[in]  num_tris    Number of triangles
 * \param[in]  vertices    Array of vertices (must be 16 bytes aligned)
 * \param[out] result      Bounding volume hierarchy object
 */
unsigned int build_bvh(const unsigned int* indices,
                       unsigned int num_tris,
                       const float* vertices,
                       bvh_obj* result);

/** Destroys a BVH
 * \param[in]   bvh        Bounding volume hierarchy object to destroy
 */
void destroy_bvh(bvh_obj* bvh);

#endif // DREAM_BVH_H
