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

#ifndef DREAM_VIEW_H
#define DREAM_VIEW_H

#include "intr.h"

typedef struct
{
    float eye[4];                       /**< Position of the camera */
    float right[4];                     /**< Unit vector in the rigth direction */
    float dir[4];                       /**< Specify the direction the camera look at */
    float up[4];                        /**< Unit vector in the up direction */
    float far[4];                       /**< Far plane equation */
    float near[4];                      /**< Near plane equation */
    float wnear, hnear, dnear, pad;     /**< Screen dimension in the scene */
} view_info;

/** Update the position and the direction of the camera
 * \param[in]     rx     rotation around x axis 
 * \param[in]     ry     rotation around y axis
 * \param[in]     tx     translation along x 
 * \param[in]     ty     translation along y 
 * \param[in,out] up     Up vector (must be normalized)
 * \param[in,out] pos    Reference point position
 * \param[in,out] eye    Eye position
 */
void update_view(float rx, float ry,
                 float tx, float ty,
                 float* up,
                 float* eye,
                 float* pos);

/** Setup view frustum planes for the given perspective view
 * \param[out] view      View information
 * \param[in]  eye       Eye position
 * \param[in]  pos       Reference point position
 * \param[in]  up        Up vector
 * \param[in]  fov       Field of view, in degrees
 * \param[in]  ratio     View ratio
 * \param[in]  near      Near plane coordinate
 * \param[in]  far       Far plane coordinate
 */
void setup_view_persp(view_info* view,
                      const float* eye,
                      const float* pos,
                      const float* up,
                      float fov, float ratio,
                      float near, float far);

/** Builds ray packets for a perspective view
 * \param[in]  view       View information
 * \param[out] packets    Resulting packets
 * \param[in]  i          Horizontal position of the packets
 * \param[in]  j          Vertical position of the packets
 * \param[in]  packet_w   Width (in rays) of the packets
 * \param[in]  packet_h   Height (in rays) of the packets
 * \param[in]  view_w     Width (in rays) of the view
 * \param[in]  view_h     Height (in rays) of the view
 */
void build_packet4_persp(const view_info* view,
                         ray_packet4* packets,
                         unsigned int i, unsigned int j,
                         unsigned int packet_w, unsigned int packet_h,
                         unsigned int view_w, unsigned int view_h);

/** Computes a ray packet bounding frustum (for a perspective view)
 * \param[in]  packets         List of ray packets
 * \param[in]  packet_w        Width (in rays) of the packets
 * \param[in]  packet_h        Height (in rays) of the packets
 * \param[in]  near_plane      Near plane
 * \param[in]  far_plane       Far plane
 * \param[out] packet_frustum  Packets frustum
 */
void build_packet4_frustum_persp(const ray_packet4* packets,
                                 unsigned int packet_w, unsigned int packet_h,
                                 const float* near_plane, const float* far_plane,
                                 float* packet_frustum);

/** Builds ray packets for shadow testing purposes
 * \param[out] packets       Resulting packets
 * \param[in]  num_packets   Number of ray packets
 * \param[in]  prev_packets  Previous packets
 * \param[in]  hits          Previous hit information
 */
void build_packet4_shadow(ray_packet4* packets,
                          unsigned int num_packets,
                          const ray_packet4* prev_packets,
                          const packet4_hit* hits,
                          const float* light_pos);

/** Computes a ray packet bounding frustum (for shadow rays)
 * \param[in]  packets         List of ray packets
 * \param[in]  num_packets     Number of ray packets
 * \param[in]  light_pos       Light position
 * \param[out] packet_frustum  Packets frustum
 */
void build_packet4_frustum_shadow(const ray_packet4* packets,
                                  unsigned int num_packets,
                                  const float* light_pos,
                                  float* packet_frustum);

#endif // DREAM_VIEW_H
