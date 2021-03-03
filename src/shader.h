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

#ifndef DREAM_SHADER_H
#define DREAM_SHADER_H

#include "intr.h"

/* Available texture modes */
enum 
{
    TEX_REPEAT = 0x01,
    TEX_CLAMP = 0x02,
    TEX_NEAREST = 0x04,
    TEX_BILINEAR = 0x08,
};

/* RGB color */
typedef union
{
    uint32_t p;
    struct
    {
        uint8_t r, g, b, a;
    } q;
} rgb;

/* Face material description */
typedef struct
{
    float diffuse[4];
    float specular[4];
    float ambient[4];
    float alpha;
    unsigned int tex_w, tex_h;
    int tex_mode;
    rgb* texture;
} material;

typedef struct 
{
    float pos[4];
    float diffuse[4];
    float specular[4];
} light;

/** Triangular face */
typedef struct
{
    unsigned int vertices[3];
    unsigned int normals[3];
    unsigned int texcoords[3];
    unsigned int material;
} face;

/** Shader data (normals must be unit length) */
typedef struct
{
    float* vertices;
    float* normals;
    float* texcoords;
    material* materials;
    face* faces;
    light* lights;
    unsigned int num_lights;
    float ambient[4];
    rgb bg_color;
} shader_info;

/** Apply Phong shading over an array of ray packets
 * \param[in]  hits             Array of primary ray hits (16 bytes aligned)
 * \param[in]  packets          Array of primary ray packets (16 bytes aligned)
 * \param[in]  shadow_hits      Array of shadow ray hits (16 bytes aligned)
 * \param[in]  shadow_packets   Array of shadow ray packets (16 bytes aligned)
 * \param[in]  num_packets      Number of packets
 * \param[in]  shader           Shading info
 * \param[out] result           Resulting color array
 */
void packet4_phong_shader(const packet4_hit* hits,
                          const ray_packet4* packet,
                          const packet4_hit* shadow_hits,
                          const ray_packet4* shadow_packet,
                          unsigned int num_packets,
                          const shader_info* shader,
                          rgb* result);

/** Apply Cel shading over an array of ray packets
 * \param[in]  hits             Array of primary ray hits (16 bytes aligned)
 * \param[in]  packets          Array of primary ray packets (16 bytes aligned)
 * \param[in]  shadow_hits      Array of shadow ray hits (16 bytes aligned)
 * \param[in]  shadow_packets   Array of shadow ray packets (16 bytes aligned)
 * \param[in]  num_packets      Number of packets
 * \param[in]  shader           Shading info
 * \param[out] result           Resulting color array
 */
void packet4_cel_shader(const packet4_hit* hits,
                        const ray_packet4* packets,
                        const packet4_hit* shadow_hits,
                        const ray_packet4* shadow_packets,
                        unsigned int num_packets,
                        const shader_info* shader,
                        rgb* result);

#endif // DREAM_SHADER_H
