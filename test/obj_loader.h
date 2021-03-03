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
 * \file obj_loader.h
 * \brief Dump OBJ mesh file
 * \author Arsène Pérard-Gayot, Camille Brugel
 * \date 10 juin 2013
 */

#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

/* Simplistic Wavefront OBJ file loader */

#define MAT_NAME 128
#define TEX_NAME 128

/**
* \struct obj_face
* Contains the mesh data */
typedef struct
{
    unsigned int vertices[3];
    unsigned int normals[3];
    unsigned int texcoords[3];
    unsigned int mat_id;
} obj_face;

/**
* \struct obj_material
* Represent a material (for the phong shader) */
typedef struct
{
    char mat_name[MAT_NAME];
    char tex_name[TEX_NAME];
    float ambient[3];
    float diffuse[3];
    float specular[3];
    float alpha;
} obj_material;

/**
* \struct obj_model
* Represent a 3d model */
typedef struct
{
    obj_face* faces;
    float* vertices;
    float* normals;
    float* texcoords;
    char* materials;

    unsigned int num_faces;
    unsigned int num_vertices;
    unsigned int num_normals;
    unsigned int num_texcoords;
    unsigned int num_materials;
} obj_model;

/**
* \struct obj_mtl
* Contains the mtl file infos */
typedef struct
{
    obj_material* materials;
    unsigned int num_materials;
} obj_mtl;

/** Loads an OBJ model from a file */
int load_obj(obj_model* model, const char* file_name);
/** Destroys a previously loaded OBJ model */
void destroy_obj(obj_model* model);
/** Loads a MTL file */
int load_mtl(obj_mtl* mtl, const char* file_name);
/** Destroys a previously loaded MTL */
void destroy_mtl(obj_mtl* mtl);

#endif // OBJ_LOADER_H

