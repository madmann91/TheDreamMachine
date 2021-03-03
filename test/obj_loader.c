#include "obj_loader.h"
#include "mem.h"
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

#define BUFFER_SIZE 256

int find_material(const char* name, const mem_buffer* buf)
{
    for (unsigned int i = 0, j = 0; i < buf->size; i += sizeof(char) * MAT_NAME, j++) {
        if (!strcmp((char*)buf->ptr + i, name)) {
            return j;
        }
    }

    return -1;
}

int read_coordinates(char** ptr, unsigned int* coord)
{
    /* Find end of coordinate block */
    char* cur = *ptr;
    while (!isspace(*cur) && *cur != '\0') {
        cur++;
    }

    char val = *cur;
    *cur = '\0';

    coord[0] = 0;
    coord[1] = 0;
    coord[2] = 0;

    if (sscanf(*ptr, "%u/%u/%u", coord, coord + 1, coord + 2) != 3 &&
        sscanf(*ptr, "%u//%u", coord, coord + 2) != 2 &&
        sscanf(*ptr, "%u/%u", coord, coord + 1) != 2 &&
        sscanf(*ptr, "%u", coord) != 1)
        return 0;

    *cur = val;
    *ptr = cur + 1;
    return 1;
}

int read_face(char* line, obj_face* faces, unsigned int max_faces)
{
    char* ptr = line;
    while (isspace(*ptr)) { ptr++; }
    if (*ptr != 'f') return 0;
    
    /* Skip spaces */
    do { ptr++; } while (isspace(*ptr));

    int num_coords = 0;
    int num_faces = 0;
    unsigned int prev_coord0[3] = {0, 0, 0};
    unsigned int prev_coord1[3] = {0, 0, 0};
    unsigned int coord[3];

    while (read_coordinates(&ptr, coord)) {
        if (num_coords >= 2) {
            if (num_faces >= max_faces) return 0;
            faces[num_faces].vertices[0] = prev_coord0[0];
            faces[num_faces].texcoords[0] = prev_coord0[1];
            faces[num_faces].normals[0] = prev_coord0[2];

            faces[num_faces].vertices[1] = prev_coord1[0];
            faces[num_faces].texcoords[1] = prev_coord1[1];
            faces[num_faces].normals[1] = prev_coord1[2];

            faces[num_faces].vertices[2] = coord[0];
            faces[num_faces].texcoords[2] = coord[1];
            faces[num_faces].normals[2] = coord[2];
            num_faces++;
        }
        
        prev_coord1[0] = coord[0];
        prev_coord1[1] = coord[1];
        prev_coord1[2] = coord[2];

        if (num_coords == 0) {
            prev_coord0[0] = coord[0];
            prev_coord0[1] = coord[1];
            prev_coord0[2] = coord[2];
        }

        num_coords++;
    }

    return num_faces;
}

int load_obj(obj_model* model, const char* file_name)
{
    FILE* fp = fopen(file_name, "r");
    if (!fp)
        return 0;

    mem_buffer vertices = {0, 0, 0};
    mem_buffer normals = {0, 0, 0};
    mem_buffer texcoords = {0, 0, 0};
    mem_buffer faces = {0, 0, 0};
    mem_buffer materials = {0, 0, 0};
    unsigned int current_mat = 0;
    char buffer[BUFFER_SIZE];
    int line = 0;

    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        unsigned int first = strspn(buffer, " \t\n");

        /* Skip empty lines */
        if (first == strlen(buffer))
            continue;

        if (buffer[first] == 'v') {
            if (isspace(buffer[first + 1])) {
                float vertex[3];
                sscanf(buffer + first, "v %f %f %f", &vertex[0], &vertex[1], &vertex[2]);
                buffer_add(&vertices, vertex, sizeof(float) * 3);
            } else if (buffer[first + 1] == 'n') {
                float normal[3];
                sscanf(buffer + first, "vn %f %f %f", &normal[0], &normal[1], &normal[2]);
                buffer_add(&normals, normal, sizeof(float) * 3);
            } else if (buffer[first + 1] == 't') {
                float texcoord[2];
                sscanf(buffer + first, "vt %f %f", &texcoord[0], &texcoord[1]);
                buffer_add(&texcoords, texcoord, sizeof(float) * 2);
            }
        } else if (buffer[first] == 'f') {
            obj_face face[8];
            unsigned int num_faces = read_face(buffer + first, face, 8);
            if (num_faces != 0) {
                for (unsigned int i = 0; i < num_faces; i++)
                    face[i].mat_id = current_mat;

                buffer_add(&faces, face, sizeof(obj_face) * num_faces);
            } else {
                printf("warning : invalid face definition\n");
            }
        } else if (!strncmp(buffer + first, "usemtl", 6)) {
            char mtl_name[MAT_NAME];
            memset(mtl_name, 0, sizeof(char) * MAT_NAME);

            if (sscanf(buffer + first, "usemtl %s", mtl_name) == 1) {
                /* Find the material or add a new one */
                int index = find_material(mtl_name, &materials);
                if (index < 0) {
                    current_mat = materials.size / (sizeof(char) * MAT_NAME);
                    buffer_add(&materials, mtl_name, sizeof(char) * MAT_NAME);
                } else {
                    current_mat = index;
                }
            } else {
                printf("warning : invalid material command\n");
            }
        } else if (buffer[first] != '#') {
            printf("warning : unknown OBJ command at line %d\n", line);
        }

        line++;
    }
    fclose(fp);

    buffer_refit(&vertices);
    buffer_refit(&normals);
    buffer_refit(&texcoords);
    buffer_refit(&faces);
    buffer_refit(&materials);

    model->faces = (obj_face*)faces.ptr;
    model->vertices = (float*)vertices.ptr;
    model->normals = (float*)normals.ptr;
    model->texcoords = (float*)texcoords.ptr;
    model->materials = (char*)materials.ptr;

    model->num_faces = faces.size / (sizeof(obj_face));
    model->num_vertices = vertices.size / (sizeof(float) * 3);
    model->num_normals = normals.size / (sizeof(float) * 3);
    model->num_texcoords = texcoords.size / (sizeof(float) * 2);
    model->num_materials = materials.size / (sizeof(char) * MAT_NAME);

    return 1;
}

void destroy_obj(obj_model* model)
{
    free(model->faces);
    free(model->vertices);
    free(model->normals);
    free(model->texcoords);
    free(model->materials);
}

int load_mtl(obj_mtl* mats, const char* file_name)
{
    FILE* fp = fopen(file_name, "r");
    if (!fp)
        return 0;

    mem_buffer materials = {0, 0, 0};
    char buffer[BUFFER_SIZE];
    int current_mat = -1;
    int line = 0;

    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        unsigned int first = strspn(buffer, " \t\n");

        /* Skip empty lines */
        if (first == strlen(buffer))
            continue;

        if (!strncmp(buffer + first, "newmtl", 6)) {
            obj_material mat;
            memset(mat.mat_name, 0, sizeof(char) * MAT_NAME);
            memset(mat.tex_name, 0, sizeof(char) * TEX_NAME);
            memset(mat.ambient, 0, sizeof(float) * 3);
            memset(mat.specular, 0, sizeof(float) * 3);
            mat.alpha = 100.0f;
            mat.diffuse[0] = 1.0f;
            mat.diffuse[1] = 1.0f;
            mat.diffuse[2] = 1.0f;

            if (sscanf(buffer + first, "newmtl %s", mat.mat_name) == 1) {
                buffer_add(&materials, &mat, sizeof(obj_material));
                current_mat++;
            } else {
                printf("warning : invalid material definition\n");
            }
        } else if (!strncmp(buffer + first, "Ka", 2)) {
            if (current_mat >= 0) {
                obj_material* mat = (obj_material*)materials.ptr + current_mat;
                sscanf(buffer + first, "Ka %f %f %f",
                       &mat->ambient[0],
                       &mat->ambient[1],
                       &mat->ambient[2]);
            } else {
                printf("warning : no material created, material command ignored\n");
            }
        } else if (!strncmp(buffer + first, "Kd", 2)) {
            if (current_mat >= 0) {
                obj_material* mat = (obj_material*)materials.ptr + current_mat;
                sscanf(buffer + first, "Kd %f %f %f",
                       &mat->diffuse[0],
                       &mat->diffuse[1],
                       &mat->diffuse[2]);
            } else {
                printf("warning : no material created, material command ignored\n");
            }
        } else if (!strncmp(buffer + first, "Ks", 2)) {
            if (current_mat >= 0) {
                obj_material* mat = (obj_material*)materials.ptr + current_mat;
                sscanf(buffer + first, "Ks %f %f %f",
                       &mat->specular[0],
                       &mat->specular[1],
                       &mat->specular[2]);
            } else {
                printf("warning : no material created, material command ignored\n");
            }
        } else if (!strncmp(buffer + first, "Ns", 2)) {
            if (current_mat >= 0) {
                obj_material* mat = (obj_material*)materials.ptr + current_mat;
                sscanf(buffer + first, "Ns %f", &mat->alpha);
            } else {
                printf("warning : no material created, material command ignored\n");
            }
        } else if (!strncmp(buffer + first, "map_Kd", 6)) {
            if (current_mat >= 0) {
                obj_material* mat = (obj_material*)materials.ptr + current_mat;
                sscanf(buffer + first, "map_Kd %s", mat->tex_name);
            } else {
                printf("warning : no material created, material command ignored\n");
            }
        } else if (buffer[first] != '#') {
            printf("warning : unknown MTL command at line %d\n", line);
        }

        line++;
    }
    fclose(fp);

    buffer_refit(&materials);
    mats->materials = (obj_material*)materials.ptr;
    mats->num_materials = materials.size / sizeof(obj_material);

    return 1;
}

void destroy_mtl(obj_mtl* mtl)
{
    free(mtl->materials);
}

