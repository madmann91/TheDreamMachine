#include "bvh.h"
#include "model.h"  
#include <time.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    bvh_obj bvh;
    unsigned int num_tris = sizeof(indices) / (sizeof(indices[0]) * 3);
    const unsigned int num_vertices =  sizeof(vertices) / (sizeof(vertices[0]) * 3);
    float* align_verts = aligned_malloc(sizeof(float) * 4 * num_vertices, 16);
    for (unsigned int i = 0; i < num_vertices; i++) {
        align_verts[i * 4 + 0] = vertices[i * 3 + 0];
        align_verts[i * 4 + 1] = vertices[i * 3 + 1];
        align_verts[i * 4 + 2] = vertices[i * 3 + 2];
        align_verts[i * 4 + 3] = 1.0f;
    }

    clock_t c = clock();
    unsigned int num_nodes = build_bvh(indices, num_tris, align_verts, &bvh);
    printf("time : %ld ms\n", (clock() - c) * 1000 / CLOCKS_PER_SEC);
    printf("nodes : %u\n", num_nodes);
    destroy_bvh(&bvh);
    aligned_free(align_verts);
}
