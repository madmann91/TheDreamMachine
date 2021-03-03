#include "../intr.h"
#include "../linalg.inl"

int intr_packet4_box(const ray_packet4* packet,
                     const float* box_min,
                     const float* box_max,
                     const float* prev_tmin)
{
    float tmin[4], tmax[4];

    {
        const float tmin_x[4] = {
            (box_min[0] - packet->org[0]) * packet->inv_dir[0],
            (box_min[0] - packet->org[1]) * packet->inv_dir[1],
            (box_min[0] - packet->org[2]) * packet->inv_dir[2],
            (box_min[0] - packet->org[3]) * packet->inv_dir[3]
        };
        
        const float tmax_x[4] = {
            (box_max[0] - packet->org[0]) * packet->inv_dir[0],
            (box_max[0] - packet->org[1]) * packet->inv_dir[1],
            (box_max[0] - packet->org[2]) * packet->inv_dir[2],
            (box_max[0] - packet->org[3]) * packet->inv_dir[3]
        };
        
        vector4_min(tmin_x, tmax_x, tmin);
        vector4_max(tmin_x, tmax_x, tmax);
    }
    
    {
        const float tmin_y[4] = {
            (box_min[1] - packet->org[4]) * packet->inv_dir[4],
            (box_min[1] - packet->org[5]) * packet->inv_dir[5],
            (box_min[1] - packet->org[6]) * packet->inv_dir[6],
            (box_min[1] - packet->org[7]) * packet->inv_dir[7]
        };
        
        const float tmax_y[4] = {
            (box_max[1] - packet->org[4]) * packet->inv_dir[4],
            (box_max[1] - packet->org[5]) * packet->inv_dir[5],
            (box_max[1] - packet->org[6]) * packet->inv_dir[6],
            (box_max[1] - packet->org[7]) * packet->inv_dir[7]
        };
        
        float min_y[4], max_y[4];
        vector4_min(tmin_y, tmax_y, min_y);
        vector4_max(tmin_y, tmax_y, max_y);
        
        vector4_max(tmin, min_y, tmin);
        vector4_min(tmax, max_y, tmax);
    }
    
    {
        const float tmin_z[4] = {
            (box_min[2] - packet->org[ 8]) * packet->inv_dir[ 8],
            (box_min[2] - packet->org[ 9]) * packet->inv_dir[ 9],
            (box_min[2] - packet->org[10]) * packet->inv_dir[10],
            (box_min[2] - packet->org[11]) * packet->inv_dir[11]
        };
        
        const float tmax_z[4] = {
            (box_max[2] - packet->org[ 8]) * packet->inv_dir[ 8],
            (box_max[2] - packet->org[ 9]) * packet->inv_dir[ 9],
            (box_max[2] - packet->org[10]) * packet->inv_dir[10],
            (box_max[2] - packet->org[11]) * packet->inv_dir[11]
        };
        
        float min_z[4], max_z[4];
        vector4_min(tmin_z, tmax_z, min_z);
        vector4_max(tmin_z, tmax_z, max_z);
        
        vector4_max(tmin, min_z, tmin);
        vector4_min(tmax, max_z, tmax);
    }
    
    return ((tmax[0] >= tmin[0]) && (tmax[0] >= 0) && (tmin[0] <= prev_tmin[0])) ||
           ((tmax[1] >= tmin[1]) && (tmax[1] >= 0) && (tmin[1] <= prev_tmin[1])) ||
           ((tmax[2] >= tmin[2]) && (tmax[2] >= 0) && (tmin[2] <= prev_tmin[2])) ||
           ((tmax[3] >= tmin[3]) && (tmax[3] >= 0) && (tmin[3] <= prev_tmin[3]));
}
