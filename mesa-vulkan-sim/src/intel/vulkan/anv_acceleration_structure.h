#ifndef ANV_ACCELERATION_SRTUCTURE_H
#define ANV_ACCELERATION_SRTUCTURE_H

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_intel.h>
#include <vulkan/vk_icd.h>

struct anv_bvh_vec4f {
   float v[4];
};

struct anv_bvh_vec3f {
   float v[3];
};

struct anv_bvh_triangle {
   struct anv_bvh_vec3f v[3];
};

struct anv_bvh_node {
   uint8_t is_leaf;

   int8_t exp_x;
   int8_t exp_y;
   int8_t exp_z;

   float origin_x;
   float origin_y;
   float origin_z;

   uint8_t lower_x[6];
   uint8_t upper_x[6];
   uint8_t lower_y[6];
   uint8_t upper_y[6];
   uint8_t lower_z[6];
   uint8_t upper_z[6];

   struct anv_bvh_node *children[6];
};

struct anv_bvh_leaf {
   uint8_t is_leaf;
   uint32_t geometry_id;
   uint32_t primitive_id;
};

#endif /* ANV_ACCELERATION_SRTUCTURE_H */
