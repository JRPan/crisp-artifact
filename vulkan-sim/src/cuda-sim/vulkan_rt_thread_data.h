#ifndef VULKAN_RT_THREAD_DATA_H
#define VULKAN_RT_THREAD_DATA_H

#include "vulkan/vulkan.h"
#include "vulkan/vulkan_intel.h"

#include "vulkan_ray_tracing.h"

// #include "ptx_ir.h"
#include <cmath>
#include <fstream>
#include <stack>
#include "../../libcuda/gpgpu_context.h"
#include "compiler/shader_enums.h"
#include "ptx_ir.h"

typedef struct variable_decleration_entry {
  uint64_t type;
  std::string name;
  uint64_t address;
  uint32_t size;
} variable_decleration_entry;

typedef struct Hit_data {
  VkGeometryTypeKHR geometryType;
  float world_min_thit;
  uint32_t geometry_index;
  uint32_t primitive_index;
  float3 intersection_point;
  float3 barycentric_coordinates;
  int32_t
      hitGroupIndex;  // Shader ID of the closest hit for procedural geometries

  uint32_t instance_index;
  float4x4 worldToObjectMatrix;
  float4x4 objectToWorldMatrix;
} Hit_data;

typedef struct Traversal_data {
  bool hit_geometry;
  Hit_data closest_hit;
  float3 ray_world_direction;
  float3 ray_world_origin;
  float Tmin;
  float Tmax;
  int32_t current_shader_counter;  // set to shader_counter in call_intersection
                                   // and -1 in call_miss and call_closest_hit

  uint32_t rayFlags;
  uint32_t cullMask;
  uint32_t sbtRecordOffset;
  uint32_t sbtRecordStride;
  uint32_t missIndex;
} Traversal_data;

typedef struct Vulkan_RT_thread_data {
  std::vector<variable_decleration_entry> variable_decleration_table;

  std::vector<Traversal_data*> traversal_data;

  variable_decleration_entry* get_variable_decleration_entry(uint64_t type,
                                                             std::string name,
                                                             uint32_t size) {
    if (type == 8192) return get_hitAttribute();

    for (int i = 0; i < variable_decleration_table.size(); i++) {
      if (variable_decleration_table[i].name == name) {
        assert(variable_decleration_table[i].address != NULL);
        return &(variable_decleration_table[i]);
      }
    }
    return NULL;
  }

  uint64_t add_variable_decleration_entry(uint64_t type, std::string name,
                                          std::string identifier, uint32_t size,
                                          uint32_t offset) {
    variable_decleration_entry entry;
    entry.type = type;
    entry.name = name;

    gpgpu_context* ctx = GPGPU_Context();
    CUctx_st* context = GPGPUSim_Context(ctx);

    std::vector<std::string> v;
    std::stringstream ss(identifier);

    while (ss.good()) {
      std::string substr;
      getline(ss, substr, '_');
      v.push_back(substr);
    }

    unsigned attrib_index = -1;
    if (name.find("\%draw_call") != std::string::npos) {
      entry.address = VulkanRayTracing::getConst();
    } else
    if (!VulkanRayTracing::is_FS) {
      if (identifier.find("VERT_ATTRIB_GENERIC") != std::string::npos) {
        // vertex shader input attributes
        assert(v.size() == 4);
        attrib_index = std::atoi(&v[2].back());
        entry.address = VulkanRayTracing::getVertexAddr(attrib_index, offset);

      } else if (identifier.find("VARYING_SLOT_POS") != std::string::npos) {
        assert(identifier.find("xyzw") != std::string::npos);
        if (VulkanRayTracing::VertexMeta->vertex_out_devptr.find(name) ==
            VulkanRayTracing::VertexMeta->vertex_out_devptr.end()) {
          uint32_t* dev_ptr = context->get_device()->get_gpgpu()->gpu_malloc(
              VulkanRayTracing::thread_count * size);
          context->get_device()
              ->get_gpgpu()
              ->valid_addr_start["VARYING_SLOT_POS_xyzw"] = (uint64_t)dev_ptr;
          context->get_device()
              ->get_gpgpu()
              ->valid_addr_end["VARYING_SLOT_POS_xyzw"] =
              ((uint64_t)dev_ptr) + VulkanRayTracing::thread_count * size;
          VulkanRayTracing::VertexMeta->vertex_out_devptr.insert(
              std::make_pair(name, dev_ptr));
          VulkanRayTracing::VertexMeta->vertex_id_map.insert(
              std::make_pair(identifier, name));
          VulkanRayTracing::VertexMeta->vertex_out_stride.insert(
              std::make_pair(name, size));
        }
        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);

      } else if (identifier.find("VARYING_SLOT_VAR") != std::string::npos) {
        if (VulkanRayTracing::VertexMeta->vertex_out_devptr.find(name) ==
            VulkanRayTracing::VertexMeta->vertex_out_devptr.end()) {
          attrib_index = std::atoi(&v[2].back());
          std::string attrib_name =
              "VARYING_SLOT_VAR" + std::to_string(attrib_index);

          uint32_t* dev_ptr = context->get_device()->get_gpgpu()->gpu_malloc(
              VulkanRayTracing::thread_count * size);
          context->get_device()->get_gpgpu()->valid_addr_start[attrib_name] =
              (uint64_t)dev_ptr;
          context->get_device()->get_gpgpu()->valid_addr_end[attrib_name] =
              ((uint64_t)dev_ptr) + VulkanRayTracing::thread_count * size;
          VulkanRayTracing::VertexMeta->vertex_out_devptr.insert(
              std::make_pair(name, dev_ptr));
          VulkanRayTracing::VertexMeta->vertex_id_map.insert(
              std::make_pair(identifier, name));
          VulkanRayTracing::VertexMeta->vertex_out_stride.insert(
              std::make_pair(name, size));
        }

        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);
      } else {
        assert(identifier.find("UNDEFINED") != std::string::npos);
        if (VulkanRayTracing::VertexMeta->vertex_out_devptr.find(name) ==
            VulkanRayTracing::VertexMeta->vertex_out_devptr.end()) {
          uint32_t* dev_ptr = context->get_device()->get_gpgpu()->gpu_malloc(
              VulkanRayTracing::thread_count * size);
          VulkanRayTracing::VertexMeta->vertex_out_devptr.insert(
              std::make_pair(name, dev_ptr));
          VulkanRayTracing::VertexMeta->vertex_out_stride.insert(
              std::make_pair(name, size));
        }

        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);
      }
    } else {
      char const *app_env = std::getenv("VULKAN_APP");
      std::string vulkan_app(app_env);
      if (identifier.find("VARYING_SLOT_VAR0_xyzw") != std::string::npos && 
      (vulkan_app == "render_passes" || vulkan_app == "instancing")) {
        entry.address = VulkanRayTracing::getVertexOutAddr("\%field0", offset);
      } else 

      // this is for pbrtexture only
      if (identifier.find("VARYING_SLOT_VAR3_xyzw") != std::string::npos && vulkan_app == "pbrtexture") {
        name = VulkanRayTracing::VertexMeta->vertex_id_map.at("VARYING_SLOT_VAR3_xyz");
        attrib_index = std::atoi(&v[2].back());
        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);
      } else 
      
      if (identifier.find("VARYING_SLOT_VAR") != std::string::npos) {
        assert(VulkanRayTracing::VertexMeta->vertex_id_map.find(identifier) !=
               VulkanRayTracing::VertexMeta->vertex_id_map.end());
        // if (VulkanRayTracing::VertexMeta->vertex_id_map.find(identifier) ==
        //        VulkanRayTracing::VertexMeta->vertex_id_map.end()) {
        //         printf("identifier: %s not found\n", identifier.c_str());
        //         fflush(stdout);
        //         assert(0);
        //        }
        name = VulkanRayTracing::VertexMeta->vertex_id_map.at(identifier);
        attrib_index = std::atoi(&v[2].back());

        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);
      } else if (identifier.find("FRAG_RESULT_DATA0_xyzw") != std::string::npos) {
        entry.address = VulkanRayTracing::getFBOAddr(offset);
      } else {
        assert(0);
      }
    }
    entry.size = size;
    variable_decleration_table.push_back(entry);

    return entry.address;
  }

  variable_decleration_entry* get_hitAttribute() {
    variable_decleration_entry* hitAttribute = NULL;
    for (int i = 0; i < variable_decleration_table.size(); i++) {
      if (variable_decleration_table[i].type == 8192) {
        assert(variable_decleration_table[i].address != NULL);
        assert(hitAttribute == NULL);  // There should be only 1 hitAttribute
        hitAttribute = &(variable_decleration_table[i]);
      }
    }
    return hitAttribute;
  }

  void set_hitAttribute(float3 barycentric) {
    variable_decleration_entry* hitAttribute = get_hitAttribute();
    float* address;
    if (hitAttribute == NULL) {
      address = (float*)add_variable_decleration_entry(8192, "attribs", "UNDEFINED", 12, 0);
    } else {
      assert(hitAttribute->type == 8192);
      assert(hitAttribute->address != NULL);
      // hitAttribute->name = name;
      address = (float*)(hitAttribute->address);
    }
    // address[0] = barycentric.x;
    // address[1] = barycentric.y;
    // address[2] = barycentric.z;
    gpgpu_context* ctx = GPGPU_Context();
    CUctx_st* context = GPGPUSim_Context(ctx);
    context->get_device()->get_gpgpu()->memcpy_to_gpu(address, &barycentric,
                                                      sizeof(float3));
  }
} Vulkan_RT_thread_data;

#endif /* VULKAN_RT_THREAD_DATA_H */