// Copyright (c) 2022, Mohammadreza Saed, Yuan Hsi Chou, Lufei Liu, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "vulkan_ray_tracing.h"
#include "vulkan_rt_thread_data.h"

#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#define __CUDA_RUNTIME_API_H__
// clang-format off
#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "../../libcuda/cuda_api.h"
#include "cudaProfiler.h"
// clang-format on
#if (CUDART_VERSION < 8000)
#include "__cudaFatFormat.h"
#endif

#include "../../libcuda/gpgpu_context.h"
#include "../../libcuda/cuda_api_object.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../cuda-sim/ptx_loader.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx_ir.h"
#include "../cuda-sim/ptx_parser.h"
#include "../gpgpusim_entrypoint.h"
#include "../stream_manager.h"
#include "../abstract_hardware_model.h"
#include "vulkan_acceleration_structure_util.h"
#include "../gpgpu-sim/vector-math.h"

//#include "intel_image_util.h"
 #include "astc_decomp.h"

// #define HAVE_PTHREAD
// #define UTIL_ARCH_LITTLE_ENDIAN 1
// #define UTIL_ARCH_BIG_ENDIAN 0
// #define signbit signbit

// #define UINT_MAX 65535
// #define GLuint MESA_GLuint
// // #include "isl/isl.h"
// // #include "isl/isl_tiled_memcpy.c"
// #include "vulkan/anv_private.h"
// #undef GLuint

// #undef HAVE_PTHREAD
// #undef UTIL_ARCH_LITTLE_ENDIAN
// #undef UTIL_ARCH_BIG_ENDIAN
// #undef signbit

// #include "vulkan/anv_public.h"
#include "intel_image.h"

// #include "anv_include.h"

VkRayTracingPipelineCreateInfoKHR* VulkanRayTracing::pCreateInfos = NULL;
VkAccelerationStructureGeometryKHR* VulkanRayTracing::pGeometries = NULL;
uint32_t VulkanRayTracing::geometryCount = 0;
VkAccelerationStructureKHR VulkanRayTracing::topLevelAS = NULL;
std::vector<std::vector<Descriptor> > VulkanRayTracing::descriptors;
std::ofstream VulkanRayTracing::imageFile;
bool VulkanRayTracing::firstTime = true;
std::vector<shader_stage_info> VulkanRayTracing::shaders;
// RayDebugGPUData VulkanRayTracing::rayDebugGPUData[2000][2000] = {0};
struct anv_descriptor_set* VulkanRayTracing::descriptorSet[MAX_DESCRIPTOR_SETS] = {NULL};
void* VulkanRayTracing::launcher_descriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS] = {NULL};
void* VulkanRayTracing::launcher_deviceDescriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS] = {NULL};
std::vector<void*> VulkanRayTracing::child_addrs_from_driver;
bool VulkanRayTracing::dumped = false;

bool use_external_launcher = false;
struct vertex_metadata* VulkanRayTracing::VertexMeta = new struct vertex_metadata;
struct FBO* VulkanRayTracing::FBO = new struct FBO;

bool VulkanRayTracing::_init_ = false;
bool VulkanRayTracing::is_FS = false;
unsigned VulkanRayTracing::texture_width = 0;
unsigned VulkanRayTracing::texture_height = 0;
VULKAN_APPS VulkanRayTracing::app_id = VULKAN_APPS_MAX;
unsigned VulkanRayTracing::draw = 0;
unsigned VulkanRayTracing::thread_count = 0;
warp_intersection_table *** VulkanRayTracing::intersection_table;
std::unordered_map<void *, unsigned> VulkanRayTracing::pipeline_shader_map;
std::unordered_map<void *, struct VertexAttrib *>
    VulkanRayTracing::pipeline_vertex_map;
std::deque<struct vertex_metadata* > VulkanRayTracing::draw_meta;
struct anv_buffer* VulkanRayTracing::index_buffer = NULL;
VkIndexType VulkanRayTracing::index_type = VK_INDEX_TYPE_MAX_ENUM;

float get_norm(float4 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}
float get_norm(float3 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float4 normalized(float4 v)
{
    float norm = get_norm(v);
    return {v.x / norm, v.y / norm, v.z / norm, v.w / norm};
}
float3 normalized(float3 v)
{
    float norm = get_norm(v);
    return {v.x / norm, v.y / norm, v.z / norm};
}

Ray make_transformed_ray(Ray &ray, float4x4 matrix, float *worldToObject_tMultiplier)
{
    Ray transformedRay;
    float4 transformedOrigin4 = matrix * float4({ray.get_origin().x, ray.get_origin().y, ray.get_origin().z, 1});
    float4 transformedDirection4 = matrix * float4({ray.get_direction().x, ray.get_direction().y, ray.get_direction().z, 0});

    float3 transformedOrigin = {transformedOrigin4.x / transformedOrigin4.w, transformedOrigin4.y / transformedOrigin4.w, transformedOrigin4.z / transformedOrigin4.w};
    float3 transformedDirection = {transformedDirection4.x, transformedDirection4.y, transformedDirection4.z};
    *worldToObject_tMultiplier = get_norm(transformedDirection);
    transformedDirection = normalized(transformedDirection);

    transformedRay.make_ray(transformedOrigin, transformedDirection, ray.get_tmin() * (*worldToObject_tMultiplier), ray.get_tmax() * (*worldToObject_tMultiplier));
    return transformedRay;
}

float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MIN_MAX(a0, a1, d);
	float t2 = MIN_MAX(b0, b1, t1);
	float t3 = MIN_MAX(c0, c1, t2);
	return t3;
}

float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MAX_MIN(a0, a1, d);
	float t2 = MAX_MIN(b0, b1, t1);
	float t3 = MAX_MIN(c0, c1, t2);
	return t3;
}

float3 get_t_bound(float3 box, float3 origin, float3 idirection)
{
    // // Avoid div by zero, returns 1/2^80, an extremely small number
    // const float ooeps = exp2f(-80.0f);

    // // Calculate inverse direction
    // float3 idir;
    // idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    // idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    // idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));

    // Calculate bounds
    float3 result;
    result.x = (box.x - origin.x) * idirection.x;
    result.y = (box.y - origin.y) * idirection.y;
    result.z = (box.z - origin.z) * idirection.z;

    // Return
    return result;
}

float3 calculate_idir(float3 direction) {
    // Avoid div by zero, returns 1/2^80, an extremely small number
    const float ooeps = exp2f(-80.0f);

    // Calculate inverse direction
    float3 idir;
    // TODO: is this wrong?
    idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));

    // idir.x = fabsf(direction.x) > ooeps ? 1.0f / direction.x : copysignf(ooeps, direction.x);
    // idir.y = fabsf(direction.y) > ooeps ? 1.0f / direction.y : copysignf(ooeps, direction.y);
    // idir.z = fabsf(direction.z) > ooeps ? 1.0f / direction.z : copysignf(ooeps, direction.z);
    return idir;
}

bool ray_box_test(float3 low, float3 high, float3 idirection, float3 origin, float tmin, float tmax, float& thit)
{
	// const float3 lo = Low * InvDir - Ood;
	// const float3 hi = High * InvDir - Ood;
    float3 lo = get_t_bound(low, origin, idirection);
    float3 hi = get_t_bound(high, origin, idirection);

    // QUESTION: max value does not match rtao benchmark, rtao benchmark converts float to int with __float_as_int
    // i.e. __float_as_int: -110.704826 => -1025677090, -24.690834 => -1044019502

	// const float slabMin = tMinFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMin);
	// const float slabMax = tMaxFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMax);
    float min = magic_max7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmin);
    float max = magic_min7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);

	// OutIntersectionDist = slabMin;
    thit = min;

	// return slabMin <= slabMax;
    return (min <= max);
}

typedef struct StackEntry {
    uint8_t* addr;
    bool topLevel;
    bool leaf;
    StackEntry(uint8_t* addr, bool topLevel, bool leaf): addr(addr), topLevel(topLevel), leaf(leaf) {}
} StackEntry;

bool find_primitive(uint8_t* address, int primitiveID, int instanceID, std::list<uint8_t *>& path, bool isTopLevel = true, bool isLeaf = false, bool isRoot = true)
{
    path.push_back(address);

    if(isRoot)
    {
        GEN_RT_BVH topBVH; //TODO: test hit with world before traversal
        GEN_RT_BVH_unpack(&topBVH, (uint8_t*)address);

        uint8_t* topRootAddr = (uint8_t*)address + topBVH.RootNodeOffset;

        if(find_primitive(topRootAddr, primitiveID, instanceID, path, isTopLevel, false, false))
            return true;
    }
    
    else if(!isLeaf) // internal nodes
    {
        struct GEN_RT_BVH_INTERNAL_NODE node;
        GEN_RT_BVH_INTERNAL_NODE_unpack(&node, address);

        uint8_t *child_addr = address + (node.ChildOffset * 64);
        for(int i = 0; i < 6; i++)
        {
            if(node.ChildSize[i] > 0)
            {
                if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                    isLeaf = true;
                else
                    isLeaf = false;

                if(find_primitive(child_addr, primitiveID, instanceID, path, isTopLevel, isLeaf, false))
                    return true;
            }

            child_addr += node.ChildSize[i] * 64;
        }
    }

    else // leaf nodes
    {
        if(isTopLevel)
        {
            GEN_RT_BVH_INSTANCE_LEAF instanceLeaf;
            GEN_RT_BVH_INSTANCE_LEAF_unpack(&instanceLeaf, address);

            float4x4 worldToObjectMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.WorldToObjectm00);
            float4x4 objectToWorldMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.ObjectToWorldm00);

            assert(instanceLeaf.BVHAddress != NULL);
            if(instanceLeaf.InstanceID != instanceID)
                return false;
            if(find_primitive(instanceLeaf.BVHAddress, primitiveID, instanceID, path, false, false, true))
                return true;
        }
        else
        {
            struct GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR leaf_descriptor;
            GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_unpack(&leaf_descriptor, address);
            
            if (leaf_descriptor.LeafType == TYPE_QUAD)
            {
                struct GEN_RT_BVH_QUAD_LEAF leaf;
                GEN_RT_BVH_QUAD_LEAF_unpack(&leaf, address);

                float3 p[3];
                for(int i = 0; i < 3; i++)
                {
                    p[i].x = leaf.QuadVertex[i].X;
                    p[i].y = leaf.QuadVertex[i].Y;
                    p[i].z = leaf.QuadVertex[i].Z;
                }

                assert(leaf.PrimitiveIndex1Delta == 0);

                if(leaf.PrimitiveIndex0 == primitiveID)
                {
                    return true;
                }
            }
            else
            {
                printf("sth is wrong here\n");
            }
        }
    }

    path.pop_back();
    return false;
}

void VulkanRayTracing::init(uint32_t launch_width, uint32_t launch_height)
{
    if(_init_)
        return;
    _init_ = true;

    uint32_t width = (launch_width + 31) / 32;
    uint32_t height = launch_height;

    if(intersectionTableType == IntersectionTableType::Baseline)
    {
        intersection_table = new Baseline_warp_intersection_table**[width];
        for(int i = 0; i < width; i++)
        {
            intersection_table[i] = new Baseline_warp_intersection_table*[height];
            for(int j = 0; j < height; j++)
                intersection_table[i][j] = new Baseline_warp_intersection_table();
        }
    }
    else
    {
        intersection_table = new Coalescing_warp_intersection_table**[width];
        for(int i = 0; i < width; i++)
        {
            intersection_table[i] = new Coalescing_warp_intersection_table*[height];
            for(int j = 0; j < height; j++)
                intersection_table[i][j] = new Coalescing_warp_intersection_table();
        }

    }
}


bool debugTraversal = false;

void VulkanRayTracing::traceRay(VkAccelerationStructureKHR _topLevelAS,
				   uint rayFlags,
                   uint cullMask,
                   uint sbtRecordOffset,
                   uint sbtRecordStride,
                   uint missIndex,
                   float3 origin,
                   float Tmin,
                   float3 direction,
                   float Tmax,
                   int payload,
                   const ptx_instruction *pI,
                   ptx_thread_info *thread)
{
    // printf("## calling trceRay function. rayFlags = %d, cullMask = %d, sbtRecordOffset = %d, sbtRecordStride = %d, missIndex = %d, origin = (%f, %f, %f), Tmin = %f, direction = (%f, %f, %f), Tmax = %f, payload = %d\n",
    //         rayFlags, cullMask, sbtRecordOffset, sbtRecordStride, missIndex, origin.x, origin.y, origin.z, Tmin, direction.x, direction.y, direction.z, Tmax, payload);
    // std::list<uint8_t *> path;
    // find_primitive((uint8_t*)_topLevelAS, 6, 2, path);

    if (!use_external_launcher && !dumped) 
    {
      assert(0);
        // dump_AS(VulkanRayTracing::descriptorSet, _topLevelAS);
        std::cout << "Trace dumped" << std::endl;
        dumped = true;
    }

    // Convert device address back to host address for func sim. This will break if the device address was modified then passed to traceRay. Should be fixable if I also record the size when I malloc then I can check the bounds of the device address.
    uint8_t* deviceAddress = nullptr;
    int64_t device_offset = 0;
    if (use_external_launcher)
    {
        deviceAddress = (uint8_t*)_topLevelAS;
        bool addressFound = false;
        for (int i = 0; i < MAX_DESCRIPTOR_SETS; i++)
        {
            for (int j = 0; j < MAX_DESCRIPTOR_SET_BINDINGS; j++)
            {
                if (launcher_deviceDescriptorSets[i][j] == (void*)_topLevelAS)
                {
                    _topLevelAS = launcher_descriptorSets[i][j];
                    addressFound = true;
                    break;
                }
            }
            if (addressFound)
                break;
        }
        if (!addressFound)
            abort();
    
        // Calculate offset between host and device for memory transactions
        device_offset = (uint64_t)deviceAddress - (uint64_t)_topLevelAS;
    }

    


    Traversal_data traversal_data;

    traversal_data.ray_world_direction = direction;
    traversal_data.ray_world_origin = origin;
    traversal_data.sbtRecordOffset = sbtRecordOffset;
    traversal_data.sbtRecordStride = sbtRecordStride;
    traversal_data.missIndex = missIndex;
    traversal_data.Tmin = Tmin;
    traversal_data.Tmax = Tmax;

    std::ofstream traversalFile;

    if (debugTraversal)
    {
        traversalFile.open("traversal.txt");
        traversalFile << "starting traversal\n";
        traversalFile << "origin = (" << origin.x << ", " << origin.y << ", " << origin.z << "), ";
        traversalFile << "direction = (" << direction.x << ", " << direction.y << ", " << direction.z << "), ";
        traversalFile << "tmin = " << Tmin << ", tmax = " << Tmax << std::endl << std::endl;
    }


    bool terminateOnFirstHit = rayFlags & SpvRayFlagsTerminateOnFirstHitKHRMask;
    bool skipClosestHitShader = rayFlags & SpvRayFlagsSkipClosestHitShaderKHRMask;

    std::vector<MemoryTransactionRecord> transactions;
    std::vector<MemoryStoreTransactionRecord> store_transactions;

    gpgpu_context *ctx = GPGPU_Context();

    if (terminateOnFirstHit) ctx->func_sim->g_n_anyhit_rays++;
    else ctx->func_sim->g_n_closesthit_rays++;

    unsigned total_nodes_accessed = 0;
    std::map<uint8_t*, unsigned> tree_level_map;
    
	// Create ray
	Ray ray;
	ray.make_ray(origin, direction, Tmin, Tmax);
    thread->add_ray_properties(ray);

	// Set thit to max
    float min_thit = ray.dir_tmax.w;
    struct GEN_RT_BVH_QUAD_LEAF closest_leaf;
    struct GEN_RT_BVH_INSTANCE_LEAF closest_instanceLeaf;    
    float4x4 closest_worldToObject, closest_objectToWorld;
    Ray closest_objectRay;
    float min_thit_object;

	// Get bottom-level AS
    //uint8_t* topLevelASAddr = get_anv_accel_address((VkAccelerationStructureKHR)_topLevelAS);
    GEN_RT_BVH topBVH; //TODO: test hit with world before traversal
    GEN_RT_BVH_unpack(&topBVH, (uint8_t*)_topLevelAS);
    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)_topLevelAS + device_offset), GEN_RT_BVH_length * 4, TransactionType::BVH_STRUCTURE));
    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;
    
    uint8_t* topRootAddr = (uint8_t*)_topLevelAS + topBVH.RootNodeOffset;

    // Get min/max
    if (!ctx->func_sim->g_rt_world_set) {
        struct GEN_RT_BVH_INTERNAL_NODE node;
        GEN_RT_BVH_INTERNAL_NODE_unpack(&node, topRootAddr);
        for(int i = 0; i < 6; i++) {
            if (node.ChildSize[i] > 0) {
                float3 idir = calculate_idir(ray.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                float3 lo, hi;
                set_child_bounds(&node, i, &lo, &hi);
                ctx->func_sim->g_rt_world_min = min(ctx->func_sim->g_rt_world_min, lo);
                ctx->func_sim->g_rt_world_max = min(ctx->func_sim->g_rt_world_max, hi);
            }
        }
        ctx->func_sim->g_rt_world_set = true;
    }

    std::list<StackEntry> stack;
    tree_level_map[topRootAddr] = 1;
    
    {
        float3 lo, hi;
        lo.x = topBVH.BoundsMin.X;
        lo.y = topBVH.BoundsMin.Y;
        lo.z = topBVH.BoundsMin.Z;
        hi.x = topBVH.BoundsMax.X;
        hi.y = topBVH.BoundsMax.Y;
        hi.z = topBVH.BoundsMax.Z;

        float thit;
        if(ray_box_test(lo, hi, calculate_idir(ray.get_direction()), ray.get_origin(), ray.get_tmin(), ray.get_tmax(), thit))
            stack.push_back(StackEntry(topRootAddr, true, false));
    }

    while (!stack.empty())
    {
        uint8_t *node_addr = NULL;
        uint8_t *next_node_addr = NULL;

        // traverse top level internal nodes
        assert(stack.back().topLevel);
        
        if(!stack.back().leaf)
        {
            next_node_addr = stack.back().addr;
            stack.pop_back();
        }

        while (next_node_addr > 0)
        {
            node_addr = next_node_addr;
            next_node_addr = NULL;
            struct GEN_RT_BVH_INTERNAL_NODE node;
            GEN_RT_BVH_INTERNAL_NODE_unpack(&node, node_addr);
            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)node_addr + device_offset), GEN_RT_BVH_INTERNAL_NODE_length * 4, TransactionType::BVH_INTERNAL_NODE));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
            total_nodes_accessed++;

            if (debugTraversal)
            {
                traversalFile << "traversing top level internal node " << (void *)node_addr << "\n";
            }

            bool child_hit[6];
            float thit[6];
            for(int i = 0; i < 6; i++)
            {
                if (node.ChildSize[i] > 0)
                {
                    float3 idir = calculate_idir(ray.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                    float3 lo, hi;
                    set_child_bounds(&node, i, &lo, &hi);

                    child_hit[i] = ray_box_test(lo, hi, idir, ray.get_origin(), ray.get_tmin(), ray.get_tmax(), thit[i]);
                    if(child_hit[i] && thit[i] >= min_thit)
                        child_hit[i] = false;

                    
                    if (debugTraversal)
                    {
                        if(child_hit[i])
                            traversalFile << "hit child number " << i << ", ";
                        else
                            traversalFile << "missed child number " << i << ", ";
                        traversalFile << "lo = (" << lo.x << ", " << lo.y << ", " << lo.z << "), ";
                        traversalFile << "hi = (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;
                    }
                }
                else
                    child_hit[i] = false;
            }

            uint8_t *child_addr = node_addr + (node.ChildOffset * 64);
            for(int i = 0; i < 6; i++)
            {
                if(child_hit[i])
                {
                    if (debugTraversal)
                    {
                        traversalFile << "add child node " << (void *)child_addr << ", child number " << i << ", type " << node.ChildType[i] << ", to stack" << std::endl;
                    }
                    if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                    {
                        assert(node.ChildType[i] == NODE_TYPE_INSTANCE);
                        stack.push_back(StackEntry(child_addr, true, true));
                        assert(tree_level_map.find(node_addr) != tree_level_map.end());
                        tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                    }
                    else
                    {
                        if(next_node_addr == NULL) {
                            next_node_addr = child_addr; // TODO: sort by thit
                            assert(tree_level_map.find(node_addr) != tree_level_map.end());
                            tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                        }
                        else {
                            stack.push_back(StackEntry(child_addr, true, false));
                            assert(tree_level_map.find(node_addr) != tree_level_map.end());
                            tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                        }
                    }
                }
                else
                {
                    if (debugTraversal)
                    {
                        traversalFile << "ignoring missed node " << (void *)child_addr << ", child number " << i << ", type " << node.ChildType[i] << std::endl;
                    }
                }
                child_addr += node.ChildSize[i] * 64;
            }

            if (debugTraversal)
            {
                traversalFile << std::endl;
            }
        }

        // traverse top level leaf nodes
        while (!stack.empty() && stack.back().leaf)
        {
            assert(stack.back().topLevel);

            uint8_t* leaf_addr = stack.back().addr;
            stack.pop_back();

            GEN_RT_BVH_INSTANCE_LEAF instanceLeaf;
            GEN_RT_BVH_INSTANCE_LEAF_unpack(&instanceLeaf, leaf_addr);
            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_INSTANCE_LEAF_length * 4, TransactionType::BVH_INSTANCE_LEAF));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INSTANCE_LEAF)]++;
            total_nodes_accessed++;

            float4x4 worldToObjectMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.WorldToObjectm00);
            float4x4 objectToWorldMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.ObjectToWorldm00);

            assert(instanceLeaf.BVHAddress != NULL);
            GEN_RT_BVH botLevelASAddr;
            GEN_RT_BVH_unpack(&botLevelASAddr, (uint8_t *)(leaf_addr + instanceLeaf.BVHAddress));
            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + instanceLeaf.BVHAddress + device_offset), GEN_RT_BVH_length * 4, TransactionType::BVH_STRUCTURE));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;

            // std::ofstream offsetfile;
            // offsetfile.open("offsets.txt", std::ios::app);
            // offsetfile << (int64_t)instanceLeaf.BVHAddress << std::endl;

            // std::ofstream leaf_addr_file;
            // leaf_addr_file.open("leaf.txt", std::ios::app);
            // leaf_addr_file << (int64_t)((uint64_t)leaf_addr - (uint64_t)_topLevelAS) << std::endl;

            float worldToObject_tMultiplier;
            Ray objectRay = make_transformed_ray(ray, worldToObjectMatrix, &worldToObject_tMultiplier);

            uint8_t * botLevelRootAddr = ((uint8_t *)((uint64_t)leaf_addr + instanceLeaf.BVHAddress)) + botLevelASAddr.RootNodeOffset;
            stack.push_back(StackEntry(botLevelRootAddr, false, false));
            assert(tree_level_map.find(leaf_addr) != tree_level_map.end());
            tree_level_map[botLevelRootAddr] = tree_level_map[leaf_addr];

            if (debugTraversal)
            {
                traversalFile << "traversing top level leaf node " << (void *)leaf_addr << " with instanceID = " << instanceLeaf.InstanceID << ", child bot root " << (void *)botLevelRootAddr << std::endl;
                traversalFile << "warped ray to object coordinates ";
                traversalFile << "origin = (" << objectRay.get_origin().x << ", " << objectRay.get_origin().y << ", " << objectRay.get_origin().z << "), ";
                traversalFile << "direction = (" << objectRay.get_direction().x << ", " << objectRay.get_direction().y << ", " << objectRay.get_direction().z << "), ";
                traversalFile << "tmin = " << objectRay.get_tmin() << ", tmax = " << objectRay.get_tmax() << std::endl << std::endl;
            }

            // traverse bottom level tree
            while (!stack.empty() && !stack.back().topLevel)
            {
                uint8_t* node_addr = NULL;
                uint8_t* next_node_addr = stack.back().addr;
                stack.pop_back();
                

                // traverse bottom level internal nodes
                while (next_node_addr > 0)
                {
                    node_addr = next_node_addr;
                    next_node_addr = NULL;

                    // if(node_addr == *(++path.rbegin()))
                    //     printf("this is where things go wrong\n");

                    struct GEN_RT_BVH_INTERNAL_NODE node;
                    GEN_RT_BVH_INTERNAL_NODE_unpack(&node, node_addr);
                    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)node_addr + device_offset), GEN_RT_BVH_INTERNAL_NODE_length * 4, TransactionType::BVH_INTERNAL_NODE));
                    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
                    total_nodes_accessed++;

                    if (debugTraversal)
                    {
                        traversalFile << "traversing bot level internal node " << (void *)node_addr << "\n";
                    }

                    bool child_hit[6];
                    float thit[6];
                    for(int i = 0; i < 6; i++)
                    {
                        if (node.ChildSize[i] > 0)
                        {
                            float3 idir = calculate_idir(objectRay.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                            float3 lo, hi;
                            set_child_bounds(&node, i, &lo, &hi);

                            child_hit[i] = ray_box_test(lo, hi, idir, objectRay.get_origin(), objectRay.get_tmin(), objectRay.get_tmax(), thit[i]);
                            if(child_hit[i] && thit[i] >= min_thit * worldToObject_tMultiplier)
                                child_hit[i] = false;

                            if (debugTraversal)
                            {
                                if(child_hit[i])
                                    traversalFile << "hit child number " << i << ", ";
                                else
                                    traversalFile << "missed child number " << i << ", ";
                                traversalFile << "lo = (" << lo.x << ", " << lo.y << ", " << lo.z << "), ";
                                traversalFile << "hi = (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;
                            }
                        }
                        else
                            child_hit[i] = false;
                    }

                    uint8_t *child_addr = node_addr + (node.ChildOffset * 64);
                    for(int i = 0; i < 6; i++)
                    {
                        if(child_hit[i])
                        {
                            if (debugTraversal)
                            {
                                traversalFile << "add child node " << (void *)child_addr << ", child number " << i << ", type " << node.ChildType[i] << ", to stack" << std::endl;
                            }

                            if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                            {
                                stack.push_back(StackEntry(child_addr, false, true));
                                assert(tree_level_map.find(node_addr) != tree_level_map.end());
                                tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                            }
                            else
                            {
                                if(next_node_addr == 0) {
                                    next_node_addr = child_addr; // TODO: sort by thit
                                    assert(tree_level_map.find(node_addr) != tree_level_map.end());
                                    tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                                }
                                else {
                                    stack.push_back(StackEntry(child_addr, false, false));
                                    assert(tree_level_map.find(node_addr) != tree_level_map.end());
                                    tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                                }
                            }
                        }
                        else
                        {
                            if (debugTraversal)
                            {
                                traversalFile << "ignoring missed node " << (void *)child_addr << ", child number " << i << ", type " << node.ChildType[i] << std::endl;
                            }
                        }
                        child_addr += node.ChildSize[i] * 64;
                    }

                    if (debugTraversal)
                    {
                        traversalFile << std::endl;
                    }
                }

                // traverse bottom level leaf nodes
                while(!stack.empty() && !stack.back().topLevel && stack.back().leaf)
                {
                    uint8_t* leaf_addr = stack.back().addr;
                    stack.pop_back();
                    struct GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR leaf_descriptor;
                    GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_unpack(&leaf_descriptor, leaf_addr);
                    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_length * 4, TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR));
                    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR)]++;

                    if (leaf_descriptor.LeafType == TYPE_QUAD)
                    {
                        struct GEN_RT_BVH_QUAD_LEAF leaf;
                        GEN_RT_BVH_QUAD_LEAF_unpack(&leaf, leaf_addr);

                        // if(leaf.PrimitiveIndex0 == 9600)
                        // {
                        //     leaf.QuadVertex[2].Z = -0.001213;
                        // }

                        float3 p[3];
                        for(int i = 0; i < 3; i++)
                        {
                            p[i].x = leaf.QuadVertex[i].X;
                            p[i].y = leaf.QuadVertex[i].Y;
                            p[i].z = leaf.QuadVertex[i].Z;
                        }

                        // Triangle intersection algorithm
                        float thit;
                        bool hit = VulkanRayTracing::mt_ray_triangle_test(p[0], p[1], p[2], objectRay, &thit);

                        assert(leaf.PrimitiveIndex1Delta == 0);

                        if (debugTraversal)
                        {
                            if(hit)
                                traversalFile << "hit quad node " << (void *)leaf_addr << " with thit " << thit << " ";
                            else
                                traversalFile << "miss quad node " << leaf_addr << " ";
                            traversalFile << "primitiveID = " << leaf.PrimitiveIndex0 << ", InstanceID = " << instanceLeaf.InstanceID << "\n";

                            traversalFile << "p[0] = (" << p[0].x << ", " << p[0].y << ", " << p[0].z << ") ";
                            traversalFile << "p[1] = (" << p[1].x << ", " << p[1].y << ", " << p[1].z << ") ";
                            traversalFile << "p[2] = (" << p[2].x << ", " << p[2].y << ", " << p[2].z << ") ";
                            traversalFile << "p[3] = (" << p[3].x << ", " << p[3].y << ", " << p[3].z << ")" << std::endl;
                        }

                        float world_thit = thit / worldToObject_tMultiplier;

                        //TODO: why the Tmin Tmax consition wasn't handled in the object coordinates?
                        if(hit && Tmin <= world_thit && world_thit <= Tmax && world_thit < min_thit)
                        {
                            if (debugTraversal)
                            {
                                traversalFile << "quad node " << (void *)leaf_addr << ", primitiveID " << leaf.PrimitiveIndex0 << " is the closest hit. world_thit " << thit / worldToObject_tMultiplier;
                            }

                            min_thit = thit / worldToObject_tMultiplier;
                            min_thit_object = thit;
                            closest_leaf = leaf;
                            closest_instanceLeaf = instanceLeaf;
                            closest_worldToObject = worldToObjectMatrix;
                            closest_objectToWorld = objectToWorldMatrix;
                            closest_objectRay = objectRay;
                            min_thit_object = thit;
                            thread->add_ray_intersect();
                            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_QUAD_LEAF_length * 4, TransactionType::BVH_QUAD_LEAF_HIT));
                            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF_HIT)]++;
                            total_nodes_accessed++;

                            if(terminateOnFirstHit)
                            {
                                stack.clear();
                            }
                        }
                        else {
                            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_QUAD_LEAF_length * 4, TransactionType::BVH_QUAD_LEAF));
                            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF)]++;
                            total_nodes_accessed++;
                        }
                        if (debugTraversal)
                        {
                            traversalFile << std::endl;
                        }
                    }
                    else
                    {
                        struct GEN_RT_BVH_PROCEDURAL_LEAF leaf;
                        GEN_RT_BVH_PROCEDURAL_LEAF_unpack(&leaf, leaf_addr);
                        transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_PROCEDURAL_LEAF_length * 4, TransactionType::BVH_PROCEDURAL_LEAF));
                        ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PROCEDURAL_LEAF)]++;
                        total_nodes_accessed++;

                        uint32_t hit_group_index = instanceLeaf.InstanceContributionToHitGroupIndex;

                        warp_intersection_table* table = intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
                        auto intersectionTransactions = table->add_intersection(hit_group_index, thread->get_tid().x, leaf.PrimitiveIndex[0], instanceLeaf.InstanceID, pI, thread); // TODO: switch these to device addresses
                        
                        // transactions.insert(transactions.end(), intersectionTransactions.first.begin(), intersectionTransactions.first.end());
                        for(auto & newTransaction : intersectionTransactions.first)
                        {
                            bool found = false;
                            for(auto & transaction : transactions)
                                if(transaction.address == newTransaction.address)
                                {
                                    found = true;
                                    break;
                                }
                            if(!found)
                                transactions.push_back(newTransaction);

                        }
                        store_transactions.insert(store_transactions.end(), intersectionTransactions.second.begin(), intersectionTransactions.second.end());
                    }
                }
            }
        }
    }

    if (min_thit < ray.dir_tmax.w)
    {
        traversal_data.hit_geometry = true;
        ctx->func_sim->g_rt_num_hits++;
        traversal_data.closest_hit.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        traversal_data.closest_hit.geometry_index = closest_leaf.LeafDescriptor.GeometryIndex;
        traversal_data.closest_hit.primitive_index = closest_leaf.PrimitiveIndex0;
        traversal_data.closest_hit.instance_index = closest_instanceLeaf.InstanceID;
        float3 intersection_point = ray.get_origin() + make_float3(ray.get_direction().x * min_thit, ray.get_direction().y * min_thit, ray.get_direction().z * min_thit);
        float3 rayatinter = ray.at(min_thit);
        // assert(intersection_point.x == ray.at(min_thit).x && intersection_point.y == ray.at(min_thit).y && intersection_point.z == ray.at(min_thit).z);
        traversal_data.closest_hit.intersection_point = intersection_point;
        traversal_data.closest_hit.worldToObjectMatrix = closest_worldToObject;
        traversal_data.closest_hit.objectToWorldMatrix = closest_objectToWorld;
        traversal_data.closest_hit.world_min_thit = min_thit;

        float3 p[3];
        for(int i = 0; i < 3; i++)
        {
            p[i].x = closest_leaf.QuadVertex[i].X;
            p[i].y = closest_leaf.QuadVertex[i].Y;
            p[i].z = closest_leaf.QuadVertex[i].Z;
        }
        float3 object_intersection_point = closest_objectRay.get_origin() + make_float3(closest_objectRay.get_direction().x * min_thit_object, closest_objectRay.get_direction().y * min_thit_object, closest_objectRay.get_direction().z * min_thit_object);
        //closest_objectRay.at(min_thit_object);
        float3 barycentric = Barycentric(object_intersection_point, p[0], p[1], p[2]);
        traversal_data.closest_hit.barycentric_coordinates = barycentric;
        thread->RT_thread_data->set_hitAttribute(barycentric);

        // store_transactions.push_back(MemoryStoreTransactionRecord(&traversal_data, sizeof(traversal_data), StoreTransactionType::Traversal_Results));
    }
    else
    {
        traversal_data.hit_geometry = false;
    }

    memory_space *mem = thread->get_global_memory();
    Traversal_data* device_traversal_data = (Traversal_data*) VulkanRayTracing::gpgpusim_alloc(sizeof(Traversal_data));
    mem->write(device_traversal_data, sizeof(Traversal_data), &traversal_data, thread, pI);
    thread->RT_thread_data->traversal_data.push_back(device_traversal_data);
    
    thread->set_rt_transactions(transactions);
    thread->set_rt_store_transactions(store_transactions);

    if (debugTraversal)
    {
        traversalFile.close();
    }

    if (total_nodes_accessed > ctx->func_sim->g_max_nodes_per_ray) {
        ctx->func_sim->g_max_nodes_per_ray = total_nodes_accessed;
    }
    ctx->func_sim->g_tot_nodes_per_ray += total_nodes_accessed;

    unsigned level = 0;
    for (auto it=tree_level_map.begin(); it!=tree_level_map.end(); it++) {
        if (it->second > level) {
            level = it->second;
        }
    }
    if (level > ctx->func_sim->g_max_tree_depth) {
        ctx->func_sim->g_max_tree_depth = level;
    }
}

void VulkanRayTracing::endTraceRay(const ptx_instruction *pI, ptx_thread_info *thread)
{
    assert(thread->RT_thread_data->traversal_data.size() > 0);
    thread->RT_thread_data->traversal_data.pop_back();
    warp_intersection_table* table = intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
    table->clear();
}

bool VulkanRayTracing::mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit)
{
    // Moller Trumbore algorithm (from scratchapixel.com)
    float3 v0v1 = p1 - p0;
    float3 v0v2 = p2 - p0;
    float3 pvec = cross(ray_properties.get_direction(), v0v2);
    float det = dot(v0v1, pvec);

    float idet = 1 / det;

    float3 tvec = ray_properties.get_origin() - p0;
    float u = dot(tvec, pvec) * idet;

    if (u < 0 || u > 1) return false;

    float3 qvec = cross(tvec, v0v1);
    float v = dot(ray_properties.get_direction(), qvec) * idet;

    if (v < 0 || (u + v) > 1) return false;

    *thit = dot(v0v2, qvec) * idet;
    return true;
}

float3 VulkanRayTracing::Barycentric(float3 p, float3 a, float3 b, float3 c)
{
    //source: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    float3 v0 = b - a;
    float3 v1 = c - a;
    float3 v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return {v, w, u};
}

void VulkanRayTracing::load_descriptor(const ptx_instruction *pI, ptx_thread_info *thread)
{

}


void VulkanRayTracing::setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos)
{
    VulkanRayTracing::pCreateInfos = pCreateInfos;
	std::cout << "gpgpusim: set pipeline" << std::endl;
}


void VulkanRayTracing::setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount)
{
    VulkanRayTracing::pGeometries = pGeometries;
    VulkanRayTracing::geometryCount = geometryCount;
	std::cout << "gpgpusim: set geometry" << std::endl;
}

void VulkanRayTracing::setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure)
{
    GEN_RT_BVH topBVH; //TODO: test hit with world before traversal
    GEN_RT_BVH_unpack(&topBVH, (uint8_t *)accelerationStructure);




    std::cout << "gpgpusim: set AS" << std::endl;
    VulkanRayTracing::topLevelAS = accelerationStructure;
}

std::string base_name(std::string & path)
{
  return path.substr(path.find_last_of("/") + 1);
}

void VulkanRayTracing::setDescriptorSet(struct anv_descriptor_set *set, unsigned set_index)
{
    VulkanRayTracing::descriptorSet[set_index] = set;
}

static bool invoked = false;

void copyHardCodedShaders()
{
    std::ifstream  src;
    std::ofstream  dst;

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_MISS_2.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_MISS_2.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();
    
    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_CLOSEST_HIT_2.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_CLOSEST_HIT_2.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_RAYGEN_0.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_RAYGEN_0.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_INTERSECTION_4.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_INTERSECTION_4.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // {
    //     std::ifstream  src("/home/mrs/emerald-ray-tracing/MESA_SHADER_MISS_0.ptx", std::ios::binary);
    //     std::ofstream  dst("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_MISS_1.ptx",   std::ios::binary);
    //     dst << src.rdbuf();
    //     src.close();
    //     dst.close();
    // }
}

uint32_t VulkanRayTracing::registerShaders(char * shaderPath, gl_shader_stage shaderType)
{
    copyHardCodedShaders();

    VulkanRayTracing::invoke_gpgpusim();
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    // Register all the ptx files in $MESA_ROOT/gpgpusimShaders by looping through them
    // std::vector <std::string> ptx_list;

    // Add ptx file names in gpgpusimShaders folder to ptx_list
    char *mesa_root = getenv("MESA_ROOT");
    char *gpgpusim_root = getenv("GPGPUSIM_ROOT");
    // char *filePath = "gpgpusimShaders/";
    // char fullPath[200];
    // snprintf(fullPath, sizeof(fullPath), "%s%s", mesa_root, filePath);
    // std::string fullPathString(fullPath);

    // for (auto &p : fs::recursive_directory_iterator(fullPathString))
    // {
    //     if (p.path().extension() == ".ptx")
    //     {
    //         //std::cout << p.path().string() << '\n';
    //         ptx_list.push_back(p.path().string());
    //     }
    // }

    std::string fullpath(shaderPath);
    std::string fullfilename = base_name(fullpath);
    std::string filenameNoExt;
    size_t start = fullfilename.find_first_not_of('.', 0);
    size_t end = fullfilename.find('.', start);
    filenameNoExt = fullfilename.substr(start, end - start);
    std::string idInString = filenameNoExt.substr(filenameNoExt.find_last_of("_") + 1);
    // Register each ptx file in ptx_list
    shader_stage_info shader;
    //shader.ID = VulkanRayTracing::shaders.size();
    shader.ID = std::stoi(idInString);
    shader.type = shaderType;
    shader.function_name = (char*)malloc(200 * sizeof(char));

    std::string deviceFunction;

    switch(shaderType) {
        case MESA_SHADER_RAYGEN:
            // shader.function_name = "raygen_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "raygen_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_RAYGEN";
            break;
        case MESA_SHADER_ANY_HIT:
            // shader.function_name = "anyhit_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "anyhit_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "";
            assert(0);
            break;
        case MESA_SHADER_CLOSEST_HIT:
            // shader.function_name = "closesthit_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "closesthit_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_CLOSEST_HIT";
            break;
        case MESA_SHADER_MISS:
            // shader.function_name = "miss_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "miss_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_MISS";
            break;
        case MESA_SHADER_INTERSECTION:
            // shader.function_name = "intersection_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "intersection_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_INTERSECTION";
            break;
        case MESA_SHADER_CALLABLE:
            // shader.function_name = "callable_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "callable_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "";
            assert(0);
            break;
        case MESA_SHADER_VERTEX:
            // shader.function_name = "callable_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "vertex");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_VERTEX";
            break;
        case MESA_SHADER_FRAGMENT:
            // shader.function_name = "callable_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "frag");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_FRAGMENT";
            break;
        default:
            assert(0);
    }
    deviceFunction += "_func" + std::to_string(shader.ID) + "_main";
    // deviceFunction += "_main";

    symbol_table *symtab;
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 20;
    unsigned selected_capability = 20;
    bool found = false;

    unsigned long long fat_cubin_handle = shader.ID;

    // PTX File
    //std::cout << itr << std::endl;
    symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(shaderPath);
    context->add_binary(symtab, fat_cubin_handle);
    // need to add all the magic registers to ptx.l to special_register, reference ayub ptx.l:225

    // PTX info
    // Run the python script and get ptxinfo
    std::cout << "GPGPUSIM: Generating PTXINFO for" << shaderPath << "info" << std::endl;
    char command[400];
    snprintf(command, sizeof(command), "python3 %s/scripts/generate_rt_ptxinfo.py %s", gpgpusim_root, shaderPath);
    int result = system(command);
    if (result != 0) {
        printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
        printf("               Ensure ptxas is in your path.\n");
        exit(1);
    }

    char ptxinfo_filename[400];
    snprintf(ptxinfo_filename, sizeof(ptxinfo_filename), "%sinfo", shaderPath);
    ctx->gpgpu_ptx_info_load_from_external_file(ptxinfo_filename); // TODO: make a version where it just loads my ptxinfo instead of generating a new one

    context->register_function(fat_cubin_handle, shader.function_name, deviceFunction.c_str());

    VulkanRayTracing::shaders.push_back(shader);
    system(("mv " + fullpath + " " + fullpath + "_translated").c_str());

    return shader.ID;
}


void VulkanRayTracing::invoke_gpgpusim()
{
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    if(!invoked)
    {
        //registerShaders();
        invoked = true;
    }
}

// int CmdTraceRaysKHRID = 0;

unsigned block_size = 64;
void VulkanRayTracing::run_shader(unsigned shader_id, unsigned thread_count) {
  gpgpu_context *ctx = GPGPU_Context();
  CUctx_st *context = GPGPUSim_Context(ctx);

  shader_stage_info shader = VulkanRayTracing::shaders[shader_id];
  function_info *entry = context->get_kernel(shader.function_name);

  if (entry->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           entry->get_name().c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           entry->get_name().c_str());
    /*
     * Some of the instructions like printf() gives the gpgpusim the wrong
     * impression that it is a function call. As printf() doesnt have a body
     * like functions do, doing pdom analysis for printf() causes a crash.
     */
    if (entry->get_function_size() > 0) entry->do_pdom();
    entry->set_pdom();
  }
  unsigned block_count = (thread_count + block_size - 1) / block_size;
//   unsigned block_count = 16 * 16;
  context->get_device()->get_gpgpu()->gtrace << "block_dim, " << block_size << std::endl;
  dim3 blockDim = dim3(block_size, 1, 1);
  dim3 gridDim = dim3(block_count, 1, 1);
  gpgpu_ptx_sim_arg_list_t args;
  kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      shader.function_name, args, gridDim, blockDim, context);

  struct CUstream_st *stream = 0;

  stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
  ctx->the_gpgpusim->g_stream_manager->push(op);

  fflush(stdout);

  while (!op.is_done() && !op.get_kernel()->done()) {
    printf("waiting for op to finish\n");
    sleep(1);
    continue;
  }
}

void print_memcpy(std::string name, unsigned addr, unsigned size,
                  unsigned per_cta_size) {
  gpgpu_context *ctx = GPGPU_Context();
  CUctx_st *context = GPGPUSim_Context(ctx);
  context->get_device()->get_gpgpu()->gtrace
      << name << ",0x" << std::hex << addr << "," << std::dec << size
      << "," << per_cta_size << std::endl;

}

const bool writeImageBinary = true;
// checkpointing to we don't have to run vertex shader every time
// unsigned draw = 0;
#define SKIP_VS false
#define SKIP_FS false
unsigned DRAW_START = 1;
unsigned DRAW_END = 2;


// workloads: 
// render_passes:       0 
// instancing draw #0:  1
// instancing draw #1:  2
// #define WORKLOAD 0
// other places need to be updated
// this file - remember to update the draw counter range
// vulkan-sim/src/cuda-sim/vulkan_rt_thread_data.h - shader input/output names

// #if WORKLOAD == 0
// render_passes
// unsigned attrib_size[] = {3};
// unsigned attrib_stride[] = {0};
// unsigned out_attrib_count = 3;
// unsigned out_attrib_size[] = {16,8,12};
// unsigned out_pos_index = 0;
// #elif WORKLOAD == 1 || WORKLOAD == 2
// instancing
// layout(location = 1) out vec3 outColor;
// layout(location = 2) out vec2 outUV;
// layout(location = 0) out vec3 outNormal;
// layout(location = 4) out vec3 outLightVec;
// layout(location = 3) out vec3 outViewVec;
// unsigned attrib_size[] = {3,3,2};
// unsigned attrib_stride[] = {0,12,24,32};
// unsigned out_attrib_count = 6;
// unsigned out_attrib_size[] = {12,12,8,12,12,16};
// unsigned out_pos_index = 5;
// #endif

// unsigned out_attrib_count = -1;
// unsigned *out_attrib_size = NULL;
// unsigned out_pos_index = -1;
unsigned tex_index = -1;
void VulkanRayTracing::clearDraws() {
  for (auto &draw : draw_meta) {
      delete draw;
  }
  draw_meta.clear();
}

void VulkanRayTracing::saveDumbDraw() {
    draw_meta.push_back(NULL);
}

void VulkanRayTracing::saveDraw(struct anv_cmd_buffer *cmd_buffer,
                                unsigned VertexCount, unsigned StartVertex,
                                unsigned instanceCount, unsigned StartInstance,
                                unsigned BaseVertex) {
  if (VertexCount == 6 && cmd_buffer->state.gfx.dynamic.depth_compare_op == VK_COMPARE_OP_NEVER) {
    return;
  }
  VertexMeta = new struct vertex_metadata();
  // VertexMeta->vbuffer = cmd_buffer->state.vertex_bindings;
  struct anv_vertex_binding *vbuffer = cmd_buffer->state.vertex_bindings;
  VertexMeta->pipeline = cmd_buffer->state.gfx.pipeline;
  assert(cmd_buffer->state.gfx.dynamic.viewport.count == 1);
  VertexMeta->viewports = *(cmd_buffer->state.gfx.dynamic.viewport.viewports);
  char const *app_env = std::getenv("VULKAN_APP");
  std::string vulkan_app(app_env);
  if (vulkan_app == "demo" or vulkan_app == "materials" or
      vulkan_app == "sponza" or vulkan_app == "platformer") {
    if (VertexMeta->viewports.width != 1152) {
      return;
    }
  }
  VertexMeta->push_constants = new float[128];
  memcpy(VertexMeta->push_constants,
         cmd_buffer->state.gfx.base.push_constants.client_data, 128 * 4);
    //   (float *)&cmd_buffer->state.gfx.base.push_constants.client_data;
  assert(pipeline_shader_map.find((void *)VertexMeta->pipeline) !=
         pipeline_shader_map.end());
  unsigned vertex_id = pipeline_shader_map.at((void *)VertexMeta->pipeline);
  unsigned frag_id = vertex_id + 1;
  VertexMeta->VertexCountPerInstance = VertexCount;
  VertexMeta->StartVertexLocation = StartVertex;
  VertexMeta->InstanceCount = instanceCount;
  VertexMeta->StartInstanceLocation = StartInstance;
  VertexMeta->BaseVertexLocation = BaseVertex;
  VertexMeta->DepthcmpOp = cmd_buffer->state.gfx.dynamic.depth_compare_op;
  VertexMeta->VertexAttrib = pipeline_vertex_map.at((void *)VertexMeta->pipeline);
  VertexMeta->index_buffer = index_buffer;
  VertexMeta->index_type = index_type;

  for (unsigned i = 0; i < MAX_VBS; i++) {
    if (vbuffer[i].buffer) {
      VertexMeta->vertex_buffers[i] = vbuffer[i].buffer;
      VertexMeta->vertex_size[i] = vbuffer[i].buffer->size;
      VertexMeta->vertex_count[i] = VertexMeta->vertex_size[i] / 4;
      VertexMeta->vertex_stride[i] = VertexMeta->pipeline->vb[i].stride;
      // dumpVertex(vbuffer[i].buffer, VertexMeta->pipeline, i);
      // assert(VertexMeta->pipeline->vb[i].stride % 4 == 0);
      // assert(vbuffer[i].buffer->size % (VertexMeta->pipeline->vb[i].stride / 4) == 0);
    }
  }

  for (unsigned i = 0; i < 8; i++) {
    if (cmd_buffer->state.gfx.base.descriptors[i] == NULL) {
      continue;
    }
    struct anv_descriptor_set *set = cmd_buffer->state.gfx.base.descriptors[i];
    for (int j = 0; j < set->layout->binding_count; j++) {
      const struct anv_descriptor_set_binding_layout *bind_layout =
          &set->layout->binding[j];
      if (bind_layout->type == (unsigned)-1) {
        continue;
      }
      
      struct anv_descriptor *desc =
          &set->descriptors[bind_layout->descriptor_index];

      struct anv_buffer_view *bview =
          &set->buffer_views[bind_layout->buffer_view_index];
      VertexMeta->decoded_descriptors[i][j] = desc;
      VertexMeta->decoded_bview[i][j] = bview;
    }
    // dump_descriptor_sets(cmd_buffer->state.gfx.base.descriptors[i], true, i);

    // anv_descriptor_set *set = new anv_descriptor_set();
    // memcpy(set, cmd_buffer->state.gfx.base.descriptors[i], sizeof(anv_descriptor_set));
    VertexMeta->descriptor_set[i] = cmd_buffer->state.gfx.base.descriptors[i];
    // VertexMeta->descriptor_set[i] = cmd_buffer->state.gfx.base.descriptors[i];
  }

  draw_meta.push_back(VertexMeta);
  // pbrtexture
  if (VertexMeta->viewports.width == 1280 && draw_meta.size() == 2 && vulkan_app == "pbrtexture") {
    draw_meta.pop_front();
    VulkanRayTracing::vkCmdDraw(NULL, 0, 0, 0, 0, 0);
  }
}

void VulkanRayTracing::vkCmdDraw(struct anv_cmd_buffer *cmd_buffer, unsigned VertexCount, unsigned StartVertex, unsigned instanceCount, unsigned StartInstance, unsigned BaseVertex) {
  // assume only vertex and frag. No geometry or tessellation
  gpgpu_context *ctx = GPGPU_Context();
  CUctx_st *context = GPGPUSim_Context(ctx);
  if (draw_meta.empty()) {
    return;
  }
  // define starting draw
  draw = 0;
  char const *start_env = std::getenv("START_DRAW");
  if (start_env != NULL) {
    draw = std::stoi(std::string(start_env));
  }
  char const *app_env = std::getenv("VULKAN_APP");
  std::string vulkan_app(app_env);


  while(true) {
    if (draw == 23 && vulkan_app == "sponza") {
      draw = 26;
    }
    if (draw == 80 && vulkan_app == "materials") {
      draw = draw_meta.size();
    }

    if(draw == 2 && vulkan_app == "instancing") {
      draw = draw_meta.size();
    }

    if (draw == draw_meta.size()) {
      context->get_device()->get_gpgpu()->gtrace.close();
      system("mv traces.traceg complete.traceg");
      exit(0);
    }

  VertexMeta = draw_meta[draw];
  
  if (!VertexMeta) {
    // draw_meta.pop_front();
    return;
  }

  struct anv_graphics_pipeline *pipeline = VertexMeta->pipeline;
  unsigned vertex_id = pipeline_shader_map.at((void*) pipeline);
  unsigned frag_id = vertex_id + 1;

  char const *app_env = std::getenv("VULKAN_APP");
  tex_index = 1;

  std::string vulkan_app(app_env);

  FILE *fp;
  // create fbo
  printf("Starting Drawcall #%u\n", draw);
  if (!FBO->fbo) {
    printf("render resolution: %u x %u\n", (unsigned) VertexMeta->viewports.width,(unsigned) VertexMeta->viewports.height);
    FBO->width = VertexMeta->viewports.width;
    FBO->height = VertexMeta->viewports.height;
    // FBO->width = 2560;
    // FBO->height = 1440;
    // FBO->width = 3840;
    // FBO->height = 2160;
    FBO->width = 640;
    FBO->height = 480;
    FBO->x = VertexMeta->viewports.x;
    FBO->y = VertexMeta->viewports.y;
    FBO->fbo_size = 4 * FBO->width * FBO->height * sizeof(float);
    FBO->fbo_count = 4 * FBO->width * FBO->height;
    FBO->fbo_stride = 16;
    FBO->fbo = new float[FBO->fbo_count]{0};
    FBO->depthout = new float[FBO->fbo_count / 4];
    for (unsigned i = 0; i < FBO->fbo_count / 4; i ++) {
      if (VertexMeta->DepthcmpOp == VK_COMPARE_OP_GREATER) {
        FBO->depthout[i] = (float) VertexMeta->viewports.minDepth;
      } else if (VertexMeta->DepthcmpOp == VK_COMPARE_OP_LESS ||
                  VertexMeta->DepthcmpOp == VK_COMPARE_OP_LESS_OR_EQUAL) {
        FBO->depthout[i] = (float) VertexMeta->viewports.maxDepth;
      } else {
        assert(0 && "unsupported depth compare op");
      }
      
    }
    FBO->fbo_dev = 0xF0000000;
    context->get_device()
      ->get_gpgpu()
      ->valid_addr_start["fbo_dev"] = (uint64_t) FBO->fbo_dev;
    context->get_device()
      ->get_gpgpu()
      ->valid_addr_end["fbo_dev"] = (uint64_t) FBO->fbo_dev + FBO->fbo_size;
  }
  assert(FBO->fbo);
  assert(FBO->depthout);
  assert(FBO->fbo_dev);
  // could be different for different type of FBO
  // dump vertex buffer

  unsigned group_size = 96;

  unsigned index_size = -1;
  char *index_buffer = anv_address_map(VertexMeta->index_buffer->address);
  if (VertexMeta->index_type == VK_INDEX_TYPE_UINT16) {
    index_size = sizeof(u_int16_t);
    if (((u_int16_t*) index_buffer)[0] == ((u_int16_t*) index_buffer)[1] &&
        ((u_int16_t*) index_buffer)[1] == ((u_int16_t*) index_buffer)[2] && 
        ((u_int16_t*) index_buffer)[0] == 0) {
      VertexMeta = NULL;
      return;
    }
  } else if (VertexMeta->index_type == VK_INDEX_TYPE_UINT32) {
    index_size = sizeof(u_int32_t);
    if (((u_int32_t*) index_buffer)[0] == ((u_int32_t*) index_buffer)[1] &&
        ((u_int32_t*) index_buffer)[1] == ((u_int32_t*) index_buffer)[2] && 
        ((u_int32_t*) index_buffer)[0] == 0) {
      VertexMeta = NULL;
      return;
    }
  } else {
    // add your type
    printf("unsupported index type\n");
    assert(0 && "unsupported index type");
  }

  unsigned num_group =
      (VertexMeta->index_buffer->size / index_size) / group_size + 1;
  VertexMeta->vertex_map.resize(num_group);
  for (unsigned i = 0; i < VertexMeta->index_buffer->size / index_size;
       i += 3) {
    for (unsigned j = 0; j < 3; j++) {
      unsigned vertex;
      if (index_size == 4) {
        vertex = ((u_int32_t*) index_buffer)[i + j];
      } else if (index_size == 2) {
        vertex = ((u_int16_t*) index_buffer)[i + j];
      }
      unsigned selected_vpc = i / group_size;
      if (VertexMeta->vertex_map[selected_vpc].find(vertex) ==
          VertexMeta->vertex_map[selected_vpc].end()) {
        VertexMeta->vertex_map[selected_vpc][vertex] = VertexMeta->vb.size();
        VertexMeta->vb.push_back(vertex);
      }
    }
  }
  printf("vertex count: %u\n", VertexMeta->vb.size());

  VulkanRayTracing::is_FS = false;
  for (auto &i : VertexMeta->VertexAttrib->binding) {
    if (VertexMeta->vertex_stride[i] != 0) {
      assert(VertexMeta->vertex_buffers[i]->size %
                 (VertexMeta->vertex_stride[i] / 4) ==
             0);
      // stride should be multilpe of 4 bytes -> vectors
      assert(VertexMeta->vertex_stride[i] % 4 == 0);
    }

      dumpVertex(VertexMeta->vertex_buffers[i], pipeline, i);
  }
  // Dump Descriptor Sets
//   return;
  for (unsigned i = 0; i < MAX_DESCRIPTOR_SETS; i++) {
    for (unsigned j = 0; j < MAX_DESCRIPTOR_SET_BINDINGS; j++) {
      dump_descriptor(i, j, VertexMeta->decoded_descriptors[i][j], VertexMeta->decoded_bview[i][j], false);
    }
  }

  thread_count = VertexMeta->vb.size() * VertexMeta->InstanceCount;

  VertexMeta->constants_dev_addr =
      context->get_device()->get_gpgpu()->gpu_malloc(128);
  context->get_device()->get_gpgpu()->memcpy_to_gpu(VertexMeta->constants_dev_addr, VertexMeta->push_constants, 128);
  context->get_device()
      ->get_gpgpu()
      ->valid_addr_start["push_constant"] = (uint64_t) VertexMeta->constants_dev_addr;
  context->get_device()
      ->get_gpgpu()
      ->valid_addr_end["push_constant"] = (uint64_t) VertexMeta->constants_dev_addr + 128;
  print_memcpy("MemcpyVulkan",VertexMeta->constants_dev_addr, 128, 0);
  if (!SKIP_VS) {
    // run vertex shader
    run_shader(vertex_id,thread_count);
  }

  if (draw == 1 && vulkan_app == "materials") {
    draw++;
    context->get_device()->get_gpgpu()->valid_addr_start.clear();
    context->get_device()->get_gpgpu()->valid_addr_end.clear();
    continue;
  }

  std::string mesa_root = getenv("MESA_ROOT");
  std::string filePath = "../fb/depth_buffer/";


  if (vulkan_app == "instancing") {
    unsigned size = 12;
    std::string name = "\%inColor";
    std::string identifier = "VARYING_SLOT_VAR1_xyz";
    uint32_t *dev_ptr = context->get_device()->get_gpgpu()->gpu_malloc(
        VulkanRayTracing::thread_count * size);
    VulkanRayTracing::VertexMeta->vertex_out_devptr.insert(
        std::make_pair(name, dev_ptr));
    VulkanRayTracing::VertexMeta->vertex_id_map.insert(
        std::make_pair(identifier, name));
    VulkanRayTracing::VertexMeta->vertex_out_stride.insert(
        std::make_pair(name, size));
  }

  for (auto out_attrib : VertexMeta->vertex_id_map) {
    std::string attrib_name = out_attrib.second;
    unsigned stride = VertexMeta->vertex_out_stride.at(attrib_name);
    VertexMeta->vertex_out_count[attrib_name] = stride / 4 * thread_count;
    VertexMeta->vertex_out_size[attrib_name] = stride * thread_count;
    VertexMeta->vertex_out[attrib_name] = new float[VertexMeta->vertex_out_count.at(attrib_name)];
  }

  for (auto out_attrib : VertexMeta->vertex_id_map) {
    std::string attrib_name = out_attrib.second;
    std::string vb = mesa_root + "../vb/" + "vb" + attrib_name + "_" +
                     std::to_string(draw) + ".bin";

    if (!SKIP_VS) {
      context->get_device()->get_gpgpu()->memcpy_from_gpu(
          VertexMeta->vertex_out.at(attrib_name), VertexMeta->vertex_out_devptr.at(attrib_name),
          VertexMeta->vertex_out_size.at(attrib_name));
      FILE *fp;
      fp = fopen(vb.c_str(), "wb+");
      fwrite(VertexMeta->vertex_out.at(attrib_name), 1, VertexMeta->vertex_out_size.at(attrib_name), fp);
      fclose(fp);
    } else {
      VulkanRayTracing::read_binary_file(vb.c_str(), VertexMeta->vertex_out.at(attrib_name),
                                         VertexMeta->vertex_out_size.at(attrib_name));
      context->get_device()->get_gpgpu()->memcpy_to_gpu(
          VertexMeta->vertex_out_devptr.at(attrib_name), VertexMeta->vertex_out.at(attrib_name),
          VertexMeta->vertex_out_size.at(attrib_name));
    }
  }

  if (vulkan_app == "instancing") {
    for (unsigned i = 0; i < VertexMeta->vertex_out_count["\%inColor"]; i++) {
        VertexMeta->vertex_out.at("\%inColor")[i] = 1.0f;
    }
  }


  // vertex-post processing
  // tranform & clipping
  std::vector<std::vector<float>> vertex_ndc;
  std::vector<std::vector<float>> vertex_screen;
  std::vector<std::vector<float>> vertex_raw;
  std::string pos_id = VertexMeta->vertex_id_map.at("VARYING_SLOT_POS_xyzw");
  for (unsigned i = 0; i < VertexMeta->vertex_out_count.at(pos_id); i += 4) {
    // transform to NDC space
    std::vector<float> ndc;
    std::vector<float> view;
    std::vector<float> raw;
    float ndc_x = (VertexMeta->vertex_out.at(pos_id)[i] /
                   VertexMeta->vertex_out.at(pos_id)[i + 3]);
    ndc.push_back(ndc_x);
    float ndc_y = (VertexMeta->vertex_out.at(pos_id)[i + 1] /
                   VertexMeta->vertex_out.at(pos_id)[i + 3]);
    ndc.push_back(ndc_y);
    float ndc_z = (VertexMeta->vertex_out.at(pos_id)[i + 2] /
                   VertexMeta->vertex_out.at(pos_id)[i + 3]);
    ndc.push_back(ndc_z);
    float ndc_w = (VertexMeta->vertex_out.at(pos_id)[i + 3] /
                   VertexMeta->vertex_out.at(pos_id)[i + 3]);
    ndc.push_back(ndc_w);
    vertex_ndc.push_back(ndc);

    // X = (X + 1) * Viewport.Width * 0.5 + Viewport.TopLeftX
    // Y = (1 - Y) * Viewport.Height * 0.5 + Viewport.TopLeftY (actually no)
    // Z = Viewport.MinDepth + Z * (Viewport.MaxDepth - Viewport.MinDepth)
    float screen_x = (ndc_x + 1) * (FBO->width / 2) + FBO->x;
    float screen_y = (ndc_y + 1) * (FBO->height / 2) + FBO->y;
    float screen_z = 0.0f + ndc_z * (1.0f - 0.0f);
    view.push_back(screen_x);
    view.push_back(screen_y);
    view.push_back(screen_z);
    view.push_back(ndc_w);
    vertex_screen.push_back(view);

    assert(!isnan(VertexMeta->vertex_out.at(pos_id)[i]));
    raw.push_back(VertexMeta->vertex_out.at(pos_id)[i]);
    raw.push_back(VertexMeta->vertex_out.at(pos_id)[i+1]);
    raw.push_back(VertexMeta->vertex_out.at(pos_id)[i+2]);
    raw.push_back(VertexMeta->vertex_out.at(pos_id)[i+3]);
    vertex_raw.push_back(raw);
  }
  assert(vertex_screen.size() == thread_count);
  assert(vertex_ndc.size() == thread_count);


  // Assemble into triangles using index buffer
  std::vector<std::vector<unsigned>> primitives;
// #if WORKLOAD == 0 || WORKLOAD == 1
  // uint16_t *index_buffer = anv_address_map(VertexMeta->index_buffer->address);
  for (unsigned instance = 0; instance < VertexMeta->InstanceCount; instance++) {
    for (unsigned i = 0; i < VertexMeta->index_buffer->size / index_size;
         i += 3) {
      unsigned selected_vpc = i / group_size;
      unsigned clipped = 0;
      std::vector<unsigned> prim;
      // 3 because primitives are triangles
      for (unsigned j = 0; j < 3; j++) {
        unsigned vertex;
        if (index_size == 4) {
          vertex = ((u_int32_t *)index_buffer)[i + j];
        } else if (index_size == 2) {
          vertex = ((u_int16_t *)index_buffer)[i + j];
        }
        unsigned batched_index = VertexMeta->vertex_map[selected_vpc][vertex] + instance * VertexMeta->vb.size();
        prim.push_back(batched_index);
        // (-w <= x,y,z <= w) equal to (fabs(x,y,z) > fbas(w))
        if (fabs(vertex_raw[batched_index][0]) >
            fabs(vertex_raw[batched_index][3])) {
          clipped++;
          continue;
        }
        if (fabs(vertex_raw[batched_index][1]) >
            fabs(vertex_raw[batched_index][3])) {
          clipped++;
          continue;
        }
        if (fabs(vertex_raw[batched_index][2]) >
            fabs(vertex_raw[batched_index][3])) {
          clipped++;
          continue;
        }
      }
      if (clipped < 3) {
        primitives.push_back(prim);
      }
    }
  }

  printf("total primitives after clipping: %u\n", primitives.size());

  unsigned tile_size = 8;
  std::vector<unsigned> drawed_pixels;
  std::unordered_map<std::string, std::vector<std::vector<float>>> attribs;
  std::vector<std::vector<unsigned>> tile_map;
  tile_map.resize(FBO->width * FBO->height / tile_size / tile_size);

  for (auto attrib : VertexMeta->vertex_id_map) {
    std::string attrib_name = attrib.second;
    attribs.insert(std::make_pair(attrib_name, std::vector<std::vector<float>>()));
  }

  for (std::vector<std::vector<unsigned>>::iterator prim = primitives.begin();
       prim < primitives.end(); prim++) {
    double x1 = vertex_screen[(*prim)[0]][0];
    double y1 = vertex_screen[(*prim)[0]][1];
    double x2 = vertex_screen[(*prim)[1]][0];
    double y2 = vertex_screen[(*prim)[1]][1];
    double x3 = vertex_screen[(*prim)[2]][0];
    double y3 = vertex_screen[(*prim)[2]][1];

    double max_x = std::min(FBO->width - 1.0, std::max(x1, std::max(x2, x3)));
    double min_x = std::max(0., std::min(x1, std::min(x2, x3)));
    double max_y = std::min(FBO->height - 1.0, std::max(y1, std::max(y2, y3)));
    double min_y = std::max(0., std::min(y1, std::min(y2, y3)));

    double r = rand() % 255 / 255.f;
    double g = rand() % 255 / 255.f;
    double b = rand() % 255 / 255.f;

    // assert(FBO->width % tile_size == 0);
    // assert(FBO->height % tile_size == 0);
    unsigned tile_id = (min_y * FBO->width + min_x) / tile_size / tile_size;
    for (int x = (int)min_x; x <= (int)max_x; x++) {
      for (int y = (int)min_y; y <= (int)max_y; y++) {
        unsigned pixel = y * FBO->width + x;
        double det = x1 * (y2 - y3) - y1 * (x2 - x3) + (x2 * y3 - y2 * x3);
        // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
        double d00 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        double d01 = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1);
        double d11 = (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1);
        double d20 = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1);
        double d21 = (x - x1) * (x3 - x1) + (y - y1) * (y3 - y1);
        double denom = d00 * d11 - d01 * d01;
        double v = (d11 * d20 - d01 * d21) / denom;
        double w = (d00 * d21 - d01 * d20) / denom;
        double u = 1.0f - v - w;
        if (det == 0.0) {
          continue;
        }
        if (u < 0 || v < 0 || w < 0) {
          continue;
        }
        float depth = u * vertex_screen[(*prim)[0]][2] +
                      v * vertex_screen[(*prim)[1]][2] +
                      w * vertex_screen[(*prim)[2]][2];
        // printf("depth is %f\n",depth);
        switch(VertexMeta->DepthcmpOp) {
          case VK_COMPARE_OP_GREATER:
            if (FBO->depthout[pixel] > depth) {
              continue;
            }
            break;
          case VK_COMPARE_OP_LESS:
            if (FBO->depthout[pixel] < depth) {
              continue;
            }
            break;
          case VK_COMPARE_OP_LESS_OR_EQUAL:
            if (FBO->depthout[pixel] <= depth) {
              continue;
            }
            break;
          // case VK_COMPARE_OP_NEVER:
            // break;
          default:
            printf("unsupported depth compare op\n");
            assert(0 && "unsupported depth compare op");
        }
        if (SKIP_FS) {
          FBO->fbo[(pixel) * 4] = r;
          FBO->fbo[(pixel) * 4 + 1] = g;
          FBO->fbo[(pixel) * 4 + 2] = b;
          FBO->fbo[(pixel) * 4 + 3] = 1.0f;
        }

        for (auto attrib : VertexMeta->vertex_id_map) {
          std::vector<float> vec;
          std::string attrib_name = attrib.second;
          switch (VertexMeta->vertex_out_stride.at(attrib_name)) {
            case 4: {
                float v0 = VertexMeta->vertex_out.at(
                                attrib_name)[1 * (*prim)[0]] *
                                u +
                            VertexMeta->vertex_out.at(
                                attrib_name)[1 * (*prim)[1]] *
                                v +
                            VertexMeta->vertex_out.at(
                                attrib_name)[1 * (*prim)[2]] *
                                w;
                vec.push_back(v0);
                break;
            }
            case 8: {
              // vec2
              float v0 = VertexMeta->vertex_out.at(attrib_name)[2 * (*prim)[0]] * u +
                        VertexMeta->vertex_out.at(attrib_name)[2 * (*prim)[1]] * v +
                        VertexMeta->vertex_out.at(attrib_name)[2 * (*prim)[2]] * w;
              float v1 = VertexMeta->vertex_out.at(attrib_name)[2 * (*prim)[0] + 1] * u +
                        VertexMeta->vertex_out.at(attrib_name)[2 * (*prim)[1] + 1] * v +
                        VertexMeta->vertex_out.at(attrib_name)[2 * (*prim)[2] + 1] * w;
              assert(!isnan(v0) && !isnan(v1));
              vec.push_back(v0);
              vec.push_back(v1);
              break;
            }
            case 12: {
              // vec3
              float v0 = VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[0]] * u +
                        VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[1]] * v +
                        VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[2]] * w;
              float v1 = VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[0] + 1] * u +
                        VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[1] + 1] * v +
                        VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[2] + 1] * w;
              float v2 = VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[0] + 2] * u +
                        VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[1] + 2] * v +
                        VertexMeta->vertex_out.at(attrib_name)[3 * (*prim)[2] + 2] * w;
              // TODO: change to the index of normal
              // if (attrib == 0) {
              //   float norm = sqrt(v0 * v0 + v1 * v1 + v2 * v2);
              //   v0 /= norm;
              //   v1 /= norm;
              //   v2 /= norm;
              // }
              vec.push_back(v0);
              vec.push_back(v1);
              vec.push_back(v2);
              break;
            }
            case 16: {
              // vec4
              float v0 = VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[0]] * u +
                        VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[1]] * v +
                        VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[2]] * w;
              float v1 = VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[0] + 1] * u +
                        VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[1] + 1] * v +
                        VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[2] + 1] * w;
              float v2 = VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[0] + 2] * u +
                        VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[1] + 2] * v +
                        VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[2] + 2] * w;
              float v3 = VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[0] + 3] * u +
                        VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[1] + 3] * v +
                        VertexMeta->vertex_out.at(attrib_name)[4 * (*prim)[2] + 3] * w;
              vec.push_back(v0);
              vec.push_back(v1);
              vec.push_back(v2);
              vec.push_back(v3);
              break;
            }
            default: {
              printf("unsupported vertex out attribute size\n");
              // just add it
              assert(0);
            }
          }
          attribs.at(attrib_name).push_back(vec);
          assert(vec.size() == attribs.at(attrib_name)[0].size());
        }
        tile_map[tile_id].push_back(attribs.at(pos_id).size() - 1);
        FBO->thread_info_pixel.push_back(pixel);
        FBO->depthout[pixel] = depth;
      }
    }
  }

  printf("total frags collected - %u\n",FBO->thread_info_pixel.size());
  if (FBO->thread_info_pixel.size() == 0) {
    draw++;
    // delete (VertexMeta);

    // VertexMeta = NULL;
    // draw_meta.pop_front();
      context->get_device()->get_gpgpu()->valid_addr_start.clear();
      context->get_device()->get_gpgpu()->valid_addr_end.clear();

    continue;
  }

  if (!SKIP_FS) {
  // copy vertex data to gpu
  VertexMeta->vertex_out_count.clear();
  VertexMeta->vertex_out_size.clear();
  for (auto attrib : VertexMeta->vertex_id_map) {
    std::string attrib_name = attrib.second;

    unsigned stride = VertexMeta->vertex_out_stride.at(attrib_name);
    VertexMeta->vertex_out_count[attrib_name] =
        attribs.at(attrib_name).size() * attribs.at(attrib_name)[0].size();
    assert(VertexMeta->vertex_out_stride.at(attrib_name) ==
           attribs.at(attrib_name)[0].size() * sizeof(float));
    VertexMeta->vertex_out_size[attrib_name] = attribs.at(attrib_name).size() *
                                               attribs.at(attrib_name)[0].size() *
                                               sizeof(float);

    delete VertexMeta->vertex_out.at(attrib_name);
    VertexMeta->vertex_out[attrib_name] = new float[VertexMeta->vertex_out_count.at(attrib_name)];
  }

  std::vector<unsigned> pixel_index = FBO->thread_info_pixel;
  std::unordered_map<unsigned, unsigned> pixel_map;
  // use map to speedup neighbour pixel lookup
  FBO->thread_info_pixel.clear();
  unsigned index = 0;
  for (unsigned tile = 0; tile < tile_map.size(); tile++) {
    for (unsigned frag = 0; frag < tile_map[tile].size(); frag++) {
      unsigned i = tile_map[tile][frag];
      for (auto attrib : VertexMeta->vertex_id_map) {
        std::string attrib_name = attrib.second;
        for (unsigned j = 0; j < attribs.at(attrib_name)[i].size(); j++) {
            VertexMeta->vertex_out.at(attrib_name)
                                  [index * attribs.at(attrib_name)[i].size() + j] =
                attribs.at(attrib_name)[i][j];
        }
      }
      // store where the pixel is in the vector
      pixel_map[pixel_index[i]] = FBO->thread_info_pixel.size();
      FBO->thread_info_pixel.push_back(pixel_index[i]);
      index++;
    }
  }
  assert(pixel_index.size() == FBO->thread_info_pixel.size());

  std::string tex_id = "UNDEFINED";
  if (VertexMeta->vertex_id_map.find("VARYING_SLOT_VAR3_xy") !=
      VertexMeta->vertex_id_map.end()) {
    tex_id = VertexMeta->vertex_id_map.at("VARYING_SLOT_VAR3_xy");
  }
  else if (vulkan_app == "render_passes") {
    tex_id = VertexMeta->vertex_id_map.at("VARYING_SLOT_VAR1_xy");
  } 
  else if (vulkan_app == "instancing") {
    if (draw == 0)
      tex_id = VertexMeta->vertex_id_map.at("VARYING_SLOT_VAR2_xy");
    else
      tex_id = VertexMeta->vertex_id_map.at("VARYING_SLOT_VAR2_xyz");
  }
  else if (vulkan_app == "pbrtexture") {
    tex_id = VertexMeta->vertex_id_map.at("VARYING_SLOT_VAR2_xy");
  } else if (vulkan_app == "demo") {
    printf("warning: undefined texture id\n");
  }
  else {
    // assert(0 && "unknown texture id");
  }
  for (unsigned i = 0; i < FBO->thread_info_pixel.size(); i ++) {
    float lod = 0;
    if (tex_id != "UNDEFINED") {
      // calcualte LOD of each pixel
      unsigned pixel = FBO->thread_info_pixel[i];
      unsigned x = pixel % FBO->width;
      unsigned y = pixel / FBO->width;
      // determine which pixel to get within the 2x2 quad
      int x_offset = 0;
      int y_offset = 0;
      x % 2 == 0 ? x_offset = 1 : x_offset = -1;
      y % 2 == 0 ? y_offset = 1 : y_offset = -1;
      // make sure the offset is within the image
      if (x + x_offset > FBO->width - 1 || x + x_offset < 0) {
        x_offset = 0;
      }
      if (y + y_offset > FBO->height - 1 || y + y_offset < 0) {
        y_offset = 0;
      }

      unsigned next_x = pixel + x_offset;
      unsigned next_y = pixel + y_offset * FBO->width;
      // real gpu always use 2x2 quad, but we don't do that here.
      // next pixel may not been drawn
      // need to make sure the next pixel is rendered
      float ddx, ddy;
      float u = VertexMeta->vertex_out.at(tex_id)[i * 2];
      float v = VertexMeta->vertex_out.at(tex_id)[i * 2 + 1];
      auto xx = pixel_map.find(next_x);
      if (xx != pixel_map.end()) {
        unsigned thread_id = pixel_map[next_x];
        float dudx = (VertexMeta->vertex_out.at(tex_id)[thread_id * 2]) - u;
        float dvdx = (VertexMeta->vertex_out.at(tex_id)[thread_id * 2 + 1]) - v;
        ddx = std::min(sqrt(dudx * dudx + dvdx * dvdx), 1.0f);
      } else {
        ddx = 1.0f / texture_width;
      }
      auto yy = pixel_map.find(next_y);
      if (yy != pixel_map.end()) {
        unsigned thread_id = pixel_map[next_y];
        float dudy = (VertexMeta->vertex_out.at(tex_id)[thread_id * 2]) - u;
        float dvdy = (VertexMeta->vertex_out.at(tex_id)[thread_id * 2 + 1]) - v;
        ddy = std::min(sqrt(dudy * dudy + dvdy * dvdy), 1.0f);
      } else {
        ddy = 1.0f / texture_height;
      }
      lod = log2(std::max(ddx * texture_width, ddy * texture_height));
      // assert(!signbit(lod));
      lod = std::max(lod, (float)0);
      assert(!isnan(lod));
    }
    FBO->thread_info_lod.push_back(lod);
    
  }

  for (auto attrib : VertexMeta->vertex_id_map) {
    std::string attrib_name = attrib.second;
    uint32_t *dev_ptr = context->get_device()->get_gpgpu()->gpu_malloc(
            VertexMeta->vertex_out_size.at(attrib_name));
    VertexMeta->vertex_out_devptr.at(attrib_name) = dev_ptr;
    context->get_device()->get_gpgpu()->valid_addr_start[attrib_name] =
        (uint64_t)dev_ptr;
    context->get_device()->get_gpgpu()->valid_addr_end[attrib_name] =
        ((uint64_t)dev_ptr) + VertexMeta->vertex_out_size.at(attrib_name);
    context->get_device()->get_gpgpu()->memcpy_to_gpu(
        VertexMeta->vertex_out_devptr.at(attrib_name),
        VertexMeta->vertex_out.at(attrib_name),
        VertexMeta->vertex_out_size.at(attrib_name));
    print_memcpy("MemcpyVulkan", VertexMeta->vertex_out_devptr.at(attrib_name),
                 VertexMeta->vertex_out_size.at(attrib_name),
                 VertexMeta->vertex_out_stride.at(attrib_name) * block_size);
  }

  for (unsigned i = 0; i < MAX_DESCRIPTOR_SETS; i++) {
    for (unsigned j = 0; j < MAX_DESCRIPTOR_SET_BINDINGS; j++) {
      dump_texture(i, j, VertexMeta->decoded_descriptors[i][j]);
    }
  }

  // set push constants
  print_memcpy("MemcpyVulkan",VertexMeta->constants_dev_addr, 1024, 0);
  

  // pixel shaders
  VulkanRayTracing::is_FS = true;
  VulkanRayTracing::thread_count = FBO->thread_info_pixel.size();
  run_shader(frag_id, thread_count);

  // copy back framebuffer and dump
  context->get_device()->get_gpgpu()->memcpy_from_gpu(FBO->fbo, FBO->fbo_dev,
                                                      FBO->fbo_size);
  }

  FBO->thread_info_pixel.clear();
  printf("Drawcall #%u Done\n", draw);

  draw++;
  
  VertexMeta = NULL;
  context->get_device()->get_gpgpu()->valid_addr_start.clear();
  context->get_device()->get_gpgpu()->valid_addr_end.clear();
  }
  
}

void VulkanRayTracing::read_binary_file(std::string path, void* ptr, unsigned size) {
    // read in before
  std::ifstream dataStream(path, std::fstream::in | std::fstream::binary);
  if (!dataStream.is_open()) {
    abort();
  }
  dataStream.read((char*) ptr, size);
  dataStream.close();
}

void VulkanRayTracing::saveIndexBuffer(struct anv_buffer *ptr, VkIndexType type) {
//   printf("Saving index buffer %p\n", ptr);
  VulkanRayTracing::index_buffer = ptr;
  VulkanRayTracing::index_type = type;
}

uint64_t VulkanRayTracing::getVertexAddr(uint32_t buffer_index,
                                         uint32_t tid) {
  // instancing (multiple vertex attributes in one vertex buffer)
  // assert(buffer_index < VertexMeta->VertexAttrib->binding.size());
  unsigned loc = -1;
  for (unsigned i = 0; i < VertexMeta->VertexAttrib->location.size(); i++) {
    if (buffer_index == VertexMeta->VertexAttrib->location[i]) {
      loc = i;
      break;
    }
  }
  assert(loc != -1);
  assert(buffer_index == VertexMeta->VertexAttrib->location[loc]);
  unsigned binding = VertexMeta->VertexAttrib->binding[loc];
  unsigned attrib_stride = VertexMeta->VertexAttrib->offset[loc];

  unsigned instance = tid / VertexMeta->vb.size();
  unsigned index = tid % VertexMeta->vb.size();
  if (tid >= VulkanRayTracing::thread_count) {
    // out of range
    return 0;
  }
  unsigned offset = 0;
  if (VertexMeta->VertexAttrib->rate[binding] == VK_VERTEX_INPUT_RATE_VERTEX) {
    offset = VertexMeta->vb[index] * VertexMeta->vertex_stride[binding] / 4;
    assert(offset < VertexMeta->vertex_count[binding] * VertexMeta->InstanceCount);
  } else if (VertexMeta->VertexAttrib->rate[binding] == VK_VERTEX_INPUT_RATE_INSTANCE) {
    offset = instance * VertexMeta->vertex_stride[binding] / 4;
    assert(offset < VertexMeta->vertex_count[binding]);
  }
  return VertexMeta->vertex_addr[binding] + offset + attrib_stride / 4;
  //   if (app_id == INSTANCING && draw == 1) {
  //     float *base = 0x0;
  //     if (buffer_index < 3) {
  //       unsigned real_id = tid % VertexMeta->vb.size();
  //       base = (float *)VertexMeta->vertex_addr[0] +
  //              VertexMeta->vb[real_id] * VertexMeta->vertex_stride[0] / 4;
  //     } else {
  //       unsigned instance = tid / VertexMeta->vb.size();
  //       base = (float *)VertexMeta->vertex_addr[1] +
  //              instance * VertexMeta->vertex_stride[1] / 4;
  //     }
  //     if (tid >= thread_count) {
  //       return 0;
  //     }
  //     return base + (attrib_stride[buffer_index] / 4);
  //   } else if (app_id == INSTANCING && draw == 0 || app_id == PBRBASIC ||
  //              app_id == PBRTEXTURE) {
  //     unsigned offset = VertexMeta->vb[tid] * VertexMeta->vertex_stride[0] /
  //     4; if (tid >= VulkanRayTracing::thread_count) {
  //       return 0;
  //     }
  //     assert(offset < VertexMeta->vertex_count[0]);
  //     return VertexMeta->vertex_addr[0] + offset +
  //            (attrib_stride[buffer_index] / 4);
  //   }

  //   // render_passes (single vertex attribute)
  //   else if (app_id == RENDER_PASSES) {
  //     unsigned offset =
  //         VertexMeta->vb[tid] * VertexMeta->vertex_stride[buffer_index] / 4;
  //     if (tid >= VulkanRayTracing::thread_count) {
  //       // out of range
  //       return 0;
  //     }
  //     assert(offset < VertexMeta->vertex_count[buffer_index]);
  //     return VertexMeta->vertex_addr[buffer_index] + offset;
  //   } else {
  //     assert(0);
  //   }
}

// uint64_t VulkanRayTracing::getVertexOutAddr(uint32_t buffer_index,
//                                             uint32_t tid) {
//   unsigned offset = (tid) * VertexMeta->vertex_out_stride[buffer_index] / 4;
//   if (tid >= VulkanRayTracing::thread_count) {
//     // out of range
//     return 0;
//   }
//   assert(offset < VertexMeta->vertex_out_count[buffer_index]);
//   return VertexMeta->vertex_out_devptr[buffer_index] + offset;
// }

uint64_t VulkanRayTracing::getVertexOutAddr(std::string identifier,
                                            uint32_t tid) {
  unsigned offset = (tid) * VertexMeta->vertex_out_stride.at(identifier) / 4;
  if (tid >= VulkanRayTracing::thread_count) {
    // out of range
    return 0;
  }
//   assert(offset < VertexMeta->vertex_out_count[buffer_index]);
  return VertexMeta->vertex_out_devptr.at(identifier) + offset;
}

uint64_t VulkanRayTracing::getFBOAddr(uint32_t offset) {
  // get pixel coord
  if((offset) >= VulkanRayTracing::thread_count) {
    return 0;
  }
  assert((FBO->thread_info_pixel[offset] * 4) < FBO->fbo_count);
  return FBO->fbo_dev + FBO->thread_info_pixel[offset] * 4;
}

void VulkanRayTracing::getFragCoord(uint32_t thread_id, uint32_t &x,
                                    uint32_t &y) {
  // get pixel coord
  if ((thread_id) >= VulkanRayTracing::thread_count) {
    x = 0;
    y = 0;
    return 0;
  }

  x = FBO->thread_info_pixel[thread_id] % FBO->width;
  y = FBO->thread_info_pixel[thread_id] / FBO->width;
}

uint64_t VulkanRayTracing::getConst() {
    return VertexMeta->constants_dev_addr;
}

float VulkanRayTracing::getTexLOD(unsigned thread_id) {
  if (thread_id >= VulkanRayTracing::thread_count) {
    return 0;
  }
    return FBO->thread_info_lod[thread_id];
}

void VulkanRayTracing::vkCmdTraceRaysKHR(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr) {
    // launch_width = 32;
    // launch_height = 32;
    init(launch_width, launch_height);
    
    // Dump Descriptor Sets
    if (!use_external_launcher) 
    {
      assert(0);
        // dump_descriptor_sets(VulkanRayTracing::descriptorSet, true);
        dump_callparams_and_sbt(raygen_sbt, miss_sbt, hit_sbt, callable_sbt, is_indirect, launch_width, launch_height, launch_depth, launch_size_addr);
    }

    // CmdTraceRaysKHRID++;
    // if(CmdTraceRaysKHRID != 1)
    //     return;
    // launch_width = 420;
    // launch_height = 320;

    if(writeImageBinary && !imageFile.is_open())
    {
        char* imageFileName;
        char defaultFileName[40] = "image.binary";
        if(getenv("VULKAN_IMAGE_FILE_NAME"))
            imageFileName = getenv("VULKAN_IMAGE_FILE_NAME");
        else
            imageFileName = defaultFileName;
        imageFile.open(imageFileName, std::ios::out | std::ios::binary);
        
        // imageFile.open("image.txt", std::ios::out);
    }
    else
        return;
    // memset(((uint8_t*)descriptors[0][1].address), uint8_t(127), launch_height * launch_width * 4);
    // return;

    // {
    //     std::ifstream infile("debug_printf.log");
    //     std::string line;
    //     while (std::getline(infile, line))
    //     {
    //         if(line == "")
    //             continue;

    //         RayDebugGPUData data;
    //         // sscanf(line.c_str(), "LaunchID:(%d,%d), InstanceCustomIndex = %d, primitiveID = %d, v0 = (%f, %f, %f), v1 = (%f, %f, %f), v2 = (%f, %f, %f), hitAttribute = (%f, %f), normalWorld = (%f, %f, %f), objectIntersection = (%f, %f, %f), worldIntersection = (%f, %f, %f), objectNormal = (%f, %f, %f), worldNormal = (%f, %f, %f), NdotL = %f",
    //         //             &data.launchIDx, &data.launchIDy, &data.instanceCustomIndex, &data.primitiveID, &data.v0pos.x, &data.v0pos.y, &data.v0pos.z, &data.v1pos.x, &data.v1pos.y, &data.v1pos.z, &data.v2pos.x, &data.v2pos.y, &data.v2pos.z, &data.attribs.x, &data.attribs.y, &data.N.x, &data.N.y, &data.N.z, &data.P_object.x, &data.P_object.y, &data.P_object.z, &data.P.x, &data.P.y, &data.P.z, &data.N_object.x, &data.N_object.y, &data.N_object.z, &data.N.x, &data.N.y, &data.N.z, &data.NdotL);
    //         sscanf(line.c_str(), "launchID = (%d, %d), hitValue = (%f, %f, %f)",
    //                     &data.launchIDx, &data.launchIDy, &data.hitValue.x, &data.hitValue.y, &data.hitValue.z);
    //         data.valid = true;
    //         assert(data.launchIDx < 2000 && data.launchIDy < 2000);
    //         // printf("#### (%d, %d)\n", data.launchIDx, data.launchIDy);
    //         // fflush(stdout);
    //         rayDebugGPUData[data.launchIDx][data.launchIDy] = data;

    //     }
    // }

    assert(launch_depth == 1);

    struct anv_descriptor desc;
    desc.image_view = NULL;

    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    unsigned long shaderId = *(uint64_t*)raygen_sbt;
    int index = 0;
    for (int i = 0; i < shaders.size(); i++) {
        if (shaders[i].ID == 0){
            index = i;
            break;
        }
    }

    shader_stage_info raygen_shader = shaders[index];
    function_info *entry = context->get_kernel(raygen_shader.function_name);
    // printf("################ number of args = %d\n", entry->num_args());

    if (entry->is_pdom_set()) {
        printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
            entry->get_name().c_str());
    } else {
        printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
            entry->get_name().c_str());
        /*
        * Some of the instructions like printf() gives the gpgpusim the wrong
        * impression that it is a function call. As printf() doesnt have a body
        * like functions do, doing pdom analysis for printf() causes a crash.
        */
        if (entry->get_function_size() > 0) entry->do_pdom();
        entry->set_pdom();
    }

    // check that number of args and return match function requirements
    //if (pI->has_return() ^ entry->has_return()) {
    //    printf(
    //        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
    //        "between\n"
    //        "               call instruction and function declaration\n");
    //    abort();
    //}
    unsigned n_return = entry->has_return();
    unsigned n_args = entry->num_args();
    //unsigned n_operands = pI->get_num_operands();

    // launch_width = 1;
    // launch_height = 1;

    dim3 blockDim = dim3(1, 1, 1);
    dim3 gridDim = dim3(1, launch_height, launch_depth);
    if(launch_width <= 32) {
        blockDim.x = launch_width;
        gridDim.x = 1;
    }
    else {
        blockDim.x = 32;
        gridDim.x = launch_width / 32;
        if(launch_width % 32 != 0)
            gridDim.x++;
    }

    gpgpu_ptx_sim_arg_list_t args;
    // kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
    //   raygen_shader.function_name, args, dim3(4, 128, 1), dim3(32, 1, 1), context);
    kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      raygen_shader.function_name, args, gridDim, blockDim, context);
    grid->vulkan_metadata.raygen_sbt = raygen_sbt;
    grid->vulkan_metadata.miss_sbt = miss_sbt;
    grid->vulkan_metadata.hit_sbt = hit_sbt;
    grid->vulkan_metadata.callable_sbt = callable_sbt;
    grid->vulkan_metadata.launch_width = launch_width;
    grid->vulkan_metadata.launch_height = launch_height;
    grid->vulkan_metadata.launch_depth = launch_depth;
    
    struct CUstream_st *stream = 0;
    stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
    ctx->the_gpgpusim->g_stream_manager->push(op);

    //printf("%d\n", descriptors[0][1].address);

    fflush(stdout);

    while(!op.is_done() && !op.get_kernel()->done()) {
        printf("waiting for op to finish\n");
        sleep(1);
        continue;
    }
    // for (unsigned i = 0; i < entry->num_args(); i++) {
    //     std::pair<size_t, unsigned> p = entry->get_param_config(i);
    //     cudaSetupArgumentInternal(args[i], p.first, p.second);
    // }
}

void VulkanRayTracing::callMissShader(const ptx_instruction *pI, ptx_thread_info *thread) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

    int32_t current_shader_counter = -1;
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &current_shader_counter, thread, pI);

    uint32_t missIndex;
    mem->read(&(traversal_data->missIndex), sizeof(traversal_data->missIndex), &missIndex);

    uint32_t shaderID = *((uint32_t *)(thread->get_kernel().vulkan_metadata.miss_sbt) + 8 * missIndex);
    
    shader_stage_info miss_shader = shaders[shaderID];

    function_info *entry = context->get_kernel(miss_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callClosestHitShader(const ptx_instruction *pI, ptx_thread_info *thread) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

    bool hit_geometry;
    mem->read(&(traversal_data->hit_geometry), sizeof(bool), &hit_geometry);
    assert(hit_geometry);

    int32_t current_shader_counter = -1;
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &current_shader_counter, thread, pI);

    VkGeometryTypeKHR geometryType;
    mem->read(&(traversal_data->closest_hit.geometryType), sizeof(traversal_data->closest_hit.geometryType), &geometryType);

    shader_stage_info closesthit_shader;
    if(geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR)
        closesthit_shader = shaders[*((uint64_t *)(thread->get_kernel().vulkan_metadata.hit_sbt))];
    else {
        int32_t hitGroupIndex;
        mem->read(&(traversal_data->closest_hit.hitGroupIndex), sizeof(traversal_data->closest_hit.hitGroupIndex), &hitGroupIndex);
        closesthit_shader = shaders[*((uint64_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex)];
    }

    function_info *entry = context->get_kernel(closesthit_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callIntersectionShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    
    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter, thread, pI);

    warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
    uint32_t hitGroupIndex = table->get_hitGroupIndex(shader_counter, thread->get_tid().x, pI, thread);

    shader_stage_info intersection_shader = shaders[*((uint64_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex + 1)];
    function_info *entry = context->get_kernel(intersection_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callAnyHitShader(const ptx_instruction *pI, ptx_thread_info *thread) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    assert(0);
}

void VulkanRayTracing::callShader(const ptx_instruction *pI, ptx_thread_info *thread, function_info *target_func) {
    static unsigned call_uid_next = 1;

  if (target_func->is_pdom_set()) {
    // printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
    //        target_func->get_name().c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           target_func->get_name().c_str());
    /*
     * Some of the instructions like printf() gives the gpgpusim the wrong
     * impression that it is a function call. As printf() doesnt have a body
     * like functions do, doing pdom analysis for printf() causes a crash.
     */
    if (target_func->get_function_size() > 0) target_func->do_pdom();
    target_func->set_pdom();
  }

  thread->set_npc(target_func->get_start_PC());

  // check that number of args and return match function requirements
  if (pI->has_return() ^ target_func->has_return()) {
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
        "between\n"
        "               call instruction and function declaration\n");
    abort();
  }
  unsigned n_return = target_func->has_return();
  unsigned n_args = target_func->num_args();
  unsigned n_operands = pI->get_num_operands();

  // TODO: why this fails?
//   if (n_operands != (n_return + 1 + n_args)) {
//     printf(
//         "GPGPU-Sim PTX: Execution error - mismatch in number of arguements "
//         "between\n"
//         "               call instruction and function declaration\n");
//     abort();
//   }

  // handle intrinsic functions
//   std::string fname = target_func->get_name();
//   if (fname == "vprintf") {
//     gpgpusim_cuda_vprintf(pI, thread, target_func);
//     return;
//   }
// #if (CUDART_VERSION >= 5000)
//   // Jin: handle device runtime apis for CDP
//   else if (fname == "cudaGetParameterBufferV2") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_getParameterBufferV2(
//         pI, thread, target_func);
//     return;
//   } else if (fname == "cudaLaunchDeviceV2") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_launchDeviceV2(
//         pI, thread, target_func);
//     return;
//   } else if (fname == "cudaStreamCreateWithFlags") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_streamCreateWithFlags(
//         pI, thread, target_func);
//     return;
//   }
// #endif

  // read source arguements into register specified in declaration of function
  arg_buffer_list_t arg_values;
  copy_args_into_buffer_list(pI, thread, target_func, arg_values);

  // record local for return value (we only support a single return value)
  const symbol *return_var_src = NULL;
  const symbol *return_var_dst = NULL;
  if (target_func->has_return()) {
    return_var_dst = pI->dst().get_symbol();
    return_var_src = target_func->get_return_var();
  }

  gpgpu_sim *gpu = thread->get_gpu();
  unsigned callee_pc = 0, callee_rpc = 0;
  /*if (gpu->simd_model() == POST_DOMINATOR)*/ { //MRS_TODO: why this fails?
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc);
    assert(callee_pc == thread->get_pc());
  }

  thread->callstack_push(callee_pc + pI->inst_size(), callee_rpc,
                         return_var_src, return_var_dst, call_uid_next++);

  copy_buffer_list_into_frame(thread, arg_values);

  thread->set_npc(target_func);
}

void VulkanRayTracing::setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type)
{
    if(descriptors.size() <= setID)
        descriptors.resize(setID + 1);
    if(descriptors[setID].size() <= descID)
        descriptors[setID].resize(descID + 1);
    
    descriptors[setID][descID].setID = setID;
    descriptors[setID][descID].descID = descID;
    descriptors[setID][descID].address = address;
    descriptors[setID][descID].size = size;
    descriptors[setID][descID].type = type;
}


void VulkanRayTracing::setDescriptorSetFromLauncher(void *address, void *deviceAddress, uint32_t setID, uint32_t descID)
{
    launcher_deviceDescriptorSets[setID][descID] = deviceAddress;
    launcher_descriptorSets[setID][descID] = address;
}

void* VulkanRayTracing::getDescriptorAddress(uint32_t setID, uint32_t binding)
{
    if (true)
    // if (use_external_launcher)
    {
        return launcher_deviceDescriptorSets[setID][binding];
        // return launcher_descriptorSets[setID][binding];
    }
    // else 
    // {
    //     // assert(setID < descriptors.size());
    //     // assert(binding < descriptors[setID].size());

    //     struct anv_descriptor_set* set = VulkanRayTracing::descriptorSet;

    //     const struct anv_descriptor_set_binding_layout *bind_layout = &set->layout->binding[binding];
    //     struct anv_descriptor *desc = &set->descriptors[bind_layout->descriptor_index];
    //     void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

    //     assert(desc->type == bind_layout->type);

    //     switch (desc->type)
    //     {
    //         case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    //         {
    //             return (void *)(desc);
    //         }
    //         case VK_DESCRIPTOR_TYPE_SAMPLER:
    //         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    //         case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    //         case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
    //         {
    //             return desc;
    //         }

    //         case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
    //         case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
    //             assert(0);
    //             break;

    //         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    //         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    //         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
    //         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
    //         {
    //             if (desc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
    //                 desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
    //             {
    //                 // MRS_TODO: account for desc->offset?
    //                 return anv_address_map(desc->buffer->address);
    //             }
    //             else
    //             {
    //                 struct anv_buffer_view *bview = &set->buffer_views[bind_layout->buffer_view_index];
    //                 return anv_address_map(bview->address);
    //             }
    //         }

    //         case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
    //             assert(0);
    //             break;

    //         case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
    //         case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
    //         {
    //             struct anv_address_range_descriptor *desc_data = desc_map;
    //             return (void *)(desc_data->address);
    //         }

    //         default:
    //             assert(0);
    //             break;
    //     }

    //     // return descriptors[setID][binding].address;
    // }
}

void VulkanRayTracing::getTexture(struct anv_descriptor *desc, 
                                    float x, float y, float level, float lod, 
                                    float &c0, float &c1, float &c2, float &c3, 
                                    std::vector<ImageMemoryTransactionRecord>& transactions,
                                    uint64_t launcher_offset)
{
    Pixel pixel;

    if (true)
    // if (use_external_launcher)
    {
      pixel = get_interpolated_pixel(
          (anv_image_view *)desc, (anv_sampler *)desc, x, y, level, lod, 
          transactions, launcher_offset);  // cast back to metadata later
    }
    else 
    {
        struct anv_image_view *image_view =  desc->image_view;
        struct anv_sampler *sampler = desc->sampler;

        const struct anv_image *image = image_view->image;
        assert(image->n_planes == 1);
        assert(image->samples == 1);
        assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
        assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
        assert(sampler->conversion == NULL);

        pixel = get_interpolated_pixel(image_view, sampler, x, y,level, lod, transactions);
    }

    TXL_DPRINTF("Setting transaction type to TEXTURE_LOAD\n");
    for(int i = 0; i < transactions.size(); i++)
        transactions[i].type = ImageTransactionType::TEXTURE_LOAD;
    
    c0 = pixel.c0;
    c1 = pixel.c1;
    c2 = pixel.c2;
    c3 = pixel.c3;


    // uint8_t* address = anv_address_map(image->planes[0].address);

    // for(int x = 0; x < image->extent.width; x++)
    // {
    //     for(int y = 0; y < image->extent.height; y++)
    //     {
    //         int blockX = x / 8;
    //         int blockY = y / 8;

    //         uint32_t offset = (blockX + blockY * (image->extent.width / 8)) * (128 / 8);

    //         uint8_t dst_colors[100];
    //         basisu::astc::decompress(dst_colors, address + offset, true, 8, 8);
    //         uint8_t* pixel_color = &dst_colors[0] + (x % 8 + (y % 8) * 8) * 4;

    //         uint32_t bit_map_offset = x + y * image->extent.width;

    //         float data[4];
    //         data[0] = pixel_color[0] / 255.0;
    //         data[1] = pixel_color[1] / 255.0;
    //         data[2] = pixel_color[2] / 255.0;
    //         data[3] = pixel_color[3] / 255.0;
    //         imageFile.write((char*) data, 3 * sizeof(float));
    //         imageFile.write((char*) (&bit_map_offset), sizeof(uint32_t));
    //         imageFile.flush();
    //     }
    // }
}

void VulkanRayTracing::image_load(struct anv_descriptor *desc, uint32_t x, uint32_t y, float &c0, float &c1, float &c2, float &c3)
{
    ImageMemoryTransactionRecord transaction;

    struct anv_image_view *image_view =  desc->image_view;
    struct anv_sampler *sampler = desc->sampler;

    const struct anv_image *image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);
    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
    assert(sampler->conversion == NULL);

    Pixel pixel = load_image_pixel(image, x, y, 0, 1, transaction);

    transaction.type = ImageTransactionType::IMAGE_LOAD;
    
    c0 = pixel.c0;
    c1 = pixel.c1;
    c2 = pixel.c2;
    c3 = pixel.c3;
}

void VulkanRayTracing::image_store(struct anv_descriptor* desc, uint32_t gl_LaunchIDEXT_X, uint32_t gl_LaunchIDEXT_Y, uint32_t gl_LaunchIDEXT_Z, uint32_t gl_LaunchIDEXT_W, 
              float hitValue_X, float hitValue_Y, float hitValue_Z, float hitValue_W, const ptx_instruction *pI, ptx_thread_info *thread)
{
    ImageMemoryTransactionRecord transaction;
    Pixel pixel = Pixel(hitValue_X, hitValue_Y, hitValue_Z, hitValue_W);

    VkFormat vk_format;
    if (use_external_launcher)
    {
        storage_image_metadata *metadata = (storage_image_metadata*) desc;
        vk_format = metadata->format;
        store_image_pixel((anv_image*) desc, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, 0, pixel, transaction);
    }
    else
    {
        assert(desc->sampler == NULL);

        struct anv_image_view *image_view = desc->image_view;
        assert(image_view != NULL);
        struct anv_image * image = image_view->image;

        vk_format = image->vk_format;

        store_image_pixel(image, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, 0, pixel, transaction);
    }

    
    transaction.type = ImageTransactionType::IMAGE_STORE;

    if(writeImageBinary && vk_format != VK_FORMAT_R32G32B32A32_SFLOAT)
    {
        uint32_t image_width = thread->get_kernel().vulkan_metadata.launch_width;
        uint32_t offset = 0;
        offset += gl_LaunchIDEXT_Y * image_width;
        offset += gl_LaunchIDEXT_X;

        float data[4];
        data[0] = hitValue_X;
        data[1] = hitValue_Y;
        data[2] = hitValue_Z;
        data[3] = hitValue_W;
        imageFile.write((char*) data, 3 * sizeof(float));
        imageFile.write((char*) (&offset), sizeof(uint32_t));
        imageFile.flush();

        // imageFile << "(" << gl_LaunchIDEXT_X << ", " << gl_LaunchIDEXT_Y << ") : (";
        // imageFile << hitValue_X << ", " << hitValue_Y << ", " << hitValue_Z << ", " << hitValue_W << ")\n";
    }

    TXL_DPRINTF("Setting transaction for image_store\n");
    thread->set_txl_transactions(transaction);

    // // if(std::abs(hitValue_X - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.x) > 0.0001 || 
    // //     std::abs(hitValue_Y - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.y) > 0.0001 ||
    // //     std::abs(hitValue_Z - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.z) > 0.0001)
    // //     {
    // //         printf("wrong value. (%d, %d): (%f, %f, %f)\n"
    // //                 , gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, hitValue_X, hitValue_Y, hitValue_Z);
    // //     }
    
    // // if (gl_LaunchIDEXT_X == 1070 && gl_LaunchIDEXT_Y == 220)
    // //     printf("this one has wrong value\n");

    // // if(hitValue_X > 1 || hitValue_Y > 1 || hitValue_Z > 1)
    // // {
    // //     printf("this one has wrong value.\n");
    // // }
}

// variable_decleration_entry* VulkanRayTracing::get_variable_decleration_entry(std::string name, ptx_thread_info *thread)
// {
//     std::vector<variable_decleration_entry>& table = thread->RT_thread_data->variable_decleration_table;
//     for (int i = 0; i < table.size(); i++) {
//         if (table[i].name == name) {
//             assert (table[i].address != NULL);
//             return &(table[i]);
//         }
//     }
//     return NULL;
// }

// void VulkanRayTracing::add_variable_decleration_entry(uint64_t type, std::string name, uint64_t address, uint32_t size, ptx_thread_info *thread)
// {
//     variable_decleration_entry entry;

//     entry.type = type;
//     entry.name = name;
//     entry.address = address;
//     entry.size = size;
//     thread->RT_thread_data->variable_decleration_table.push_back(entry);
// }

void VulkanRayTracing::dumpVertex(struct anv_buffer *vbuffer, struct anv_graphics_pipeline * pipeline, uint32_t setID) {


    float* address = anv_address_map(vbuffer->address);
    unsigned size = vbuffer->size;
    // unsigned stride = pipeline->vb[setID].stride;
    // unsigned attrib_per_vertex = stride /4;
    assert(size % 4 == 0);
    // Data to dump
    // FILE *fp,*mp;
    // char *mesa_root = getenv("MESA_ROOT");
    // char *filePath = "gpgpusimShaders/";
    // char *extension = ".vkvertexbuffer";

    // Vertex data
    // if you want to use dumped vertex data, you need to either dump batched
    // vertex (after indexing) or perform vertex batching on loaded vertex data.
    // Right here it's saving raw vertex buffer from Mesa3D
    // i.e. currently it can only be launched directly from Mesa3D.
    // char fullPath[200];
    // char metaPath[200];
    // snprintf(fullPath, sizeof(fullPath), "%s%s_%d_%d_%d.vkvertexdata", mesa_root, filePath, setID, size, VertexMeta->vertex_stride[setID]);
    // snprintf(metaPath, sizeof(fullPath), "%s%s_%d_%d_%d.vkvertexmeta", mesa_root, filePath, setID, size, VertexMeta->vertex_stride[setID]);
    // fp = fopen(fullPath, "wb+");
    // mp = fopen(metaPath, "w+");
    // fwrite(address, 1, size, fp);
    // fclose(fp);
    // fprintf(mp,"%u",pipeline->vb[setID].stride);
    // fclose(mp);

    u_int32_t *devPtr;
    gpgpu_context *ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
    context->get_device()
        ->get_gpgpu()
        ->valid_addr_start["vb" + std::to_string(setID)] = (uint64_t)devPtr;

    context->get_device()
        ->get_gpgpu()
        ->valid_addr_end["vb" + std::to_string(setID)] = ((uint64_t)devPtr) + size;
    context->get_device()->get_gpgpu()->memcpy_to_gpu(devPtr, address, size);
    VertexMeta->vertex_addr[setID] = devPtr;
    // VertexMeta->vertex_size[setID] = size;
    // VertexMeta->vertex_count[setID] = size / 4;
    // TODO: multiple by CTA size here. Should printer byte / CTA
    print_memcpy("MemcpyVulkan",devPtr, VertexMeta->vertex_size[setID], VertexMeta->vertex_stride[setID]);
    // context->get_device()->get_gpgpu()->gtrace
    //     << "MemcpyHtoD," << std::hex << devPtr << "," << std::dec
    //     << VertexMeta->vertex_size[setID] << std::endl;

    // VertexMeta->vertex_count[setID] = VertexMeta->vb.size() * attrib_per_vertex;
    // VertexMeta->vertex_size[setID] = VertexMeta->vertex_count[setID] * sizeof(float);

    // float *buffer = new float[VertexMeta->vertex_count[setID]];
    // for (int i = 0; i < VertexMeta->vb.size(); i++) {
    //   unsigned index = VertexMeta->vb[i];
    //   for (int j = 0; j < attrib_per_vertex; j++) {
    //     buffer[i * attrib_per_vertex + j] =
    //         address[index * attrib_per_vertex + j];
    //     assert(index < (size / stride));
    //     assert(i < VertexMeta->vertex_count[setID]);
    //   }
    // }

    // u_int32_t *devPtr;
    // gpgpu_context *ctx = GPGPU_Context();
    // CUctx_st *context = GPGPUSim_Context(ctx);
    // devPtr = context->get_device()->get_gpgpu()->gpu_malloc(
    //     VertexMeta->vertex_size[setID]);
    // context->get_device()->get_gpgpu()->gtrace
    //     << "MemcpyHtoD, 0x" << std::hex << devPtr << "," << std::dec
    //     << VertexMeta->vertex_size[setID] << std::endl;
    // context->get_device()->get_gpgpu()->memcpy_to_gpu(
    //     devPtr, buffer, VertexMeta->vertex_size[setID]);
    // VertexMeta->vertex_addr[setID] = devPtr;
    // delete[] buffer;

}

void VulkanRayTracing::dumpTextures(struct anv_descriptor *desc, uint32_t setID, uint32_t descID, uint32_t binding, VkDescriptorType type)
{
    anv_descriptor *desc_offset = ((anv_descriptor*)((void*)desc)); // offset for raytracing_extended
    struct anv_image_view *image_view =  desc_offset->image_view;
    struct anv_sampler *sampler = desc_offset->sampler;
    if (!image_view) {
      return;
    }

    const struct anv_image *image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);
    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
    // assert(sampler->conversion == NULL);

    uint8_t* address = anv_address_map(image->planes[0].address);
    uint32_t image_extent_width = image->extent.width;
    uint32_t image_extent_height = image->extent.height;
    VkFormat format = image->vk_format;
    uint64_t size = image->size;

    texture_width = image_extent_width;
    texture_height = image_extent_height;

    VkFilter filter;
    // if(sampler->conversion == NULL)
    filter = VK_FILTER_NEAREST;

    // Data to dump
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsettexturedata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            VkDescriptorTypeNum = 0;
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            VkDescriptorTypeNum = 1;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            VkDescriptorTypeNum = 2;
            break;
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            VkDescriptorTypeNum = 10;
            break;
        default:
            abort(); // should not be here!
    }

    // Texture data
    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vktexturedata", mesa_root, filePath, setID, binding);
    // File name format: setID_descID.vktexturedata

    fp = fopen(fullPath, "wb+");
    fwrite(address, 1, size, fp);
    fclose(fp);

    // Texture metadata
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vktexturemetadata", mesa_root, filePath, setID, binding);
    fp = fopen(fullPath, "w+");
    // File name format: setID_descID.vktexturemetadata

    fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", size, 
                                                 image_extent_width, 
                                                 image_extent_height, 
                                                 format, 
                                                 VkDescriptorTypeNum, 
                                                 image->n_planes, 
                                                 image->samples, 
                                                 image->tiling, 
                                                 image->planes[0].surface.isl.tiling,
                                                 image->planes[0].surface.isl.row_pitch_B,
                                                 filter);
    fclose(fp);
    

    u_int32_t *devPtr;
    gpgpu_context *ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
    context->get_device()->get_gpgpu()->memcpy_to_gpu(devPtr, address, size);
    context->get_device()
        ->get_gpgpu()
        ->valid_addr_start["tex" + std::to_string(setID) +
                           std::to_string(descID)] = (uint64_t)devPtr;
    context->get_device()
        ->get_gpgpu()
        ->valid_addr_end["tex" + std::to_string(setID) +
                           std::to_string(descID)] = ((uint64_t)devPtr) + size;

    setTextureFromLauncher(address, devPtr, setID, descID, size,
                           image_extent_width, image_extent_height, format,
                           VkDescriptorTypeNum, image->n_planes, image->samples,
                           image->tiling, image->planes[0].surface.isl.tiling,
                           image->planes[0].surface.isl.row_pitch_B, filter, image->levels);
    print_memcpy("dumpTextures", devPtr, size, 0);
}


void VulkanRayTracing::dumpStorageImage(struct anv_descriptor *desc, uint32_t setID, uint32_t binding, VkDescriptorType type)
{
    assert(type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    assert(desc->sampler == NULL);

    struct anv_image_view *image_view = desc->image_view;
    assert(image_view != NULL);
    struct anv_image * image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);

    void* mem_address = anv_address_map(image->planes[0].address);

    VkFormat format = image->vk_format;
    VkImageTiling tiling = image->tiling;
    isl_tiling isl_tiling_mode = image->planes[0].surface.isl.tiling;
    uint32_t row_pitch_B  = image->planes[0].surface.isl.row_pitch_B;

    uint32_t width = image->extent.width;
    uint32_t height = image->extent.height;

    // Dump storage image metadata
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum = 3;

    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vkstorageimagemetadata", mesa_root, filePath, setID, binding);
    fp = fopen(fullPath, "w+");
    // File name format: setID_descID.vktexturemetadata

    fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d",   width, 
                                                height, 
                                                format, 
                                                VkDescriptorTypeNum, 
                                                image->n_planes, 
                                                image->samples, 
                                                tiling, 
                                                isl_tiling_mode,
                                                row_pitch_B);
    fclose(fp);
}


void VulkanRayTracing::dump_descriptor_set_for_AS(uint32_t setID, uint32_t descID, void *address, uint32_t desc_size, VkDescriptorType type, uint32_t backwards_range, uint32_t forward_range, bool split_files, VkAccelerationStructureKHR _topLevelAS)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            VkDescriptorTypeNum = 1000150000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            VkDescriptorTypeNum = 1000165000;
            break;
        default:
            abort(); // should not be here!
    }

    char fullPath[200];
    int result;

    int64_t max_backwards; // negative number
    int64_t min_backwards; // negative number
    int64_t min_forwards;
    int64_t max_forwards;
    int64_t back_buffer_amount = 0; //20kB buffer just in case
    int64_t front_buffer_amount = 1024*20; //20kB buffer just in case
    findOffsetBounds(max_backwards, min_backwards, min_forwards, max_forwards, _topLevelAS);

    bool haveBackwards = (max_backwards != 0) && (min_backwards != 0);
    bool haveForwards = (min_forwards != 0) && (max_forwards != 0);
    
    if (split_files) // Used when the AS is too far apart between top tree and BVHAddress and cant just dump the whole thing
    {
        // Main Top Level
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asmain", mesa_root, filePath, setID, descID);
        fp = fopen(fullPath, "wb+");
        result = fwrite(address, 1, desc_size, fp);
        assert(result == desc_size);
        fclose(fp);

        // Bot level whose address is smaller than top level
        if (haveBackwards)
        {
            snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asback", mesa_root, filePath, setID, descID);
            fp = fopen(fullPath, "wb+");
            result = fwrite(address + max_backwards, 1, min_backwards - max_backwards + back_buffer_amount, fp);
            assert(result == min_backwards - max_backwards + back_buffer_amount);
            fclose(fp);
        }

        // Bot level whose address is larger than top level
        if (haveForwards)
        {
            snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asfront", mesa_root, filePath, setID, descID);
            fp = fopen(fullPath, "wb+");
            result = fwrite(address + min_forwards, 1, max_forwards - min_forwards + front_buffer_amount, fp);
            assert(result == max_forwards - min_forwards + front_buffer_amount);
            fclose(fp);
        }

        // AS metadata
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asmetadata", mesa_root, filePath, setID, descID);
        fp = fopen(fullPath, "w+");
        fprintf(fp, "%d,%d,%ld,%ld,%ld,%ld,%ld,%ld,%d,%d", desc_size,
                                                            VkDescriptorTypeNum,
                                                            max_backwards,
                                                            min_backwards,
                                                            min_forwards,
                                                            max_forwards,
                                                            back_buffer_amount,
                                                            front_buffer_amount,
                                                            haveBackwards,
                                                            haveForwards);
        fclose(fp);

        
        // uint64_t total_size = (desc_size + backwards_range + forward_range);
        // uint64_t chunk_size = 1024*1024*20; // 20MB chunks
        // int totalFiles =  (total_size + chunk_size) / chunk_size; // rounds up

        // for (int i = 0; i < totalFiles; i++)
        // {
        //     // if split_files is 1, then look at the next number to see what the file part number is
        //     snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d_%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, desc_size, VkDescriptorTypeNum, backwards_range, forward_range, split_files, i, extension);
        //     fp = fopen(fullPath, "wb+");
        //     int result = fwrite(address-(uint64_t)backwards_range + chunk_size * i, 1, chunk_size, fp);
        //     printf("File part %d, %d bytes written, starting address 0x%.12" PRIXPTR "\n", i, result, (uintptr_t)(address-(uint64_t)backwards_range + chunk_size * i));
        //     fclose(fp);
        // }
    }
    else 
    {
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, desc_size, VkDescriptorTypeNum, backwards_range, forward_range, extension);
        // File name format: setID_descID_SizeInBytes_VkDescriptorType_desired_range.vkdescrptorsetdata

        fp = fopen(fullPath, "wb+");
        int result = fwrite(address-(uint64_t)backwards_range, 1, desc_size + backwards_range + forward_range, fp);
        fclose(fp);
    }
}


void VulkanRayTracing::dump_descriptor_set(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            VkDescriptorTypeNum = 0;
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            VkDescriptorTypeNum = 1;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            VkDescriptorTypeNum = 2;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            VkDescriptorTypeNum = 3;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            VkDescriptorTypeNum = 4;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            VkDescriptorTypeNum = 5;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            VkDescriptorTypeNum = 6;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            VkDescriptorTypeNum = 7;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            VkDescriptorTypeNum = 8;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            VkDescriptorTypeNum = 9;
            break;
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            VkDescriptorTypeNum = 10;
            break;
        case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
            VkDescriptorTypeNum = 1000138000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            VkDescriptorTypeNum = 1000150000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            VkDescriptorTypeNum = 1000165000;
            break;
        case VK_DESCRIPTOR_TYPE_MUTABLE_VALVE:
            VkDescriptorTypeNum = 1000351000;
            break;
        case VK_DESCRIPTOR_TYPE_MAX_ENUM:
            VkDescriptorTypeNum = 0x7FFFFFF;
            break;
        default:
            abort(); // should not be here!
    }

    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, size, VkDescriptorTypeNum, extension);
    // File name format: setID_descID_SizeInBytes_VkDescriptorType.vkdescrptorsetdata

    fp = fopen(fullPath, "wb+");
    fwrite(address, 1, size, fp);
    fclose(fp);
    // VertexMeta->decoded_descriptors[setID][descID].addr = address;
    // VertexMeta->decoded_descriptors[setID][descID].size = size;

    

    u_int32_t *devPtr;
    gpgpu_context *ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size * sizeof(float));
    context->get_device()->get_gpgpu()->memcpy_to_gpu(devPtr, address, size * sizeof(float));
    setDescriptorSetFromLauncher(address,devPtr,setID,descID);
    context->get_device()->get_gpgpu()->valid_addr_start["desc" + std::to_string(setID) + std::to_string(descID)] = (uint64_t)devPtr;
    context->get_device()->get_gpgpu()->valid_addr_end["desc" + std::to_string(setID) + std::to_string(descID)] =  ((uint64_t)devPtr) +  size * sizeof(float);
    print_memcpy("MemcpyVulkan",devPtr, size, 0);
}

void VulkanRayTracing::dump_texture(struct anv_descriptor_set *set, unsigned set_index) {
  for (int i = 0; i < set->descriptor_count; i++) {
      if (VulkanRayTracing::app_id == RENDER_PASSES ||
          VulkanRayTracing::app_id == INSTANCING) {
          if (i == 2) {
              i = 4;
          }
      }
      // struct anv_descriptor_set *set = VulkanRayTracing::descriptorSet[set_index];

      const struct anv_descriptor_set_binding_layout *bind_layout =
          &set->layout->binding[i];
      struct anv_descriptor *desc =
          &set->descriptors[bind_layout->descriptor_index];
      void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

      if (bind_layout->type == (unsigned) -1) {
        continue;
      }
      assert(desc->type == bind_layout->type);

      switch (desc->type) {
          case VK_DESCRIPTOR_TYPE_SAMPLER:
          case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
          case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
          case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
              dumpTextures(desc, set_index, i, i, desc->type);
              break;
          }
      }
  }
}

void VulkanRayTracing::dump_texture(unsigned set_index, unsigned desc_index,  struct anv_descriptor *desc) {
  if(!desc) {
    return;
  }
  switch (desc->type) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
      dumpTextures(desc, set_index, desc_index, desc_index, desc->type);
      break;
    }
  }
}

void VulkanRayTracing::dump_descriptor(unsigned set_index, unsigned desc_index,  struct anv_descriptor *desc, struct anv_buffer_view *bview, bool dump_texture) {
  if (!desc) {
    return;
  }
  switch (desc->type) {
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
        // return (void *)(desc);
        dumpStorageImage(desc, 0, desc_index, desc->type);
        break;
      }
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
        anv_descriptor *desc_offset = ((
            anv_descriptor *)((void *)desc));  // offset for raytracing_extended
        if (!desc_offset->image_view || !desc_offset->image_view->image) {
          break;
        }
        if (dump_texture) {
          dumpTextures(desc, set_index, desc_index, desc_index, desc->type);
        } else {
        // anv_descriptor *desc_offset = ((
        //     anv_descriptor *)((void *)desc));  // offset for raytracing_extended
        struct anv_image_view *image_view = desc_offset->image_view;
        const struct anv_image *image = image_view->image;
        texture_width = image->extent.width;
        texture_height = image->extent.height;
        }
        break;
      }

            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            {
                if (desc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                    desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                {
                    // MRS_TODO: account for desc->offset?
                    //return anv_address_map(desc->buffer->address);
                    dump_descriptor_set(set_index, desc_index, anv_address_map(desc->buffer->address), desc->buffer->size, desc->type);
                    break;
                }
                else
                {
                    //return anv_address_map(bview->address);
                    dump_descriptor_set(set_index, desc_index, anv_address_map(bview->address), bview->range, desc->type);
                    break;
                }
            }

            case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
              assert(0);
                // struct anv_address_range_descriptor *desc_data = desc_map;
                //return (void *)(desc_data->address);
                //dump_descriptor_set_for_AS(0, i, (void *)(desc_data->address), desc_data->range, set->descriptors[i].type, 1024*1024*10, 1024*1024*10, true);
                break;
            }

            default:
                // assert(0);
                break;
        }

}

bool VulkanRayTracing::check_descriptor_sets(struct anv_descriptor_set *set, unsigned set_index) {
  for (int i = 0; i < set->layout->binding_count; i++) {
    const struct anv_descriptor_set_binding_layout *bind_layout =
        &set->layout->binding[i];
    struct anv_descriptor *desc =
        &set->descriptors[bind_layout->descriptor_index];
    void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

    if (bind_layout->type == (unsigned) -1) {
        continue;
    }

    switch (desc->type) {
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
        return true;
      }
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
          anv_descriptor *desc_offset = ((anv_descriptor *)((
              void *)desc));  // offset for raytracing_extended
          // if (desc_offset->image_view == NULL) {
          //   return false;
          // }
            return true;
      }

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        assert(0);
        break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
        if (desc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
            desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
          return true;
        } else {
          struct anv_buffer_view *bview =
              &set->buffer_views[bind_layout->buffer_view_index];
          if (!bview) {
            return false;
          }
          return true;
        }
      }

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
        assert(0);
        break;

      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV: {
        break;
      }

      default:
        return false;
        break;
    }
  }
}

void VulkanRayTracing::dump_descriptor_sets(struct anv_descriptor_set *set, bool dump_texture, unsigned set_index)
{
  for (int i = 0; i < set->layout->binding_count; i++) {
    if (VulkanRayTracing::app_id == RENDER_PASSES ||
        VulkanRayTracing::app_id == INSTANCING) {
      if (i == 2) {
        i = 4;
        // for some reason raytracing_extended skipped binding = 3
        // and somehow they have 34 descriptor sets but only 10 are used
        // so we just skip those
        // continue;
      }
    }

    // struct anv_descriptor_set *set = VulkanRayTracing::descriptorSet[set_index];

    const struct anv_descriptor_set_binding_layout *bind_layout =
        &set->layout->binding[i];
    struct anv_descriptor *desc =
        &set->descriptors[bind_layout->descriptor_index];
    void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;
    if (set_index == 1 && i == 0) {
      printf("descriptor %p \n", desc);
    }

    // if (bind_layout->type > (unsigned)VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
    //     continue;
    // }
    if (bind_layout->type == (unsigned) -1) {
        continue;
    }
    // assert(desc->type == bind_layout->type);

    switch (desc->type) {
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
        // return (void *)(desc);
        dumpStorageImage(desc, 0, i, desc->type);
        break;
      }
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
        // return desc;
        if (dump_texture) {
          dumpTextures(desc, set_index, i, i, desc->type);
        } else {
        anv_descriptor *desc_offset = ((
            anv_descriptor *)((void *)desc));  // offset for raytracing_extended
        if (desc_offset->image_view == NULL) {
            continue;
        }
        struct anv_image_view *image_view = desc_offset->image_view;
        const struct anv_image *image = image_view->image;
        texture_width = image->extent.width;
        texture_height = image->extent.height;
        }
        break;
      }

            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            {
                if (desc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                    desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                {
                    // MRS_TODO: account for desc->offset?
                    //return anv_address_map(desc->buffer->address);
                    dump_descriptor_set(set_index, i, anv_address_map(desc->buffer->address), set->descriptors[bind_layout->descriptor_index].buffer->size, set->descriptors[bind_layout->descriptor_index].type);
                    break;
                }
                else
                {
                    struct anv_buffer_view *bview = &set->buffer_views[bind_layout->buffer_view_index];
                    //return anv_address_map(bview->address);
                    dump_descriptor_set(set_index, i, anv_address_map(bview->address), bview->range, set->descriptors[bind_layout->descriptor_index].type);
                    break;
                }
            }

            case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
                struct anv_address_range_descriptor *desc_data = desc_map;
                //return (void *)(desc_data->address);
                //dump_descriptor_set_for_AS(0, i, (void *)(desc_data->address), desc_data->range, set->descriptors[i].type, 1024*1024*10, 1024*1024*10, true);
                break;
            }

            default:
                assert(0);
                break;
        }
  }
}

void VulkanRayTracing::dump_AS(struct anv_descriptor_set *set, VkAccelerationStructureKHR _topLevelAS)
{
   for(int i = 0; i < set->descriptor_count; i++)
   {
    //    if(i == 3 || i > 9)
    //    {
    //         // for some reason raytracing_extended skipped binding = 3
    //         // and somehow they have 34 descriptor sets but only 10 are used
    //         // so we just skip those
    //         continue;
    //    }
    assert(0);
        // struct anv_descriptor_set* set = VulkanRayTracing::descriptorSet;

        const struct anv_descriptor_set_binding_layout *bind_layout = &set->layout->binding[i];
        struct anv_descriptor *desc = &set->descriptors[bind_layout->descriptor_index];
        void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

        assert(desc->type == bind_layout->type);

        switch (desc->type)
        {
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
                struct anv_address_range_descriptor *desc_data = desc_map;
                //return (void *)(desc_data->address);
                dump_descriptor_set_for_AS(0, i, (void *)(desc_data->address), desc_data->range, set->descriptors[i].type, 1024*1024*10, 1024*1024*10, true, _topLevelAS);
                break;
            }

            default:
                break;
        }
   }
}

void VulkanRayTracing::dump_callparams_and_sbt(void *raygen_sbt, void *miss_sbt, void *hit_sbt, void *callable_sbt, bool is_indirect, uint32_t launch_width, uint32_t launch_height, uint32_t launch_depth, uint32_t launch_size_addr)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";

    char call_params_filename [200];
    int trace_rays_call_count = 0; // just a placeholder for now
    snprintf(call_params_filename, sizeof(call_params_filename), "%s%s%d.callparams", mesa_root, filePath, trace_rays_call_count);
    fp = fopen(call_params_filename, "w+");
    fprintf(fp, "%d,%d,%d,%d,%lu", is_indirect, launch_width, launch_height, launch_depth, launch_size_addr);
    fclose(fp);

    // TODO: Is the size always 32?
    int sbt_size = 64 *sizeof(uint64_t);
    if (raygen_sbt) {
        char raygen_sbt_filename [200];
        snprintf(raygen_sbt_filename, sizeof(raygen_sbt_filename), "%s%s%d.raygensbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(raygen_sbt_filename, "wb+");
        fwrite(raygen_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (miss_sbt) {
        char miss_sbt_filename [200];
        snprintf(miss_sbt_filename, sizeof(miss_sbt_filename), "%s%s%d.misssbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(miss_sbt_filename, "wb+");
        fwrite(miss_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (hit_sbt) {
        char hit_sbt_filename [200];
        snprintf(hit_sbt_filename, sizeof(hit_sbt_filename), "%s%s%d.hitsbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(hit_sbt_filename, "wb+");
        fwrite(hit_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (callable_sbt) {
        char callable_sbt_filename [200];
        snprintf(callable_sbt_filename, sizeof(callable_sbt_filename), "%s%s%d.callablesbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(callable_sbt_filename, "wb+");
        fwrite(callable_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }
}

void VulkanRayTracing::setStorageImageFromLauncher(void *address, 
                                                void *deviceAddress, 
                                                uint32_t setID, 
                                                uint32_t descID, 
                                                uint32_t width,
                                                uint32_t height,
                                                VkFormat format,
                                                uint32_t VkDescriptorTypeNum,
                                                uint32_t n_planes,
                                                uint32_t n_samples,
                                                VkImageTiling tiling,
                                                uint32_t isl_tiling_mode, 
                                                uint32_t row_pitch_B)
{
    storage_image_metadata *storage_image = new storage_image_metadata;
    storage_image->address = address;
    storage_image->setID = setID;
    storage_image->descID = descID;
    storage_image->width = width;
    storage_image->height = height;
    storage_image->format = format;
    storage_image->VkDescriptorTypeNum = VkDescriptorTypeNum;
    storage_image->n_planes = n_planes;
    storage_image->n_samples = n_samples;
    storage_image->tiling = tiling;
    storage_image->isl_tiling_mode = isl_tiling_mode; 
    storage_image->row_pitch_B = row_pitch_B;
    storage_image->deviceAddress = deviceAddress;

    launcher_descriptorSets[setID][descID] = (void*) storage_image;
    launcher_deviceDescriptorSets[setID][descID] = (void*) storage_image;
}

void VulkanRayTracing::setTextureFromLauncher(void *address, 
                                            void *deviceAddress, 
                                            uint32_t setID, 
                                            uint32_t descID, 
                                            uint64_t size,
                                            uint32_t width,
                                            uint32_t height,
                                            VkFormat format,
                                            uint32_t VkDescriptorTypeNum,
                                            uint32_t n_planes,
                                            uint32_t n_samples,
                                            VkImageTiling tiling,
                                            uint32_t isl_tiling_mode,
                                            uint32_t row_pitch_B,
                                            uint32_t filter, 
                                            uint32_t mip_level)
{
    texture_metadata *texture = new texture_metadata;
    texture->address = address;
    texture->setID = setID;
    texture->descID = descID;
    texture->size = size;
    texture->width = width;
    texture->height = height;
    texture->format = format;
    texture->VkDescriptorTypeNum = VkDescriptorTypeNum;
    texture->n_planes = n_planes;
    texture->n_samples = n_samples;
    texture->tiling = tiling;
    texture->isl_tiling_mode = isl_tiling_mode;
    texture->row_pitch_B = row_pitch_B;
    texture->filter = filter;
    texture->deviceAddress = deviceAddress;
    texture->mip_level = mip_level;

    // VertexMeta->decoded_descriptors[setID][descID].is_texture = true;
    // VertexMeta->decoded_descriptors[setID][descID].addr = (void*) texture;
    // VertexMeta->decoded_descriptors[setID][descID].size = size;
    launcher_descriptorSets[setID][descID] = (void*) texture;
    launcher_deviceDescriptorSets[setID][descID] = (void*) texture;
}

void VulkanRayTracing::pass_child_addr(void *address)
{
    child_addrs_from_driver.push_back(address);
}

void VulkanRayTracing::findOffsetBounds(int64_t &max_backwards, int64_t &min_backwards, int64_t &min_forwards, int64_t &max_forwards, VkAccelerationStructureKHR _topLevelAS)
{
    // uint64_t current_min_backwards = 0;
    // uint64_t current_max_backwards = 0;
    // uint64_t current_min_forwards = 0;
    // uint64_t current_max_forwards = 0;
    int64_t offset;

    std::vector<int64_t> positive_offsets;
    std::vector<int64_t> negative_offsets;

    for (auto addr : child_addrs_from_driver)
    {
        offset = (uint64_t)addr - (uint64_t)_topLevelAS;
        if (offset >= 0)
            positive_offsets.push_back(offset);
        else
            negative_offsets.push_back(offset);
    }

    sort(positive_offsets.begin(), positive_offsets.end());
    sort(negative_offsets.begin(), negative_offsets.end());

    if (negative_offsets.size() > 0)
    {
        max_backwards = negative_offsets.front();
        min_backwards = negative_offsets.back();
    }
    else
    {
        max_backwards = 0;
        min_backwards = 0;
    }

    if (positive_offsets.size() > 0)
    {
        min_forwards = positive_offsets.front();
        max_forwards = positive_offsets.back();
    }
    else
    {
        min_forwards = 0;
        max_forwards = 0;
    }
}


void* VulkanRayTracing::gpgpusim_alloc(uint32_t size)
{
    gpgpu_context *ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    void* devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
    // for (unsigned i = 0; i < size / 4; i ++) {
    //     context->get_device()->get_gpgpu()->ignore_addr[(unsigned long) devPtr + i * 4] = true;
    // }
    if (g_debug_execution >= 3) {
        printf("GPGPU-Sim PTX: gpgpusim_allocing %zu bytes starting at 0x%llx..\n",
            size, (unsigned long long)devPtr);
        ctx->api->g_mallocPtr_Size[(unsigned long long)devPtr] = size;
    }
    assert(devPtr);
    return devPtr;
}

void VulkanRayTracing::map_pipeline_shader(void *ptr, unsigned shader_index) {
    pipeline_shader_map.insert(std::make_pair(ptr, shader_index));
}

void VulkanRayTracing::map_pipeline_info(void *ptr, VkGraphicsPipelineCreateInfo *pCreateinfo) {
    struct VertexAttrib *vertexAttrib = new struct VertexAttrib;
    unsigned save = false;
    for (unsigned i = 0;
         i < pCreateinfo->pVertexInputState->vertexAttributeDescriptionCount;
         i++) {
        vertexAttrib->location.push_back(
            pCreateinfo->pVertexInputState->pVertexAttributeDescriptions[i]
                .location);
        vertexAttrib->binding.push_back(
            pCreateinfo->pVertexInputState->pVertexAttributeDescriptions[i]
                .binding);
        vertexAttrib->offset.push_back(
            pCreateinfo->pVertexInputState->pVertexAttributeDescriptions[i]
                .offset);
        save = true;
    }
    for (unsigned i = 0; i < pCreateinfo->pVertexInputState->vertexBindingDescriptionCount; i++) {
        assert(pCreateinfo->pVertexInputState->pVertexBindingDescriptions[i]
                   .binding == i);
        vertexAttrib->rate.push_back(
            pCreateinfo->pVertexInputState->pVertexBindingDescriptions[i]
                .inputRate);
    }
    // if (save) {
    pipeline_vertex_map.insert(std::make_pair(ptr, vertexAttrib));
    // }
}