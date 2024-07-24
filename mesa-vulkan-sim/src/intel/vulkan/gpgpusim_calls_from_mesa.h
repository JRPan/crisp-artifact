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

#ifndef GPGPUSIM_CALLS_FROM_MESA_H
#define GPGPUSIM_CALLS_FROM_MESA_H

#include "vulkan/vulkan_core.h"
#include "anv_private.h"

//extern void gpgpusim_init();
extern void gpgpusim_setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos);
extern void gpgpusim_setGeometries(const VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount);
extern void gpgpusim_addTreelets(VkAccelerationStructureKHR accelerationStructure);
extern void gpgpusim_testTraversal(struct anv_bvh_node* root);
extern uint32_t gpgpusim_registerShader(char * shaderPath, uint32_t shader_type);

extern void gpgpusim_vkCmdTraceRaysKHR(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);

extern void gpgpusim_vkCmdDraw(struct anv_cmd_buffer *cmd_buffer, unsigned VertexCountPerInstance, unsigned StartVertex, unsigned instanceCount, unsigned StartInstance, unsigned BaseVertex);
extern void gpgpusim_clearDraws();
extern void gpgpusim_saveDumbDraw();
extern void gpgpusim_saveDraw(struct anv_cmd_buffer *cmd_buffer, unsigned VertexCountPerInstance, unsigned StartVertex, unsigned instanceCount, unsigned StartInstance, unsigned BaseVertex);
extern void gpgpusim_saveIndexBuffer(struct anv_buffer *ptr, VkIndexType type);

extern void gpgpusim_setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
extern void gpgpusim_setDescriptorSet(struct anv_descriptor_set *set, unsigned set_index);

// For trace runner
extern void gpgpusim_addTreelets_cpp(VkAccelerationStructureKHR accelerationStructure);
extern uint32_t gpgpusim_registerShader_cpp(char * shaderPath, uint32_t shader_type);

extern void gpgpusim_vkCmdTraceRaysKHR_cpp(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);

extern void gpgpusim_setDescriptorSet_cpp(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);

extern void gpgpusim_pass_child_addr(void* address);
extern void gpgpusim_map_pipeline_shader(void *ptr, unsigned shader_index);
extern void gpgpusim_map_pipeline_info(void *ptr, VkGraphicsPipelineCreateInfo *pCreateInfo);
#endif /* GPGPUSIM_CALLS_FROM_MESA_H */
