// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
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

#include "mem_fetch.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"

unsigned mem_fetch::sm_next_mf_request_uid = 1;

mem_fetch::mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
                     unsigned ctrl_size, unsigned wid, unsigned sid,
                     unsigned tpc, const memory_config *config,
                     unsigned long long cycle, unsigned kernel_uid,
                     mem_fetch *m_original_mf, mem_fetch *m_original_wr_mf)
    : m_access(access)

{
  m_request_uid = sm_next_mf_request_uid++;
  m_access = access;
  if (inst) {
    m_inst = *inst;
    assert(wid == m_inst.warp_id());
  }
  m_data_size = access.get_size();
  m_ctrl_size = ctrl_size;
  m_sid = sid;
  m_tpc = tpc;
  m_wid = wid;
  config->m_address_mapping.addrdec_tlx(access.get_addr(), &m_raw_addr);

        
  if (config->m_shader_config->gpgpu_concurrent_mig && inst) {
    const gpgpu_sim *gpu = config->get_gpgpu_sim();
    float dynamic_ratio = (float)gpu->dynamic_sm_count / gpu->concurrent_granularity;
    unsigned sub_partition = m_raw_addr.sub_partition;
    if (is_graphics()) {
      sub_partition = sub_partition * dynamic_ratio;
    } else {
      unsigned avail_sm =
          config->m_shader_config->num_shader() * (1.0f - dynamic_ratio);
      unsigned start = config->m_n_mem_sub_partition * dynamic_ratio;
      sub_partition = start + sub_partition * avail_sm /
                                  config->m_shader_config->num_shader();
    }
    unsigned chip =
        sub_partition / config->m_n_sub_partition_per_memory_channel;
    assert(chip < config->m_n_mem);
    assert(sub_partition < config->m_n_mem_sub_partition);
    m_raw_addr.chip = chip;
    m_raw_addr.sub_partition = sub_partition;
  }
  m_partition_addr =
      config->m_address_mapping.partition_address(access.get_addr());
  m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
  m_timestamp = cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = cycle;
  m_mem_config = config;
  icnt_flit_size = config->icnt_flit_size;
  m_kernel_uid = kernel_uid;
  original_mf = m_original_mf;
  original_wr_mf = m_original_wr_mf;
  if (m_original_mf) {
    m_raw_addr.chip = m_original_mf->get_tlx_addr().chip;
    m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;
  }
}

mem_fetch::~mem_fetch() { m_status = MEM_FETCH_DELETED; }

#define MF_TUP_BEGIN(X) static const char *Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) \
  }                   \
  ;
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

void mem_fetch::print(FILE *fp, bool print_inst) const {
  if (this == NULL) {
    fprintf(fp, " <NULL mem_fetch pointer>\n");
    return;
  }
  fprintf(fp, "  mf: uid=%6u, sid%02u:w%02u, part=%u, ", m_request_uid, m_sid,
          m_wid, m_raw_addr.chip);
  m_access.print(fp);
  if ((unsigned)m_status < NUM_MEM_REQ_STAT)
    fprintf(fp, " status = %s (%llu), ", Status_str[m_status], m_status_change);
  else
    fprintf(fp, " status = %u??? (%llu), ", m_status, m_status_change);
  if (!m_inst.empty() && print_inst)
    m_inst.print(fp);
  else
    fprintf(fp, "\n");
}

void mem_fetch::set_status(enum mem_fetch_status status,
                           unsigned long long cycle) {
  m_status = status;
  m_status_change = cycle;
}

bool mem_fetch::isatomic() const {
  if (m_inst.empty()) return false;
  return m_inst.isatomic();
}

void mem_fetch::do_atomic() { m_inst.do_atomic(m_access.get_warp_mask()); }

bool mem_fetch::istexture() const {
  if (m_inst.empty()) return false;
  return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const {
  if (m_inst.empty()) return false;
  return (m_inst.space.get_type() == const_space) ||
         (m_inst.space.get_type() == param_space_kernel);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the
/// direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem) {
  unsigned sz = 0;
  // If atomic, write going to memory, or read coming back from memory, size =
  // ctrl + data. Else, only ctrl
  if (isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()))
    sz = size();
  else
    sz = get_ctrl_size();

  return (sz / icnt_flit_size) + ((sz % icnt_flit_size) ? 1 : 0);
}